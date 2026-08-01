"""Preprocessing pipeline (Phase 3).

Critical contract: the ``Preprocessor`` is **fit on UNSW-train only**, then applied unchanged
to UNSW-test and to TON_IoT. Never refit on the target set or you leak future information into
the cross-era measurement.

Two things make "fit on UNSW-train only" harder than it reads, and both fail *silently*:

* ``unsw_common.parquet`` is the **concatenation** of the two delivered UNSW partitions
  (train 175,341 + test 82,332 = 257,673 rows), tagged by the ``split`` column -- see
  ``schema_map.build_common_frames``. Fitting on the whole frame quietly folds UNSW-*test*
  statistics into every mean, std and one-hot vocabulary and raises nothing. :func:`load_source`
  filters to ``split == "train"`` and asserts the row count, and :meth:`Preprocessor.fit` refuses
  a frame carrying any other split, so the mistake becomes an exception rather than a number.
* The stratified train/validation split is drawn *after* that filter, and the fit sees the
  **train fold only** -- so validation statistics do not leak into standardization either.

Order of numeric operations is load-bearing: impute (raw-space train-fold median) -> ``log1p``
-> z-score. Standardizing in log space is the point; z-scoring the raw heavy tails first would
leave a scaler that maps almost all of TON_IoT into one bin (``schema_map`` FEATURE_MAP notes on
``flow_duration``).
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .config import (
    PREPROCESSOR,
    RANDOM_SEED,
    TONIOT_COMMON,
    UNSW_COMMON,
    set_seeds,
)
from .schema_map import (
    BINARY_LABEL_COL,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RARE_BUCKET,
    ZERO_DURATION_FLAG,
)

# --- Column sets, all derived from schema_map -------------------------------------------
# Nothing below spells a column name of its own: the feature sets are schema_map's constants,
# so a Phase 2 change to the shared subspace propagates here instead of silently disagreeing.

# The harmonized frames carry one binary label column, spelled the same on both sides. Derived
# rather than written out so a future per-side rename raises here instead of KeyError-ing deep
# inside a split.
_LABEL_COLS = set(BINARY_LABEL_COL.values())
if len(_LABEL_COLS) != 1:  # pragma: no cover - guards a schema_map edit, not a runtime path
    raise ValueError(
        "schema_map.BINARY_LABEL_COL no longer names a single harmonized label column: "
        f"{sorted(_LABEL_COLS)}. Phase 3 assumes one shared binary label."
    )
LABEL_COL: str = _LABEL_COLS.pop()

# ``zero_duration`` is a 0/1 indicator, not a measurement: it is passed through untouched rather
# than z-scored, so it stays readable in coefficients and stays trivially ablatable in Phase 6
# (its meaning inverts across eras -- see schema_map's ZERO_DURATION_FLAG note).
PASSTHROUGH_FEATURES: tuple[str, ...] = (ZERO_DURATION_FLAG,)

# Every column in NUMERIC_FEATURES is a non-negative count, duration or rate with a multi-decade
# heavy tail (UNSW `src_bytes` spans 28 B .. 1.30e7; `bytes_per_sec` spans 0 .. 1.50e9), so log1p
# applies to all of them -- `log1p` is defined and finite on the whole support, and fit() asserts
# non-negativity rather than trusting it. This is a superset of the "byte/packet counts" the plan
# names: `flow_duration` is included because schema_map's FEATURE_MAP entry asks for it by name
# ("log1p, then clip/winsorize the TON_IoT tail before z-scoring"), and the two DERIVED_FEATURES
# rates are included because they are quotients of columns that are themselves log-scaled.
LOG1P_FEATURES: tuple[str, ...] = NUMERIC_FEATURES

# The `split` tags written by schema_map.build_common_frames(). The UNSW pair is spelled there as
# string literals; they are repeated once here and then *validated* against the delivered frame in
# load_source(), so a divergence raises instead of filtering to an empty (or over-full) frame.
SOURCE_SPLIT: str = "train"
HOLDOUT_SPLIT: str = "test"

# data/README.md, verified 2026-07-29. The whole point of asserting it: filtering `split` wrong
# yields 257,673 rows and no error at all, and every statistic downstream is then contaminated.
EXPECTED_SOURCE_ROWS: int = 175_341

VALIDATION_FRACTION: float = 0.20


class Preprocessor:
    """Fit-on-source preprocessor: one-hot categoricals, z-score numerics, impute, log-transform.

    Numeric standardization and categorical vocabularies are learned from UNSW-train statistics
    and frozen; ``transform`` applies those frozen parameters to any later frame.

    Concretely, ``fit`` learns and freezes four things from the UNSW train fold:

    * ``impute_values_`` -- raw-space median per numeric column, used for missing values only.
      ``-`` in ``service`` is *not* missing: schema_map keeps it as a real level ("Zeek detected
      no application protocol"), it is modal on both sides, and it is one-hot encoded like any
      other level.
    * ``means_`` / ``scales_`` -- z-score parameters computed **after** ``log1p``, so
      standardization lives in log space. A zero-variance column takes scale 1.0 and comes out
      as a constant 0 rather than a NaN.
    * ``encoder_`` -- a ``OneHotEncoder`` with a fixed vocabulary and ``handle_unknown="ignore"``.
      A level absent from the fit (UNSW-test's train-only/test-only ``state`` codes, or any
      TON_IoT level the source era never showed) encodes as all-zeros: no error, and critically
      no extra column, so UNSW and TON_IoT transform to the identical matrix width and order that
      Phase 6's zero-shot comparison requires.
    * ``feature_names_`` -- the frozen output column order.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self.numeric_features_: tuple[str, ...] = tuple(NUMERIC_FEATURES)
        self.categorical_features_: tuple[str, ...] = tuple(CATEGORICAL_FEATURES)
        self.passthrough_features_: tuple[str, ...] = tuple(PASSTHROUGH_FEATURES)
        self.log1p_features_: tuple[str, ...] = tuple(LOG1P_FEATURES)
        self.impute_values_: dict[str, float] = {}
        self.means_: np.ndarray | None = None
        self.scales_: np.ndarray | None = None
        self.encoder_: OneHotEncoder | None = None
        self.feature_names_: tuple[str, ...] = ()

    # --- Fit ----------------------------------------------------------------------------

    def fit(self, X_source: Any) -> "Preprocessor":
        """Learn scaling/encoding parameters from the source (UNSW-train) frame only."""
        frame = self._require_columns(X_source)
        self._reject_non_source_rows(frame)

        raw = self._raw_numeric(frame)
        if np.isnan(raw).any():
            # Median over the observed values; a fully-missing column would give NaN, which the
            # check below turns into an error rather than an all-NaN feature.
            medians = np.nanmedian(raw, axis=0)
        else:
            medians = np.median(raw, axis=0)
        if not np.isfinite(medians).all():
            bad = [c for c, m in zip(self.numeric_features_, medians) if not np.isfinite(m)]
            raise ValueError(f"no finite median for numeric column(s) {bad} -- cannot impute")
        self.impute_values_ = {
            col: float(value) for col, value in zip(self.numeric_features_, medians)
        }

        numeric = self._numeric_block(frame)
        self.means_ = numeric.mean(axis=0)
        deviations = numeric.std(axis=0, ddof=0)
        # A constant column (scale 0) would divide to NaN/inf. Scale 1.0 leaves it at a constant
        # 0 after centering -- inert, but not a NaN detonating in every downstream model.
        self.scales_ = np.where(deviations > 0.0, deviations, 1.0)

        self.encoder_ = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float64
        )
        self.encoder_.fit(self._categorical_block(frame))

        self.feature_names_ = (
            *self.numeric_features_,
            *(str(name) for name in self.encoder_.get_feature_names_out(
                self.categorical_features_
            )),
            *self.passthrough_features_,
        )
        self._fitted = True
        return self

    def transform(self, X: Any) -> Any:
        """Apply the frozen fit parameters to any frame (UNSW-test or TON_IoT)."""
        if not self._fitted:
            raise RuntimeError("Preprocessor.transform called before fit")
        frame = self._require_columns(X)

        numeric = (self._numeric_block(frame) - self.means_) / self.scales_
        categorical = self.encoder_.transform(self._categorical_block(frame))
        passthrough = np.column_stack(
            [
                pd.to_numeric(frame[col], errors="raise").to_numpy(dtype="float64")
                for col in self.passthrough_features_
            ]
        ) if self.passthrough_features_ else np.empty((len(frame), 0), dtype="float64")

        matrix = np.hstack([numeric, categorical, passthrough])
        if matrix.shape[1] != len(self.feature_names_):
            raise RuntimeError(
                f"transform produced {matrix.shape[1]} columns, fit froze "
                f"{len(self.feature_names_)}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("transform produced non-finite values -- impute/scale is broken")
        return pd.DataFrame(matrix, columns=list(self.feature_names_), index=frame.index)

    def fit_transform(self, X_source: Any) -> Any:
        return self.fit(X_source).transform(X_source)

    # --- Persistence --------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the fitted preprocessor for reproducibility."""
        if not self._fitted:
            raise RuntimeError("refusing to serialize an unfitted Preprocessor")
        import joblib

        from pathlib import Path as _Path

        destination = _Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str) -> "Preprocessor":
        import joblib

        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not hold a Preprocessor (got {type(obj).__name__})")
        if not obj._fitted:
            raise ValueError(f"{path} holds an unfitted Preprocessor")
        return obj

    # --- Internals ----------------------------------------------------------------------

    def _required_columns(self) -> tuple[str, ...]:
        return (
            *self.numeric_features_,
            *self.categorical_features_,
            *self.passthrough_features_,
        )

    def _require_columns(self, X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"expected a pandas DataFrame, got {type(X).__name__}")
        missing = [col for col in self._required_columns() if col not in X.columns]
        if missing:
            raise ValueError(f"frame is missing shared-subspace column(s) {missing}")
        return X

    @staticmethod
    def _reject_non_source_rows(frame: pd.DataFrame) -> None:
        """Refuse to fit on anything but UNSW-train rows.

        This is the guard for the failure the module docstring opens with: fitting on the
        concatenated UNSW frame, or on TON_IoT, produces perfectly plausible statistics and no
        error whatsoever. Turn it into an exception.
        """
        if "split" not in frame.columns:
            return
        splits = sorted(set(frame["split"].dropna().astype(str).unique()))
        if splits != [SOURCE_SPLIT]:
            raise ValueError(
                f"fit() received rows with split(s) {splits}; the Preprocessor is fit on "
                f"{SOURCE_SPLIT!r} rows only. Filter before fitting -- fitting on the "
                "concatenated frame leaks held-out statistics silently."
            )

    def _raw_numeric(self, frame: pd.DataFrame) -> np.ndarray:
        """Numeric columns as float64, un-imputed and un-transformed."""
        return np.column_stack(
            [
                pd.to_numeric(frame[col], errors="raise").to_numpy(dtype="float64")
                for col in self.numeric_features_
            ]
        )

    def _numeric_block(self, frame: pd.DataFrame) -> np.ndarray:
        """Impute (raw-space train-fold median) -> log1p. Standardization is applied by caller."""
        block = self._raw_numeric(frame)
        if self.impute_values_:
            fill = np.array(
                [self.impute_values_[col] for col in self.numeric_features_], dtype="float64"
            )
            block = np.where(np.isnan(block), fill, block)
        if np.isnan(block).any():
            raise ValueError("numeric block still holds NaN after imputation")

        log_mask = np.array(
            [col in set(self.log1p_features_) for col in self.numeric_features_], dtype=bool
        )
        if log_mask.any():
            selected = block[:, log_mask]
            if (selected < 0.0).any():
                negative = [
                    col
                    for col, keep, minimum in zip(
                        self.numeric_features_, log_mask, block.min(axis=0)
                    )
                    if keep and minimum < 0.0
                ]
                raise ValueError(
                    f"log1p requested for column(s) {negative} holding negative values; "
                    "every shared-subspace numeric is documented non-negative"
                )
            block = block.copy()
            block[:, log_mask] = np.log1p(selected)
        if not np.isfinite(block).all():
            raise ValueError("numeric block is non-finite after log1p")
        return block

    def _categorical_block(self, frame: pd.DataFrame) -> np.ndarray:
        """Categoricals as a plain string array.

        Missing values become ``RARE_BUCKET`` -- the same bucket schema_map already uses for
        out-of-vocabulary levels. Note this does **not** touch ``-`` in ``service``: that is a
        delivered level meaning "no application protocol detected", not a missing value.
        """
        columns = [
            frame[col].astype("string").fillna(RARE_BUCKET).astype(str)
            for col in self.categorical_features_
        ]
        return np.column_stack([col.to_numpy() for col in columns])


# --- Data loading + splitting ------------------------------------------------------------


def load_source() -> pd.DataFrame:
    """Load ``unsw_common.parquet`` and return the UNSW-**train** rows only.

    The delivered parquet is train+test concatenated (257,673 rows). Everything downstream of a
    missed filter here is contaminated and nothing raises, so this asserts both the vocabulary of
    the ``split`` column and the resulting row count.
    """
    frame = pd.read_parquet(UNSW_COMMON)
    splits = sorted(set(frame["split"].dropna().astype(str).unique()))
    if splits != sorted({SOURCE_SPLIT, HOLDOUT_SPLIT}):
        raise ValueError(
            f"{UNSW_COMMON} carries split values {splits}; expected "
            f"{sorted({SOURCE_SPLIT, HOLDOUT_SPLIT})}. Rebuild with `python -m src.schema_map "
            "--build`."
        )
    source = frame.loc[frame["split"].astype(str) == SOURCE_SPLIT].reset_index(drop=True)
    if len(source) != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"UNSW split=={SOURCE_SPLIT!r} yielded {len(source):,} rows, expected "
            f"{EXPECTED_SOURCE_ROWS:,} (data/README.md). Refusing to fit on an unexpected frame."
        )
    return source


def load_holdout() -> pd.DataFrame:
    """Load the held-out UNSW-test partition. Transform-only -- never fit on this."""
    frame = pd.read_parquet(UNSW_COMMON)
    return frame.loc[frame["split"].astype(str) == HOLDOUT_SPLIT].reset_index(drop=True)


def load_target() -> pd.DataFrame:
    """Load the TON_IoT frame. Transform-only -- never fit on this, that is the whole RQ1."""
    return pd.read_parquet(TONIOT_COMMON)


def split_source(
    source: pd.DataFrame, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified ~80/20 train/validation split of UNSW-train on the binary label."""
    train_fold, val_fold = train_test_split(
        source,
        test_size=VALIDATION_FRACTION,
        random_state=seed,
        stratify=source[LABEL_COL],
        shuffle=True,
    )
    return train_fold.reset_index(drop=True), val_fold.reset_index(drop=True)


def fit_preprocessor(seed: int = RANDOM_SEED) -> tuple[Preprocessor, dict[str, pd.DataFrame]]:
    """Phase 3 end to end: split UNSW-train, fit on the train fold, transform the other three.

    Returns the fitted ``Preprocessor`` and the four *raw* frames it was applied to, keyed by
    the names Phases 4-7 use. The fit touches ``train`` and nothing else.
    """
    source = load_source()
    train_fold, val_fold = split_source(source, seed=seed)

    preprocessor = Preprocessor().fit(train_fold)

    frames = {
        "train": train_fold,
        "val": val_fold,
        "unsw_test": load_holdout(),
        "toniot": load_target(),
    }
    return preprocessor, frames


# --- Entry point --------------------------------------------------------------------------


def _report(preprocessor: Preprocessor, frames: dict[str, pd.DataFrame]) -> None:
    """One-screen Phase 3 report + the invariants Phases 4-7 silently depend on."""
    print(f"fitted feature schema: {len(preprocessor.feature_names_)} columns")
    print(f"    numeric (log1p -> z-score): {len(preprocessor.numeric_features_)}")
    n_onehot = (
        len(preprocessor.feature_names_)
        - len(preprocessor.numeric_features_)
        - len(preprocessor.passthrough_features_)
    )
    print(f"    one-hot                   : {n_onehot}")
    print(f"    passthrough               : {len(preprocessor.passthrough_features_)}")
    for column, levels in zip(
        preprocessor.categorical_features_, preprocessor.encoder_.categories_
    ):
        print(f"        {column}: {list(levels)}")

    reference: list[str] | None = None
    for name, frame in frames.items():
        matrix = preprocessor.transform(frame)
        columns = list(matrix.columns)
        if reference is None:
            reference = columns
        elif columns != reference:
            raise RuntimeError(f"{name}: feature columns diverged from the train fold")
        finite = bool(np.isfinite(matrix.to_numpy()).all())
        normal = float((frame[LABEL_COL] == 0).mean())
        print(
            f"    {name:<10} n={len(frame):>9,}  cols={matrix.shape[1]:>3}  "
            f"normal={normal:6.2%}  finite={finite}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.preprocess",
        description=(
            "Phase 3: fit the Preprocessor on the UNSW-train fold only and serialize it."
        ),
    )
    parser.parse_args(argv)

    set_seeds()

    preprocessor, frames = fit_preprocessor()
    _report(preprocessor, frames)

    preprocessor.save(str(PREPROCESSOR))
    print(f"fitted preprocessor -> {PREPROCESSOR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
