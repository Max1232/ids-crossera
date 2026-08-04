"""Evaluation — metrics and the two regimes (Phase 6, RQ1).

Lead with F1 and ROC-AUC (accuracy is misleading under imbalance), plus precision/recall,
confusion matrices, and per-shared-family breakdowns. Headline number = the delta between the
in-distribution and cross-era (zero-shot) regimes. All runs are logged to reports/metrics.csv.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import METRICS_CSV, PREPROCESSOR, RANDOM_SEED, set_seeds

# The ten original columns are FROZEN — never reorder or rename them. `log_metrics()` rewrites the
# whole file against this header on every call but carries already-logged rows forward *by column
# name*, so adding a column is safe while renaming or reordering one silently rewrites history:
# every prior row would be re-emitted with the renamed field blank. The four appended columns
# landed in Phase 2 (before Phase 4 logs anything) for exactly that reason:
#   balanced_accuracy, macro_f1  -- prevalence-robust cross-checks Phase 6 needs, because the two
#                                   test sets do not share a class balance (45% vs 24% normal).
#   n_test, positive_rate        -- record the class balance each row was measured against, so a
#                                   delta can never be read without the prevalence it was taken at.
METRICS_HEADER = [
    "run_id", "model", "regime", "seed",
    "accuracy", "precision", "recall", "f1", "roc_auc", "notes",
    "balanced_accuracy", "macro_f1", "n_test", "positive_rate",
]


#: The binary label encoding both harmonized frames carry (schema_map: normal 0 / attack 1). Every
#: prevalence-bearing number below -- ``precision``, ``recall``, ``f1``, ``positive_rate`` -- is
#: reported for the **attack** class, so a "positive rate" in metrics.csv is an attack share.
POSITIVE_LABEL: int = 1

#: The metrics a Phase 6 delta is taken over, ordered as the headline reads them: ROC-AUC first
#: (prevalence-insensitive, so it is the one the drift claim leads with), F1 second (promised by
#: the proposal, but part of any F1 delta is the 55% -> 76% attack-share change rather than
#: drift), then the two prevalence-robust cross-checks, then accuracy -- which is logged for
#: completeness and is never the headline under this imbalance.
DELTA_METRICS: tuple[str, ...] = (
    "roc_auc", "f1", "balanced_accuracy", "macro_f1", "precision", "recall", "accuracy",
)


def positive_scores(model: Any, X: Any) -> np.ndarray:
    """Continuous positive-class score for ROC-AUC, whichever interface the estimator exposes.

    ``predict_proba`` where it exists; otherwise ``decision_function``. This is what keeps the
    linear SVM honest without dragging in probability calibration: ``roc_auc_score`` only needs a
    *ranking*, and a signed margin ranks identically to any monotone calibration of it, so
    wrapping ``LinearSVC`` in ``CalibratedClassifierCV`` would buy a k-fold refit and change no
    AUC (see ``models.baselines.make_svm``).
    """
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        classes = list(getattr(model, "classes_", [0, 1]))
        if POSITIVE_LABEL not in classes:
            raise ValueError(
                f"estimator was fit on classes {classes}; no positive label {POSITIVE_LABEL} "
                "column to score. Refusing to guess which column is the attack class."
            )
        return proba[:, classes.index(POSITIVE_LABEL)]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).ravel()
    raise TypeError(
        f"{type(model).__name__} exposes neither predict_proba nor decision_function, so no "
        "ROC-AUC is computable. Every model in this project must offer one of the two."
    )


def evaluate(model: Any, X: Any, y: Any) -> dict[str, Any]:
    """Compute accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix.

    Keys are spelled to match :data:`METRICS_HEADER`, so the result can be handed straight to
    :func:`log_metrics` (plus ``confusion_matrix``, which the CSV has no column for and Phase 6
    renders separately).

    ``zero_division=0`` on precision/recall/F1 is there for the Dummy floor specifically: a
    ``most_frequent`` classifier that never predicts the positive class has no defined precision,
    and the honest reading of that is 0 -- not a crash, and not the 1.0 sklearn's warning path
    might suggest. The floor has to be *loggable* for the "every real model clears it" check to
    mean anything.

    Implemented in Phase 4 rather than Phase 6 for the same reason ``METRICS_HEADER`` was extended
    in Phase 2: Phase 4 logs the first rows, so it needs the metric function that produces them,
    and a private duplicate in ``models/baselines.py`` would be free to drift away from whatever
    Phase 6 later computed. :func:`run_regimes` is still Phase 6's.
    """
    y_true = np.asarray(y).ravel()
    y_pred = np.asarray(model.predict(X)).ravel()
    scores = positive_scores(model, X)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_test": int(y_true.shape[0]),
        "positive_rate": float((y_true == POSITIVE_LABEL).mean()),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, POSITIVE_LABEL]).tolist(),
    }


class LeakageError(RuntimeError):
    """A fit was attempted against a set that may only ever be transformed and scored.

    Phase 6's entire claim is that **one** UNSW-train-fitted model is carried unchanged onto
    UNSW-test and onto TON_IoT. Nothing in Python enforces that, and the failure is silent: a
    refit on the target returns better numbers and raises nothing, so the drift measurement would
    quietly become a measurement of how well a model fits the era it was just trained on. The
    objects that *could* be refit are therefore sealed for the duration of the measurement (see
    :func:`sealed`) and raise this instead.
    """


#: Method names :func:`sealed` shadows. ``fit_transform`` is here because
#: ``Preprocessor.fit_transform`` refits, and ``partial_fit`` because it is the incremental
#: spelling of the identical mistake (it is what an ``SGDClassifier`` would offer, were the linear
#: SVM ever swapped for one -- see ``models.baselines.make_svm``).
FIT_METHODS: tuple[str, ...] = ("fit", "partial_fit", "fit_transform")


@contextlib.contextmanager
def sealed(*objects: Any, reason: str) -> Iterator[None]:
    """Make every name in :data:`FIT_METHODS` raise :class:`LeakageError` on ``objects``.

    A structural guard, not a comment: the instance's ``__dict__`` shadows the bound class method
    for the duration of the block and the original is restored on exit (including on an
    exception). Read-only use -- ``predict``, ``predict_proba``, ``decision_function``,
    ``transform`` -- is untouched, which is exactly the set of operations a *measurement* needs.

    Used on two things, for the two ways this phase could leak:

    * the **model**, for the span of both :func:`evaluate` calls in :func:`run_regimes`, so a
      refit against UNSW-test or TON_IoT is impossible rather than merely absent;
    * the **Preprocessor**, for the whole of :func:`run_phase6`, because the drift number is only
      meaningful if the target era is transformed through the frozen Phase 3 artifact.

    Fitting the models on the UNSW **train fold** is expected and correct -- ``baselines`` and the
    scratch models persist nothing, so every run re-fits from the factories -- and that fit
    happens *outside* these blocks, before the seal is applied.
    """
    restore: list[tuple[Any, str, bool, Any]] = []

    def _forbid(owner: Any, method: str) -> Callable[..., Any]:
        def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
            raise LeakageError(
                f"{type(owner).__name__}.{method}() was called inside a sealed block: {reason}"
            )

        return _forbidden

    try:
        for obj in objects:
            namespace = getattr(obj, "__dict__", None)
            if namespace is None:  # pragma: no cover - no __slots__ object reaches this
                raise TypeError(
                    f"cannot seal {type(obj).__name__}: it has no instance __dict__, so its fit "
                    "methods cannot be shadowed. Refusing to run unguarded."
                )
            for method in FIT_METHODS:
                if not hasattr(obj, method):
                    continue
                restore.append((obj, method, method in namespace, namespace.get(method)))
                setattr(obj, method, _forbid(obj, method))
        yield
    finally:
        for obj, method, existed, previous in reversed(restore):
            if existed:
                setattr(obj, method, previous)
            else:
                obj.__dict__.pop(method, None)


def run_regimes(model: Any, X_unsw_test: Any, y_unsw_test: Any,
                X_toniot: Any, y_toniot: Any) -> dict[str, dict[str, Any]]:
    """Evaluate one trained model in both regimes side by side.

    Returns ``{"in_distribution": {...}, "cross_era": {...}}``. The reported drift is the
    per-metric delta (in_distribution - cross_era).

    ``model`` arrives **already fitted on the UNSW train fold** and is sealed here (see
    :func:`sealed`) for the span of both evaluations, so neither regime can refit it. That is the
    whole experiment: ``in_distribution`` is that model on UNSW-test, ``cross_era`` is the *same*
    object on TON_IoT with no retraining, no warm start and no refit of the ``Preprocessor`` --
    the target era is transform-only through the Phase 3 artifact. Phase 7 is where target labels
    are allowed to touch a model, and it does so under its own ``run_id``.

    Every returned row already carries ``n_test`` and ``positive_rate`` (the *attack* share, see
    :data:`POSITIVE_LABEL`) for the set it was measured on, because the two sets do not share a
    class balance -- UNSW-test is 55.06% attack, TON_IoT 76.31% -- and no delta between them can
    be read without both. Lead the drift claim with ``roc_auc``, which is insensitive to that
    difference; report ``f1`` beside it with the prevalence caveat attached; treat
    ``balanced_accuracy`` and ``macro_f1`` as the supplementary cross-checks. Never accuracy.

    **``run_id`` scheme** (this function does not log -- :func:`run_phase6` does -- but the pair it
    returns is what a ``run_id`` names, so the convention belongs with it). Per
    :func:`log_metrics`, rows key on ``(run_id, model, regime)``, so one ``run_id`` per
    experimental *condition*, and both halves of a delta live under the same one::

        phase4-baselines            # Phase 4 owns the four baselines' in_distribution rows
        phase6-crossera             # Phase 6: cross_era for all six models, plus the two
                                    #   scratch models' in_distribution rows (new -- Phase 5
                                    #   logged nothing). The baselines' in_distribution halves
                                    #   are NOT re-logged here; they are phase4-baselines'.
        phase6-crossera-no_proto    # the proto ablation -- a matched in_distribution +
                                    #   cross_era pair for all six models, retrained on the
                                    #   train fold without the protocol one-hots, so the
                                    #   quantity of interest is the difference of the deltas
        phase7-recovery-f0.05       # Phase 7 extends the same scheme: one run_id per budget
        phase7-recovery-f0.25

    A condition that changes what was measured without changing ``model`` or ``regime`` must take
    a new ``run_id`` or the upsert silently overwrites the row it should sit beside.
    """
    with sealed(
        model,
        reason=(
            "Phase 6 scores a UNSW-train-fitted model on UNSW-test and TON_IoT unchanged. "
            "Fitting against either set is the leakage that would make the RQ1 delta meaningless"
        ),
    ):
        return {
            "in_distribution": evaluate(model, X_unsw_test, y_unsw_test),
            "cross_era": evaluate(model, X_toniot, y_toniot),
        }


def metric_deltas(regimes: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Per-metric ``in_distribution - cross_era`` — the headline quantity of RQ1.

    Only the float metrics are differenced. ``n_test`` and ``positive_rate`` describe the two sets
    rather than the model, so subtracting them would be meaningless; they are reported *alongside*
    every delta instead, which is why they have columns in :data:`METRICS_HEADER`.
    """
    in_distribution, cross_era = regimes["in_distribution"], regimes["cross_era"]
    return {
        metric: float(in_distribution[metric]) - float(cross_era[metric])
        for metric in DELTA_METRICS
    }


#: Decimal places metric floats are rounded to on the way into ``reports/metrics.csv``. Six is far
#: past anything reportable (the report quotes three) and past any delta that could be argued about,
#: but it keeps the committed run log readable and diffable instead of a wall of float64 repr --
#: which matters because that file *is* the run log and a re-run's reproducibility is checked by
#: diffing it.
METRIC_DECIMALS: int = 6


def round_metrics(row: dict[str, Any], decimals: int = METRIC_DECIMALS) -> dict[str, Any]:
    """Round the float metrics in a row for logging. Ints and strings pass through untouched.

    Applied by every phase at the ``log_metrics`` boundary so rows from Phase 4 and Phase 6 are
    formatted identically and their delta can be read off the CSV directly.
    """
    return {
        key: round(value, decimals) if isinstance(value, float) else value
        for key, value in row.items()
    }


#: The identity of a logged row: one row per ``(run_id, model, regime)``. :func:`log_metrics`
#: upserts on this key, which is what makes ``run.sh`` idempotent — see the docstring below for the
#: convention that keeps distinct experimental conditions from colliding on it.
METRICS_KEY: tuple[str, ...] = ("run_id", "model", "regime")

#: Line terminator for ``reports/metrics.csv``. Pinned to ``\n`` rather than ``csv``'s default
#: ``\r\n`` because this file is committed and its whole purpose is to be diffable: a re-run must
#: produce a byte-identical file, and mixed endings make ``git diff`` unreadable.
METRICS_LINETERMINATOR: str = "\n"


def read_metrics(path: Any = METRICS_CSV) -> dict[tuple[str, ...], dict[str, str]]:
    """Read ``reports/metrics.csv`` into ``{(run_id, model, regime): row}``.

    Values come back as strings, exactly as they were written, so a row that is carried forward
    untouched is re-emitted byte-for-byte. A missing or empty file is an empty log, not an error —
    that is the first-ever-run case.

    Raises if the file's header is not :data:`METRICS_HEADER`. A stale header means the frozen
    column set changed under an existing log, and silently reconciling that is how every row in the
    file gets quietly misaligned; refuse instead.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != METRICS_HEADER:
            raise ValueError(
                f"{path} has header {reader.fieldnames}, expected {METRICS_HEADER}. The metrics "
                "header is frozen; reconcile the file by hand rather than letting a run rewrite "
                "it, or every existing row silently misaligns."
            )
        return {
            tuple(record.get(key) or "" for key in METRICS_KEY): {
                column: (record.get(column) or "") for column in METRICS_HEADER
            }
            for record in reader
        }


def log_metrics(row: dict[str, Any], path: Any = METRICS_CSV) -> None:
    """Upsert one run's metrics into ``reports/metrics.csv``, keyed on ``(run_id, model, regime)``.

    An **upsert, not an append**: the existing log is read, the incoming row replaces any row
    carrying the same :data:`METRICS_KEY` (and is inserted otherwise), and the whole file is
    rewritten under :data:`METRICS_HEADER` sorted by that key. So ``./run.sh`` is idempotent —
    running it twice leaves ``reports/metrics.csv`` byte-identical instead of doubling every row,
    which is what makes "did this reproduce?" a `git diff` and keeps the committed run log from
    growing a duplicate block per re-run. Rows belonging to other keys are carried forward
    unmodified.

    **Convention — one ``run_id`` per experimental condition.** ``model`` and ``regime`` alone do
    not identify a run, so anything that changes what was measured without changing those two must
    encode itself into ``run_id`` or it will *overwrite* the row it should sit beside. Ablations and
    transfer fractions are exactly that case::

        "phase4-baselines"                  # Phase 4, in-distribution ceiling
        "phase6-crossera"                   # Phase 6, full feature set
        "phase6-crossera-no_proto"          # same model + regime, `proto` ablated
        "phase7-recovery-f0.05"             # same model + regime, 5% fine-tune budget
        "phase7-recovery-f0.25"             # ... and 25%

    Without the suffixes, every ``proto`` ablation would land on its unablated row and the recovery
    curve would collapse to whichever fraction ran last. The key is deliberately *not* a timestamp:
    a fixed label is what lets a re-run be diffed against the committed file at all.
    """
    path = Path(path)
    rows = read_metrics(path)
    incoming = {column: row.get(column, "") for column in METRICS_HEADER}
    rows[tuple(str(incoming[key]) for key in METRICS_KEY)] = incoming

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=METRICS_HEADER, lineterminator=METRICS_LINETERMINATOR
        )
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])


# --- Phase 6 — the zero-shot cross-era run (RQ1) -------------------------------------------
#
# One `run_id` per condition; see the scheme documented in `run_regimes`. The two conditions this
# phase owns:
#
#   phase6-crossera           full shared feature set (d=22)
#   phase6-crossera-no_proto  the `proto` ablation (d=18), retrained on the train fold
#
# WHY THE ABLATION IS A RETRAIN AND NOT A TEST-TIME MASK. README's Limitations measures the
# hazard: 18.31% of UNSW train rows use a protocol TON_IoT never contains, those rows are 91%
# attack, and both sides collapse to {tcp, udp, icmp, other} -- so a model can learn
# "`other` -> attack" from a bucket that is 91% attack in training and 0% of rows at test time,
# and part of the RQ1 drop would be that signal going inert rather than attacker evolution.
#
# Zeroing the `protocol` columns at test time on a model that was *trained* with them does not
# measure that. It evaluates a with-proto model on inputs no era ever produces -- an all-zero
# one-hot block is off the training manifold for reasons that have nothing to do with drift -- so
# the resulting drop confounds "the proto signal went inert" with "the model was perturbed". The
# defensible ablation removes the feature from the *hypothesis class*: retrain on the UNSW train
# fold with the protocol one-hots excluded (a train-fold-only refit, which is not leakage -- no
# model is persisted here and every condition re-fits from the factories anyway) and then run that
# model through both regimes. The quantity of interest is then the **difference of the deltas**:
# how much of the with-proto drop survives when `proto` was never available to be learned. That is
# why the ablation needs a matched in_distribution + cross_era pair under its own `run_id` -- a
# delta requires both halves, and an ablated cross_era row alone would be uncomparable to
# anything.
#
# Dropping the four `protocol_*` columns from the transformed matrix is *exactly* equivalent to
# refitting the Preprocessor without `protocol` -- one-hot encoding of one categorical column is
# independent of the others, and the numeric z-score parameters do not see it at all -- so it
# needs no second preprocessor artifact and cannot disturb the frozen Phase 3 fit.

#: Phase 6's two conditions. Fixed labels, never timestamps (see :func:`log_metrics`).
RUN_ID: str = "phase6-crossera"
ABLATION_RUN_ID: str = f"{RUN_ID}-no_proto"

#: The regime labels, spelled identically to ``models.baselines.REGIME`` so Phase 4's
#: in_distribution rows and Phase 6's cross_era rows join on the pair.
IN_DISTRIBUTION: str = "in_distribution"
CROSS_ERA: str = "cross_era"

#: The categorical column the ablation removes, as ``schema_map`` spells it. The one-hot columns
#: it expands to are derived from the fitted encoder rather than pattern-matched on a prefix, so a
#: renamed level cannot silently ablate nothing.
PROTOCOL_FEATURE: str = "protocol"

#: Row counts and class balances every Phase 6 frame is checked against, from ``data/README.md``
#: and the Phase 3 fold split (verified 2026-07-29 / 2026-08-01). These are asserted rather than
#: trusted because every way of getting them wrong is silent: an unfiltered UNSW frame, a re-drawn
#: fold boundary, or a TON_IoT frame that quietly lost rows all produce plausible numbers. The
#: TON_IoT shortfall (50,000 normal against a documented 300,000) is upstream's and is part of the
#: measurement -- see README's Limitations.
EXPECTED_ROWS: dict[str, tuple[int, int]] = {  # frame -> (n_rows, n_attack)
    "train": (140_272, 95_472),
    "unsw_test": (82_332, 45_332),
    "toniot": (211_043, 161_043),
}

#: Tolerance for the check that Phase 6's recomputed in-distribution baselines still match the
#: rows Phase 4 committed. Anything under this is float noise across BLAS/thread configurations;
#: anything over it means the two phases are no longer training the same model on the same fold,
#: which would make the delta a comparison between different experiments.
PHASE4_AGREEMENT_TOLERANCE: float = 1e-4


def load_preprocessor() -> Any:
    """Load the serialized Phase 3 ``Preprocessor``. Never fits one.

    ``preprocess.fit_preprocessor()`` would refit -- on the same fold, so the numbers would look
    right -- and that makes "was this the artifact the cross-era run used?" unanswerable. Phase 4
    and Phase 5 load it for the same reason.
    """
    from .preprocess import Preprocessor  # noqa: PLC0415 - keeps pandas out of the import graph

    if not PREPROCESSOR.exists():
        raise FileNotFoundError(
            f"{PREPROCESSOR} not found -- Phase 3 has not run. Run `python -m src.preprocess` "
            "first; Phase 6 loads that artifact and must never fit its own."
        )
    return Preprocessor.load(str(PREPROCESSOR))


def transform_regime_frames(preprocessor: Any) -> dict[str, tuple[Any, Any]]:
    """Transform the three frames Phase 6 needs: ``train``, ``unsw_test``, ``toniot``.

    Returns ``{name: (X, y)}``. ``train`` is the fold every model is fitted on; the other two are
    scored and nothing else. **TON_IoT is opened here for the first time in the pipeline** and is
    transform-only -- it goes through the frozen Phase 3 parameters (including the ``flow_duration``
    upper clip learned from the *source* era) exactly as UNSW-test does.

    The fold boundary is reproduced from ``config.RANDOM_SEED`` rather than re-drawn, which is why
    the seed is not a parameter, and every frame's row count and attack count are asserted against
    :data:`EXPECTED_ROWS` -- a mis-filtered ``split``, a re-drawn split, or a truncated target
    frame are all silent otherwise.
    """
    from .preprocess import (  # noqa: PLC0415 - keeps pandas out of the import graph
        LABEL_COL,
        load_holdout,
        load_source,
        load_target,
        split_source,
    )

    train_fold, _val_fold = split_source(load_source(), seed=RANDOM_SEED)
    raw = {"train": train_fold, "unsw_test": load_holdout(), "toniot": load_target()}

    frames: dict[str, tuple[Any, Any]] = {}
    reference: list[str] | None = None
    for name, frame in raw.items():
        y = frame[LABEL_COL].astype(int)
        expected = EXPECTED_ROWS[name]
        measured = (int(len(frame)), int((y == POSITIVE_LABEL).sum()))
        if measured != expected:
            raise ValueError(
                f"{name} frame is {measured[0]:,} rows / {measured[1]:,} attack, expected "
                f"{expected[0]:,} / {expected[1]:,} (data/README.md + the seeded fold split). "
                "Refusing to report a drift number measured against an unexpected frame."
            )
        X = preprocessor.transform(frame)
        if reference is None:
            reference = list(X.columns)
        elif list(X.columns) != reference:
            raise RuntimeError(
                f"{name} transformed to a different feature schema than the train fold. The "
                "zero-shot comparison requires one identical matrix width and column order."
            )
        frames[name] = (X, y)
    return frames


def protocol_columns(preprocessor: Any) -> tuple[str, ...]:
    """The transformed column names the ``protocol`` one-hot expands to, from the fitted encoder.

    Derived from ``encoder_.categories_`` positionally rather than by matching a ``protocol_``
    prefix, so the ablation cannot silently drop nothing (a renamed feature) or too much (a level
    that happens to share the prefix). Raises if the column set it computes is not actually in the
    frozen ``feature_names_``.
    """
    categoricals = list(preprocessor.categorical_features_)
    if PROTOCOL_FEATURE not in categoricals:
        raise ValueError(
            f"{PROTOCOL_FEATURE!r} is not among the Preprocessor's categorical features "
            f"{categoricals}; the proto ablation has nothing to remove."
        )
    names = [str(name) for name in
             preprocessor.encoder_.get_feature_names_out(tuple(categoricals))]
    widths = [len(levels) for levels in preprocessor.encoder_.categories_]
    index = categoricals.index(PROTOCOL_FEATURE)
    start = sum(widths[:index])
    columns = tuple(names[start:start + widths[index]])

    frozen = set(preprocessor.feature_names_)
    missing = [column for column in columns if column not in frozen]
    if missing or not columns:
        raise RuntimeError(
            f"derived protocol one-hot columns {columns} are not all in the fitted feature "
            f"schema (missing {missing}) -- the encoder and feature_names_ disagree."
        )
    return columns


def phase6_models() -> dict[str, Callable[[int], Any]]:
    """Every model from Phases 4-5, as ``name -> factory(n_features)``.

    The factories are the locked ones -- ``baselines.BASELINE_FACTORIES`` (tuned on the UNSW val
    fold in Phase 4) and the two scratch factories (locked in Phase 5) -- so Phase 6 re-instantiates
    identical estimators and **tunes nothing**. Re-tuning in-distribution would desync the rows
    Phase 4 already committed, and re-tuning cross-era would be target leakage of the plainest kind.

    ``n_features`` exists only for the MLP, whose input layer is dimensioned on the feature width:
    the ablation runs at d=18 rather than d=22, so its input layer is rebuilt while the *selected*
    hidden widths ``(44, 22)`` and the locked schedule are carried across unchanged. The
    ``scratch_mlp.INPUT_DIM`` comment anticipates exactly this. Every other factory ignores it.

    Imports are function-local: ``models.baselines`` imports from this module, so a module-level
    import here would be circular.
    """
    from .models.baselines import BASELINE_FACTORIES  # noqa: PLC0415 - circular at module level
    from .models.scratch_logreg import make_scratch_logreg  # noqa: PLC0415
    from .models.scratch_mlp import TUNED_PARAMS as MLP_PARAMS  # noqa: PLC0415
    from .models.scratch_mlp import make_scratch_mlp  # noqa: PLC0415

    def _mlp(n_features: int) -> Any:
        hidden_and_out = tuple(MLP_PARAMS["layer_sizes"])[1:]
        return make_scratch_mlp(layer_sizes=(int(n_features), *hidden_and_out))

    ordered = ("dummy", "decision_tree", "random_forest", "svm")
    factories: dict[str, Callable[[int], Any]] = {
        name: (lambda _n, _factory=BASELINE_FACTORIES[name]: _factory()) for name in ordered
    }
    factories["scratch_logreg"] = lambda _n: make_scratch_logreg()
    factories["scratch_mlp"] = _mlp
    return factories


def _check_against_phase4(model_name: str, scores: dict[str, Any]) -> None:
    """Cross-check a recomputed baseline in-distribution score against Phase 4's committed row.

    Phase 6 re-fits the same locked estimator on the same train fold, so its in-distribution
    numbers must reproduce ``phase4-baselines``' to the logged precision. They are *not* re-logged
    (Phase 4 owns those rows, and a second copy under a Phase 6 ``run_id`` would double-count the
    baselines in the results table) -- but they are computed anyway for the delta, so checking
    them is free and catches the two failures that would otherwise be invisible: a factory or
    ``TUNED_PARAMS`` edit that moved a model out from under the committed rows, and a fold split
    that stopped reproducing.
    """
    from .models.baselines import RUN_ID as PHASE4_RUN_ID  # noqa: PLC0415 - circular at top level

    logged = read_metrics().get((PHASE4_RUN_ID, model_name, IN_DISTRIBUTION))
    if logged is None:
        print(f"    note: no {PHASE4_RUN_ID} row for {model_name} to cross-check against")
        return
    for metric in ("f1", "roc_auc"):
        drift = abs(float(scores[metric]) - float(logged[metric]))
        if drift > PHASE4_AGREEMENT_TOLERANCE:
            raise RuntimeError(
                f"{model_name} in-distribution {metric} is {scores[metric]:.6f} here but "
                f"{logged[metric]} in the committed {PHASE4_RUN_ID} row (drift {drift:.2e}). "
                "Phase 4 and Phase 6 are no longer measuring the same model on the same fold; "
                "reconcile before reporting a delta between them."
            )
        if drift > 0.0:
            print(f"    note: {model_name} {metric} differs from {PHASE4_RUN_ID} by {drift:.2e}")


#: The models whose in-distribution rows Phase 4 owns, so Phase 6 must not re-log them. Spelled
#: out rather than imported from ``BASELINE_FACTORIES`` (which would be a circular import at
#: module level) and checked against that registry in :func:`run_phase6`.
_PHASE4_MODELS: frozenset[str] = frozenset({"dummy", "decision_tree", "random_forest", "svm"})

#: What each logged row's ``notes`` records about how it was produced. The leakage contract is
#: written into the row itself so a reader of ``reports/metrics.csv`` does not have to take the
#: code's word for it.
_REGIME_NOTES: dict[str, str] = {
    IN_DISTRIBUTION: "fit on UNSW train fold, scored on UNSW-test",
    CROSS_ERA: "same fit, zero-shot on TON_IoT (transform-only, no refit of model or preprocessor)",
}


def run_condition(
    run_id: str,
    frames: dict[str, tuple[Any, Any]],
    columns: tuple[str, ...],
    *,
    condition_note: str,
    log: bool = True,
    log_in_distribution: bool = True,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Fit every Phase 4-5 model on the train fold and run it through both regimes.

    Returns ``{model_name: {"in_distribution": {...}, "cross_era": {...}}}`` and upserts one row
    per (model, regime) under ``run_id``.

    ``columns`` selects the feature subset -- the full 22 for the headline condition, the 18
    without the protocol one-hots for the ablation -- and is applied identically to all three
    frames, so the model, UNSW-test and TON_IoT always agree on the matrix.

    ``log_in_distribution=False`` computes the in-distribution half (the delta needs it, and
    :func:`_check_against_phase4` verifies it) but does not write it: in the unablated condition
    the four baselines' in-distribution rows belong to ``phase4-baselines``, and re-logging them
    under a Phase 6 ``run_id`` would make the results table count them twice. It does not apply to
    the scratch models, which have no Phase 4 row -- Phase 5 deliberately logged nothing -- so
    Phase 6 is where their in-distribution rows are created.
    """
    selected = list(columns)
    X_train, y_train = frames["train"]
    X_train = X_train[selected]
    regime_frames = {
        IN_DISTRIBUTION: (frames["unsw_test"][0][selected], frames["unsw_test"][1]),
        CROSS_ERA: (frames["toniot"][0][selected], frames["toniot"][1]),
    }
    expected_share = {
        regime: float((y == POSITIVE_LABEL).mean()) for regime, (_X, y) in regime_frames.items()
    }

    if len(X_train) != EXPECTED_ROWS["train"][0]:  # pragma: no cover - asserted upstream too
        raise LeakageError(
            f"the training matrix has {len(X_train):,} rows, not the UNSW train fold's "
            f"{EXPECTED_ROWS['train'][0]:,}. Models in this phase are fit on the train fold and "
            "nothing else."
        )

    print(
        f"\n=== {run_id} — {condition_note} (d={len(selected)}) ===\n"
        f"    train fold n={len(X_train):,}   "
        f"UNSW-test n={len(regime_frames[IN_DISTRIBUTION][0]):,} "
        f"(attack {expected_share[IN_DISTRIBUTION]:.4f})   "
        f"TON_IoT n={len(regime_frames[CROSS_ERA][0]):,} "
        f"(attack {expected_share[CROSS_ERA]:.4f})"
    )

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for name, factory in phase6_models().items():
        model = factory(len(selected))
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        regimes = run_regimes(
            model,
            *regime_frames[IN_DISTRIBUTION],
            *regime_frames[CROSS_ERA],
        )
        results[name] = regimes

        print(f"\n  {name}  [{type(model).__name__}]  fit {fit_seconds:.1f}s")
        for regime in (IN_DISTRIBUTION, CROSS_ERA):
            scores = regimes[regime]
            if abs(scores["positive_rate"] - expected_share[regime]) > 1e-9:
                raise RuntimeError(
                    f"{name}/{regime} logged positive_rate {scores['positive_rate']:.6f} against "
                    f"a set whose attack share is {expected_share[regime]:.6f} -- the metrics were "
                    "measured against different labels than the frame they claim. Refusing to log."
                )
            print(
                f"    {regime:<16} n={scores['n_test']:>7,}  attack={scores['positive_rate']:.4f}  "
                f"f1={scores['f1']:.4f}  roc_auc={scores['roc_auc']:.4f}  "
                f"precision={scores['precision']:.4f}  recall={scores['recall']:.4f}\n"
                f"    {'':<16} accuracy={scores['accuracy']:.4f}  "
                f"balanced_accuracy={scores['balanced_accuracy']:.4f}  "
                f"macro_f1={scores['macro_f1']:.4f}  "
                f"confusion [[tn, fp], [fn, tp]]={scores['confusion_matrix']}"
            )
        deltas = metric_deltas(regimes)
        print(
            f"    {'Δ (in − cross)':<16} roc_auc={deltas['roc_auc']:+.4f}  "
            f"f1={deltas['f1']:+.4f}  balanced_accuracy={deltas['balanced_accuracy']:+.4f}  "
            f"macro_f1={deltas['macro_f1']:+.4f}"
        )

        skip_in_distribution = not log_in_distribution and name in _PHASE4_MODELS
        if skip_in_distribution:
            _check_against_phase4(name, regimes[IN_DISTRIBUTION])

        if not log:
            continue
        for regime in (IN_DISTRIBUTION, CROSS_ERA):
            if regime == IN_DISTRIBUTION and skip_in_distribution:
                continue
            log_metrics(
                round_metrics(
                    {
                        "run_id": run_id,
                        "model": name,
                        "regime": regime,
                        "seed": RANDOM_SEED,
                        "notes": f"{condition_note}; {_REGIME_NOTES[regime]}; d={len(selected)}",
                        # The confusion matrix has no column in the frozen 14-field header, so it
                        # goes to stdout only; Phase 9 renders it as a figure.
                        **{k: v for k, v in regimes[regime].items() if k != "confusion_matrix"},
                    }
                )
            )
    return results


def run_phase6(log: bool = True) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Phase 6 end to end: both conditions, both regimes, all six models, logged.

    Returns ``{run_id: {model: {regime: metrics}}}``. The ``Preprocessor`` is sealed for the whole
    run (see :func:`sealed`), so no path through this function can refit it -- the target era is
    transformed through the frozen Phase 3 artifact or the run raises.
    """
    preprocessor = load_preprocessor()
    with sealed(
        preprocessor,
        reason=(
            "the Preprocessor is fit on the UNSW train fold in Phase 3 and applied unchanged "
            "thereafter; refitting it on UNSW-test or TON_IoT invalidates the drift measurement"
        ),
    ):
        frames = transform_regime_frames(preprocessor)
        full_columns = tuple(preprocessor.feature_names_)
        proto = protocol_columns(preprocessor)
        ablated_columns = tuple(column for column in full_columns if column not in set(proto))
        if len(ablated_columns) != len(full_columns) - len(proto):  # pragma: no cover
            raise RuntimeError("the proto ablation removed an unexpected number of columns")

        from .models.baselines import BASELINE_FACTORIES  # noqa: PLC0415 - circular at top level

        if set(BASELINE_FACTORIES) != _PHASE4_MODELS:
            raise RuntimeError(
                f"baselines registry {sorted(BASELINE_FACTORIES)} no longer matches the models "
                f"whose in_distribution rows Phase 4 owns {sorted(_PHASE4_MODELS)}; the "
                "in-distribution rows would be double-logged or silently dropped."
            )

        print(
            f"feature schema: {len(full_columns)} columns; ablation drops {len(proto)} "
            f"protocol one-hots {list(proto)} -> {len(ablated_columns)} columns"
        )

        results = {
            RUN_ID: run_condition(
                RUN_ID,
                frames,
                full_columns,
                condition_note="full shared feature set",
                log=log,
                # The four baselines' in-distribution rows are phase4-baselines'. Only the two
                # scratch models get an in-distribution row from this condition.
                log_in_distribution=False,
            ),
            ABLATION_RUN_ID: run_condition(
                ABLATION_RUN_ID,
                frames,
                ablated_columns,
                condition_note="proto ablation: protocol one-hots dropped, retrained on train fold",
                log=log,
                # A delta needs both halves, and no other run_id holds an ablated in-distribution
                # row, so this condition logs its own matched pair for every model.
                log_in_distribution=True,
            ),
        }

    _print_headline(results[RUN_ID])
    _print_ablation_contrast(results[RUN_ID], results[ABLATION_RUN_ID])
    return results


def _print_headline(results: dict[str, dict[str, dict[str, Any]]]) -> None:
    """The RQ1 table: per model, both regimes' ROC-AUC and F1 and the delta, with both balances."""
    first = next(iter(results.values()))
    in_share = first[IN_DISTRIBUTION]["positive_rate"]
    cross_share = first[CROSS_ERA]["positive_rate"]
    print(
        f"\n{'=' * 100}\nRQ1 headline — {RUN_ID}\n"
        f"  in_distribution: UNSW-test  n={first[IN_DISTRIBUTION]['n_test']:,}  "
        f"normal {1 - in_share:.2%} / attack {in_share:.2%}\n"
        f"  cross_era:       TON_IoT    n={first[CROSS_ERA]['n_test']:,}  "
        f"normal {1 - cross_share:.2%} / attack {cross_share:.2%}\n"
        f"  Lead with ROC-AUC (prevalence-insensitive). Part of every F1 delta is the balance "
        f"change, not drift.\n{'-' * 100}\n"
        f"{'model':<16}{'AUC in':>9}{'AUC cross':>11}{'Δ AUC':>9}"
        f"{'F1 in':>9}{'F1 cross':>10}{'Δ F1':>9}{'Δ bal-acc':>11}{'Δ macro-F1':>12}"
    )
    for name, regimes in results.items():
        deltas = metric_deltas(regimes)
        print(
            f"{name:<16}{regimes[IN_DISTRIBUTION]['roc_auc']:>9.4f}"
            f"{regimes[CROSS_ERA]['roc_auc']:>11.4f}{deltas['roc_auc']:>+9.4f}"
            f"{regimes[IN_DISTRIBUTION]['f1']:>9.4f}{regimes[CROSS_ERA]['f1']:>10.4f}"
            f"{deltas['f1']:>+9.4f}{deltas['balanced_accuracy']:>+11.4f}"
            f"{deltas['macro_f1']:>+12.4f}"
        )
    dummy = results.get("dummy")
    if dummy is not None:
        print(
            f"{'-' * 100}\n"
            f"Prevalence artifact: the majority-class dummy's F1 "
            f"{'rises' if dummy[CROSS_ERA]['f1'] > dummy[IN_DISTRIBUTION]['f1'] else 'falls'} "
            f"{dummy[IN_DISTRIBUTION]['f1']:.4f} -> {dummy[CROSS_ERA]['f1']:.4f} cross-era at "
            f"ROC-AUC {dummy[CROSS_ERA]['roc_auc']:.4f}, on class balance alone. Any real model's "
            f"F1 delta must be read against that {dummy[CROSS_ERA]['f1'] - dummy[IN_DISTRIBUTION]['f1']:+.4f}."
        )


def _print_ablation_contrast(
    base: dict[str, dict[str, dict[str, Any]]],
    ablated: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """The difference of the deltas: how much of the drop survives dropping ``proto``."""
    print(
        f"\n{'=' * 100}\nproto ablation — {ABLATION_RUN_ID} vs {RUN_ID}\n"
        "  Each side is its own matched in_distribution/cross_era pair, so the comparable\n"
        "  quantity is the difference of the deltas: how much of the with-proto drop survives\n"
        "  when the protocol one-hots were never available to be learned.\n"
        f"{'-' * 100}\n"
        f"{'model':<16}{'Δ AUC proto':>13}{'Δ AUC no_proto':>16}{'Δ of Δ':>10}"
        f"{'Δ F1 proto':>13}{'Δ F1 no_proto':>15}{'Δ of Δ':>10}"
    )
    for name in base:
        with_proto = metric_deltas(base[name])
        without = metric_deltas(ablated[name])
        print(
            f"{name:<16}{with_proto['roc_auc']:>+13.4f}{without['roc_auc']:>+16.4f}"
            f"{with_proto['roc_auc'] - without['roc_auc']:>+10.4f}"
            f"{with_proto['f1']:>+13.4f}{without['f1']:>+15.4f}"
            f"{with_proto['f1'] - without['f1']:>+10.4f}"
        )
    print(
        "  A Δ-of-Δ near zero means the drop is not the `other`-bucket artifact; a large positive\n"
        "  one means part of the with-proto drop was that learned signal going inert."
    )


# --- Entry point -------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description=(
            "Phase 6: evaluate every Phase 4-5 model in both regimes (in-distribution on "
            "UNSW-test, zero-shot cross-era on TON_IoT) plus the proto ablation, and upsert the "
            "rows into reports/metrics.csv. Nothing is refit on UNSW-test or TON_IoT."
        ),
    )
    parser.add_argument(
        "--regimes",
        action="store_true",
        help="run the two-regime cross-era evaluation (RQ1) and log it",
    )
    args = parser.parse_args(argv)
    if not args.regimes:
        parser.error("nothing to do: pass --regimes to run the Phase 6 cross-era evaluation")

    set_seeds()
    results = run_phase6()
    rows = sum(
        len(regimes) for condition in results.values() for regimes in condition.values()
    ) - len(_PHASE4_MODELS)  # the baselines' in-distribution halves stay Phase 4's
    print(f"\nlogged {rows} rows across {len(results)} run_ids -> {METRICS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
