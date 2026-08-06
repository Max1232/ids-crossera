"""Transfer-learning correction — RQ2 recovery curve (Phase 7).

Fine-tune each model on small stratified fractions of TON_IoT labels (1%, 5%, 10%, 25%); the
rest of TON_IoT stays as the test set. MLP: freeze early layers, retrain the classifier head.
Classical models: warm-start / retrain on the small modern sample. Plot post-transfer
F1/ROC-AUC vs fraction of modern data used.

--------------------------------------------------------------------------------------------
The frozen test set — the rule the experiment is invalid without
--------------------------------------------------------------------------------------------

TON_IoT is split **once, up front** into a permanent test half and a fine-tune pool
(:func:`frozen_split_indices`, stratified on the binary label, seeded from
``config.RANDOM_SEED``). Every fine-tune fraction is drawn from the pool and **every** point on
the curve — fraction 0 through the ceiling — is scored on the *same* frozen test half. That is
what makes the x-axis mean "labelled modern rows spent" and nothing else: if the test set moved
with the budget, the first segment of the curve would conflate "gained fine-tune data" with
"changed test set".

Two consequences worth naming, because both are silent if they go wrong:

* **The fine-tune data and the frozen test half are disjoint, asserted on row indices** rather
  than on values. ~52% of the UNSW train fold are duplicate feature vectors in the shared
  11-column subspace and TON_IoT is no better, so a value-based disjointness check would report
  false overlaps on rows that are genuinely different records.
* **Phase 6's ``cross_era`` rows are not the fraction-0 point of this curve.** They were measured
  on the full 211,043-row TON_IoT, i.e. on a *different* test set that contains this phase's
  fine-tune pool. The zero-shot point is therefore re-measured here, on the frozen test half,
  under its own ``run_id`` (:data:`ZERO_SHOT_LABEL`). See ``deviations.md`` §3.12.

--------------------------------------------------------------------------------------------
Adaptation, per model — and why the mechanisms differ
--------------------------------------------------------------------------------------------

Nothing is re-tuned per fraction: every model is re-instantiated from the Phase 4/5 factories at
its locked ``TUNED_PARAMS`` (via ``evaluate.phase6_models``), so the data budget is the only
variable along the curve. Class weights stay on throughout, and on both scratch models
``class_weight=None``/``"balanced"`` *is* the weighted setting (the inversion of sklearn's default
documented in ``scratch_logreg._sample_weights``) — which matters here, because the pool is 76.31%
attack.

* **``scratch_mlp`` — freeze the hidden layers, retrain the head** (``fit(freeze_hidden=True)``).
  As committed, that freezes **both** hidden layers: ``fit`` marks only ``head_keys()`` trainable,
  so the entire ``(22, 44, 22)`` stack keeps its UNSW weights and acts as a fixed feature
  extractor while 23 of the net's 2,025 parameters (1.1%) move. There is no "freeze only the
  first layer" mode and the interface is deliberately not widened to add one: the head-only
  variant is the strongest form of the data-efficiency claim (the modern budget buys a new
  decision boundary in a 2015 feature space, nothing more) and it is the variant the Phase 5
  docstring committed to. The head is *continued* from its fitted values rather than
  re-initialized, so a 1% budget is an adjustment to the source fit and not a fresh random head.
* **``scratch_logreg`` — warm start** (``fit(warm_start=True)``): gradient descent continues from
  the UNSW-fitted weights on the target slice, which is what that parameter exists for.
* **``dummy``, ``decision_tree``, ``random_forest``, ``svm`` — refit on the small target sample.**
  None of the three real ones has an incremental interface that preserves a fit (sklearn's
  ``RandomForestClassifier.warm_start`` grows more trees on the *same* data; ``DecisionTreeClassifier``
  and ``LinearSVC`` have none), so "retrain on the small sample" from the stub docstring is the
  available mechanism, and a plain ``fit`` on a fitted sklearn estimator is exactly that.

**Target-only rather than (source + small target), deliberately.** The alternative — pooling the
140,272-row UNSW train fold with the target slice — was rejected on two grounds. It is not a
data-budget curve: at the 1% budget the fit is 140,272 source rows against 1,055 target rows, so
the measured quantity would be dominated by the source era at every point a reader cares about.
And it does not converge to the ceiling: the ceiling this curve is read against is a
**full-target-trained** model, so a pooled curve would approach a *different* asymptote and "how
close does 25% get to the ceiling" would no longer be a well-posed question. Each model's ceiling
is that model's own adaptation mechanism run at the full budget (fraction 1.0 of the pool), which
is what makes the curve continuous into its own upper bound. Recorded as ``deviations.md`` §3.13.

The MLP is the one model whose ceiling is set by its *mechanism* rather than by the data, so the
freeze is measured rather than asserted: :func:`freeze_cost_control` retrains the same net end to
end on the same pool under its own ``run_id`` (:data:`FREEZE_CONTROL_LABEL`), and the difference
between the two rows is what holding the 2015 feature space fixed costs. The classical models need
no such control (their ceilings already *are* full target retrains) and neither does
``scratch_logreg`` (its objective is convex, so a warm start and a cold start reach the same
optimum at the full budget).

--------------------------------------------------------------------------------------------
Reading the curve — the dummy floor comes first
--------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------
Reading the curve — the dummy floor comes first
--------------------------------------------------------------------------------------------

The majority-class dummy scores **F1 0.8656 at ROC-AUC exactly 0.5000** on TON_IoT, because the
target era is 76.31% attack (``deviations.md`` §3.8). So an F1 climbing toward 0.87 may have
recovered *nothing*, and RQ2 leads with **ROC-AUC** for the same reason RQ1 does. Three reference
lines are measured on the frozen test half and logged beside every recovery point: the dummy floor
(the ``dummy`` model's own row at every budget), the zero-shot point (fraction 0), and the ceiling.
The interesting threshold on AUC is not "higher than before" but **0.5000** — Phase 6 found the
2015 ranking *inverted* cross-era (§3.10), so a model only stops being anti-correlated with ground
truth once its AUC clears the dummy's.
"""

from __future__ import annotations

import argparse
import copy
import time
from typing import Any

import numpy as np

from .config import METRICS_CSV, PER_FAMILY_CSV, RANDOM_SEED, set_seeds
from .evaluate import (
    CROSS_ERA,
    NATIVE_FAMILY_SET,
    NORMAL_FAMILY,
    POSITIVE_LABEL,
    evaluate,
    load_preprocessor,
    log_metrics,
    per_family_rows,
    phase6_models,
    read_metrics,
    regime_families,
    round_metrics,
    sealed,
    transform_regime_frames,
    write_per_family_metrics,
)

# Modern-data budgets for the recovery curve.
TRANSFER_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25)

#: Share of TON_IoT reserved as the **permanent** test half. Split once, stratified, seeded; the
#: other half is the fine-tune pool every fraction is drawn from. ~50% keeps the test half large
#: enough (105,521 rows) that a metric moves on the model rather than on sampling noise, while
#: leaving a pool whose 25% budget (26,380 rows) is still a plausible "small modern sample".
TEST_FRACTION: float = 0.50

#: The regime label Phase 7's rows carry. Deliberately **not** ``cross_era``: those rows are Phase
#: 6's, measured on the full 211,043-row TON_IoT with no adaptation, and re-using the spelling
#: would make a Phase 7 row look like a Phase 6 row on any join over ``(model, regime)``. It is
#: not ``in_distribution`` either — the models are fitted on 2015 data and (from fraction 0.01 up)
#: adapted on a slice of the target era, so neither Phase 6 label describes what was measured.
#: Every row under this label was scored on the frozen TON_IoT test half and nothing else.
REGIME: str = "target_frozen_test"

#: ``run_id`` prefix. One ``run_id`` per budget, per the convention in ``evaluate.log_metrics``:
#: ``(run_id, model, regime)`` is the upsert key, so two fractions sharing a ``run_id`` would
#: silently overwrite each other and the curve would collapse to whichever ran last.
RUN_ID_PREFIX: str = "phase7-recovery"

#: Label for the re-measured zero-shot point: the UNSW-train-fitted models, unadapted, on the
#: frozen test half. Spelled as a fraction so it sorts and reads as the curve's left endpoint.
ZERO_SHOT_LABEL: str = "f0.00"

#: Label for the full-budget reference — each model's own adaptation mechanism run on the entire
#: fine-tune pool. The upper bound the curve approaches.
CEILING_LABEL: str = "ceiling"
CEILING_FRACTION: float = 1.0

#: The freeze-cost control's ``run_id``: the from-scratch MLP **re-initialized and retrained** on
#: the entire fine-tune pool, no layer frozen. One extra row, and it is what makes the freeze a
#: measured choice rather than an asserted one -- the MLP is the only model whose ceiling is limited
#: by its adaptation mechanism (the classical models' ceilings are already full target retrains, and
#: ``scratch_logreg``'s objective is convex, so its warm-started ceiling *is* the cold-start
#: optimum). Its own ``run_id`` because it is a different condition at the same budget.
FREEZE_CONTROL_LABEL: str = f"{CEILING_LABEL}-no_freeze"

#: The model the freeze-cost control applies to. Named rather than inferred so adding a second
#: neural model cannot silently leave it unmeasured.
FREEZE_CONTROL_MODEL: str = "scratch_mlp"

#: How far a stratified half or draw may drift from the source frame's attack share before the
#: split is treated as broken. Stratification here is exact up to the per-class floor division, so
#: the achievable drift is O(1/n) -- 5e-5 on the real 1% draw (805 attack of 1,055) and ~1e-6 on
#: the frozen half. The floor below is what a *broken* split has to beat; :func:`_stratification_
#: tolerance` widens it on small frames, where floor rounding alone can move the share by a percent
#: (the synthetic frames in ``tests/test_transfer.py``, not anything in the pipeline).
STRATIFICATION_TOLERANCE: float = 1e-3


def _stratification_tolerance(n_rows: int) -> float:
    """The drift a correctly stratified draw of ``n_rows`` can still show, from rounding alone."""
    return max(STRATIFICATION_TOLERANCE, 1.0 / max(int(n_rows), 1))

#: The AUC line every recovery point is read against: the majority-class dummy's ROC-AUC is
#: exactly 0.5 at any prevalence, and Phase 6 found every real model *below* it cross-era. Used
#: only for reporting where each model crosses back over it.
NO_SKILL_AUC: float = 0.5


def fraction_label(fraction: float) -> str:
    """``0.05 -> "f0.05"`` — the budget as it appears in a ``run_id`` and on the curve's x-axis."""
    return f"f{float(fraction):.2f}"


def run_id_for(label: str) -> str:
    """``"f0.05" -> "phase7-recovery-f0.05"``. One ``run_id`` per budget; see :data:`RUN_ID_PREFIX`."""
    return f"{RUN_ID_PREFIX}-{label}"


# --- The frozen split ----------------------------------------------------------------------


def _labels(y: Any) -> np.ndarray:
    """Labels as a 0/1 int vector, positional — index-free by design.

    Every split in this module is computed and asserted on **positional** row indices rather than
    on a pandas index or on feature values: the frames arriving from the Phase 3 ``Preprocessor``
    carry duplicate feature vectors in bulk, so a value-based identity would produce false overlap
    reports, and a positional index is the one thing that survives ``.iloc`` slicing unambiguously.
    """
    labels = np.asarray(getattr(y, "to_numpy", lambda: y)()).ravel().astype("int64")
    unexpected = sorted(set(np.unique(labels)) - {0, POSITIVE_LABEL})
    if unexpected:
        raise ValueError(
            f"labels {unexpected} are outside [0, {POSITIVE_LABEL}]; the harmonized frames carry "
            "a binary normal(0)/attack(1) label"
        )
    return labels


def _take(data: Any, positions: np.ndarray) -> Any:
    """Positional row selection that works on a DataFrame/Series and on a bare ndarray."""
    if hasattr(data, "iloc"):
        return data.iloc[positions]
    return np.asarray(data)[positions]


def _positive_rate(y: Any) -> float:
    labels = _labels(y)
    return float((labels == POSITIVE_LABEL).mean())


def frozen_split_indices(y: Any, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Split TON_IoT once into ``(pool_positions, test_positions)`` — stratified, seeded, permanent.

    :data:`TEST_FRACTION` of **each class** goes to the test half, so both sides carry the source
    frame's attack share to within a row. Returns sorted positional indices into ``y``.

    This function is the frozen test set: it depends only on the label vector and the seed, so
    every fraction, every model and every re-run resolve the identical test half. Nothing else in
    this module is allowed to re-draw it.

    The per-class count uses ``int(n * TEST_FRACTION)`` (floor) rather than ``round``, so the
    result does not depend on how a ``.5`` ties — 161,043 attack rows would otherwise sit on a
    banker's-rounding boundary.
    """
    labels = _labels(y)
    if labels.size == 0:
        raise ValueError("cannot split an empty label vector")

    pool_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for label in (0, POSITIVE_LABEL):
        positions = np.flatnonzero(labels == label)
        if positions.size < 2:
            raise ValueError(
                f"class {label} has {positions.size} row(s); a stratified 50/50 split needs at "
                "least two per class so both halves see both classes"
            )
        shuffled = positions[rng.permutation(positions.size)]
        n_test = int(positions.size * TEST_FRACTION)
        test_parts.append(shuffled[:n_test])
        pool_parts.append(shuffled[n_test:])

    pool = np.sort(np.concatenate(pool_parts))
    test = np.sort(np.concatenate(test_parts))

    # The three things that make this a *frozen test set* rather than a split.
    if np.intersect1d(pool, test, assume_unique=True).size:
        raise RuntimeError("the fine-tune pool and the frozen test half overlap")
    if pool.size + test.size != labels.size:
        raise RuntimeError(
            f"the split covers {pool.size + test.size:,} of {labels.size:,} rows; every row must "
            "land in exactly one half"
        )
    source_rate = float((labels == POSITIVE_LABEL).mean())
    for name, half in (("fine-tune pool", pool), ("frozen test half", test)):
        drift = abs(float((labels[half] == POSITIVE_LABEL).mean()) - source_rate)
        if drift > _stratification_tolerance(half.size):
            raise RuntimeError(
                f"the {name}'s attack share drifts {drift:.2e} from the source frame's "
                f"{source_rate:.6f}; the split is no longer stratified"
            )
    return pool, test


def fraction_indices(
    y: Any, pool: np.ndarray, fraction: float, seed: int = RANDOM_SEED
) -> np.ndarray:
    """Stratified draw of ``fraction`` of the fine-tune ``pool``, as sorted positional indices.

    Drawn from the pool **only** — the frozen test half is never a candidate — and **nested**
    across fractions by construction: each class's pool rows are permuted once from ``seed``
    (independently of ``fraction``) and the draw is a prefix of that permutation, so the 1% sample
    is a subset of the 5% sample is a subset of the 10%. Nesting is not required by the design, but
    it removes an entire class of confound: without it, a non-monotone point on the curve could be
    a different *sample* rather than a different *budget*.

    ``fraction=1.0`` returns the whole pool, which is how the ceiling is computed — the same
    mechanism at the full budget rather than a separate code path.
    """
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"fraction={fraction} must be in (0, 1]")
    labels = _labels(y)
    rng = np.random.default_rng(seed)

    drawn: list[np.ndarray] = []
    for label in (0, POSITIVE_LABEL):
        positions = pool[labels[pool] == label]
        if positions.size == 0:
            raise ValueError(f"the fine-tune pool holds no class-{label} rows")
        shuffled = positions[rng.permutation(positions.size)]
        # At least one row per class: a single-class fine-tune slice cannot carry balanced class
        # weights (scratch_logreg._sample_weights raises) and is not a fit anyone means to run.
        n_draw = max(1, int(positions.size * float(fraction)))
        drawn.append(shuffled[:n_draw])

    return np.sort(np.concatenate(drawn))


def sample_fraction(
    X: Any, y: Any, fraction: float, seed: int = RANDOM_SEED
) -> tuple[Any, Any, Any, Any]:
    """Stratified split of TON_IoT into a fine-tune fraction and the remaining test set.

    Returns ``(X_finetune, y_finetune, X_test, y_test)``.

    The "remaining test set" is the **frozen half**, not "whatever this fraction did not consume":
    :func:`frozen_split_indices` reserves :data:`TEST_FRACTION` of the target up front and the
    fine-tune rows come out of the other half only. So the returned test set is bit-identical for
    every ``fraction`` at a given ``seed``, which is the property the whole curve rests on — the
    data budget is the only thing that moves along the x-axis.

    Asserted here rather than trusted, because each failure is silent:

    * the fine-tune rows and the test rows are **disjoint on positional indices** (not on values —
      duplicate feature vectors are abundant in this subspace, see :func:`_labels`);
    * both sides carry the source frame's attack share to within :data:`STRATIFICATION_TOLERANCE`;
    * the two feature matrices share an identical column set and order, the same invariant
      ``evaluate.transform_regime_frames`` asserts across regimes.
    """
    pool, test = frozen_split_indices(y, seed=seed)
    finetune_positions = fraction_indices(y, pool, fraction, seed=seed)

    if np.intersect1d(finetune_positions, test, assume_unique=True).size:
        raise RuntimeError(
            "the fine-tune sample intersects the frozen test half; every recovery point would be "
            "scored partly on rows it was fitted on"
        )

    X_finetune, y_finetune = _take(X, finetune_positions), _take(y, finetune_positions)
    X_test, y_test = _take(X, test), _take(y, test)

    if hasattr(X_finetune, "columns") and hasattr(X_test, "columns"):
        if list(X_finetune.columns) != list(X_test.columns):
            raise RuntimeError(
                "the fine-tune matrix and the frozen test matrix have different feature schemas; "
                "an adapted model would be scored against a matrix it was not fitted for"
            )
    elif np.asarray(X_finetune).shape[1] != np.asarray(X_test).shape[1]:
        raise RuntimeError("the fine-tune and frozen test matrices differ in width")

    for name, labels in (("fine-tune sample", y_finetune), ("frozen test half", y_test)):
        drift = abs(_positive_rate(labels) - _positive_rate(y))
        if drift > _stratification_tolerance(len(labels)):
            raise RuntimeError(
                f"the {name}'s attack share drifts {drift:.2e} from the target frame's; the draw "
                "is no longer stratified"
            )

    return X_finetune, y_finetune, X_test, y_test


# --- Adaptation ----------------------------------------------------------------------------

#: How each model spends its modern budget, for the ``notes`` column and the stdout table. The
#: mechanisms and the reasoning behind them are in the module docstring.
ADAPTATION_NOTES: dict[str, str] = {
    "scratch_mlp": "head-only refit, both hidden layers frozen at their UNSW weights",
    "scratch_logreg": "warm-started gradient descent from the UNSW-fitted weights",
    "classical": "refit on the target sample alone (no incremental interface preserves a fit)",
}


def adaptation_note(model: Any) -> str:
    """Which of :data:`ADAPTATION_NOTES` applies to ``model``."""
    from .models.scratch_logreg import ScratchLogReg  # noqa: PLC0415 - keeps pandas out
    from .models.scratch_mlp import ScratchMLP  # noqa: PLC0415

    if isinstance(model, ScratchMLP):
        return ADAPTATION_NOTES["scratch_mlp"]
    if isinstance(model, ScratchLogReg):
        return ADAPTATION_NOTES["scratch_logreg"]
    return ADAPTATION_NOTES["classical"]


def finetune(model: Any, X_ft: Any, y_ft: Any) -> Any:
    """Fine-tune a trained model on the modern fraction.

    MLP -> freeze hidden layers, retrain head (``ScratchMLP.fit(freeze_hidden=True)``).
    Classical -> warm-start / retrain on the small sample.

    ``model`` must arrive **fitted on the UNSW train fold** and is mutated in place, so callers
    pass a copy per fraction (see :func:`recovery_curve`) — otherwise the 5% point would adapt a
    model the 1% point had already moved, and the curve would measure a cumulative schedule rather
    than a budget.

    Three mechanisms, dispatched on the model rather than configured, because the choice is not
    free per model — it is whatever preserves the source fit through an adaptation on as few as
    1,055 target rows:

    * ``ScratchMLP`` -> ``fit(freeze_hidden=True)``: the hidden stack keeps its UNSW weights and
      only ``W{L}``/``b{L}`` move, continued from their fitted values. As committed this freezes
      **all** hidden layers (``fit`` marks only ``head_keys()`` trainable); there is no
      freeze-the-first-layer-only mode and none is added here.
    * ``ScratchLogReg`` -> ``fit(warm_start=True)``: gradient descent continues from the fitted
      weights instead of restarting from zeros.
    * everything else (sklearn) -> a plain ``fit`` on the target sample. ``RandomForestClassifier``'s
      ``warm_start`` grows more trees on the same data rather than carrying a fit onto new data,
      and ``DecisionTreeClassifier``/``LinearSVC``/``DummyClassifier`` expose nothing at all, so
      "retrain on the small sample" is the mechanism available.

    Class weights are re-derived from ``y_ft`` by every path (both scratch models do it per fit;
    the sklearn factories carry ``class_weight="balanced"``), which is what they have to be: the
    fine-tune pool is 76.31% attack, not the train fold's 68.06%.
    """
    from .models.scratch_logreg import ScratchLogReg  # noqa: PLC0415 - keeps pandas out
    from .models.scratch_mlp import ScratchMLP  # noqa: PLC0415

    if isinstance(model, ScratchMLP):
        return model.fit(X_ft, y_ft, freeze_hidden=True)
    if isinstance(model, ScratchLogReg):
        return model.fit(X_ft, y_ft, warm_start=True)
    return model.fit(X_ft, y_ft)


# --- The curve -----------------------------------------------------------------------------


def recovery_curve(
    model_factory: Any,
    X_toniot: Any,
    y_toniot: Any,
    *,
    source: tuple[Any, Any] | None = None,
    families: Any = None,
    fractions: tuple[float, ...] = TRANSFER_FRACTIONS,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> Any:
    """Sweep TRANSFER_FRACTIONS, return post-transfer metrics per fraction for plotting.

    Also report how close each model gets to a full-TON_IoT-trained ceiling, and at what budget.

    Returns ``{label: metrics}`` for one model, ordered left to right along the curve::

        {"f0.00": {...}, "f0.01": {...}, "f0.05": {...}, "f0.10": {...}, "f0.25": {...},
         "ceiling": {...}}

    Every value is an ``evaluate.evaluate()`` row for the **frozen TON_IoT test half** plus
    ``fraction``, ``n_finetune`` and ``adaptation``. Nothing is logged here — :func:`run_phase7`
    owns the ``run_id`` scheme and the upserts, exactly as ``run_regimes`` leaves logging to
    ``run_phase6``.

    The three reference points the curve is read against all come out of this one call, on one
    test set: ``f0.00`` is the source model unadapted (the re-measured zero-shot point, *not*
    Phase 6's ``cross_era`` row — see the module docstring), ``ceiling`` is the same adaptation
    mechanism at the full budget, and the dummy floor is the ``dummy`` model's own curve.

    ``source`` is the transformed UNSW **train fold** ``(X, y)``. It is keyword-only and appended
    to the stub's signature because every point here — including fraction 0 — starts from a
    source-era fit, and the factory alone cannot produce one.

    ``families`` is the per-row family vector for the **whole** TON_IoT frame (aligned to
    ``y_toniot``); it is sliced to the frozen test half here and every point then carries a
    ``per_family`` block. The vocabulary is TON_IoT's own (:data:`evaluate.NATIVE_FAMILY_SET`) --
    this is a within-era breakdown of *what the modern budget buys per attack type*, so the
    shared three-family map would throw away five of the eight 20,000-row families before the
    question is even asked.

    **Where the leakage seal sits.** Unlike Phase 6, this phase legitimately fits on target data,
    so a seal spanning the fit would raise on the experiment itself. Each model is therefore
    sealed across its :func:`~src.evaluate.evaluate` calls only — the frozen test half can be
    transformed and scored but never fitted — while the fine-tune step runs outside the seal. The
    ``Preprocessor`` is sealed for the whole run one level up, in :func:`run_phase7`, because
    *that* must never be refit at all.
    """
    if source is None:
        raise ValueError(
            "recovery_curve needs the UNSW train fold as source=(X, y): the zero-shot point and "
            "every adaptation start from a source-era fit, which the factory alone cannot produce"
        )
    X_train, y_train = source

    pool, test = frozen_split_indices(y_toniot, seed=seed)
    X_test, y_test = _take(X_toniot, test), _take(y_toniot, test)
    # Sliced to the frozen half ONCE, from the same positional indices the test matrix is taken
    # with, so every budget's per-family breakdown is over the identical rows -- which is the same
    # invariant the curve itself rests on, applied to the family labels.
    families_test = None if families is None else _take(families, test)

    model = model_factory(int(np.shape(X_train)[1]))
    started = time.perf_counter()
    model.fit(X_train, y_train)
    source_seconds = time.perf_counter() - started
    mechanism = adaptation_note(model)
    if verbose:
        print(
            f"\n  {type(model).__name__}  source fit on the UNSW train fold "
            f"(n={len(X_train):,}) in {source_seconds:.1f}s\n"
            f"    adaptation: {mechanism}"
        )

    curve: dict[str, dict[str, Any]] = {}

    # Fraction 0 -- the source model, unadapted, on the frozen half. Sealed for the span of the
    # evaluation only: nothing is fitted here, and the seal proves it rather than asserting it.
    with sealed(
        model,
        reason=(
            "the fraction-0 point is the UNSW-train-fitted model scored zero-shot on the frozen "
            "TON_IoT test half; fitting it here would make the curve's left endpoint an adapted "
            "model and every recovery number relative to it meaningless"
        ),
    ):
        curve[ZERO_SHOT_LABEL] = {
            **evaluate(model, X_test, y_test, families=families_test),
            "fraction": 0.0,
            "n_finetune": 0,
            "adaptation": "none (zero-shot: the UNSW-train fit, unadapted)",
        }

    budgets = [(fraction_label(fraction), float(fraction)) for fraction in fractions]
    budgets.append((CEILING_LABEL, CEILING_FRACTION))
    for label, fraction in budgets:
        X_ft, y_ft, X_frozen, y_frozen = sample_fraction(X_toniot, y_toniot, fraction, seed=seed)
        # The frozen half must be the *same* rows for every budget, not merely the same size.
        if not np.array_equal(np.asarray(_labels(y_frozen)), np.asarray(_labels(y_test))):
            raise RuntimeError(
                f"{label}: sample_fraction returned a different test half than the frozen split; "
                "the test set moved with the budget"
            )
        if hasattr(X_ft, "columns") and list(X_ft.columns) != list(X_test.columns):
            # pragma: no cover - sample_fraction asserts this too; kept because the failure it
            # guards (an adapted model scored against a different matrix) is silent in numpy.
            raise RuntimeError(f"{label}: fine-tune and test matrices disagree on the schema")

        # deepcopy so each budget adapts the *source* model rather than the previous budget's.
        adapted = copy.deepcopy(model)
        started = time.perf_counter()
        finetune(adapted, X_ft, y_ft)
        fit_seconds = time.perf_counter() - started

        with sealed(
            adapted,
            reason=(
                "a recovery point is scored on the frozen TON_IoT test half, which is disjoint "
                "from the fine-tune pool and must never be fitted on"
            ),
        ):
            scores = evaluate(adapted, X_frozen, y_frozen, families=families_test)
        curve[label] = {
            **scores,
            "fraction": fraction,
            "n_finetune": int(len(X_ft)),
            "adaptation": mechanism,
        }
        if verbose:
            print(
                f"    {label:<8} n_ft={len(X_ft):>7,} ({fraction:6.2%} of the pool, "
                f"attack {_positive_rate(y_ft):.4f})  fit {fit_seconds:5.1f}s  "
                f"roc_auc={scores['roc_auc']:.4f}  f1={scores['f1']:.4f}"
            )

    return curve


def freeze_cost_control(
    X_toniot: Any, y_toniot: Any, *, families: Any = None, seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> dict[str, Any]:
    """The MLP at the ceiling budget with **nothing frozen** — what the freeze costs, measured.

    Same architecture, same locked schedule, same fine-tune pool, same frozen test half; the only
    difference from :data:`CEILING_LABEL` is that ``fit`` re-initializes and trains every layer
    instead of only the head. The difference between the two rows is the price of holding the 2015
    feature space fixed, which is the one claim the freeze design needs evidence for.

    Returns one metrics row, logged by :func:`run_phase7` under :data:`FREEZE_CONTROL_LABEL`.
    """
    from .models.scratch_mlp import TUNED_PARAMS, make_scratch_mlp  # noqa: PLC0415

    X_ft, y_ft, X_test, y_test = sample_fraction(
        X_toniot, y_toniot, CEILING_FRACTION, seed=seed
    )
    _pool, test = frozen_split_indices(y_toniot, seed=seed)
    families_test = None if families is None else _take(families, test)
    hidden_and_out = tuple(TUNED_PARAMS["layer_sizes"])[1:]
    model = make_scratch_mlp(layer_sizes=(int(np.shape(X_ft)[1]), *hidden_and_out))

    started = time.perf_counter()
    model.fit(X_ft, y_ft)  # freeze_hidden defaults to False: a fresh fit, every layer trainable
    fit_seconds = time.perf_counter() - started

    with sealed(
        model,
        reason=(
            "the freeze-cost control is scored on the same frozen TON_IoT test half as the rest "
            "of the curve and must not be fitted on it"
        ),
    ):
        scores = evaluate(model, X_test, y_test, families=families_test)
    if verbose:
        print(
            f"\n  freeze-cost control  [{type(model).__name__} {model.layer_sizes}, no layer "
            f"frozen, target-only]\n"
            f"    {FREEZE_CONTROL_LABEL:<18} n_ft={len(X_ft):>7,}  fit {fit_seconds:5.1f}s  "
            f"roc_auc={scores['roc_auc']:.4f}  f1={scores['f1']:.4f}"
        )
    return {
        **scores,
        "fraction": CEILING_FRACTION,
        "n_finetune": int(len(X_ft)),
        "adaptation": (
            "control: every layer re-initialized and retrained on the target pool (no freeze)"
        ),
    }


def duplicate_overlap(X_ft: Any, X_test: Any) -> tuple[float, float]:
    """``(share of test rows whose exact feature vector is in the fine-tune sample, distinct share)``.

    An **interpretation** diagnostic, not a leakage check: the split is disjoint on row indices by
    construction (:func:`sample_fraction`), which is the correct invariant, but TON_IoT is heavily
    redundant in the 22-column shared subspace and a value-level duplicate is not a leak — it is
    two genuinely distinct flows that happen to be indistinguishable to the model. Reported because
    it is what a recovery curve that saturates at the smallest budget has to be read against — but
    it does **not** bound memorization, and the memorization reading is falsified: re-scoring the
    1%-budget fit on only the test rows whose exact vector is absent from the draw (64.22% of the
    half) costs it 0.0021 ROC-AUC against a recovery of 0.786, so the recovery generalizes. What
    this quantity actually measures is how near-deterministic the label is in the 22-column
    subspace — a per-vector majority vote would score 99.8953% on the whole target frame — which is
    why ~1% of the target labels already suffices, and why 1% is a lower bound for a feature space
    with genuine class overlap (``deviations.md`` §2.3).
    """
    finetune = np.ascontiguousarray(np.asarray(X_ft, dtype="float64"))
    test = np.ascontiguousarray(np.asarray(X_test, dtype="float64"))
    _, inverse = np.unique(np.vstack([finetune, test]), axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    ft_ids, test_ids = inverse[: len(finetune)], inverse[len(finetune) :]
    reproduced = float(np.isin(test_ids, ft_ids).mean())
    distinct = float(np.unique(test_ids).size / max(test_ids.size, 1))
    return reproduced, distinct


# --- Phase 7 driver ------------------------------------------------------------------------

#: What each logged row records about how it was produced, so a reader of ``reports/metrics.csv``
#: does not have to take the code's word for the leakage contract.
_ROW_NOTE = (
    "fit on UNSW train fold, {adaptation}; scored on the frozen 50% TON_IoT test half "
    "(disjoint from the fine-tune pool); budget {fraction:.2%} of the pool, n_ft={n_finetune}; "
    "d={n_features}"
)


def run_phase7(log: bool = True) -> dict[str, dict[str, dict[str, Any]]]:
    """Phase 7 end to end: the frozen split, every model's recovery curve, logged.

    Returns ``{model: {label: metrics}}``: one curve per model, plus the freeze-cost control as an
    extra :data:`FREEZE_CONTROL_LABEL` entry on :data:`FREEZE_CONTROL_MODEL`'s curve (its own
    ``run_id``, so it neither overwrites the ceiling nor is mistaken for a point on the curve).

    The ``Preprocessor`` is sealed for the whole run, so no path through this function can refit it
    — TON_IoT is transformed through the frozen Phase 3 artifact or the run raises. Models *are*
    fitted (that is the phase), but only outside the per-evaluation seals in
    :func:`recovery_curve`.

    ``log`` gates both on-disk outputs: the ``reports/metrics.csv`` upserts and the
    ``reports/per_family_metrics.csv`` rows (this phase's block, keyed on TON_IoT's own attack
    types, sitting beside Phase 6's shared-family block in the same file).
    """
    preprocessor = load_preprocessor()
    with sealed(
        preprocessor,
        reason=(
            "the Preprocessor is fit on the UNSW train fold in Phase 3 and applied unchanged "
            "thereafter. Phase 7 fine-tunes models on target labels, but refitting the "
            "preprocessor on the target era would leak its statistics into every point of the "
            "recovery curve, including fraction 0"
        ),
    ):
        frames = transform_regime_frames(preprocessor)
        X_train, y_train = frames["train"]
        X_toniot, y_toniot = frames["toniot"]
        n_features = int(X_train.shape[1])
        # TON_IoT's OWN attack types, not the three shared families: this breakdown asks what the
        # modern budget buys per modern attack class, and five of the eight 20,000-row families
        # (`ddos`, `injection`, `password`, `ransomware`, `xss`) have no 2015 counterpart at all --
        # the shared map would drop them before the question was asked. Aligned element-wise to
        # `y_toniot` by `regime_families`.
        native_families = regime_families(frames)["toniot"][NATIVE_FAMILY_SET]

        pool, test = frozen_split_indices(y_toniot, seed=RANDOM_SEED)
        target_rate = _positive_rate(y_toniot)
        pool_rate = float((_labels(y_toniot)[pool] == POSITIVE_LABEL).mean())
        test_rate = float((_labels(y_toniot)[test] == POSITIVE_LABEL).mean())

        print(
            f"TON_IoT n={len(X_toniot):,} (attack {target_rate:.4f}), split once on the binary "
            f"label from seed {RANDOM_SEED}:\n"
            f"    frozen test half   n={test.size:,}  normal {1 - test_rate:.2%} / attack "
            f"{test_rate:.2%}   <- every point below is scored on these rows\n"
            f"    fine-tune pool     n={pool.size:,}  normal {1 - pool_rate:.2%} / attack "
            f"{pool_rate:.2%}\n"
            f"    budgets: "
            + ", ".join(
                f"{fraction:.0%} -> n={fraction_indices(y_toniot, pool, fraction).size:,}"
                for fraction in TRANSFER_FRACTIONS
            )
            + f", ceiling -> n={pool.size:,}\n"
            f"    disjoint on row indices: "
            f"{np.intersect1d(pool, test, assume_unique=True).size == 0}   "
            f"feature schema d={n_features}, identical for both halves"
        )

        def _log(run_label: str, model_name: str, scores: dict[str, Any]) -> None:
            if not log:
                return
            log_metrics(
                round_metrics(
                    {
                        "run_id": run_id_for(run_label),
                        "model": model_name,
                        "regime": REGIME,
                        "seed": RANDOM_SEED,
                        "notes": _ROW_NOTE.format(
                            adaptation=scores["adaptation"],
                            fraction=scores["fraction"],
                            n_finetune=scores["n_finetune"],
                            n_features=n_features,
                        ),
                        # The confusion matrix has no column in the frozen 14-field header, so it
                        # stays out of the log; Phase 9 renders it as a figure. `fraction`,
                        # `n_finetune` and `adaptation` are encoded in `run_id` and `notes`.
                        **{
                            key: value
                            for key, value in scores.items()
                            if key
                            not in {"confusion_matrix", "fraction", "n_finetune", "adaptation"}
                        },
                    }
                )
            )

        results: dict[str, dict[str, dict[str, Any]]] = {}
        for name, factory in phase6_models().items():
            curve = recovery_curve(
                factory, X_toniot, y_toniot, source=(X_train, y_train),
                families=native_families, seed=RANDOM_SEED,
            )
            results[name] = curve
            for label, scores in curve.items():
                _log(label, name, scores)

        control = freeze_cost_control(
            X_toniot, y_toniot, families=native_families, seed=RANDOM_SEED
        )
        results[FREEZE_CONTROL_MODEL][FREEZE_CONTROL_LABEL] = control
        _log(FREEZE_CONTROL_LABEL, FREEZE_CONTROL_MODEL, control)

        family_rows = [
            row
            for name, curve in results.items()
            for label, scores in curve.items()
            for row in per_family_rows(
                scores,
                run_id=run_id_for(label),
                model=name,
                regime=REGIME,
                family_set=NATIVE_FAMILY_SET,
                note=(
                    f"one-vs-normal against the frozen half's {NORMAL_FAMILY} rows; "
                    f"budget {scores['fraction']:.2%} of the pool, n_ft={scores['n_finetune']}; "
                    "TON_IoT's own attack types (schema_map.FAMILY_NATIVE_COL)"
                ),
            )
        ]
        if log:
            print(
                f"\nper-family metrics -> {write_per_family_metrics(family_rows)}  "
                f"({len(family_rows)} rows)"
            )

        _print_redundancy(X_toniot, y_toniot)

    _print_per_family_recovery(family_rows)
    _print_recovery_table(results)
    _print_freeze_cost(results)
    _print_zero_shot_contrast(results)
    return results


def _curve_labels(results: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    return list(next(iter(results.values())))


def _print_redundancy(X_toniot: Any, y_toniot: Any) -> None:
    """How much of the frozen test half each budget already contains verbatim.

    See :func:`duplicate_overlap` — this is the caveat the saturating curve is read against, not a
    leakage report. The split is disjoint on row indices; these rows are distinct records that are
    identical in the 22-column shared subspace.
    """
    print(
        f"\n{'=' * 108}\ntarget-era redundancy — how much of the frozen test half each budget "
        "already contains verbatim\n"
        "  Disjoint on row indices (asserted); this counts *value*-level duplicates, which in a "
        "22-column subspace are\n  distinct flows the model cannot tell apart. It does NOT bound "
        "memorization: re-scoring the 1% fit on only\n  the rows whose vector is absent from the "
        "draw costs 0.0021 ROC-AUC, so the recovery generalizes. What it\n  shows is that the "
        "label is near-deterministic in these 22 features (a per-vector majority vote scores\n"
        "  99.8953% on the frame), which is why ~1% of the target labels suffices, and why 1% is a "
        f"lower bound.\n{'-' * 108}"
    )
    for fraction in (*TRANSFER_FRACTIONS, CEILING_FRACTION):
        X_ft, _y_ft, X_test, _y_test = sample_fraction(X_toniot, y_toniot, fraction)
        reproduced, distinct = duplicate_overlap(X_ft, X_test)
        label = CEILING_LABEL if fraction == CEILING_FRACTION else fraction_label(fraction)
        print(
            f"    {label:<8} n_ft={len(X_ft):>7,}  {reproduced:6.2%} of the "
            f"{len(X_test):,} test rows have an exact feature-vector match in the fine-tune sample"
            + (
                f"   (the test half itself holds only {distinct:.2%} distinct vectors)"
                if fraction == TRANSFER_FRACTIONS[0]
                else ""
            )
        )


#: Models summarized in the per-family recovery printout. The dummy is the floor, not a competitor,
#: and is printed on its own line beneath each family rather than folded into the range.
_REAL_MODELS: tuple[str, ...] = (
    "random_forest", "decision_tree", "scratch_mlp", "svm", "scratch_logreg",
)


def _print_per_family_recovery(rows: list[dict[str, Any]]) -> None:
    """Per-TON_IoT-family F1 across the budgets, as the range over the five real models.

    The full table is 6 models x 6 budgets x every family; the file
    ``reports/per_family_metrics.csv`` carries all of it and Phase 9's figure draws it. What is
    worth reading in a run log is which families the modern budget *fails* to recover, so this
    prints the min-max band across the five real models per budget, with the majority-class floor
    beneath it -- an F1 band sitting on that floor is a family nobody has learned to detect.
    """
    if not rows:  # pragma: no cover - run_phase7 always produces rows
        return
    by_key = {(row["run_id"], row["model"], row["family"]): row for row in rows}
    labels = [ZERO_SHOT_LABEL, *(fraction_label(f) for f in TRANSFER_FRACTIONS), CEILING_LABEL]
    families = sorted({row["family"] for row in rows})
    print(
        f"\n{'=' * 108}\nper-family recovery — TON_IoT's own attack types on the frozen test half\n"
        "  Each family is scored one-vs-normal (the family's rows plus every normal row of the "
        "frozen half), so F1 and\n  the majority-class floor stay defined and each family has its "
        "own class balance. F1 shown as the min-max\n  band over the five real models; `dummy` is "
        f"the floor for that family.\n{'-' * 108}\n"
        f"{'family':<13}{'n':>8}  " + "".join(f"{label:>15}" for label in labels)
    )
    for family in families:
        cells = []
        for label in labels:
            values = [
                float(by_key[(run_id_for(label), model, family)]["f1"])
                for model in _REAL_MODELS
                if (run_id_for(label), model, family) in by_key
            ]
            cells.append(f"{min(values):.3f}-{max(values):.3f}" if values else "--")
        reference = by_key[(run_id_for(labels[0]), _REAL_MODELS[0], family)]
        floor = by_key.get((run_id_for(labels[0]), "dummy", family))
        print(
            f"{family:<13}{int(reference['n_family']):>8,}  "
            + "".join(f"{cell:>15}" for cell in cells)
            + (f"   floor {float(floor['f1']):.3f}" if floor is not None else "")
        )
    print(
        f"{'-' * 108}\n  n is the family's row count in the frozen half; each band is measured "
        f"against that family plus the half's\n  "
        f"{int(reference['n_normal']):,} {NORMAL_FAMILY} rows. Full per-model rows: "
        f"{PER_FAMILY_CSV}."
    )


def _print_freeze_cost(results: dict[str, dict[str, dict[str, Any]]]) -> None:
    """The frozen-head ceiling against the same net retrained end to end on the same pool."""
    frozen = results[FREEZE_CONTROL_MODEL][CEILING_LABEL]
    control = results[FREEZE_CONTROL_MODEL][FREEZE_CONTROL_LABEL]
    print(
        f"\n{'=' * 108}\nfreeze cost — {FREEZE_CONTROL_MODEL} at the ceiling budget "
        f"(n_ft={frozen['n_finetune']:,}), head-only vs no freeze\n"
        f"{'-' * 108}\n"
        f"    {'head-only (both hidden layers frozen)':<44} roc_auc={frozen['roc_auc']:.4f}  "
        f"f1={frozen['f1']:.4f}\n"
        f"    {'every layer retrained (control)':<44} roc_auc={control['roc_auc']:.4f}  "
        f"f1={control['f1']:.4f}\n"
        f"    {'difference (control - head-only)':<44} roc_auc="
        f"{control['roc_auc'] - frozen['roc_auc']:+.4f}  "
        f"f1={control['f1'] - frozen['f1']:+.4f}\n"
        "  The gap is what holding the 2015 feature space fixed costs at an unlimited modern "
        "budget. Only the MLP\n  needs this control: the classical models' ceilings are already "
        "full target retrains, and scratch_logreg's\n  objective is convex, so its warm-started "
        "ceiling is the cold-start optimum."
    )


def _print_recovery_table(results: dict[str, dict[str, dict[str, Any]]]) -> None:
    """The RQ2 table: ROC-AUC then F1 per budget, read against the dummy floor and the ceiling."""
    labels = _curve_labels(results)
    first = next(iter(results.values()))[ZERO_SHOT_LABEL]
    floor = results.get("dummy", {}).get(ZERO_SHOT_LABEL)

    print(
        f"\n{'=' * 108}\nRQ2 recovery curve — {RUN_ID_PREFIX}-*  (regime {REGIME})\n"
        f"  frozen TON_IoT test half: n={first['n_test']:,}  "
        f"normal {1 - first['positive_rate']:.2%} / attack {first['positive_rate']:.2%}  "
        f"— identical for every budget\n"
        f"  Lead with ROC-AUC. The 0.5000 line is the no-skill floor, and Phase 6 found every "
        f"model *below* it\n  zero-shot, so a model only stops being anti-correlated once it "
        f"clears 0.5000."
    )
    if floor is not None:
        print(
            f"  Dummy floor on this half: F1 {floor['f1']:.4f} at ROC-AUC {floor['roc_auc']:.4f} "
            f"— an F1 climbing toward {floor['f1']:.2f} may have recovered nothing."
        )

    for metric in ("roc_auc", "f1"):
        header = "".join(f"{label:>10}" for label in labels)
        print(
            f"{'-' * 108}\n{metric:<16}{header}{'gap@0.25':>11}{'recovered':>11}\n"
            f"{'-' * 108}"
        )
        for name, curve in results.items():
            row = "".join(f"{curve[label][metric]:>10.4f}" for label in labels)
            zero, top = curve[ZERO_SHOT_LABEL][metric], curve[CEILING_LABEL][metric]
            quarter = curve[fraction_label(TRANSFER_FRACTIONS[-1])][metric]
            span = top - zero
            # The dummy has no gap to recover (its score is prevalence, not skill), so the share
            # is undefined rather than 0 or 100% -- printed as n/a instead of a nan.
            recovered = f"{(quarter - zero) / span:>10.1%}" if abs(span) > 1e-12 else f"{'n/a':>10}"
            print(f"{name:<16}{row}{top - quarter:>+11.4f}{recovered}")

    print(f"{'-' * 108}")
    print(
        "  gap@0.25 = ceiling - 25% budget (how much the last three quarters of the pool would "
        "still buy).\n  recovered = share of the (ceiling - zero-shot) gap the 25% budget closes."
    )
    if floor is not None:
        print("\n  Budget at which each model's ROC-AUC first clears the dummy floor "
              f"({floor['roc_auc']:.4f}) on the frozen half:")
        for name, curve in results.items():
            crossing = next(
                (
                    label
                    for label in labels
                    if label != CEILING_LABEL and curve[label]["roc_auc"] > NO_SKILL_AUC
                ),
                None,
            )
            if crossing is None:
                verdict = (
                    f"never within the curve (ceiling {curve[CEILING_LABEL]['roc_auc']:.4f})"
                    if curve[CEILING_LABEL]["roc_auc"] <= NO_SKILL_AUC
                    else f"only at the ceiling ({curve[CEILING_LABEL]['roc_auc']:.4f})"
                )
            else:
                verdict = (
                    f"{crossing} (n_ft={curve[crossing]['n_finetune']:,}, "
                    f"roc_auc={curve[crossing]['roc_auc']:.4f})"
                )
            print(f"    {name:<16}{verdict}")


def _print_zero_shot_contrast(results: dict[str, dict[str, dict[str, Any]]]) -> None:
    """Fraction 0 re-measured on the frozen half vs Phase 6's full-TON_IoT ``cross_era`` rows.

    Phase 6 measured the same unadapted models on all 211,043 rows, which *includes* this phase's
    fine-tune pool, so those rows cannot be this curve's left endpoint (``deviations.md`` §3.12).
    They are still the right thing to compare against: a large gap would mean the frozen half is
    not representative of the target era, which would undercut every recovery number.
    """
    logged = read_metrics()
    from .evaluate import RUN_ID as PHASE6_RUN_ID  # noqa: PLC0415 - avoids shadowing RUN_ID_PREFIX

    print(
        f"\n{'=' * 108}\nfraction-0 sanity check — re-measured on the frozen half vs "
        f"{PHASE6_RUN_ID} on the full 211,043 rows\n"
        "  Not the same test set, so not the same number; the point is that the difference is "
        "small enough for the\n  frozen half to stand in for the target era.\n"
        f"{'-' * 108}\n"
        f"{'model':<16}{'AUC frozen':>12}{'AUC full':>10}{'Δ':>9}"
        f"{'F1 frozen':>12}{'F1 full':>10}{'Δ':>9}{'n frozen':>11}{'n full':>10}"
    )
    for name, curve in results.items():
        row = logged.get((PHASE6_RUN_ID, name, CROSS_ERA))
        zero = curve[ZERO_SHOT_LABEL]
        if row is None or not row.get("roc_auc"):
            print(f"{name:<16}{zero['roc_auc']:>12.4f}{'--':>10}{'--':>9}"
                  f"{zero['f1']:>12.4f}{'--':>10}{'--':>9}{zero['n_test']:>11,}{'--':>10}")
            continue
        print(
            f"{name:<16}{zero['roc_auc']:>12.4f}{float(row['roc_auc']):>10.4f}"
            f"{zero['roc_auc'] - float(row['roc_auc']):>+9.4f}"
            f"{zero['f1']:>12.4f}{float(row['f1']):>10.4f}"
            f"{zero['f1'] - float(row['f1']):>+9.4f}"
            f"{zero['n_test']:>11,}{int(row['n_test']):>10,}"
        )
    print(
        f"{'-' * 108}\n  The full-set rows are Phase 6's and stay Phase 6's: they are not "
        "re-logged under a Phase 7 run_id."
    )


# --- Entry point ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.transfer",
        description=(
            "Phase 7: freeze half of TON_IoT as a permanent test set, adapt every Phase 4-5 "
            "model on stratified 1/5/10/25% fractions of the other half, and upsert the recovery "
            "curve into reports/metrics.csv (one run_id per budget, plus the re-measured "
            "zero-shot point and the full-budget ceiling). The Preprocessor is never refit."
        ),
    )
    parser.parse_args(argv)

    set_seeds()
    results = run_phase7()
    rows = sum(len(curve) for curve in results.values())
    run_ids = len({run_id_for(label) for curve in results.values() for label in curve})
    print(
        f"\nlogged {rows} rows across {run_ids} run_ids -> {METRICS_CSV}\n"
        f"per-family rows (TON_IoT's own attack types, frozen test half) -> {PER_FAMILY_CSV}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
