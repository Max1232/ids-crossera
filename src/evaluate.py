"""Evaluation — metrics and the two regimes (Phase 6, RQ1).

Lead with F1 and ROC-AUC (accuracy is misleading under imbalance), plus precision/recall,
confusion matrices, and per-shared-family breakdowns. Headline number = the delta between the
in-distribution and cross-era (zero-shot) regimes. All runs are logged to reports/metrics.csv.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import heapq
import json
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
    roc_curve,
)

from .config import (
    CONFUSION_JSON,
    METRICS_CSV,
    PER_FAMILY_CSV,
    PREPROCESSOR,
    RANDOM_SEED,
    ROC_JSON,
    set_seeds,
)

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


# --- ROC curve vertices, for Phase 9's ROC figure ------------------------------------------
#
# `evaluate` reports ROC-AUC as a scalar; the figure needs the curve behind it. Recomputing that in
# Phase 9 would mean re-fitting eighteen models, so -- exactly as with the confusion matrices -- the
# curve is captured where it is already paid for and persisted to a sidecar (`write_roc_curves`).
#
# TWO CHOICES MAKE THAT SIDECAR SMALL WITHOUT MOVING THE NUMBER IT ILLUSTRATES.
#
# 1. COUNTS, NOT RATES. A vertex is stored as the integer (false positives, true positives) pair,
#    with the two class sizes recorded once per curve; `fpr = fp / n_negative` and
#    `tpr = tp / n_positive` reconstruct sklearn's output EXACTLY. Storing rounded floats instead
#    would introduce a rounding error of its own on top of the simplification below, and the counts
#    are also shorter to write down.
#
# 2. AREA-PRESERVING SIMPLIFICATION. The raw vertex lists are large -- the MLP's cross-era curve has
#    33,413 of them, and all 36 (three conditions x six models x two regimes) come to ~450,000
#    vertices, i.e. several MB of committed JSON for a figure 600 px wide. They are therefore thinned
#    to at most `ROC_CURVE_BUDGET` vertices each. The thinning rule is the load-bearing part: a
#    figure whose curve encloses a different area than the AUC printed beside it is wrong in exactly
#    the way a persisted artifact is supposed to prevent. Measured on this project's twelve
#    full-condition curves, at a comparable point budget:
#
#      uniform spacing along the curve   worst |Δ AUC| 6.5e-05
#      Douglas-Peucker (perpendicular)   worst |Δ AUC| 5.8e-06
#      area greedy, sign-aware           worst |Δ AUC| 1.5e-07   <- `simplify_roc`
#
#    Perpendicular-distance simplification is the wrong objective here: it bounds how far the drawn
#    line strays, not how much area that costs, and on a curve that is concave (or, cross-era,
#    convex) throughout, every deviation has the same sign and they accumulate. `simplify_roc`
#    optimises the quantity that actually matters and lets the residuals cancel; the result is
#    checked against `roc_auc_score` before anything is written.

#: Maximum vertices kept per stored ROC curve. 512 holds the twelve full-condition curves to a worst
#: area error of 1.5e-07 -- two orders inside the log's six-decimal precision -- in ~4,600 vertices
#: total, and is already far more resolution than a 6.5-inch-wide figure can show.
ROC_CURVE_BUDGET: int = 512

#: How far the simplified curve's trapezoidal area may sit from ``roc_auc_score`` before
#: :func:`roc_points` refuses to write it. Set to the log's own precision: a stored curve that
#: disagrees with the logged scalar in a digit the log records is not an illustration of that scalar.
ROC_CURVE_TOLERANCE: float = 1e-6

# WHY THE SCORES ARE SNAPPED BEFORE THE CURVE IS BUILT -- AND WHY ONLY SOMETIMES.
#
# `RandomForestClassifier(n_jobs=-1).predict_proba` accumulates the per-tree probabilities in
# whatever order the worker threads finish, so the SAME fitted forest returns scores that differ by
# up to 4.4e-16 between two calls in one process. Nothing logged can see that: every scalar in
# `metrics.csv` is rounded to six decimals and the confusion matrix counts hard predictions. A ROC
# curve can, because it resolves individual scores -- the last-bit noise splits and re-merges tied
# thresholds, so the raw vertex list came out different on every run and the committed sidecar
# showed a spurious diff after each `./run.sh`. Snapping to `ROC_SCORE_DECIMALS` removes it.
# Measured over three separate processes, all six forest curves come out identical at 13, 12 and 11
# decimals and still vary at 14 -- so the snap has to be coarse enough to clear the noise by a wide
# margin, not merely coarser than it.
#
# The snap is NOT unconditional, and the reason is a measurement rather than caution. The decision
# tree's leaf fractions contain 110 pairs that are mathematically equal but land ~7e-18 apart in
# float64; `roc_auc_score` treats those as distinct thresholds, so the AUC already committed to the
# log reflects the split, and merging them moves the curve's area by 2.24e-05 -- twenty times past
# what this sidecar promises. Rather than name a model, the snap is applied only where it
# demonstrably does not move the AUC past `ROC_CURVE_TOLERANCE`, and each curve records which it
# got, so the file says out loud what was done to it.

#: Decimals at which two scores are taken to be the same operating point. 12 is ~2,300x the observed
#: 4.4e-16 reduction noise and sits in the middle of the 13-to-11 band that reproduced across
#: processes, while still being far finer than any real gap between two forest probabilities -- it
#: separates float noise from signal rather than coarsening the curve.
ROC_SCORE_DECIMALS: int = 12


def _trapezoid_delta(x: Any, y: Any, left: int, mid: int, right: int) -> float:
    """Signed change in trapezoidal area from dropping vertex ``mid`` out of ``left-mid-right``."""
    kept = (x[right] - x[left]) * (y[left] + y[right]) / 2.0
    original = ((x[mid] - x[left]) * (y[left] + y[mid]) / 2.0
                + (x[right] - x[mid]) * (y[mid] + y[right]) / 2.0)
    return float(kept - original)


def simplify_roc(fpr: Any, tpr: Any, budget: int = ROC_CURVE_BUDGET) -> np.ndarray:
    """Indices of a ``<= budget``-vertex subsequence of the ROC polyline that keeps its area.

    A Visvalingam-style greedy: repeatedly drop the interior vertex whose removal changes the
    trapezoidal area least, keeping the two endpoints. The refinement that does the real work is
    **sign awareness** -- each removal's area change is signed, and the running total is tracked, so
    at every step the cheapest available removal *of the sign that pulls the total back toward zero*
    is preferred. Without it the residuals of a uniformly concave (or convex) curve all point the
    same way and simply add up; with it they cancel, which is the difference between a worst-case
    2e-06 and 1.5e-07 on this project's curves at the same budget.

    ``fpr``/``tpr`` are the rate vectors as :func:`sklearn.metrics.roc_curve` returns them. Returns
    the kept indices in increasing order; a curve already at or under ``budget`` is returned whole.
    """
    x, y = np.asarray(fpr, dtype=float), np.asarray(tpr, dtype=float)
    n = x.shape[0]
    if n <= max(budget, 2):
        return np.arange(n)

    previous, following = np.arange(-1, n - 1), np.arange(1, n + 1)
    alive = np.ones(n, dtype=bool)
    version = np.zeros(n, dtype=np.int64)  # lazy-deletion stamp: a stale heap entry is discarded
    positive: list[tuple[float, int, int]] = []
    negative: list[tuple[float, int, int]] = []

    def offer(index: int) -> None:
        if 0 < index < n - 1 and alive[index]:
            change = _trapezoid_delta(x, y, previous[index], index, following[index])
            heapq.heappush(
                positive if change >= 0 else negative, (abs(change), index, version[index])
            )

    for index in range(1, n - 1):
        offer(index)

    def cheapest(heap: list[tuple[float, int, int]]) -> tuple[float, int] | None:
        while heap:
            magnitude, index, stamp = heap[0]
            if alive[index] and stamp == version[index]:
                return magnitude, index
            heapq.heappop(heap)
        return None

    running, remaining = 0.0, n
    while remaining > budget:
        candidates = [
            (cheapest(positive), 1), (cheapest(negative), -1),
        ]
        available = [(entry, sign) for entry, sign in candidates if entry is not None]
        if not available:  # pragma: no cover - only if every interior vertex is already gone
            break
        if len(available) == 2:
            wanted = -1 if running > 0 else 1
            available.sort(key=lambda pair: (pair[1] != wanted, pair[0][0]))
        (magnitude, index), sign = available[0]
        heapq.heappop(positive if sign == 1 else negative)
        running += sign * magnitude
        alive[index] = False
        remaining -= 1
        before, after = int(previous[index]), int(following[index])
        following[before] = after
        if after < n:
            previous[after] = before
        version[before] += 1
        offer(before)
        if after < n - 1:
            version[after] += 1
            offer(after)
    return np.flatnonzero(alive)


def roc_points(y_true: Any, scores: Any, budget: int = ROC_CURVE_BUDGET) -> dict[str, Any]:
    """One ROC curve, as integer count vertices, ready for the sidecar.

    ``scores`` must be the continuous positive-class score :func:`positive_scores` returns -- the
    same array :func:`evaluate` hands to ``roc_auc_score`` -- so the ``roc_auc`` reported here is
    bit-for-bit the value that reaches ``reports/metrics.csv``, not a second measurement of it.

    Refuses to return a curve whose own trapezoidal area drifts past
    :data:`ROC_CURVE_TOLERANCE` from that scalar; see the block comments above for why the
    simplification -- and the score snap that makes the result reproducible -- can promise that.
    """
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores).ravel()
    n_positive = int((y_true == POSITIVE_LABEL).sum())
    n_negative = int(y_true.shape[0] - n_positive)
    # From the RAW scores, so this is the identical float `evaluate` reports and `log_metrics`
    # rounds. The snap below only ever affects which vertices the curve is drawn through.
    auc = float(roc_auc_score(y_true, scores))

    snapped = np.round(scores, ROC_SCORE_DECIMALS)
    use_snapped = abs(float(roc_auc_score(y_true, snapped)) - auc) <= ROC_CURVE_TOLERANCE
    fpr, tpr, _ = roc_curve(y_true, snapped if use_snapped else scores)
    # Back to exact counts. sklearn divides by the class sizes, so multiplying through recovers
    # integers to float noise; assert that rather than trusting it, since a non-integral result
    # would mean the curve is not the one these class sizes describe.
    false_positives = np.rint(fpr * n_negative)
    true_positives = np.rint(tpr * n_positive)
    if (max(np.abs(fpr * n_negative - false_positives).max(),
            np.abs(tpr * n_positive - true_positives).max()) > 1e-6):
        raise RuntimeError(  # pragma: no cover - would mean roc_curve changed its contract
            "ROC vertices do not resolve to integer counts against "
            f"n_negative={n_negative}, n_positive={n_positive}; refusing to store them as counts."
        )

    keep = simplify_roc(fpr, tpr, budget)
    drawn = float(np.trapz(tpr[keep], fpr[keep]))
    if abs(drawn - auc) > ROC_CURVE_TOLERANCE:
        raise RuntimeError(
            f"simplifying this ROC curve to {len(keep)} of {fpr.shape[0]} vertices moved its area "
            f"from {auc:.9f} to {drawn:.9f} (drift {abs(drawn - auc):.2e} > "
            f"{ROC_CURVE_TOLERANCE:.0e}). Raise ROC_CURVE_BUDGET rather than storing a curve that "
            "disagrees with the ROC-AUC logged beside it."
        )
    return {
        "n_negative": n_negative,
        "n_positive": n_positive,
        "n_vertices": int(fpr.shape[0]),
        "scores_snapped": bool(use_snapped),
        "false_positives": [int(value) for value in false_positives[keep]],
        "true_positives": [int(value) for value in true_positives[keep]],
        "roc_auc": auc,
        "auc_simplified": drawn,
    }


# --- Per-attack-family breakdowns, for Phase 9's per-family figures ------------------------
#
# WHAT A "PER-FAMILY F1" IS HERE, AND WHY IT NEEDS DEFINING. An attack family is a set of rows that
# are ALL label 1 -- `dos` rows are attacks by construction. F1 for the attack class on that set
# alone is degenerate: precision is 1 whenever anything is predicted positive, so the "F1" would be
# a monotone function of recall and nothing else, and it could never be compared against the
# majority-class floor the rest of this project reads every F1 against.
#
# So each family is scored ONE-VS-NORMAL: the family's rows PLUS every normal row of the same
# evaluation set. The question that answers is the one the report asks -- "can this detector still
# separate DoS traffic from benign traffic in 2019-20?" -- and it keeps precision, F1, ROC-AUC and
# the dummy floor all defined and all comparable to the aggregate row they decompose. Each family's
# subset has its OWN class balance (a 10,000-row family against 25,000 normals is not the aggregate
# 76.31% attack), so `n_family`, `n_normal` and `positive_rate` are recorded per row and no
# per-family F1 may be read without them.
#
# TWO FAMILY VOCABULARIES, and conflating them is the trap this project has already tripped over
# once (see schema_map.FAMILY_NATIVE_COL):
#
#   "shared"  -- schema_map's SHARED_FAMILIES map. THREE attack families, and only three:
#                `dos`, `scanning`, `backdoor`. This is the only vocabulary in which a cross-era
#                comparison is meaningful, because it is the only one both eras speak.
#   "native"  -- each dataset's own delivered levels. Used for the WITHIN-era Phase 7 recovery
#                breakdown over TON_IoT's own attack types, which the shared map cannot express.
#
# Rows carry `family_set` for exactly that reason: a join that mixed the two would silently compare
# UNSW `reconnaissance` against TON_IoT `scanning` as if they were different families, or read a
# TON_IoT-only family as a cross-era result.

#: The benign level in both vocabularies -- the comparison class every attack family is scored
#: against, never a family in its own right.
NORMAL_FAMILY: str = "normal"

#: The two vocabularies above, as they are spelled in the ``family_set`` column.
SHARED_FAMILY_SET: str = "shared"
NATIVE_FAMILY_SET: str = "native"


def per_family_metrics(
    y_true: Any, y_pred: Any, scores: Any, families: Any
) -> dict[str, dict[str, Any]]:
    """One-vs-normal metrics for every attack family present in ``families``.

    ``families`` is a per-row label vector aligned to ``y_true``: :data:`NORMAL_FAMILY` on the
    benign rows, the family name on an attack row, and missing on a row belonging to no family this
    vocabulary names (which is every non-shared attack level under the ``shared`` map -- 101,043 of
    TON_IoT's rows). Missing rows are excluded from every subset rather than pooled into one, which
    is what "the per-family analysis is restricted to the shared families" means operationally.

    Takes the already-computed ``y_pred`` and ``scores`` rather than a model, so a per-family
    breakdown costs a boolean mask per family and **no second prediction pass** over a 211,043-row
    frame -- the same argument that keeps the confusion matrix inside :func:`evaluate`.

    Two integrity checks, both for failures that would otherwise produce a plausible number:

    * a family's rows must be **all attack** and the normal rows **all benign**. If they are not,
      the family column and the binary label disagree, and every "family vs normal" subset below
      would be mislabelled. ``schema_map`` cross-tabbed these at build time (§3.4); this re-checks
      it against the actual evaluation rows.
    * a vocabulary with no normal rows at all cannot express one-vs-normal, so it raises rather than
      returning a table of degenerate single-class scores.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    scores = np.asarray(scores).ravel()
    # `.to_numpy(na_value=None)` for the pandas "string" dtype schema_map emits (whose NA is
    # `pd.NA`, which is not comparable with `==`); a bare object array otherwise.
    labels = (
        families.to_numpy(dtype=object, na_value=None)
        if hasattr(families, "to_numpy")
        else np.asarray(families, dtype=object)
    )
    labels = np.asarray(labels, dtype=object).ravel()
    if labels.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"the family vector has {labels.shape[0]:,} entries against {y_true.shape[0]:,} "
            "labels; a per-family breakdown would be scored against the wrong rows"
        )

    named = np.array([item is not None and item == item for item in labels], dtype=bool)
    is_normal = named & (labels == NORMAL_FAMILY)
    if not is_normal.any():
        raise ValueError(
            "the family vector holds no `normal` rows, so no family can be scored one-vs-normal. "
            "Pass the vocabulary's benign level as "
            f"{NORMAL_FAMILY!r}, not as a missing value."
        )
    if (y_true[is_normal] != 0).any():
        raise ValueError(
            f"{int((y_true[is_normal] != 0).sum()):,} rows are family {NORMAL_FAMILY!r} but carry "
            "the attack label; the family column and the binary label disagree"
        )

    out: dict[str, dict[str, Any]] = {}
    for family in sorted({str(item) for item in labels[named]} - {NORMAL_FAMILY}):
        in_family = named & (labels == family)
        if (y_true[in_family] != POSITIVE_LABEL).any():
            raise ValueError(
                f"{int((y_true[in_family] != POSITIVE_LABEL).sum()):,} rows of family {family!r} "
                "carry the normal label; the family column and the binary label disagree"
            )
        subset = in_family | is_normal
        y_sub, pred_sub, score_sub = y_true[subset], y_pred[subset], scores[subset]
        out[family] = {
            "n_family": int(in_family.sum()),
            "n_normal": int(is_normal.sum()),
            "n_test": int(subset.sum()),
            "positive_rate": float((y_sub == POSITIVE_LABEL).mean()),
            "accuracy": float(accuracy_score(y_sub, pred_sub)),
            "precision": float(
                precision_score(y_sub, pred_sub, pos_label=POSITIVE_LABEL, zero_division=0)
            ),
            "recall": float(
                recall_score(y_sub, pred_sub, pos_label=POSITIVE_LABEL, zero_division=0)
            ),
            "f1": float(f1_score(y_sub, pred_sub, pos_label=POSITIVE_LABEL, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_sub, score_sub)),
            "balanced_accuracy": float(balanced_accuracy_score(y_sub, pred_sub)),
            "macro_f1": float(f1_score(y_sub, pred_sub, average="macro", zero_division=0)),
        }
    return out


def evaluate(
    model: Any, X: Any, y: Any, *, with_roc_curve: bool = False, families: Any = None
) -> dict[str, Any]:
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

    ``with_roc_curve`` adds a ``roc_curve`` key holding the curve behind the ``roc_auc`` scalar (see
    :func:`roc_points`), for Phase 9's ROC figure. It changes **nothing** that is logged: like
    ``confusion_matrix`` it has no column in the frozen :data:`METRICS_HEADER` and is stripped at the
    logging boundary, and the scalar it sits beside is the very ``roc_auc`` this function already
    returns. It defaults off because :func:`run_regimes` is the only caller that needs it and the
    curve costs a sort plus a simplification pass per evaluation -- Phase 7 runs 37 of those and
    renders none of them.

    ``families`` behaves the same way: pass the per-row family vector (see
    :func:`per_family_metrics`) and the result gains a ``per_family`` key. It is computed from the
    ``y_pred`` and ``scores`` this call already produced, so it costs no second pass over the
    frame, and -- like the two above -- it has no column in the frozen header and is stripped at the
    logging boundary, reaching disk through ``reports/per_family_metrics.csv`` instead.
    """
    y_true = np.asarray(y).ravel()
    y_pred = np.asarray(model.predict(X)).ravel()
    scores = positive_scores(model, X)

    return {
        **({"roc_curve": roc_points(y_true, scores)} if with_roc_curve else {}),
        **(
            {"per_family": per_family_metrics(y_true, y_pred, scores, families)}
            if families is not None
            else {}
        ),
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
                X_toniot: Any, y_toniot: Any,
                with_roc_curve: bool = True,
                families: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Evaluate one trained model in both regimes side by side.

    Returns ``{"in_distribution": {...}, "cross_era": {...}}``. The reported drift is the
    per-metric delta (in_distribution - cross_era).

    ``with_roc_curve`` is on by default because Phase 9's ROC figure needs both regimes' curves and
    this is the one place they can be taken from the same single fit and the same score pass that
    produced the logged ``roc_auc``. Collecting them here is deliberate: the curve is derived from
    ``predict_proba``/``decision_function``, which :func:`sealed` leaves untouched, so it is a
    read-only measurement taken *inside* the seal rather than a second, unsealed pass over the two
    evaluation sets.

    ``families`` is optional and keyed by regime (``{IN_DISTRIBUTION: ..., CROSS_ERA: ...}``); each
    value is the per-row family vector for that regime's frame, and it produces the ``per_family``
    breakdown Phase 9's cross-era per-family figure reads. Note the two vectors are **not** the same
    vocabulary applied twice by accident: the shared map is the only one both eras speak, and
    :func:`regime_families` is what produces the aligned pair.

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
        phase6-crossera-no_conn_state  # the conn_state ablation, same design. It shares the
                                    #   proto ablation's WIDTH (both d=18) and is a different
                                    #   experiment; only the run_id separates them.
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
        families = families or {}
        return {
            "in_distribution": evaluate(
                model, X_unsw_test, y_unsw_test, with_roc_curve=with_roc_curve,
                families=families.get(IN_DISTRIBUTION),
            ),
            "cross_era": evaluate(
                model, X_toniot, y_toniot, with_roc_curve=with_roc_curve,
                families=families.get(CROSS_ERA),
            ),
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
        "phase6-crossera-no_conn_state"     # same model + regime, `conn_state` ablated (also d=18)
        "phase7-recovery-f0.05"             # same model + regime, 5% fine-tune budget
        "phase7-recovery-f0.25"             # ... and 25%

    Without the suffixes, every ablation would land on its unablated row and the recovery
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


# --- The per-family sidecar ----------------------------------------------------------------
#
# WHY THIS IS NOT reports/metrics.csv. Two reasons, either sufficient. The run log's 14-column
# header is FROZEN and carries no family dimension. And its upsert key is (run_id, model, regime):
# a `dos` row and the aggregate row it decomposes would collide on that key, so the last one written
# would silently replace the other -- the headline cross-era number overwritten by a
# 10,000-row family's. Per-family rows therefore live in their own file with their own key.
#
# Everything else is deliberately identical to `log_metrics`, because the same properties are wanted:
# a frozen header, an upsert rather than an append (so `./run.sh` twice leaves the file
# byte-identical and reproducibility is a `git diff`), sorted keys, and floats rounded to
# METRIC_DECIMALS. Two phases write into it -- Phase 6 the cross-era shared-family rows, Phase 7 the
# frozen-test-half native-family rows -- and the upsert is what lets the second one land beside the
# first instead of on top of it.

#: The per-family header. Frozen on the same terms as :data:`METRICS_HEADER`: rows are carried
#: forward by column name, so appending is safe and renaming or reordering silently blanks a field
#: in every row already committed.
PER_FAMILY_HEADER = [
    "run_id", "model", "regime", "family_set", "family", "seed",
    "n_family", "n_normal", "n_test", "positive_rate",
    "precision", "recall", "f1", "roc_auc", "accuracy", "balanced_accuracy", "macro_f1",
    "notes",
]

#: The identity of a per-family row. ``family_set`` is part of the key, not decoration: the shared
#: and native vocabularies both contain ``dos``, ``scanning`` and ``backdoor``, measured over
#: different row populations, and without it the two would overwrite each other.
PER_FAMILY_KEY: tuple[str, ...] = ("run_id", "model", "regime", "family_set", "family")


def read_per_family_metrics(path: Any = PER_FAMILY_CSV) -> dict[tuple[str, ...], dict[str, str]]:
    """Read ``reports/per_family_metrics.csv`` into ``{key: row}``. Same contract as
    :func:`read_metrics`: strings out, a missing file is an empty log, a stale header raises."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != PER_FAMILY_HEADER:
            raise ValueError(
                f"{path} has header {reader.fieldnames}, expected {PER_FAMILY_HEADER}. The "
                "per-family header is frozen; reconcile the file by hand rather than letting a run "
                "rewrite it, or every existing row silently misaligns."
            )
        return {
            tuple(record.get(key) or "" for key in PER_FAMILY_KEY): {
                column: (record.get(column) or "") for column in PER_FAMILY_HEADER
            }
            for record in reader
        }


def write_per_family_metrics(
    rows: Any, path: Any = PER_FAMILY_CSV
) -> Path:
    """Upsert per-family rows into ``reports/per_family_metrics.csv``, keyed on
    :data:`PER_FAMILY_KEY`.

    Takes the whole batch a phase produced rather than one row at a time (Phase 6 hands over 36
    rows, Phase 7 over 288), because the file is rewritten in full on every write and doing that
    once per phase rather than once per row is the difference between one pass and three hundred.
    Rows belonging to other keys are carried forward unmodified, so the two phases' blocks coexist.
    """
    path = Path(path)
    existing = read_per_family_metrics(path)
    for row in rows:
        incoming = {column: row.get(column, "") for column in PER_FAMILY_HEADER}
        existing[tuple(str(incoming[key]) for key in PER_FAMILY_KEY)] = incoming

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=PER_FAMILY_HEADER, lineterminator=METRICS_LINETERMINATOR
        )
        writer.writeheader()
        for key in sorted(existing):
            writer.writerow(existing[key])
    return path


def per_family_rows(
    scores: dict[str, Any], *, run_id: str, model: str, regime: str, family_set: str, note: str
) -> list[dict[str, Any]]:
    """Turn one :func:`evaluate` result's ``per_family`` block into rows for the sidecar.

    Spelled once here rather than inline in each phase so the two producers cannot drift on the
    seed, the rounding or the column set.
    """
    return [
        round_metrics({
            "run_id": run_id,
            "model": model,
            "regime": regime,
            "family_set": family_set,
            "family": family,
            "seed": RANDOM_SEED,
            "notes": note,
            **metrics,
        })
        for family, metrics in sorted(scores.get("per_family", {}).items())
    ]


# --- Phase 6 — the zero-shot cross-era run (RQ1) -------------------------------------------
#
# One `run_id` per condition; see the scheme documented in `run_regimes`. The three conditions this
# phase owns:
#
#   phase6-crossera                full shared feature set (d=22)
#   phase6-crossera-no_proto       the `proto` ablation (d=18), retrained on the train fold
#   phase6-crossera-no_conn_state  the `conn_state` ablation (d=18), same design
#
# THE TWO ABLATIONS ARE DIFFERENT EXPERIMENTS THAT HAPPEN TO SHARE A WIDTH. `protocol` and
# `conn_state` each encode to exactly four one-hot levels, so both conditions run at d=18. `d`
# therefore does NOT identify a condition -- only the `run_id` and the `notes` string do. Never
# write "the d=18 ablation"; name the feature.
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
# Dropping one feature's one-hot columns from the transformed matrix is *exactly* equivalent to
# refitting the Preprocessor without that feature -- one-hot encoding of one categorical column is
# independent of the others, and the numeric z-score parameters do not see it at all -- so it
# needs no second preprocessor artifact and cannot disturb the frozen Phase 3 fit. That holds for
# both ablated categoricals.
#
# WHY `conn_state` GETS THE SAME TREATMENT. Unlike `proto`, whose hazard is a learned shortcut going
# inert, the `conn_state` hazard is that the feature is *ours*: UNSW ships Argus `state` codes and
# TON_IoT ships Zeek `conn_state` codes, the two vocabularies share ZERO tokens, and the coarse
# completed / reset / no-response collapse that bridges them is a modelling decision we invented
# (schema_map.STATE_COLLAPSE, and see deviations.md 3.2). It is also badly asymmetric across the two
# eras -- measured on the train fold against TON_IoT, `reset` is 0.0421% vs 23.6757% (~560x) and the
# RARE_BUCKET `other` level is 0.0057% vs 11.0556% (~1,900x). So part of the RQ1 drop could be our
# own collapse carrying an instrumentation change rather than attacker evolution, and the only way
# to bound that is to retrain without the feature and difference the deltas. The proposal committed
# to evaluating the cross-era run with and without it; this condition is that commitment.

#: Phase 6's three conditions. Fixed labels, never timestamps (see :func:`log_metrics`).
#:
#: The two ablation values are load-bearing STRINGS, not just identifiers: ``log_metrics`` upserts on
#: ``(run_id, model, regime)`` and never deletes, so editing one of these spellings orphans the rows
#: already committed under the old one and silently duplicates the condition under the new one.
RUN_ID: str = "phase6-crossera"
PROTO_ABLATION_RUN_ID: str = f"{RUN_ID}-no_proto"
CONN_STATE_ABLATION_RUN_ID: str = f"{RUN_ID}-no_conn_state"

#: The regime labels, spelled identically to ``models.baselines.REGIME`` so Phase 4's
#: in_distribution rows and Phase 6's cross_era rows join on the pair.
IN_DISTRIBUTION: str = "in_distribution"
CROSS_ERA: str = "cross_era"

#: The categorical columns the two ablations remove, as ``schema_map`` spells them. The one-hot
#: columns each expands to are derived from the fitted encoder rather than pattern-matched on a
#: prefix, so a renamed level cannot silently ablate nothing.
PROTOCOL_FEATURE: str = "protocol"
CONN_STATE_FEATURE: str = "conn_state"

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


def regime_raw_frames() -> dict[str, Any]:
    """The three **harmonized but untransformed** frames Phase 6 works over, in one place.

    ``{"train": ..., "unsw_test": ..., "toniot": ...}``, straight off the Phase 2 parquets. Factored
    out of :func:`transform_regime_frames` because :func:`regime_families` needs the same three
    frames for their *label* columns, and two independent spellings of "the train fold" -- one of
    which re-draws the seeded split -- is precisely how the family vector would end up aligned to
    different rows than the metrics it decomposes.
    """
    from .preprocess import (  # noqa: PLC0415 - keeps pandas out of the import graph
        load_holdout,
        load_source,
        load_target,
        split_source,
    )

    train_fold, _val_fold = split_source(load_source(), seed=RANDOM_SEED)
    return {"train": train_fold, "unsw_test": load_holdout(), "toniot": load_target()}


def regime_families(frames: dict[str, tuple[Any, Any]]) -> dict[str, dict[str, Any]]:
    """Per-row family vectors aligned to :func:`transform_regime_frames`'s three frames.

    Returns ``{frame_name: {family_set: labels}}`` over both vocabularies
    (:data:`SHARED_FAMILY_SET` and :data:`NATIVE_FAMILY_SET`; see ``schema_map``).

    The ``(X, y)`` pairs :func:`transform_regime_frames` returns carry no family column -- the
    ``Preprocessor`` emits the model matrix and nothing else -- so this re-reads the same three
    frames through :func:`regime_raw_frames` and hands back their label columns. **Alignment is
    then asserted element-wise against the ``y`` the metrics were computed from**, not assumed from
    the row count: a reordering, a re-drawn fold or a rebuilt parquet would all leave the lengths
    equal while silently attributing one family's predictions to another, and the resulting figure
    would look entirely reasonable.
    """
    raw = regime_raw_frames()
    from .preprocess import LABEL_COL  # noqa: PLC0415 - keeps pandas out of the import graph
    from .schema_map import FAMILY_LABEL_COL, FAMILY_NATIVE_COL  # noqa: PLC0415

    out: dict[str, dict[str, Any]] = {}
    for name, frame in raw.items():
        _X, y = frames[name]
        labels = np.asarray(y).ravel().astype("int64")
        family_labels = np.asarray(frame[LABEL_COL]).ravel().astype("int64")
        if labels.shape != family_labels.shape or not np.array_equal(labels, family_labels):
            raise RuntimeError(
                f"the {name} family vector does not align row-for-row with the frame the metrics "
                f"were measured on ({family_labels.shape[0]:,} vs {labels.shape[0]:,} rows, labels "
                "equal: "
                f"{labels.shape == family_labels.shape and bool(np.array_equal(labels, family_labels))}). "
                "A per-family breakdown drawn from this would attribute predictions to the wrong "
                "families. Rebuild the parquets with `python -m src.schema_map --build`."
            )
        out[name] = {
            SHARED_FAMILY_SET: frame[FAMILY_LABEL_COL],
            NATIVE_FAMILY_SET: frame[FAMILY_NATIVE_COL],
        }
    return out


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
    from .preprocess import LABEL_COL  # noqa: PLC0415 - keeps pandas out of the import graph

    raw = regime_raw_frames()

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


def categorical_columns(preprocessor: Any, feature: str) -> tuple[str, ...]:
    """The transformed column names one categorical feature's one-hot block expands to.

    Derived from ``encoder_.categories_`` **positionally** rather than by matching a
    ``f"{feature}_"`` prefix, so an ablation cannot silently drop nothing (a renamed feature) or too
    much (a level of a *different* feature that happens to share the prefix -- ``protocol_other``,
    ``service_other`` and ``conn_state_other`` all exist in the fitted schema).

    Parameterized over ``feature`` rather than duplicated per ablation: Phase 6 ablates two of the
    three categoricals, and a second copy of the guards below would be free to drift away from this
    one -- the same argument :func:`evaluate` makes for not keeping a private metric helper.

    Three guards, and note which failure each owns -- the second and third are NOT redundant:

    * ``feature`` must be one of the fitted categoricals, or the ablation has nothing to remove.
    * ``categorical_features_`` and ``categories_`` must be the same length, because ``start`` is a
      prefix sum over the widths of every *earlier* feature. That is vacuous for ``protocol`` at
      index 0 (``start`` is 0 however wrong the widths are) and load-bearing for ``conn_state`` at
      index 2, whose block starts at 10.
    * every derived name must be in the frozen ``feature_names_`` **and** carry ``feature``'s own
      prefix. Membership alone does not suffice: a mis-ordered width would slide the slice onto a
      different but *valid* one-hot block, whose names are all in ``feature_names_``, and the
      ablation would then quietly remove the wrong feature under the right ``run_id``. The prefix is
      a cross-check on the positional result, not the derivation -- ``get_feature_names_out`` emits
      ``f"{input_feature}_{level}"``, so the two must agree or the encoder is not what this function
      thinks it is.
    """
    categoricals = list(preprocessor.categorical_features_)
    if feature not in categoricals:
        raise ValueError(
            f"{feature!r} is not among the Preprocessor's categorical features {categoricals}; "
            f"the {feature} ablation has nothing to remove."
        )
    widths = [len(levels) for levels in preprocessor.encoder_.categories_]
    if len(widths) != len(categoricals):
        raise RuntimeError(
            f"the encoder holds {len(widths)} category blocks for {len(categoricals)} categorical "
            f"features {categoricals}; the positional offset of every block after the first would "
            "be wrong."
        )
    names = [str(name) for name in
             preprocessor.encoder_.get_feature_names_out(tuple(categoricals))]
    index = categoricals.index(feature)
    start = sum(widths[:index])
    columns = tuple(names[start:start + widths[index]])

    frozen = set(preprocessor.feature_names_)
    missing = [column for column in columns if column not in frozen]
    misprefixed = [column for column in columns if not column.startswith(f"{feature}_")]
    if not columns or missing or misprefixed:
        raise RuntimeError(
            f"derived {feature} one-hot columns {columns} do not agree with the fitted feature "
            f"schema (missing {missing}, wrong prefix {misprefixed}) -- the encoder and "
            "feature_names_ disagree, or the block offsets have shifted."
        )
    return columns


def ablated_columns(
    full_columns: tuple[str, ...], dropped: tuple[str, ...], feature: str
) -> tuple[str, ...]:
    """``full_columns`` minus one feature's one-hot block, with the width change asserted.

    The assertion is the point: a silent mismatch between "the columns we derived" and "the columns
    actually removed" would report an ablation that did not happen at the width it claims.
    """
    remaining = tuple(column for column in full_columns if column not in set(dropped))
    if len(remaining) != len(full_columns) - len(dropped):
        raise RuntimeError(
            f"the {feature} ablation removed {len(full_columns) - len(remaining)} columns, not the "
            f"{len(dropped)} derived one-hots {list(dropped)}"
        )
    return remaining


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
    families: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Fit every Phase 4-5 model on the train fold and run it through both regimes.

    Returns ``{model_name: {"in_distribution": {...}, "cross_era": {...}}}`` and upserts one row
    per (model, regime) under ``run_id``.

    ``columns`` selects the feature subset -- the full 22 for the headline condition, or 18 for
    either ablation (dropping the ``protocol`` or the ``conn_state`` one-hots; the two are different
    experiments at the same width) -- and is applied identically to all three frames, so the model,
    UNSW-test and TON_IoT always agree on the matrix.

    ``log_in_distribution=False`` computes the in-distribution half (the delta needs it, and
    :func:`_check_against_phase4` verifies it) but does not write it: in the unablated condition
    the four baselines' in-distribution rows belong to ``phase4-baselines``, and re-logging them
    under a Phase 6 ``run_id`` would make the results table count them twice. It does not apply to
    the scratch models, which have no Phase 4 row -- Phase 5 deliberately logged nothing -- so
    Phase 6 is where their in-distribution rows are created.

    ``families``, when given, is the per-regime family vector pair from :func:`regime_families` and
    adds a ``per_family`` block to every returned row (stripped at the logging boundary -- it has no
    column in the frozen header and reaches disk through ``reports/per_family_metrics.csv``). Only
    the full condition passes it: the two ablations are matched d=18 experiments whose quantity of
    interest is the difference of the *aggregate* deltas, and per-family scoring them would double
    this phase's prediction cost to answer a question nothing asks.
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
            families=families,
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
                        # Neither the confusion matrix nor the ROC curve has a column in the frozen
                        # 14-field header, so both are stripped here; they reach disk through the
                        # two sidecars `run_phase6` writes (`write_confusion_matrices`,
                        # `write_roc_curves`), which is what Phase 9 renders. `log_metrics` would
                        # drop them anyway -- it projects onto METRICS_HEADER -- but relying on that
                        # would leave the row this phase *claims* to log different from the one it
                        # does.
                        **{
                            key: value
                            for key, value in regimes[regime].items()
                            if key not in {"confusion_matrix", "roc_curve", "per_family"}
                        },
                    }
                )
            )
    return results


# --- The confusion-matrix sidecar ----------------------------------------------------------
#
# WHY A SECOND ARTIFACT RATHER THAN FOUR MORE COLUMNS. :func:`evaluate` already computes a confusion
# matrix for every (condition, model, regime) and :func:`run_condition` prints it, but
# :data:`METRICS_HEADER` is frozen and a 2x2 integer matrix is not a scalar metric -- widening the
# committed run log by four count columns for one figure's sake is exactly the kind of change that
# header comment forbids. So the matrices go to their own file, beside the log.
#
# WHY PERSIST AT ALL. Phase 9 renders them, and `python -m src.plots` must run standalone in
# seconds. Recomputing them there would mean re-fitting eighteen models (~3 min) *and* re-running
# the phase whose output it is meant to be illustrating -- and inside `./run.sh`, which already runs
# Phase 6 immediately before Phase 9, it would run the same three conditions twice. Writing what
# was already computed costs one small file and keeps the figure impossible to disagree with the
# run it came from.
#
# Same idempotence contract as `log_metrics`: keys are sorted, floats are rounded to
# METRIC_DECIMALS, and the fits upstream are seeded, so re-running leaves the file byte-identical.

#: Version tag written into (and required back out of) the sidecar. A reader that finds a different
#: tag is looking at a file some other version of this code wrote, and the safe thing to do with a
#: figure's only data source is refuse it rather than guess at the shape.
CONFUSION_SCHEMA: str = "ids-crossera/confusion-matrices/1"

#: The per-(model, regime) fields the sidecar carries. ``n_test`` and ``positive_rate`` ride along
#: for the same reason they have columns in ``METRICS_HEADER``: they let a reader check that a
#: matrix sums to the set it claims to have been measured on, without a join.
CONFUSION_FIELDS: tuple[str, ...] = ("confusion_matrix", "n_test", "positive_rate")


def write_confusion_matrices(
    results: dict[str, dict[str, dict[str, dict[str, Any]]]], path: Any = CONFUSION_JSON
) -> Path:
    """Persist :func:`run_phase6`'s confusion matrices to ``reports/confusion_matrices.json``.

    Takes the whole ``{run_id: {model: {regime: metrics}}}`` return value and writes all three
    conditions -- the full feature set and both ablations -- because the file is a record of what
    the phase measured, not a feed for one figure. Selecting the condition is the *reader's* job
    (Phase 9's confusion figure renders ``phase6-crossera`` only, and asserts it).
    """
    payload = {
        "schema": CONFUSION_SCHEMA,
        "written_by": "src.evaluate.run_phase6",
        "seed": RANDOM_SEED,
        "labels": [0, POSITIVE_LABEL],
        "cell_order": [["tn", "fp"], ["fn", "tp"]],
        "conditions": {
            run_id: {
                model: {
                    regime: {
                        field: (
                            round(scores[field], METRIC_DECIMALS)
                            if isinstance(scores[field], float)
                            else scores[field]
                        )
                        for field in CONFUSION_FIELDS
                    }
                    for regime, scores in regimes.items()
                }
                for model, regimes in condition.items()
            }
            for run_id, condition in results.items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def read_confusion_matrices(path: Any = CONFUSION_JSON) -> dict[str, dict[str, dict[str, Any]]]:
    """Read the sidecar back as ``{run_id: {model: {regime: {...}}}}``.

    Raises with an actionable message when the file is absent: nothing downstream recomputes these,
    by design, so the fix is always "run Phase 6".
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- Phase 6 has not been run in this working tree. The confusion "
            "matrices are computed and persisted by `evaluate.run_phase6()`; nothing downstream "
            "recomputes them, because that would mean re-fitting every model. Run "
            "`python -m src.evaluate --regimes`, or `./run.sh`, which runs Phase 6 before Phase 9."
        )
    payload = json.loads(path.read_text())
    if payload.get("schema") != CONFUSION_SCHEMA:
        raise ValueError(
            f"{path} declares schema {payload.get('schema')!r}, expected {CONFUSION_SCHEMA!r}. "
            "Regenerate it with `python -m src.evaluate --regimes` rather than reading a shape "
            "this code does not know."
        )
    if payload.get("labels") != [0, POSITIVE_LABEL]:
        raise ValueError(
            f"{path} was written with labels {payload.get('labels')}, expected "
            f"{[0, POSITIVE_LABEL]}; the cell order [[tn, fp], [fn, tp]] would not hold."
        )
    return payload["conditions"]


# --- The ROC-curve sidecar -----------------------------------------------------------------
#
# Same shape, same contract and same reasons as the confusion sidecar above; see the block comment
# at `roc_points` for why the curves are stored as simplified integer counts. All three conditions
# are written, as there: the file records what the phase measured, and choosing a condition is the
# reader's job (Phase 9's ROC figure renders `phase6-crossera` only, and asserts it).

#: Version tag for the ROC sidecar, checked on read for the same reason as
#: :data:`CONFUSION_SCHEMA`.
ROC_SCHEMA: str = "ids-crossera/roc-curves/1"

#: The per-(model, regime) fields the ROC sidecar carries -- exactly :func:`roc_points`' keys.
ROC_FIELDS: tuple[str, ...] = (
    "n_negative", "n_positive", "n_vertices", "scores_snapped",
    "false_positives", "true_positives", "roc_auc", "auc_simplified",
)

#: The two count vectors, written as one space-separated line each rather than as JSON arrays.
#: ``json.dumps(indent=2)`` puts every array element on its own line, which for ~14,000 vertices is
#: 520 KB of committed whitespace against 160 KB for the same numbers; and one line per curve is the
#: more diffable of the two anyway, which is the point of committing the file at all.
#: :func:`read_roc_curves` parses them back, so no caller sees the encoding.
ROC_ARRAY_FIELDS: tuple[str, ...] = ("false_positives", "true_positives")

#: Decimals for ``auc_simplified``. Deliberately finer than :data:`METRIC_DECIMALS`: it exists to
#: show how far the stored curve's area sits from the logged ``roc_auc``, and rounding both to six
#: places makes a sub-1e-6 agreement read as a 1e-6 disagreement.
ROC_AUC_CHECK_DECIMALS: int = 9


def _roc_field(field: str, value: Any) -> Any:
    """One sidecar field, in its on-disk form: counts joined, ``roc_auc`` at the log's precision."""
    if field in ROC_ARRAY_FIELDS:
        return " ".join(str(int(item)) for item in value)
    if field == "auc_simplified":
        return round(float(value), ROC_AUC_CHECK_DECIMALS)
    # `roc_auc` is rounded exactly as `log_metrics` rounds it, so the sidecar's copy and the
    # committed metrics.csv row are the same string of digits rather than merely close.
    return round(value, METRIC_DECIMALS) if isinstance(value, float) else value


def write_roc_curves(
    results: dict[str, dict[str, dict[str, dict[str, Any]]]], path: Any = ROC_JSON
) -> Path:
    """Persist :func:`run_phase6`'s ROC curves to ``reports/roc_curves.json``.

    Takes the whole ``{run_id: {model: {regime: metrics}}}`` return value. Every entry must carry a
    ``roc_curve`` -- a missing one means :func:`run_regimes` was called with ``with_roc_curve=False``
    and the figure would be drawn from a partial file, so it raises rather than writing a hole.
    """
    conditions: dict[str, Any] = {}
    for run_id, condition in results.items():
        conditions[run_id] = {}
        for model, regimes in condition.items():
            conditions[run_id][model] = {}
            for regime, scores in regimes.items():
                if "roc_curve" not in scores:
                    raise KeyError(
                        f"{run_id}/{model}/{regime} carries no ROC curve. `run_regimes` must be "
                        "called with with_roc_curve=True for every row this sidecar records."
                    )
                curve = scores["roc_curve"]
                conditions[run_id][model][regime] = {
                    field: _roc_field(field, curve[field]) for field in ROC_FIELDS
                }
    payload = {
        "schema": ROC_SCHEMA,
        "written_by": "src.evaluate.run_phase6",
        "seed": RANDOM_SEED,
        "positive_label": POSITIVE_LABEL,
        # Spelled out in the file itself: a reader should not have to open this module to know that
        # the vertices are counts and how to turn them back into the rates a ROC plot wants.
        "vertex_encoding": (
            "false_positives/true_positives are space-separated integer counts, one entry per "
            "retained ROC vertex; fpr = false_positives / n_negative and "
            "tpr = true_positives / n_positive reproduce sklearn.metrics.roc_curve exactly."
        ),
        "simplification": (
            f"each curve is reduced from n_vertices to at most {ROC_CURVE_BUDGET} vertices by "
            "evaluate.simplify_roc, an area-preserving greedy; auc_simplified is the trapezoidal "
            f"area of the stored vertices and is within {ROC_CURVE_TOLERANCE:.0e} of roc_auc, "
            "which is the same scalar reports/metrics.csv records."
        ),
        "score_snapping": (
            f"scores_snapped records whether the curve was built from scores rounded to "
            f"{ROC_SCORE_DECIMALS} decimals. That removes the last-bit non-determinism of "
            "RandomForestClassifier(n_jobs=-1).predict_proba, which would otherwise make this file "
            "differ after every run; it is skipped wherever it would move the ROC-AUC by more than "
            f"{ROC_CURVE_TOLERANCE:.0e}. roc_auc itself is always computed from the raw scores."
        ),
        "conditions": conditions,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def read_roc_curves(path: Any = ROC_JSON) -> dict[str, dict[str, dict[str, Any]]]:
    """Read the ROC sidecar back as ``{run_id: {model: {regime: {...}}}}``.

    The two count vectors come back as ``list[int]`` -- :data:`ROC_ARRAY_FIELDS` is a storage
    encoding and stops at this boundary. Every curve is checked for the two ways a hand-edited or
    truncated file would otherwise draw something plausible: the vectors must be the same length,
    and each must end on its own class size (a ROC curve ends at ``(1, 1)``).

    Raises with an actionable message when the file is absent: nothing downstream recomputes these,
    by design, so the fix is always "run Phase 6".
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- Phase 6 has not been run in this working tree. The ROC curves are "
            "computed and persisted by `evaluate.run_phase6()`; nothing downstream recomputes them, "
            "because that would mean re-fitting every model. Run `python -m src.evaluate "
            "--regimes`, or `./run.sh`, which runs Phase 6 before Phase 9."
        )
    payload = json.loads(path.read_text())
    if payload.get("schema") != ROC_SCHEMA:
        raise ValueError(
            f"{path} declares schema {payload.get('schema')!r}, expected {ROC_SCHEMA!r}. "
            "Regenerate it with `python -m src.evaluate --regimes` rather than reading a shape "
            "this code does not know."
        )
    if payload.get("positive_label") != POSITIVE_LABEL:
        raise ValueError(
            f"{path} was written with positive_label {payload.get('positive_label')}, expected "
            f"{POSITIVE_LABEL}; every true_positives count would be against the wrong class."
        )
    conditions = payload["conditions"]
    for run_id, condition in conditions.items():
        for model, regimes in condition.items():
            for regime, curve in regimes.items():
                for field in ROC_ARRAY_FIELDS:
                    curve[field] = [int(item) for item in str(curve[field]).split()]
                fps, tps = curve["false_positives"], curve["true_positives"]
                if (len(fps) != len(tps) or not fps
                        or fps[-1] != curve["n_negative"] or tps[-1] != curve["n_positive"]):
                    raise ValueError(
                        f"{path}: the {run_id}/{model}/{regime} curve has {len(fps)} false-positive "
                        f"and {len(tps)} true-positive counts ending at ({fps[-1:]}, {tps[-1:]}), "
                        f"which is not a complete ROC curve over "
                        f"{curve['n_negative']} negatives and {curve['n_positive']} positives. "
                        "Regenerate it with `python -m src.evaluate --regimes`."
                    )
    return conditions


def run_phase6(log: bool = True) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Phase 6 end to end: all three conditions, both regimes, all six models, logged.

    Returns ``{run_id: {model: {regime: metrics}}}``. The ``Preprocessor`` is sealed for the whole
    run (see :func:`sealed`), so no path through this function can refit it -- the target era is
    transformed through the frozen Phase 3 artifact or the run raises.

    The two ablations (`proto`, `conn_state`) both land at d=18 and are **not** the same experiment;
    each is comparable to the full d=22 condition and to nothing else.

    ``log`` gates **all four** on-disk outputs: the ``reports/metrics.csv`` upserts, the
    ``reports/confusion_matrices.json`` sidecar Phase 9's confusion figure reads, the
    ``reports/roc_curves.json`` sidecar its ROC figure reads, and the
    ``reports/per_family_metrics.csv`` rows its cross-era per-family figure reads (see
    :func:`write_confusion_matrices`, :func:`write_roc_curves` and :func:`write_per_family_metrics`
    for why none of the three fits in the frozen metrics header). ``log=False`` measures and prints
    without touching any of them.
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
        # The SHARED vocabulary, and only the shared one: a cross-era per-family comparison is
        # meaningful in no other. Aligned to the two evaluation frames element-wise by
        # `regime_families`; the train fold's vector is not needed here (nothing is scored on it).
        all_families = regime_families(frames)
        shared_families = {
            IN_DISTRIBUTION: all_families["unsw_test"][SHARED_FAMILY_SET],
            CROSS_ERA: all_families["toniot"][SHARED_FAMILY_SET],
        }
        full_columns = tuple(preprocessor.feature_names_)
        proto = categorical_columns(preprocessor, PROTOCOL_FEATURE)
        conn_state = categorical_columns(preprocessor, CONN_STATE_FEATURE)
        # One line, and it closes the only way the two conditions could silently become
        # near-duplicates of each other under a future schema_map edit.
        if set(proto) & set(conn_state):  # pragma: no cover
            raise RuntimeError(
                f"the {PROTOCOL_FEATURE} and {CONN_STATE_FEATURE} one-hot blocks overlap on "
                f"{sorted(set(proto) & set(conn_state))}; the two ablations would not be "
                "independent conditions."
            )
        no_proto = ablated_columns(full_columns, proto, PROTOCOL_FEATURE)
        no_conn_state = ablated_columns(full_columns, conn_state, CONN_STATE_FEATURE)

        from .models.baselines import BASELINE_FACTORIES  # noqa: PLC0415 - circular at top level

        if set(BASELINE_FACTORIES) != _PHASE4_MODELS:
            raise RuntimeError(
                f"baselines registry {sorted(BASELINE_FACTORIES)} no longer matches the models "
                f"whose in_distribution rows Phase 4 owns {sorted(_PHASE4_MODELS)}; the "
                "in-distribution rows would be double-logged or silently dropped."
            )

        print(
            f"feature schema: {len(full_columns)} columns\n"
            f"  {PROTOCOL_FEATURE} ablation drops {len(proto)} one-hots {list(proto)} "
            f"-> {len(no_proto)} columns\n"
            f"  {CONN_STATE_FEATURE} ablation drops {len(conn_state)} one-hots "
            f"{list(conn_state)} -> {len(no_conn_state)} columns\n"
            "  Both land at the same width and are NOT the same experiment; the run_id and the "
            "notes column distinguish them, not d."
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
                families=shared_families,
            ),
            PROTO_ABLATION_RUN_ID: run_condition(
                PROTO_ABLATION_RUN_ID,
                frames,
                no_proto,
                condition_note="proto ablation: protocol one-hots dropped, retrained on train fold",
                log=log,
                # A delta needs both halves, and no other run_id holds an ablated in-distribution
                # row, so this condition logs its own matched pair for every model.
                log_in_distribution=True,
            ),
            CONN_STATE_ABLATION_RUN_ID: run_condition(
                CONN_STATE_ABLATION_RUN_ID,
                frames,
                no_conn_state,
                condition_note=(
                    "conn_state ablation: connection-state one-hots dropped, retrained on train "
                    "fold"
                ),
                log=log,
                # Same reason as the proto ablation. Note that no_proto's in-distribution rows are
                # NOT a substitute despite also being d=18: different feature set, different fitted
                # model. `log_in_distribution=False` is also not merely "log less" here -- it would
                # route the four baselines into _check_against_phase4(), whose 1e-4 tolerance an
                # ablated model legitimately blows past, and the run would raise.
                log_in_distribution=True,
            ),
        }

    rows = [
        row
        for model, regimes in results[RUN_ID].items()
        for regime, scores in regimes.items()
        for row in per_family_rows(
            scores,
            run_id=RUN_ID,
            model=model,
            regime=regime,
            family_set=SHARED_FAMILY_SET,
            note=(
                f"one-vs-normal on the {regime} set; {_REGIME_NOTES[regime]}; "
                "shared-family vocabulary (schema_map.SHARED_FAMILIES)"
            ),
        )
    ]
    if log:
        # Outside the sealed block and after every condition has run: the sidecars are a record of
        # the whole phase, not of one condition.
        print(f"\nconfusion matrices -> {write_confusion_matrices(results)}")
        print(f"ROC curves         -> {write_roc_curves(results)}")
        print(f"per-family metrics -> {write_per_family_metrics(rows)}  ({len(rows)} rows)")

    _print_headline(results[RUN_ID])
    _print_per_family(rows)
    _print_ablation_contrast(
        results[RUN_ID],
        results[PROTO_ABLATION_RUN_ID],
        run_id=PROTO_ABLATION_RUN_ID,
        label=PROTOCOL_FEATURE,
        interpretation=(
            "A Δ-of-Δ near zero means the drop is not the `other`-bucket artifact; a large positive\n"
            "  one means part of the with-protocol drop was that learned shortcut going inert."
        ),
    )
    _print_ablation_contrast(
        results[RUN_ID],
        results[CONN_STATE_ABLATION_RUN_ID],
        run_id=CONN_STATE_ABLATION_RUN_ID,
        label=CONN_STATE_FEATURE,
        interpretation=(
            "A Δ-of-Δ near zero means the drop is not our Argus->Zeek state collapse; a large\n"
            "  positive one means part of the with-conn_state drop was the hand-written collapse --\n"
            "  `reset` 0.0421% of UNSW train rows vs 23.6757% of TON_IoT's, `other` 0.0057% vs\n"
            "  11.0556% -- carrying an instrumentation change rather than attacker evolution."
        ),
    )
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
        f"change, not drift.\n"
        f"  Δ = in_distribution − cross_era throughout (metric_deltas), so Δ is what the model LOST:"
        f" positive = degraded, negative = improved.\n{'-' * 100}\n"
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
        # Δ from metric_deltas, NOT an inline cross - in: this footnote used to compute the
        # difference the other way round, so it printed the opposite sign to the table directly
        # above it. One convention per run (in - cross, i.e. "what was lost").
        dummy_deltas = metric_deltas(dummy)
        print(
            f"{'-' * 100}\n"
            f"Prevalence artifact: the majority-class dummy's F1 "
            f"{'rises' if dummy[CROSS_ERA]['f1'] > dummy[IN_DISTRIBUTION]['f1'] else 'falls'} "
            f"{dummy[IN_DISTRIBUTION]['f1']:.4f} -> {dummy[CROSS_ERA]['f1']:.4f} cross-era at "
            f"ROC-AUC {dummy[CROSS_ERA]['roc_auc']:.4f}, on class balance alone. Any real model's "
            f"F1 delta must be read against that Δ F1 = {dummy_deltas['f1']:+.4f} "
            f"(negative because a model that GAINS F1 lost nothing)."
        )


def _print_per_family(rows: list[dict[str, Any]]) -> None:
    """The three shared families' F1 and ROC-AUC per model, both regimes, with the Δ.

    Each family is scored **one-vs-normal** on its own regime's set (see
    :func:`per_family_metrics`), so the two halves of a Δ are the same family against that era's own
    benign traffic. Δ keeps the project-wide sign: ``in_distribution − cross_era``, i.e. what the
    model lost, positive = degraded.
    """
    if not rows:  # pragma: no cover - run_phase6 always passes the full condition's rows
        return
    by_key = {(row["model"], row["regime"], row["family"]): row for row in rows}
    models = list(dict.fromkeys(row["model"] for row in rows))
    families = sorted({row["family"] for row in rows})
    print(
        f"\n{'=' * 104}\nper-family breakdown — {RUN_ID}, shared families only "
        f"({', '.join(families)})\n"
        "  Each family is scored one-vs-normal: the family's rows plus every normal row of the "
        "same evaluation set,\n  so precision, F1 and the dummy floor stay defined. Every subset "
        "has its own class balance -- n below.\n"
        "  Δ = in_distribution − cross_era (what the model lost; positive = degraded).\n"
        f"{'-' * 104}"
    )
    for family in families:
        sizes = [
            (regime, by_key[(models[0], regime, family)])
            for regime in (IN_DISTRIBUTION, CROSS_ERA)
            if (models[0], regime, family) in by_key
        ]
        detail = "   ".join(
            f"{regime}: {row['n_family']:,} {family} vs {row['n_normal']:,} normal "
            f"(attack {float(row['positive_rate']):.2%})"
            for regime, row in sizes
        )
        print(f"\n  {family}   {detail}\n"
              f"{'model':<16}{'F1 in':>9}{'F1 cross':>10}{'Δ F1':>9}"
              f"{'AUC in':>9}{'AUC cross':>11}{'Δ AUC':>9}")
        for model in models:
            in_row = by_key.get((model, IN_DISTRIBUTION, family))
            cross_row = by_key.get((model, CROSS_ERA, family))
            if in_row is None or cross_row is None:  # pragma: no cover - both regimes always run
                continue
            in_f1, cross_f1 = float(in_row["f1"]), float(cross_row["f1"])
            in_auc, cross_auc = float(in_row["roc_auc"]), float(cross_row["roc_auc"])
            print(
                f"{model:<16}{in_f1:>9.4f}{cross_f1:>10.4f}{in_f1 - cross_f1:>+9.4f}"
                f"{in_auc:>9.4f}{cross_auc:>11.4f}{in_auc - cross_auc:>+9.4f}"
            )


def _print_ablation_contrast(
    base: dict[str, dict[str, dict[str, Any]]],
    ablated: dict[str, dict[str, dict[str, Any]]],
    *,
    run_id: str,
    label: str,
    interpretation: str,
) -> None:
    """The difference of the deltas: how much of the drop survives dropping one feature.

    ``label`` MUST name the ablated feature. Both ablations run at d=18, so the width is not a
    disambiguator and an unlabelled table is unreadable next to its sibling.

    Only ever call this with ``base`` = the FULL condition: each ablation is comparable to
    ``phase6-crossera`` and to nothing else. ``no_proto`` vs ``no_conn_state`` is not a valid
    contrast (neither nests inside the other and they differ in two ways at once), which is why this
    function has no mode that prints it.

    Δ is ``in_distribution − cross_era`` on both sides, so Δ-of-Δ is
    ``metric_deltas(full) − metric_deltas(ablated)``: positive means the full-feature model lost more
    than the ablated one, i.e. the removed feature was carrying part of the drop.
    """
    print(
        f"\n{'=' * 104}\n{label} ablation — {run_id} vs {RUN_ID}\n"
        "  Each side is its own matched in_distribution/cross_era pair, so the comparable\n"
        f"  quantity is the difference of the deltas: how much of the with-{label} drop survives\n"
        f"  when the {label} one-hots were never available to be learned.\n"
        f"  Δ = in − cross (what was lost); Δ-of-Δ = Δ full − Δ no_{label}.\n"
        f"{'-' * 104}\n"
        # Field width 22: the longest label this renders is `Δ AUC no_conn_state` at 19 characters,
        # and at width 19 it consumed the whole field and butted against the column to its left.
        f"{'model':<16}{'Δ AUC full':>13}{f'Δ AUC no_{label}':>22}{'Δ of Δ':>10}"
        f"{'Δ F1 full':>13}{f'Δ F1 no_{label}':>22}{'Δ of Δ':>10}"
    )
    for name in base:
        full = metric_deltas(base[name])
        without = metric_deltas(ablated[name])
        print(
            f"{name:<16}{full['roc_auc']:>+13.4f}{without['roc_auc']:>+22.4f}"
            f"{full['roc_auc'] - without['roc_auc']:>+10.4f}"
            f"{full['f1']:>+13.4f}{without['f1']:>+22.4f}"
            f"{full['f1'] - without['f1']:>+10.4f}"
        )
    print(
        f"  {interpretation}"
    )


# --- Entry point -------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description=(
            "Phase 6: evaluate every Phase 4-5 model in both regimes (in-distribution on "
            "UNSW-test, zero-shot cross-era on TON_IoT) plus the `proto` and `conn_state` "
            "ablations, and upsert the rows into reports/metrics.csv. Nothing is refit on "
            "UNSW-test or TON_IoT."
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
    ) - len(_PHASE4_MODELS)  # the baselines' in-distribution halves stay Phase 4's -- subtracting a
    # flat len(_PHASE4_MODELS) is correct only while EXACTLY ONE condition passes
    # log_in_distribution=False. Add a second such condition and this undercounts.
    print(
        f"\nlogged {rows} rows across {len(results)} run_ids -> {METRICS_CSV}\n"
        f"confusion matrices for all {len(results)} conditions -> {CONFUSION_JSON} "
        "(read by Phase 9's confusion figure; the frozen 14-column header has no room for them)\n"
        f"ROC curves for all {len(results)} conditions -> {ROC_JSON} "
        "(read by Phase 9's ROC figure; likewise not a scalar)\n"
        f"per-family metrics for {RUN_ID} -> {PER_FAMILY_CSV} "
        "(read by Phase 9's cross-era per-family figure; the metrics key has no family dimension)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
