"""Evaluation — metrics and the two regimes (Phase 6, RQ1).

Lead with F1 and ROC-AUC (accuracy is misleading under imbalance), plus precision/recall,
confusion matrices, and per-shared-family breakdowns. Headline number = the delta between the
in-distribution and cross-era (zero-shot) regimes. All runs are logged to reports/metrics.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

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

from .config import METRICS_CSV, RANDOM_SEED

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


def run_regimes(model: Any, X_unsw_test: Any, y_unsw_test: Any,
                X_toniot: Any, y_toniot: Any) -> dict[str, dict[str, Any]]:
    """Evaluate one trained model in both regimes side by side.

    Returns ``{"in_distribution": {...}, "cross_era": {...}}``. The reported drift is the
    per-metric delta (in_distribution - cross_era).
    """
    raise NotImplementedError("Phase 6: in-distribution vs cross-era")


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


_ = RANDOM_SEED  # runs are seeded via config
