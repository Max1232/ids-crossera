"""Evaluation — metrics and the two regimes (Phase 6, RQ1).

Lead with F1 and ROC-AUC (accuracy is misleading under imbalance), plus precision/recall,
confusion matrices, and per-shared-family breakdowns. Headline number = the delta between the
in-distribution and cross-era (zero-shot) regimes. All runs are logged to reports/metrics.csv.
"""

from __future__ import annotations

import csv
from typing import Any

from .config import METRICS_CSV, RANDOM_SEED

# The ten original columns are FROZEN — never reorder or rename them. `log_metrics()` writes the
# header only when the file is empty, so any change after the first logged run silently misaligns
# every existing row. The four appended columns landed in Phase 2 (before Phase 4 logs anything)
# for exactly that reason:
#   balanced_accuracy, macro_f1  -- prevalence-robust cross-checks Phase 6 needs, because the two
#                                   test sets do not share a class balance (45% vs 24% normal).
#   n_test, positive_rate        -- record the class balance each row was measured against, so a
#                                   delta can never be read without the prevalence it was taken at.
METRICS_HEADER = [
    "run_id", "model", "regime", "seed",
    "accuracy", "precision", "recall", "f1", "roc_auc", "notes",
    "balanced_accuracy", "macro_f1", "n_test", "positive_rate",
]


def evaluate(model: Any, X: Any, y: Any) -> dict[str, Any]:
    """Compute accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix."""
    raise NotImplementedError("Phase 6: metric computation")


def run_regimes(model: Any, X_unsw_test: Any, y_unsw_test: Any,
                X_toniot: Any, y_toniot: Any) -> dict[str, dict[str, Any]]:
    """Evaluate one trained model in both regimes side by side.

    Returns ``{"in_distribution": {...}, "cross_era": {...}}``. The reported drift is the
    per-metric delta (in_distribution - cross_era).
    """
    raise NotImplementedError("Phase 6: in-distribution vs cross-era")


def log_metrics(row: dict[str, Any], path: Any = METRICS_CSV) -> None:
    """Append one run's metrics to reports/metrics.csv, writing the header if absent."""
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRICS_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in METRICS_HEADER})


_ = RANDOM_SEED  # runs are seeded via config
