"""Figures for the report and presentation (Phase 9) -> reports/figures/.

Every figure needs clear labels, legends, and captions (graded). Produces:
  - in-distribution vs cross-era metric bars
  - confusion matrices (per model, per regime)
  - ROC curves
  - per-shared-family F1
  - the transfer-learning recovery curve (RQ2 secondary headline)
"""

from __future__ import annotations

from typing import Any

from .config import FIGURES


def plot_indist_vs_crossera(metrics: Any, out=FIGURES) -> None:
    """Grouped bar chart: in-distribution vs cross-era F1/ROC-AUC per model (RQ1 headline)."""
    raise NotImplementedError("Phase 9: drift bar chart")


def plot_confusion_matrices(results: Any, out=FIGURES) -> None:
    raise NotImplementedError("Phase 9: confusion matrices")


def plot_roc_curves(results: Any, out=FIGURES) -> None:
    raise NotImplementedError("Phase 9: ROC curves")


def plot_per_family_f1(results: Any, out=FIGURES) -> None:
    raise NotImplementedError("Phase 9: per-family F1")


def plot_recovery_curve(recovery: Any, out=FIGURES) -> None:
    """F1/ROC-AUC vs fraction of modern data used (RQ2 secondary headline)."""
    raise NotImplementedError("Phase 9: transfer recovery curve")
