"""Project-wide configuration: the single random seed and canonical paths.

Everything that needs reproducibility (numpy, stdlib random, sklearn splits) pulls the seed
from here so there is exactly one source of truth (Phase 0).
"""

from __future__ import annotations

from pathlib import Path

# --- Reproducibility -------------------------------------------------------------------
RANDOM_SEED: int = 42

# --- Paths -----------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
REPORTS: Path = PROJECT_ROOT / "reports"
FIGURES: Path = REPORTS / "figures"
METRICS_CSV: Path = REPORTS / "metrics.csv"

# Phase 6's confusion-matrix sidecar, written by `evaluate.run_phase6()` and read by
# `plots.plot_confusion_matrices()`. It exists because the 14-column `METRICS_HEADER` is frozen and
# has no room for a 2x2 integer matrix, and because Phase 9 must not re-fit eighteen models (~3 min)
# to redraw one figure. Committed for the same reason `metrics.csv` is: checking that a run
# reproduced should be a `git diff`, and a re-run rewrites it byte-identically.
CONFUSION_JSON: Path = REPORTS / "confusion_matrices.json"

# Phase 6's ROC-curve sidecar, written by `evaluate.run_phase6()` and read by
# `plots.plot_roc_curves()`. Same argument as the confusion sidecar -- a curve is not a scalar and
# the frozen `METRICS_HEADER` has no room for one, and Phase 9 must not re-fit eighteen models to
# redraw a figure. It carries the (false-positive, true-positive) COUNT vertices rather than
# floating-point rates, so the stored curve is exact, and each curve is simplified to at most
# `evaluate.ROC_CURVE_BUDGET` vertices by an area-preserving greedy (see `evaluate.simplify_roc`)
# so the file stays a few hundred KB instead of the ~6 MB the raw vertices would take.
ROC_JSON: Path = REPORTS / "roc_curves.json"

# The per-attack-family metrics sidecar, written by `evaluate.run_phase6()` (cross-era, the three
# shared families) and `transfer.run_phase7()` (the frozen TON_IoT test half, that dataset's own
# families) and read by `plots.plot_per_family_f1()` / `plots.plot_per_family_recovery()`.
#
# NOT rows of `metrics.csv`: that log's header is frozen at 14 columns and its upsert key
# `(run_id, model, regime)` has no family dimension, so a per-family row would collide with the
# aggregate row it decomposes. This file therefore carries its own frozen header and its own key,
# `(run_id, model, regime, family_set, family)`, and upserts on it exactly as `log_metrics` does --
# so a re-run rewrites it byte-identically and "did this reproduce?" stays a `git diff`. Committed
# for that reason.
PER_FAMILY_CSV: Path = REPORTS / "per_family_metrics.csv"

# Expected raw filenames (see data/README.md).
UNSW_TRAIN_CSV: Path = DATA_RAW / "UNSW_NB15_training-set.csv"
UNSW_TEST_CSV: Path = DATA_RAW / "UNSW_NB15_testing-set.csv"
TONIOT_CSV: Path = DATA_RAW / "Train_Test_Network.csv"

# Harmonized outputs (Phase 2).
UNSW_COMMON: Path = DATA_PROCESSED / "unsw_common.parquet"
TONIOT_COMMON: Path = DATA_PROCESSED / "toniot_common.parquet"

# Fitted preprocessor (Phase 3). Lives under data/processed/ so `.gitignore`'s
# `data/processed/*` rule covers it -- the artifact is a build product of `run.sh`, reproducible
# from the raw CSVs and the seed above, and binaries are never committed to this repo.
PREPROCESSOR: Path = DATA_PROCESSED / "preprocessor.joblib"


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Seed stdlib ``random`` and numpy. Call once at the start of every entry point."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy not yet installed during Phase 0
        pass
