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
