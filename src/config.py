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

# Expected raw filenames (see data/README.md).
UNSW_TRAIN_CSV: Path = DATA_RAW / "UNSW_NB15_training-set.csv"
UNSW_TEST_CSV: Path = DATA_RAW / "UNSW_NB15_testing-set.csv"
TONIOT_CSV: Path = DATA_RAW / "Train_Test_Network.csv"

# Harmonized outputs (Phase 2).
UNSW_COMMON: Path = DATA_PROCESSED / "unsw_common.parquet"
TONIOT_COMMON: Path = DATA_PROCESSED / "toniot_common.parquet"


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Seed stdlib ``random`` and numpy. Call once at the start of every entry point."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy not yet installed during Phase 0
        pass
