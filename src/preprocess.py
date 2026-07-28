"""Preprocessing pipeline (Phase 3).

Critical contract: the ``Preprocessor`` is **fit on UNSW-train only**, then applied unchanged
to UNSW-test and to TON_IoT. Never refit on the target set or you leak future information into
the cross-era measurement.
"""

from __future__ import annotations

from typing import Any


class Preprocessor:
    """Fit-on-source preprocessor: one-hot categoricals, z-score numerics, impute, log-transform.

    Numeric standardization and categorical vocabularies are learned from UNSW-train statistics
    and frozen; ``transform`` applies those frozen parameters to any later frame.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        # TODO Phase 3: hold fitted scalers, one-hot vocabularies, impute values.

    def fit(self, X_source: Any) -> "Preprocessor":
        """Learn scaling/encoding parameters from the source (UNSW-train) frame only."""
        # TODO Phase 3: impute missing/'-'; log-transform heavy-tailed byte/pkt counts;
        # fit one-hot on small categorical set; fit z-score on numerics.
        raise NotImplementedError("Phase 3: fit on UNSW-train")

    def transform(self, X: Any) -> Any:
        """Apply the frozen fit parameters to any frame (UNSW-test or TON_IoT)."""
        raise NotImplementedError("Phase 3: transform")

    def fit_transform(self, X_source: Any) -> Any:
        return self.fit(X_source).transform(X_source)

    def save(self, path: str) -> None:
        """Serialize the fitted preprocessor for reproducibility."""
        raise NotImplementedError("Phase 3: serialize")

    @classmethod
    def load(cls, path: str) -> "Preprocessor":
        raise NotImplementedError("Phase 3: deserialize")
