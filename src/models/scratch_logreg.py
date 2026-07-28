"""From-scratch logistic regression in numpy (Phase 5 — lower-risk fallback).

Clean math, guaranteed to converge on this imbalanced tabular data. Either this or the
from-scratch MLP satisfies the from-scratch requirement. Supports class weighting for
imbalance and warm-start for RQ2 transfer.
"""

from __future__ import annotations

from typing import Any

from ..config import RANDOM_SEED


class ScratchLogReg:
    """Binary logistic regression trained by gradient descent."""

    def __init__(
        self,
        lr: float = 0.01,
        n_epochs: int = 200,
        l2: float = 0.0,
        class_weight: dict[int, float] | None = None,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.lr = lr
        self.n_epochs = n_epochs
        self.l2 = l2
        self.class_weight = class_weight
        self.seed = seed
        self.weights: Any = None  # TODO Phase 5
        self.bias: float = 0.0

    def fit(self, X: Any, y: Any, warm_start: bool = False) -> "ScratchLogReg":
        """Gradient-descent fit. ``warm_start=True`` keeps existing weights (RQ2 transfer)."""
        raise NotImplementedError("Phase 5: logistic regression fit")

    def predict_proba(self, X: Any) -> Any:
        raise NotImplementedError("Phase 5: predict_proba")

    def predict(self, X: Any, threshold: float = 0.5) -> Any:
        raise NotImplementedError("Phase 5: predict")
