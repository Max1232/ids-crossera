"""From-scratch multilayer perceptron in numpy (Phase 5).

Implements the neural axis *and* the base for RQ2 transfer learning (freeze-and-retrain needs
our own layers). Forward pass, backpropagation, mini-batch training loop, sigmoid/softmax
output. Must converge on a toy separable dataset (unit test) and land near the sklearn
equivalent in-distribution.
"""

from __future__ import annotations

from typing import Any

from ..config import RANDOM_SEED


class ScratchMLP:
    """Fully-connected MLP trained by hand-written backprop.

    Parameters
    ----------
    layer_sizes : tuple[int, ...]
        Units per layer, input -> ... -> output.
    lr : float
        Learning rate.
    n_epochs, batch_size : int
        Mini-batch training schedule.
    seed : int
        Weight-init seed (defaults to the project seed).
    """

    def __init__(
        self,
        layer_sizes: tuple[int, ...],
        lr: float = 0.01,
        n_epochs: int = 50,
        batch_size: int = 256,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.params: dict[str, Any] = {}  # TODO Phase 5: init weights/biases

    def forward(self, X: Any) -> Any:
        """Forward pass; cache activations for backprop."""
        raise NotImplementedError("Phase 5: forward pass")

    def backward(self, y_true: Any, cache: Any) -> dict[str, Any]:
        """Backpropagate loss, return parameter gradients."""
        raise NotImplementedError("Phase 5: backpropagation")

    def fit(self, X: Any, y: Any, freeze_hidden: bool = False) -> "ScratchMLP":
        """Mini-batch training loop.

        ``freeze_hidden=True`` retrains only the classifier head — used by RQ2 transfer (Phase 7).
        """
        raise NotImplementedError("Phase 5: training loop")

    def predict_proba(self, X: Any) -> Any:
        raise NotImplementedError("Phase 5: predict_proba")

    def predict(self, X: Any) -> Any:
        raise NotImplementedError("Phase 5: predict")
