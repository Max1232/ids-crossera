"""Transfer-learning correction — RQ2 recovery curve (Phase 7).

Fine-tune each model on small stratified fractions of TON_IoT labels (1%, 5%, 10%, 25%); the
rest of TON_IoT stays as the test set. MLP: freeze early layers, retrain the classifier head.
Classical models: warm-start / retrain on the small modern sample. Plot post-transfer
F1/ROC-AUC vs fraction of modern data used.
"""

from __future__ import annotations

from typing import Any

from .config import RANDOM_SEED

# Modern-data budgets for the recovery curve.
TRANSFER_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25)


def sample_fraction(X: Any, y: Any, fraction: float, seed: int = RANDOM_SEED) -> tuple[Any, Any, Any, Any]:
    """Stratified split of TON_IoT into a fine-tune fraction and the remaining test set."""
    raise NotImplementedError("Phase 7: stratified fraction sample")


def finetune(model: Any, X_ft: Any, y_ft: Any) -> Any:
    """Fine-tune a trained model on the modern fraction.

    MLP -> freeze hidden layers, retrain head (``ScratchMLP.fit(freeze_hidden=True)``).
    Classical -> warm-start / retrain on the small sample.
    """
    raise NotImplementedError("Phase 7: fine-tune")


def recovery_curve(model_factory: Any, X_toniot: Any, y_toniot: Any) -> Any:
    """Sweep TRANSFER_FRACTIONS, return post-transfer metrics per fraction for plotting.

    Also report how close each model gets to a full-TON_IoT-trained ceiling, and at what budget.
    """
    raise NotImplementedError("Phase 7: recovery curve")
