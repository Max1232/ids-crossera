"""Library baseline models (Phase 4) — the classical axis and sanity floor.

Random Forest, linear SVM, Decision Tree (library implementations) plus a majority-class Dummy.
Class weights are used from the start to handle imbalance; hyperparameters are tuned on the
validation split *before* any cross-era run.
"""

from __future__ import annotations

from typing import Any

from ..config import RANDOM_SEED


def make_dummy() -> Any:
    """Majority-class baseline — the sanity floor."""
    raise NotImplementedError("Phase 4: DummyClassifier(strategy='most_frequent')")


def make_random_forest(**params: Any) -> Any:
    """RandomForestClassifier with balanced class weights and the project seed."""
    raise NotImplementedError("Phase 4: RandomForestClassifier(class_weight='balanced', ...)")


def make_svm(**params: Any) -> Any:
    """Linear SVM with balanced class weights — LinearSVC, or SGDClassifier(loss='hinge').

    Must be a *linear* SVM, not a kernel `SVC`. UNSW-NB15's training split is 175,341 rows, so a
    kernel SVC's n x n Gram matrix is ~2.5e10 entries (hundreds of GB in float64) and libsvm's
    training cost is roughly quadratic-to-cubic in n — it does not finish at this data size.
    `LinearSVC` (liblinear, primal) and `SGDClassifier(loss='hinge')` (out-of-core, partial_fit)
    both scale linearly and are the two supported options.

    Note for the evaluation code: neither exposes `predict_proba`. ROC-AUC must come from
    `decision_function` scores (`roc_auc_score` accepts them directly), or the estimator must be
    wrapped in `CalibratedClassifierCV` if genuinely calibrated probabilities are needed.
    """
    raise NotImplementedError(
        "Phase 4: LinearSVC(class_weight='balanced', ...) or "
        "SGDClassifier(loss='hinge', class_weight='balanced', ...); "
        "ROC-AUC from decision_function, or wrap in CalibratedClassifierCV for probabilities"
    )


def make_decision_tree(**params: Any) -> Any:
    """DecisionTreeClassifier with balanced class weights and tuned depth."""
    raise NotImplementedError("Phase 4: DecisionTreeClassifier(class_weight='balanced', ...)")


# Registry so run.sh / evaluate.py can iterate over the classical models uniformly.
BASELINE_FACTORIES = {
    "dummy": make_dummy,
    "random_forest": make_random_forest,
    "svm": make_svm,
    "decision_tree": make_decision_tree,
}

_ = RANDOM_SEED  # seed threaded into each factory in Phase 4
