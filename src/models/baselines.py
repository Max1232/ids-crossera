"""Library baseline models (Phase 4) — the classical axis and sanity floor.

Random Forest, SVM, Decision Tree (library implementations) plus a majority-class Dummy.
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
    """SVM (probability-enabled) with balanced class weights."""
    raise NotImplementedError("Phase 4: SVC(class_weight='balanced', probability=True, ...)")


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
