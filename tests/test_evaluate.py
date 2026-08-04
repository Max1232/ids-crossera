"""Phase 6 contracts: the leakage seal, the two regimes, and the metrics-log key.

Every test here is data-free by construction (see ``tests/README.md``): the real CSVs are
git-ignored and absent from a fresh clone, so the regime behaviour is exercised on a tiny
synthetic frame whose two "eras" differ in class balance the way UNSW-test and TON_IoT do. What is
being checked is the *contract*, not the numbers -- that a fit against a test set raises instead of
succeeding quietly, that each regime's row carries the balance of the set it was measured on, and
that two conditions differing only in ``run_id`` coexist in the log rather than overwriting each
other.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import RANDOM_SEED
from src.evaluate import (
    ABLATION_RUN_ID,
    CROSS_ERA,
    IN_DISTRIBUTION,
    METRICS_HEADER,
    RUN_ID,
    LeakageError,
    log_metrics,
    metric_deltas,
    read_metrics,
    run_regimes,
    sealed,
)
from src.models.scratch_logreg import ScratchLogReg


@pytest.fixture
def eras() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """A trained model plus two test "eras" with deliberately different class balances.

    ``in_distribution`` is 50% positive, ``cross_era`` is 80% positive -- the same direction as the
    real 55.06% -> 76.31% attack-share change, which is the whole reason ``positive_rate`` is a
    logged column.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    def era(n_positive: int, n_negative: int) -> tuple[np.ndarray, np.ndarray]:
        X = np.vstack(
            [
                rng.normal(loc=+2.0, scale=1.0, size=(n_positive, 3)),
                rng.normal(loc=-2.0, scale=1.0, size=(n_negative, 3)),
            ]
        )
        y = np.concatenate([np.ones(n_positive, dtype=int), np.zeros(n_negative, dtype=int)])
        return X, y

    return {"train": era(120, 120), IN_DISTRIBUTION: era(50, 50), CROSS_ERA: era(80, 20)}


@pytest.fixture
def fitted(eras: dict[str, tuple[np.ndarray, np.ndarray]]) -> ScratchLogReg:
    X_train, y_train = eras["train"]
    return ScratchLogReg(lr=0.5, n_epochs=200, l2=1e-4).fit(X_train, y_train)


def test_run_regimes_reports_each_set_s_own_prevalence(fitted, eras) -> None:
    """Both regimes come back, each carrying the n and attack share of the set it saw."""
    regimes = run_regimes(fitted, *eras[IN_DISTRIBUTION], *eras[CROSS_ERA])

    assert set(regimes) == {IN_DISTRIBUTION, CROSS_ERA}
    assert regimes[IN_DISTRIBUTION]["n_test"] == 100
    assert regimes[CROSS_ERA]["n_test"] == 100
    assert regimes[IN_DISTRIBUTION]["positive_rate"] == pytest.approx(0.50)
    assert regimes[CROSS_ERA]["positive_rate"] == pytest.approx(0.80)
    # A constant is exactly what a run that reused one set's labels would produce.
    assert regimes[IN_DISTRIBUTION]["positive_rate"] != regimes[CROSS_ERA]["positive_rate"]


def test_run_regimes_refuses_to_refit_the_model(fitted, eras) -> None:
    """The model is sealed for the span of both evaluations -- a refit raises, and is restored."""
    X_cross, y_cross = eras[CROSS_ERA]

    with sealed(fitted, reason="test"):
        with pytest.raises(LeakageError, match="sealed block"):
            fitted.fit(X_cross, y_cross)

    # The seal is scoped: outside the block the real method is back, unshadowed.
    assert "fit" not in vars(fitted)
    fitted.fit(*eras["train"])


def test_sealed_restores_after_an_exception(fitted) -> None:
    with pytest.raises(ValueError):
        with sealed(fitted, reason="test"):
            raise ValueError("boom")
    assert "fit" not in vars(fitted)


def test_metric_deltas_are_in_distribution_minus_cross_era(fitted, eras) -> None:
    regimes = run_regimes(fitted, *eras[IN_DISTRIBUTION], *eras[CROSS_ERA])
    deltas = metric_deltas(regimes)

    for metric, delta in deltas.items():
        assert delta == pytest.approx(
            regimes[IN_DISTRIBUTION][metric] - regimes[CROSS_ERA][metric]
        )
    # The set-descriptors are not differenced -- they are reported alongside the delta instead.
    assert "n_test" not in deltas
    assert "positive_rate" not in deltas


def test_conditions_differing_only_in_run_id_coexist(tmp_path) -> None:
    """The ablation's rows must sit beside the unablated ones, not overwrite them.

    This is the contract the ``run_id``-per-condition convention rests on: ``(run_id, model,
    regime)`` is the key, so the same model and regime under two conditions is two rows.
    """
    path = tmp_path / "metrics.csv"
    base = {column: "" for column in METRICS_HEADER}
    for run_id, f1 in ((RUN_ID, 0.30), (ABLATION_RUN_ID, 0.40)):
        log_metrics({**base, "run_id": run_id, "model": "svm", "regime": CROSS_ERA, "f1": f1}, path)

    rows = read_metrics(path)
    assert len(rows) == 2
    assert rows[(RUN_ID, "svm", CROSS_ERA)]["f1"] == "0.3"
    assert rows[(ABLATION_RUN_ID, "svm", CROSS_ERA)]["f1"] == "0.4"


def test_logging_the_same_key_twice_is_idempotent(tmp_path) -> None:
    """Re-running a phase upserts; the committed run log must not double."""
    path = tmp_path / "metrics.csv"
    row = {**{column: "" for column in METRICS_HEADER}, "run_id": RUN_ID, "model": "dummy",
           "regime": CROSS_ERA, "f1": 0.865622}
    log_metrics(row, path)
    once = path.read_bytes()
    log_metrics(row, path)

    assert path.read_bytes() == once
