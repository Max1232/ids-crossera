"""Phase 6 contracts: the leakage seal, the two regimes, and the metrics-log key.

Every test here is data-free by construction (see ``tests/README.md``): the real CSVs are
git-ignored and absent from a fresh clone, so the regime behaviour is exercised on a tiny
synthetic frame whose two "eras" differ in class balance the way UNSW-test and TON_IoT do. What is
being checked is the *contract*, not the numbers -- that a fit against a test set raises instead of
succeeding quietly, that each regime's row carries the balance of the set it was measured on, and
that conditions differing only in ``run_id`` coexist in the log rather than overwriting each other.

Also covers the two ablations' feature-block derivation. Those tests exist because ``protocol`` and
``conn_state`` both encode to four one-hot levels, so both ablations run at d=18: the width cannot
tell them apart, and every guard that keeps them from being silently swapped is asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

import copy

from src.config import RANDOM_SEED
from src.evaluate import (
    CONN_STATE_ABLATION_RUN_ID,
    CONN_STATE_FEATURE,
    CROSS_ERA,
    IN_DISTRIBUTION,
    METRICS_HEADER,
    PROTO_ABLATION_RUN_ID,
    PROTOCOL_FEATURE,
    RUN_ID,
    LeakageError,
    ablated_columns,
    categorical_columns,
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


@pytest.fixture(scope="module")
def preprocessor():
    """A ``Preprocessor`` fitted on a synthetic frame that reproduces the real 22-column schema.

    Data-free by construction, per ``tests/README.md``: the parquets are git-ignored and
    ``data/processed/preprocessor.joblib`` is a build product absent from a fresh clone, so neither
    may be loaded here. Every level of all three categoricals appears, so the fitted widths are the
    real ``(4, 6, 4)`` and the derived one-hot names are the real ones -- which is what makes the
    positional-offset assertions below meaningful rather than self-fulfilling.
    """
    import pandas as pd

    from src.preprocess import Preprocessor
    from src.schema_map import NUMERIC_FEATURES, RARE_BUCKET, ZERO_DURATION_FLAG

    rng = np.random.default_rng(RANDOM_SEED)
    frame = pd.DataFrame({column: rng.random(12) * 10.0 for column in NUMERIC_FEATURES})
    frame[PROTOCOL_FEATURE] = ["tcp", "udp", "icmp", RARE_BUCKET] * 3
    frame["service"] = ["-", "dns", "http", "ftp", "ssl", RARE_BUCKET] * 2
    frame[CONN_STATE_FEATURE] = ["completed", "no_response", RARE_BUCKET, "reset"] * 3
    frame[ZERO_DURATION_FLAG] = [0, 1] * 6
    return Preprocessor().fit(frame)


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
    """Each ablation's rows must sit beside the unablated ones, not overwrite them.

    This is the contract the ``run_id``-per-condition convention rests on: ``(run_id, model,
    regime)`` is the key, so the same model and regime under three conditions is three rows. The
    third condition matters specifically because ``phase6-crossera-no_conn_state`` is a strict string
    extension of ``phase6-crossera`` -- if anything in the log ever matched on a prefix, this is the
    test that would fail.
    """
    path = tmp_path / "metrics.csv"
    base = {column: "" for column in METRICS_HEADER}
    conditions = (
        (RUN_ID, 0.30),
        (PROTO_ABLATION_RUN_ID, 0.40),
        (CONN_STATE_ABLATION_RUN_ID, 0.50),
    )
    for run_id, f1 in conditions:
        log_metrics({**base, "run_id": run_id, "model": "svm", "regime": CROSS_ERA, "f1": f1}, path)

    rows = read_metrics(path)
    assert len(rows) == 3
    assert rows[(RUN_ID, "svm", CROSS_ERA)]["f1"] == "0.3"
    assert rows[(PROTO_ABLATION_RUN_ID, "svm", CROSS_ERA)]["f1"] == "0.4"
    assert rows[(CONN_STATE_ABLATION_RUN_ID, "svm", CROSS_ERA)]["f1"] == "0.5"


def test_the_three_phase6_run_ids_are_distinct() -> None:
    """The two ablation labels are load-bearing strings, not just identifiers.

    ``log_metrics`` upserts on ``(run_id, model, regime)`` and never deletes, so changing one of
    these spellings would orphan the rows already committed under the old one and silently duplicate
    the condition under the new one. The literal assertions are the guard against exactly that.
    """
    ids = (RUN_ID, PROTO_ABLATION_RUN_ID, CONN_STATE_ABLATION_RUN_ID)
    assert len(set(ids)) == 3
    assert RUN_ID == "phase6-crossera"
    assert PROTO_ABLATION_RUN_ID == "phase6-crossera-no_proto"
    assert CONN_STATE_ABLATION_RUN_ID == "phase6-crossera-no_conn_state"


def test_categorical_columns_derives_each_feature_s_own_block(preprocessor) -> None:
    """Each ablation gets its own one-hot block -- disjoint, and at the same width.

    The literal names are pinned deliberately: ``protocol_other``, ``service_other`` and
    ``conn_state_other`` all exist in this schema, so a regression to prefix-matching would pick up
    columns from all three features and this is what catches it.
    """
    proto = categorical_columns(preprocessor, PROTOCOL_FEATURE)
    state = categorical_columns(preprocessor, CONN_STATE_FEATURE)

    assert proto == ("protocol_icmp", "protocol_other", "protocol_tcp", "protocol_udp")
    assert state == (
        "conn_state_completed",
        "conn_state_no_response",
        "conn_state_other",
        "conn_state_reset",
    )
    # Different experiments ...
    assert not set(proto) & set(state)
    # ... that happen to share a width, which is why d never identifies a condition.
    assert len(proto) == len(state) == 4

    frozen = set(preprocessor.feature_names_)
    assert set(proto) <= frozen
    assert set(state) <= frozen


def test_categorical_columns_rejects_an_absent_feature(preprocessor) -> None:
    """A feature the encoder never saw has no block to remove, so the ablation cannot be built."""
    for absent in ("protocol_", "attack_cat", ""):
        with pytest.raises(ValueError, match="not among"):
            categorical_columns(preprocessor, absent)


def test_categorical_columns_raises_when_the_encoder_and_schema_disagree(preprocessor) -> None:
    """Derived names missing from the frozen schema mean the encoder is not what we think."""
    broken = copy.deepcopy(preprocessor)  # deepcopy: the fixture is module-scoped
    broken.feature_names_ = tuple(
        name for name in broken.feature_names_ if not name.startswith(f"{CONN_STATE_FEATURE}_")
    )
    with pytest.raises(RuntimeError, match="missing"):
        categorical_columns(broken, CONN_STATE_FEATURE)


def test_categorical_columns_catches_a_shifted_block(preprocessor) -> None:
    """The guard the generalization adds, and it is only reachable past index 0.

    ``start`` is a prefix sum over every earlier feature's width, so for ``protocol`` at index 0 it
    is 0 however wrong the widths are. ``conn_state`` sits at index 2 with its block starting at 10,
    so a category/feature-count mismatch would slide its slice onto another feature's columns -- all
    of which are legitimately in ``feature_names_``, so the membership check alone would pass it.
    """
    broken = copy.deepcopy(preprocessor)
    broken.encoder_.categories_ = broken.encoder_.categories_[:2]
    with pytest.raises(RuntimeError, match="positional offset"):
        categorical_columns(broken, CONN_STATE_FEATURE)


def test_the_two_ablations_drop_the_same_width_but_different_columns(preprocessor) -> None:
    """22 -> 18 for both, and the two survivor sets are not the same 18 columns."""
    full = tuple(preprocessor.feature_names_)
    assert len(full) == 22

    survivors = {}
    for feature in (PROTOCOL_FEATURE, CONN_STATE_FEATURE):
        dropped = categorical_columns(preprocessor, feature)
        survivors[feature] = ablated_columns(full, dropped, feature)
        assert len(survivors[feature]) == 18

    assert set(survivors[PROTOCOL_FEATURE]) != set(survivors[CONN_STATE_FEATURE])


def test_ablated_columns_raises_if_it_removed_the_wrong_number(preprocessor) -> None:
    """A column that is not in the frame cannot be 'removed'; the width assertion catches it."""
    full = tuple(preprocessor.feature_names_)
    with pytest.raises(RuntimeError, match="not the"):
        ablated_columns(full, ("a_column_that_is_not_in_the_schema",), PROTOCOL_FEATURE)


def test_logging_the_same_key_twice_is_idempotent(tmp_path) -> None:
    """Re-running a phase upserts; the committed run log must not double."""
    path = tmp_path / "metrics.csv"
    row = {**{column: "" for column in METRICS_HEADER}, "run_id": RUN_ID, "model": "dummy",
           "regime": CROSS_ERA, "f1": 0.865622}
    log_metrics(row, path)
    once = path.read_bytes()
    log_metrics(row, path)

    assert path.read_bytes() == once
