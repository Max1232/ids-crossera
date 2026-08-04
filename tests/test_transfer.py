"""Phase 7 contracts: the frozen test set and the stratified fraction draws.

The recovery curve is only a measurement of a *data budget* if the test set never moves and the
fine-tune rows never appear in it. Neither property is visible in a metric -- a curve drawn against
a leaking or drifting test set looks like a better result, not like a bug -- so both are asserted
here, on the split functions rather than on the pipeline.

Data-free by construction (see ``tests/README.md``): the frames are synthesized at TON_IoT's real
76.31% attack share, and deliberately carry **duplicate feature vectors**, because that is the
property that makes a value-based disjointness check the wrong instrument (~52% of the real target
rows repeat in the 22-column shared subspace). Nothing here writes, and nothing here reads
``reports/metrics.csv`` or ``data/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import RANDOM_SEED
from src.transfer import (
    CEILING_FRACTION,
    TEST_FRACTION,
    TRANSFER_FRACTIONS,
    fraction_indices,
    frozen_split_indices,
    sample_fraction,
)

#: TON_IoT's delivered attack share, which the split has to preserve on both sides.
TARGET_ATTACK_SHARE = 0.7631
N_ROWS = 4_000


@pytest.fixture
def target() -> tuple[pd.DataFrame, pd.Series]:
    """A synthetic target frame at TON_IoT's balance, with duplicate feature vectors by design."""
    rng = np.random.default_rng(RANDOM_SEED)
    n_attack = int(N_ROWS * TARGET_ATTACK_SHARE)
    y = pd.Series(
        np.concatenate([np.ones(n_attack, dtype=int), np.zeros(N_ROWS - n_attack, dtype=int)])
    )
    # Small integer support -> many rows collide exactly, the way the real 22-column subspace does.
    X = pd.DataFrame(rng.integers(0, 4, size=(N_ROWS, 3)), columns=["a", "b", "c"])
    return X, y


def test_pool_and_frozen_test_half_are_disjoint_and_cover_every_row(target) -> None:
    """The two halves partition the target: no shared row index, nothing dropped, nothing doubled."""
    X, y = target
    pool, test = frozen_split_indices(y, seed=RANDOM_SEED)

    assert np.intersect1d(pool, test).size == 0
    assert np.array_equal(np.union1d(pool, test), np.arange(len(y)))
    assert pool.size + test.size == len(y)
    assert test.size == pytest.approx(len(y) * TEST_FRACTION, abs=2)

    # The reason disjointness is asserted on *indices*: the frames hold duplicate feature vectors,
    # so the two halves overlap heavily in value space while sharing no row at all. A value-based
    # check would report a leak that is not one.
    duplicated = int(pd.concat([X.iloc[pool], X.iloc[test]]).duplicated().sum())
    assert duplicated > 0


def test_the_split_preserves_the_label_balance_in_both_halves(target) -> None:
    _X, y = target
    pool, test = frozen_split_indices(y, seed=RANDOM_SEED)

    share = float(y.mean())
    assert float(y.iloc[pool].mean()) == pytest.approx(share, abs=1e-3)
    assert float(y.iloc[test].mean()) == pytest.approx(share, abs=1e-3)
    # Both classes are present on both sides, or a balanced class weight cannot be derived.
    assert set(y.iloc[pool].unique()) == {0, 1}
    assert set(y.iloc[test].unique()) == {0, 1}


def test_the_same_seed_reproduces_the_same_split(target) -> None:
    """Reproducible, and actually seed-dependent -- a constant split would also pass the former."""
    _X, y = target
    pool, test = frozen_split_indices(y, seed=RANDOM_SEED)
    again = frozen_split_indices(y, seed=RANDOM_SEED)
    other = frozen_split_indices(y, seed=RANDOM_SEED + 1)

    assert np.array_equal(pool, again[0])
    assert np.array_equal(test, again[1])
    assert not np.array_equal(test, other[1])


def test_fraction_draws_come_only_from_the_pool_and_are_nested(target) -> None:
    """Every budget is drawn from the fine-tune half, and each is a superset of the smaller ones."""
    _X, y = target
    pool, test = frozen_split_indices(y, seed=RANDOM_SEED)

    previous: np.ndarray | None = None
    for fraction in (*TRANSFER_FRACTIONS, CEILING_FRACTION):
        drawn = fraction_indices(y, pool, fraction, seed=RANDOM_SEED)

        assert np.setdiff1d(drawn, pool).size == 0
        assert np.intersect1d(drawn, test).size == 0
        assert drawn.size == pytest.approx(pool.size * fraction, rel=0.02, abs=2)
        if previous is not None:
            assert np.setdiff1d(previous, drawn).size == 0, "budgets must nest, not resample"
        previous = drawn

    assert previous is not None and previous.size == pool.size  # the ceiling is the whole pool


def test_every_budget_is_scored_on_the_identical_frozen_test_half(target) -> None:
    """The x-axis is the only thing that moves: same test rows, same n, same balance, disjoint."""
    X, y = target
    _pool, test = frozen_split_indices(y, seed=RANDOM_SEED)

    reference: pd.Index | None = None
    for fraction in (*TRANSFER_FRACTIONS, CEILING_FRACTION):
        X_ft, y_ft, X_test, y_test = sample_fraction(X, y, fraction, seed=RANDOM_SEED)

        assert np.array_equal(X_test.index.to_numpy(), test)
        assert list(X_ft.columns) == list(X_test.columns)
        assert np.intersect1d(X_ft.index.to_numpy(), X_test.index.to_numpy()).size == 0
        if reference is None:
            reference = X_test.index
        else:
            assert X_test.index.equals(reference)
        assert len(y_test) == test.size
        # Stratified to within the per-class floor rounding, which on a 19-row 1% draw of this
        # small synthetic pool is a percent or two; on the real 1,055-row draw it is 5e-5.
        assert float(y_ft.mean()) == pytest.approx(
            float(y.mean()), abs=max(1e-3, 1.0 / len(y_ft))
        )


def test_a_fraction_outside_zero_to_one_is_refused(target) -> None:
    """Fraction 0 is the zero-shot point, not a draw, and >1 cannot come out of the pool."""
    _X, y = target
    pool, _test = frozen_split_indices(y, seed=RANDOM_SEED)

    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="fraction"):
            fraction_indices(y, pool, bad, seed=RANDOM_SEED)
