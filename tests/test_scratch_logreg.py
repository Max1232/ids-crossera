"""Phase 5a unit tests for the from-scratch logistic regression.

Three tests, and each one exists because a specific failure would otherwise be invisible:

1. :func:`test_converges_on_toy_separable_set` — the gradient is right. A hand-written gradient
   that is subtly wrong (a sign, a missing 1/n, a transpose) still *runs* and still produces
   plausible-looking scores on real data. On a trivially separable toy set it cannot hide: the
   loss must fall monotonically to ~0 and the fit must classify every row correctly.
2. :func:`test_class_weighting_recovers_minority_recall` — the class weighting actually bites.
   ``tests/README.md`` calls this the single most important test in the directory: under this
   project's imbalance an unweighted fit predicts the majority class almost everywhere and posts a
   *deceptively high accuracy*, a failure that looks like success in every metric except the ones
   the report leads with. This test pins both halves of that — the weighted model's minority recall
   and the unweighted model's higher accuracy.
3. :func:`test_matches_sklearn_in_distribution` — the model is not just self-consistent but
   *correct*, benchmarked against ``sklearn.linear_model.LogisticRegression`` on the real UNSW
   train/val folds. This is the one test that needs the Phase 3 artifacts and it **skips** rather
   than fails when they are absent (see :func:`_processed_artifacts_missing`).

sklearn appears in this file, and only in this file's third test, purely as the reference
implementation. The model under test imports numpy and nothing else.

Determinism: every toy fixture draws from ``np.random.default_rng(RANDOM_SEED)`` and the optimizer
is full-batch gradient descent from a zero initialization, so there is no run-to-run variation to
average over -- a failure here is reproducible by construction. Nothing in this file writes to
disk, so no ``tmp_path`` fixture is needed and ``reports/metrics.csv`` is never touched.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.config import RANDOM_SEED
from src.evaluate import POSITIVE_LABEL, evaluate
from src.models.scratch_logreg import (
    TUNED_PARAMS,
    ScratchLogReg,
    make_scratch_logreg,
    _sample_weights,
)

# --- Tolerances ---------------------------------------------------------------------------

#: Absolute tolerance on the scratch-vs-sklearn gap in **F1 and ROC-AUC**, in-distribution.
#:
#: Two points. The plan's done-when is "within a few points of the sklearn equivalent", not
#: bit-equality, and bit-equality is not even the right target: the reference minimizes the *sum* of
#: losses with lbfgs (a quasi-Newton method using curvature) at ``C=1.0``, while this model
#: minimizes the *mean* with fixed-step first-order descent at ``l2=1e-4``. Different objective
#: scaling, different optimizer, so the two land at nearby-but-distinct optima.
#:
#: The measured gap (2026-08-03, UNSW val fold) is **0.19 F1 points and 0.16 ROC-AUC points**, so
#: this tolerance carries a ~10x margin. It is set at 2 points rather than tightened to the measured
#: value on purpose: this test's job is to catch a *broken* from-scratch model (a wrong gradient, an
#: unweighted loss, a diverged fit -- all of which cost whole points), not to freeze the third
#: decimal of a number that legitimately moves with the BLAS version.
PARITY_TOLERANCE: float = 0.02

#: Minimum minority-class recall the **weighted** fit must reach on the imbalanced toy set, and the
#: maximum the **unweighted** fit may reach. The gap between them is asserted separately. Measured:
#: weighted 0.800, unweighted 0.100 -- so both thresholds sit far from the wire, which is what keeps
#: this test a detector of the failure mode rather than a fragile pin on two floats.
MIN_WEIGHTED_RECALL: float = 0.60
MAX_UNWEIGHTED_RECALL: float = 0.25
MIN_RECALL_GAP: float = 0.40


# --- Toy fixtures -------------------------------------------------------------------------


def _toy_separable(n_per_class: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Two tight, well-separated 2-D Gaussian blobs. Linearly separable with a wide margin.

    Class 0 sits at (-2, -2) and class 1 (the positive/attack class) at (+2, +2) with sd 0.5, so
    the classes are ~8 sd apart along the diagonal and no draw from this seed overlaps.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    normal = rng.normal(loc=(-2.0, -2.0), scale=0.5, size=(n_per_class, 2))
    attack = rng.normal(loc=(2.0, 2.0), scale=0.5, size=(n_per_class, 2))
    X = np.vstack([normal, attack])
    y = np.array([0] * n_per_class + [POSITIVE_LABEL] * n_per_class)
    shuffled = rng.permutation(len(y))
    return X[shuffled], y[shuffled]


def _toy_imbalanced(
    n_majority: int = 1000, n_minority: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Heavily imbalanced, deliberately *overlapping* 2-D blobs: 3.85% positive.

    Both design choices matter. The imbalance (26:1) is what pulls an unweighted fit toward the
    majority class, and the overlap (unit-variance blobs 1.2 apart, so ~0.6 sd) is what makes the
    pull decisive: with a clean margin even an unweighted fit would find the boundary and the test
    would prove nothing. This is a scaled-down caricature of the real problem -- TON_IoT is 76.31%
    attack against UNSW-train's 68.06%, and the shared 22-column subspace is far from separable.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    majority = rng.normal(loc=0.0, scale=1.0, size=(n_majority, 2))
    minority = rng.normal(loc=1.2, scale=1.0, size=(n_minority, 2))
    X = np.vstack([majority, minority])
    y = np.array([0] * n_majority + [POSITIVE_LABEL] * n_minority)
    shuffled = rng.permutation(len(y))
    return X[shuffled], y[shuffled]


def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Positive-class (attack) recall, computed by hand to keep sklearn out of tests 1 and 2."""
    positives = y_true == POSITIVE_LABEL
    return float((y_pred[positives] == POSITIVE_LABEL).mean())


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


# --- 1. Convergence -----------------------------------------------------------------------


def test_converges_on_toy_separable_set() -> None:
    """The loss falls monotonically to ~0 and the fit separates the classes perfectly.

    Also pins the first loss value at ``log 2``, which is a direct check on three things at once:
    the zero initialization (so every ``p_i`` starts at exactly 0.5), the cross-entropy formula
    itself, and the fact that balanced class weights average to 1.0 over the training set. If any
    of the three were wrong, ``-log(0.5) = 0.693147...`` would not come out on the nose.
    """
    X, y = _toy_separable()

    # l2=0 so the objective is pure cross-entropy and the achievable loss really is ~0 (a ridge
    # term would floor it at a positive value on separable data, since ||w|| must grow to drive the
    # loss down). tol is set below anything reachable in 2000 iterations, making the iteration cap
    # the stopping rule -- this test is about the descent trajectory, not about early stopping.
    model = ScratchLogReg(lr=0.5, n_epochs=2000, l2=0.0, tol=1e-12).fit(X, y)

    history = np.asarray(model.loss_history_)
    assert len(history) == model.n_iter_ == 2000

    # The uninformed starting point, exactly.
    assert history[0] == pytest.approx(math.log(2.0), abs=1e-12)

    # Monotone decrease at *every* iteration, not just end-to-end: a step size above the stability
    # limit can still end lower than it started while oscillating on the way.
    steps = np.diff(history)
    assert (steps < 0).all(), f"loss increased at {int((steps >= 0).sum())} iteration(s)"

    # Near-zero training loss and a perfect fit on trivially fittable data.
    assert history[-1] < 0.01, f"final loss {history[-1]:.6f} is not near zero"
    predictions = model.predict(X)
    assert _accuracy(y, predictions) == 1.0
    assert _recall(y, predictions) == 1.0

    # The learned separator is the (1, 1) diagonal the blobs are arranged along, and the margin
    # ranks every row on the correct side of zero -- i.e. ROC-AUC would be 1.0.
    margins = np.asarray(model.decision_function(X))
    assert (margins[y == POSITIVE_LABEL] > 0).all()
    assert (margins[y == 0] < 0).all()
    assert (np.asarray(model.weights) > 0).all()

    # predict_proba's contract: (n, 2), rows summing to 1, column 1 = attack, and consistent with
    # the margin. evaluate.positive_scores() depends on exactly this shape and ordering.
    proba = np.asarray(model.predict_proba(X))
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(model.classes_) == [0, POSITIVE_LABEL]
    assert ((proba[:, 1] >= 0.5) == (margins >= 0)).all()


# --- 2. Class weighting -------------------------------------------------------------------


def test_class_weighting_recovers_minority_recall() -> None:
    """Weighted vs unweighted on a 26:1 toy set: the weighted fit recovers minority recall.

    And the unweighted fit posts the *higher accuracy* while doing it, which is the trap this
    project's metric choices exist to avoid. Both are asserted, because "the weighted model is
    better" is only half the finding — the other half is that the standard metric would have told
    you the broken model was the good one.
    """
    X, y = _toy_imbalanced()
    assert y.mean() == pytest.approx(40 / 1040)

    # Identical everything except the loss weights, so the recall difference cannot be attributed
    # to the optimizer, the schedule, or the initialization.
    schedule = {"lr": 0.5, "n_epochs": 3000, "l2": 0.0, "tol": 1e-10}
    weighted = ScratchLogReg(class_weight="balanced", **schedule).fit(X, y)
    unweighted = ScratchLogReg(class_weight={0: 1.0, POSITIVE_LABEL: 1.0}, **schedule).fit(X, y)

    weighted_recall = _recall(y, weighted.predict(X))
    unweighted_recall = _recall(y, unweighted.predict(X))

    assert weighted_recall >= MIN_WEIGHTED_RECALL, (
        f"weighted minority recall {weighted_recall:.3f} < {MIN_WEIGHTED_RECALL}"
    )
    assert unweighted_recall <= MAX_UNWEIGHTED_RECALL, (
        f"unweighted minority recall {unweighted_recall:.3f} > {MAX_UNWEIGHTED_RECALL} -- the toy "
        "set is no longer hard enough for this test to prove anything"
    )
    assert weighted_recall - unweighted_recall >= MIN_RECALL_GAP

    # The failure mode, stated as an assertion: the collapsed model looks *better* on accuracy.
    assert _accuracy(y, unweighted.predict(X)) > _accuracy(y, weighted.predict(X))

    # The weighting is in the LOSS, not a post-hoc threshold shift. Two independent checks:
    #
    # (a) the two fits converged to genuinely different parameters -- had the weighting only moved
    #     a decision threshold, the fitted separator would be identical;
    assert not np.allclose(weighted.weights, unweighted.weights)
    assert weighted.bias != unweighted.bias
    #
    # (b) the per-row weights are the balanced formula c_k = n / (n_classes * n_k), derived from
    #     the training labels and nothing else.
    row_weights = _sample_weights(y, "balanced")
    assert row_weights[y == 0][0] == pytest.approx(1040 / (2 * 1000))
    assert row_weights[y == POSITIVE_LABEL][0] == pytest.approx(1040 / (2 * 40))
    assert row_weights.mean() == pytest.approx(1.0)
    # ... and None means balanced, not unweighted (see _sample_weights' docstring).
    assert np.array_equal(_sample_weights(y, None), row_weights)


# --- 3. In-distribution parity against sklearn --------------------------------------------


def _processed_artifacts_missing() -> str:
    """Reason to skip the parity test, or ``""`` if the Phase 3 artifacts are all present.

    ``tests/README.md`` requires that no test depend on ``data/raw/`` -- the CSVs are git-ignored and
    absent from a fresh clone. This test does not read ``data/raw/``, but it does need Phase 3's
    *derived* artifacts (``unsw_common.parquet`` and the serialized ``Preprocessor``), which are
    build products of ``./run.sh`` and are git-ignored for the same reason. So it **skips** on a
    fresh clone rather than failing: a missing build product is not a broken model, and a test suite
    that cannot go green without a 75 MB download is a test suite people stop running. Tests 1 and 2
    -- the ones that actually verify the gradient and the weighting -- are pure synthetic fixtures
    and always run.

    Rebuilding the artifacts (and un-skipping this test) is ``./run.sh``, or Phases 2-3 alone:
    ``python -m src.schema_map --build && python -m src.preprocess``.
    """
    from src.config import PREPROCESSOR, UNSW_COMMON

    missing = [path for path in (UNSW_COMMON, PREPROCESSOR) if not path.exists()]
    if not missing:
        return ""
    return (
        "Phase 3 artifacts absent ("
        + ", ".join(str(path.relative_to(path.parent.parent.parent)) for path in missing)
        + "); run `./run.sh` (or `python -m src.schema_map --build && python -m src.preprocess`) "
        "to build them. Skipped rather than failed: they are git-ignored build products."
    )


def test_matches_sklearn_in_distribution() -> None:
    """Scratch vs ``sklearn.linear_model.LogisticRegression`` on the real UNSW train/val folds.

    Fit both on the **train** fold, score both on the **val** fold, require F1 and ROC-AUC within
    :data:`PARITY_TOLERANCE`. UNSW-test is not opened (that is Phase 6's single scoring pass) and
    neither is TON_IoT.

    The reference is configured to be as comparable as the two formulations allow:
    ``class_weight="balanced"`` (the same weights, by the same formula) and the same seed. What
    cannot be matched is the objective scaling and the optimizer -- see :data:`PARITY_TOLERANCE`.

    Takes ~40 s, nearly all of it the 4,346 full-batch iterations the scratch model needs to hit
    its tolerance. That is the honest cost of a fixed-step first-order method against a
    quasi-Newton one, and it is the number the report quotes.
    """
    skip_reason = _processed_artifacts_missing()
    if skip_reason:
        pytest.skip(skip_reason)

    from sklearn.linear_model import LogisticRegression

    from src.config import set_seeds
    from src.models.scratch_logreg import load_in_distribution_folds

    set_seeds()
    folds = load_in_distribution_folds()
    X_train, y_train = folds["train"]
    X_val, y_val = folds["val"]
    assert len(X_train) == 140_272 and len(X_val) == 35_069

    scratch = make_scratch_logreg().fit(X_train, y_train)
    # The locked schedule must converge on its tolerance, not run out of iterations -- "it stopped
    # because the cap was reached" is a materially weaker claim than "it converged", and Phase 5's
    # done-when is the latter.
    assert scratch.converged_, (
        f"locked schedule hit the {scratch.n_epochs}-iteration cap without converging to "
        f"tol={scratch.tol}"
    )
    assert scratch.n_iter_ < scratch.n_epochs
    assert (np.diff(np.asarray(scratch.loss_history_)) < 0).all()

    reference = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
    ).fit(X_train, y_train)

    scratch_scores = evaluate(scratch, X_val, y_val)
    reference_scores = evaluate(reference, X_val, y_val)

    for metric in ("f1", "roc_auc"):
        gap = abs(scratch_scores[metric] - reference_scores[metric])
        assert gap <= PARITY_TOLERANCE, (
            f"{metric}: scratch {scratch_scores[metric]:.6f} vs sklearn "
            f"{reference_scores[metric]:.6f} -- gap {gap:.6f} exceeds {PARITY_TOLERANCE}"
        )

    # Sanity floor, from the same argument baselines._assert_clears_floor makes: UNSW val is 68.06%
    # attack, so predicting "attack" everywhere scores F1 ~0.81 with ROC-AUC 0.50. A model that has
    # genuinely learned a ranking must clear the AUC floor by a wide margin.
    assert scratch_scores["roc_auc"] > 0.85
    assert TUNED_PARAMS["class_weight"] == "balanced"
