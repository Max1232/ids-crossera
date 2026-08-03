"""Phase 5b unit tests for the from-scratch MLP.

Four tests, and each one exists because a specific failure would otherwise be invisible:

1. :func:`test_gradient_check_matches_finite_differences` — **the load-bearing one.** Every entry
   of every analytic ``dJ/dW^{(l)}`` and ``dJ/db^{(l)}`` is compared against a central finite
   difference of the *class-weighted* loss. Unlike the logistic regression, this model has no
   convexity to lean on and no monotone-descent guarantee to assert, so nothing else in this file
   can tell a correct backprop from a subtly wrong one: a flipped sign, a missing transpose, a
   ``1/n`` applied twice or a ReLU mask taken from the wrong layer all still *run*, still train to
   something plausible, and still pass tests 2-4. A finite difference does not care what the
   derivation was supposed to be.
2. :func:`test_converges_on_toy_separable_set` — the whole loop works end to end. On trivially
   separable data the loss must fall to ~0 and the fit must classify every row correctly; if it
   cannot fit data that is trivially fittable, no in-distribution score from it means anything.
3. :func:`test_class_weighting_recovers_minority_recall` — the class weighting actually bites.
   ``tests/README.md`` calls this the single most important test in the directory: under this
   project's imbalance an unweighted net predicts the majority class almost everywhere and posts a
   *deceptively high accuracy*, a failure that looks like success in every metric except the ones
   the report leads with. This test pins both halves of that, plus the fact that the weight is
   inside the objective (it changes the gradient) rather than a threshold moved afterwards.
4. :func:`test_matches_sklearn_in_distribution` — the model is not just self-consistent but
   *correct*, benchmarked against ``sklearn.neural_network.MLPClassifier`` on the real UNSW
   train/val folds. This is the one test that needs the Phase 3 artifacts and it **skips** rather
   than fails when they are absent (see :func:`_processed_artifacts_missing`).

sklearn appears in this file, and only in this file's fourth test, purely as the reference
implementation. The model under test imports numpy and its pure-numpy sibling, nothing else.

Determinism: every toy fixture draws from ``np.random.default_rng(RANDOM_SEED)``, and the model's
He initialization *and* its per-epoch batch permutations both come from a single
``default_rng(seed)`` created inside ``fit``, so two runs of this file produce bit-identical
numbers -- a failure here is reproducible by construction. Nothing in this file writes to disk, so
no ``tmp_path`` fixture is needed and ``reports/metrics.csv`` is never touched.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import RANDOM_SEED
from src.evaluate import POSITIVE_LABEL, evaluate
from src.models.scratch_mlp import (
    INPUT_DIM,
    TUNED_PARAMS,
    ScratchMLP,
    make_scratch_mlp,
)
from src.models.scratch_logreg import _sample_weights

# --- Tolerances ---------------------------------------------------------------------------

#: Step size for the central finite difference ``(J(w + h) - J(w - h)) / 2h``.
#:
#: Chosen, not guessed. A central difference carries two competing errors: truncation, which is
#: ``O(h^2)`` (~1e-10 here), and floating-point cancellation, which is ``O(eps_machine / h)``
#: (~1e-16/1e-5 = 1e-11). 1e-5 sits near the minimum of their sum for a loss of order 1 in float64.
#: 1e-6 also passes but an order of magnitude less comfortably (worst observed relative error
#: 1.2e-7 against 2.0e-8 at 1e-5), which is exactly the cancellation term growing.
GRAD_CHECK_EPS: float = 1e-5

#: Maximum relative error between the analytic and numerical gradient of any single parameter
#: block, as ``||analytic - numerical|| / (||analytic|| + ||numerical||)`` -- the standard
#: gradient-check statistic, normalized so it cannot be flattered by a block whose gradient happens
#: to be large.
#:
#: 1e-6 is a *tight* tolerance and is meant to be: the conventional reading is that below 1e-7 the
#: gradient is correct, 1e-5 to 1e-3 is suspicious, and above 1e-3 it is wrong. The measured worst
#: case across all six parameter blocks and all three weighting schemes is **2.0e-8**, so this
#: carries a ~50x margin while still sitting far below the level any real backprop bug could hide
#: under -- an off-by-one in a transpose produces relative errors of order 1, not 1e-6.
GRADIENT_TOLERANCE: float = 1e-6

#: How many times the worst-case perturbation every *hidden* pre-activation must clear zero by, in
#: the gradient-check fixture.
#:
#: ReLU has a kink at 0 where the analytic derivative (taken as 0, see ``scratch_mlp._relu_grad``)
#: and a finite difference straddling the kink legitimately disagree — that would be a false
#: failure, not a caught bug. Nudging one weight by ``±GRAD_CHECK_EPS`` moves a pre-activation by at
#: most ``GRAD_CHECK_EPS * max|A|`` (~3.4e-5 on this fixture), so the guard is written against that
#: quantity rather than a hardcoded distance: it stays correct if the fixture's scale ever changes.
#: Measured slack at the checked point is ~380x for the tightest of the three weighting arms.
#: Asserted rather than assumed, so a future edit to the fixture cannot silently invalidate the
#: test by parking a unit on the kink.
KINK_SAFETY_FACTOR: float = 100.0

#: Absolute tolerance on the scratch-vs-sklearn gap in **F1 and ROC-AUC**, in-distribution, with
#: the two models solving the *same* objective — i.e. this model's unweighted control arm against
#: ``MLPClassifier``, which has no ``class_weight`` parameter at all and is therefore always
#: unweighted.
#:
#: The measured gap (2026-08-03, UNSW val fold, ``(22, 44, 22, 1)`` vs
#: ``hidden_layer_sizes=(44, 22)``) is **0.52 F1 points and 0.42 ROC-AUC points**, so this carries
#: a ~4x margin. It is not tightened to the measured value on purpose: the job here is to catch a
#: *broken* from-scratch model -- a wrong gradient, an unweighted loss, a diverged fit, all of which
#: cost whole points -- not to freeze the third decimal of a number that legitimately moves with the
#: BLAS version. Bit-equality is not the target either: the reference uses Adam with an adaptive
#: per-parameter step and its own early stopping, this model uses fixed-step mini-batch SGD for a
#: fixed 40 epochs, so the two land at nearby-but-distinct optima of a non-convex objective.
PARITY_TOLERANCE: float = 0.02

#: The same tolerance for the **locked, class-weighted** model — the one Phase 6 actually ships.
#: Wider, and the extra width is a known, measured, *intended* effect rather than slack:
#: ``MLPClassifier`` cannot express class weighting, so the reference is necessarily unweighted,
#: and re-weighting a 68%-attack training set deliberately trades attack-class F1 for balanced
#: accuracy. Measured gap: **1.84 F1 points and 0.11 ROC-AUC points**, of which the F1 component is
#: almost entirely the weighting (the unweighted control closes it to 0.52 -- see
#: :data:`PARITY_TOLERANCE`), while the model's *balanced accuracy* moves the other way, 0.9071 ->
#: 0.9204. Both arms are asserted so the difference between "the implementation is wrong" and "the
#: objective is deliberately different" stays legible.
WEIGHTED_PARITY_TOLERANCE: float = 0.03

#: Minimum minority-class recall the **weighted** fit must reach on the imbalanced toy set, and the
#: maximum the **unweighted** fit may reach. The gap between them is asserted separately. Measured:
#: weighted 0.800, unweighted 0.100 -- so both thresholds sit far from the wire, which is what keeps
#: this test a detector of the failure mode rather than a fragile pin on two floats. Identical
#: values to ``test_scratch_logreg``, on an identical fixture, because the failure being detected
#: is identical.
MIN_WEIGHTED_RECALL: float = 0.60
MAX_UNWEIGHTED_RECALL: float = 0.25
MIN_RECALL_GAP: float = 0.40


# --- Toy fixtures -------------------------------------------------------------------------


def _toy_separable(n_per_class: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Two tight, well-separated 2-D Gaussian blobs. Linearly separable with a wide margin.

    Class 0 sits at (-2, -2) and class 1 (the positive/attack class) at (+2, +2) with sd 0.5, so
    the classes are ~8 sd apart along the diagonal and no draw from this seed overlaps. Identical
    to ``test_scratch_logreg``'s fixture, deliberately: the two from-scratch models are being held
    to the same "can it fit the trivially fittable" bar.
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
    would prove nothing. A net is if anything *more* prone to this than a linear model -- it has
    the capacity to carve the 40 minority points out and simply has no incentive to.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    majority = rng.normal(loc=0.0, scale=1.0, size=(n_majority, 2))
    minority = rng.normal(loc=1.2, scale=1.0, size=(n_minority, 2))
    X = np.vstack([majority, minority])
    y = np.array([0] * n_majority + [POSITIVE_LABEL] * n_minority)
    shuffled = rng.permutation(len(y))
    return X[shuffled], y[shuffled]


def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Positive-class (attack) recall, computed by hand to keep sklearn out of tests 1-3."""
    positives = y_true == POSITIVE_LABEL
    return float((y_pred[positives] == POSITIVE_LABEL).mean())


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


# --- 1. Gradient check --------------------------------------------------------------------


def _grad_check_fixture(
    class_weight: dict[int, float] | str | None,
) -> tuple[ScratchMLP, np.ndarray, np.ndarray]:
    """A small two-hidden-layer net trained a few steps, plus the data it was trained on.

    Deliberately small (6 rows, 4 -> 5 -> 3 -> 1) so a full ``2 * 44``-evaluation numerical
    gradient is instant, and deliberately **two** hidden layers so the recursive step
    ``delta^{(l-1)} = (delta^{(l)} W^{(l)T}) ⊙ relu'(Z^{(l-1)})`` is exercised at least once rather
    than only its base case.

    The check runs at a *trained* point rather than at initialization. At init the biases are
    exactly zero and the head's gradient is near-degenerate, so a check there is weaker than it
    looks; ten epochs of SGD move every parameter to a generic point where all six blocks carry
    real signal, while still leaving every hidden unit comfortably clear of ReLU's kink (see
    :data:`KINK_SAFETY_FACTOR`, which asserts exactly that rather than trusting it).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    X = rng.normal(size=(6, 4))
    y = np.array([0, POSITIVE_LABEL, 0, POSITIVE_LABEL, POSITIVE_LABEL, 0])

    model = ScratchMLP(
        layer_sizes=(4, 5, 3, 1),
        lr=0.05,
        n_epochs=10,
        batch_size=3,
        seed=RANDOM_SEED,
        class_weight=class_weight,
    ).fit(X, y)
    return model, X, y


def _numerical_gradient(
    model: ScratchMLP, key: str, X: np.ndarray, y: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Central finite difference of the class-weighted loss w.r.t. every entry of ``params[key]``.

    Perturbs one scalar at a time, restores it exactly, and re-runs the *whole* forward pass -- so
    this is an end-to-end numerical derivative of the same quantity ``backward`` claims to
    differentiate, sharing no code with it beyond ``forward`` and ``_loss``.
    """
    block = model.params[key]
    numerical = np.zeros_like(block)
    iterator = np.nditer(block, flags=["multi_index"])
    while not iterator.finished:
        index = iterator.multi_index
        original = block[index]

        block[index] = original + GRAD_CHECK_EPS
        plus = model._loss(model.forward(X)[1]["Z"][-1], y, c)
        block[index] = original - GRAD_CHECK_EPS
        minus = model._loss(model.forward(X)[1]["Z"][-1], y, c)
        block[index] = original

        numerical[index] = (plus - minus) / (2.0 * GRAD_CHECK_EPS)
        iterator.iternext()
    return numerical


def _relative_error(analytic: np.ndarray, numerical: np.ndarray) -> float:
    """``||a - n|| / (||a|| + ||n||)`` — the standard gradient-check statistic."""
    denominator = np.linalg.norm(analytic) + np.linalg.norm(numerical)
    if denominator == 0.0:
        return 0.0
    return float(np.linalg.norm(analytic - numerical) / denominator)


@pytest.mark.parametrize(
    "class_weight",
    ["balanced", {0: 1.0, POSITIVE_LABEL: 1.0}, {0: 2.3, POSITIVE_LABEL: 0.4}],
    ids=["balanced", "unweighted", "skewed"],
)
def test_gradient_check_matches_finite_differences(
    class_weight: dict[int, float] | str,
) -> None:
    """Analytic backprop == numerical gradient, every layer, every parameter, weighted loss.

    Run three times over three weighting schemes -- balanced, the explicit unweighted control, and
    a deliberately lopsided ``{0: 2.3, 1: 0.4}`` -- because the class weight enters at
    ``delta^{(L)} = (1/n) c ⊙ (p - y)`` and then rides down the entire chain. Checking only the
    unweighted case would verify a loss this project never optimizes; checking only the balanced
    case would pass even if ``c`` were applied as a single scalar rather than per row, since
    balanced weights are nearly equal on a set this small. The skewed arm is what makes the
    per-row-ness observable.
    """
    model, X, y = _grad_check_fixture(class_weight)
    row_weights = model._row_weights(y)

    probabilities, cache = model.forward(X)
    assert probabilities.shape == (len(y), 1)

    # The ReLU kink guard -- see KINK_SAFETY_FACTOR. cache["Z"][:-1] is the hidden pre-activations;
    # the output pre-activation is excluded because sigmoid is smooth everywhere and routinely sits
    # at exactly 0 for a row whose hidden activations all clipped to zero.
    closest = min(float(np.abs(z).min()) for z in cache["Z"][:-1])
    perturbation = GRAD_CHECK_EPS * max(float(np.abs(a).max()) for a in cache["A"][:-1])
    assert closest > KINK_SAFETY_FACTOR * perturbation, (
        f"a hidden pre-activation sits {closest:.2e} from ReLU's kink at 0, under the "
        f"{KINK_SAFETY_FACTOR:.0f}x margin over the {perturbation:.2e} shift a finite-difference "
        "step can cause; the fixture is no longer valid for a gradient check"
    )

    analytic = model.backward(y, cache)
    assert set(analytic) == set(model.params)

    for key in sorted(model.params):
        numerical = _numerical_gradient(model, key, X, y, row_weights)
        assert numerical.shape == analytic[key].shape == model.params[key].shape
        error = _relative_error(analytic[key], numerical)
        assert error < GRADIENT_TOLERANCE, (
            f"{key}: analytic vs numerical gradient relative error {error:.3e} exceeds "
            f"{GRADIENT_TOLERANCE:.0e} -- backprop does not compute the derivative of the loss "
            f"it is supposed to.\nanalytic:\n{analytic[key]}\nnumerical:\n{numerical}"
        )
        # Non-degenerate: a block of zeros would match any numerical gradient trivially.
        assert np.abs(analytic[key]).max() > 0.0


# --- 2. Convergence -----------------------------------------------------------------------


def test_converges_on_toy_separable_set() -> None:
    """The loss falls to ~0 and the fit separates the classes perfectly.

    Also checks ``predict_proba``'s contract, because ``evaluate.positive_scores`` depends on
    exactly that shape and column ordering to find the attack score for ROC-AUC.
    """
    X, y = _toy_separable()

    model = ScratchMLP(
        layer_sizes=(2, 8, 1), lr=0.1, n_epochs=200, batch_size=32, seed=RANDOM_SEED
    ).fit(X, y)

    history = np.asarray(model.loss_history_)
    assert len(history) == model.n_iter_ == 200

    # Near-zero training loss and a perfect fit on trivially fittable data.
    assert history[-1] < 0.01, f"final loss {history[-1]:.6f} is not near zero"
    predictions = model.predict(X)
    assert _accuracy(y, predictions) == 1.0
    assert _recall(y, predictions) == 1.0

    # Mini-batch SGD on a non-convex objective carries no monotone-descent *guarantee* -- unlike
    # scratch_logreg, where the same assertion is a theorem. On this fixture it is nonetheless
    # monotone at every epoch, and since the run is bit-deterministic under the fixed seed that is
    # a fact worth pinning: an optimizer bug that made the loss bounce would still reach a low
    # final value here and would otherwise go unnoticed.
    steps = np.diff(history)
    assert (steps < 0).all(), f"loss increased at {int((steps >= 0).sum())} epoch(s)"

    # predict_proba's contract: (n, 2), rows summing to 1, column 1 = attack, consistent with
    # predict at the 0.5 threshold.
    proba = np.asarray(model.predict_proba(X))
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(model.classes_) == [0, POSITIVE_LABEL]
    assert ((proba[:, 1] >= 0.5) == (predictions == POSITIVE_LABEL)).all()

    # Per-layer parameter access, which Phase 7's freeze-and-retrain is built on.
    assert sorted(model.params) == ["W1", "W2", "b1", "b2"]
    assert model.params["W1"].shape == (2, 8) and model.params["W2"].shape == (8, 1)
    assert model.head_keys() == ("W2", "b2")


# --- 3. Class weighting -------------------------------------------------------------------


def test_class_weighting_recovers_minority_recall() -> None:
    """Weighted vs unweighted on a 26:1 toy set: the weighted net recovers minority recall.

    And the unweighted net posts the *higher accuracy* while doing it, which is the trap this
    project's metric choices exist to avoid. Both are asserted, because "the weighted model is
    better" is only half the finding — the other half is that the standard metric would have told
    you the broken model was the good one.
    """
    X, y = _toy_imbalanced()
    assert y.mean() == pytest.approx(40 / 1040)

    # Identical everything except the loss weights, same seed, so the two nets start from the same
    # He draw and see the same batch order: the recall difference cannot be attributed to the
    # optimizer, the schedule, or the initialization.
    schedule = {
        "layer_sizes": (2, 8, 1),
        "lr": 0.05,
        "n_epochs": 300,
        "batch_size": 64,
        "seed": RANDOM_SEED,
    }
    weighted = ScratchMLP(class_weight="balanced", **schedule).fit(X, y)
    unweighted = ScratchMLP(class_weight={0: 1.0, POSITIVE_LABEL: 1.0}, **schedule).fit(X, y)

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

    # The weighting is in the LOSS and the GRADIENT, not a post-hoc threshold shift. Three checks:
    #
    # (a) the two fits converged to genuinely different parameters -- had the weighting only moved
    #     a decision threshold, the fitted network would be identical;
    assert not np.allclose(weighted.params["W1"], unweighted.params["W1"])
    assert not np.allclose(weighted.params["W2"], unweighted.params["W2"])
    #
    # (b) the resolved per-class weights are the balanced formula c_k = n / (n_classes * n_k),
    #     derived from the training labels and nothing else -- and by the same function the
    #     from-scratch logistic regression uses, so the two models cannot drift apart on it;
    row_weights = _sample_weights(y, "balanced")
    assert weighted.class_weight_[0] == pytest.approx(1040 / (2 * 1000))
    assert weighted.class_weight_[POSITIVE_LABEL] == pytest.approx(1040 / (2 * 40))
    assert row_weights.mean() == pytest.approx(1.0)
    assert unweighted.class_weight_ == {0: 1.0, POSITIVE_LABEL: 1.0}
    #
    # ... and None means balanced, not unweighted. This is the convention both from-scratch models
    # share and deliberately invert relative to sklearn's; a regression here would silently ship an
    # all-majority net from a caller that simply omitted the argument.
    defaulted = ScratchMLP(class_weight=None, **schedule)
    defaulted._resolve_class_weights(y)
    assert defaulted.class_weight_ == weighted.class_weight_
    assert np.array_equal(_sample_weights(y, None), row_weights)
    #
    # (c) at *identical parameters*, the analytic gradient differs between the two weightings --
    #     i.e. the weight is genuinely inside dJ/dW and not applied somewhere after the fit.
    probe = ScratchMLP(class_weight="balanced", **schedule)
    probe.initialize_params()
    _, cache = probe.forward(X)
    probe._resolve_class_weights(y)
    balanced_grads = probe.backward(y, cache)
    probe.class_weight_ = {0: 1.0, POSITIVE_LABEL: 1.0}
    unweighted_grads = probe.backward(y, cache)
    assert not np.allclose(balanced_grads["W1"], unweighted_grads["W1"])
    assert not np.allclose(balanced_grads["b2"], unweighted_grads["b2"])


# --- 4. In-distribution parity against sklearn --------------------------------------------


def _processed_artifacts_missing() -> str:
    """Reason to skip the parity test, or ``""`` if the Phase 3 artifacts are all present.

    ``tests/README.md`` requires that no test depend on ``data/raw/`` -- the CSVs are git-ignored and
    absent from a fresh clone. This test does not read ``data/raw/``, but it does need Phase 3's
    *derived* artifacts (``unsw_common.parquet`` and the serialized ``Preprocessor``), which are
    build products of ``./run.sh`` and are git-ignored for the same reason. So it **skips** on a
    fresh clone rather than failing: a missing build product is not a broken model, and a test suite
    that cannot go green without a 75 MB download is a test suite people stop running. Tests 1-3 --
    including the gradient check, which is what actually verifies the backprop -- are pure synthetic
    fixtures and always run.

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
    """Scratch vs ``sklearn.neural_network.MLPClassifier`` on the real UNSW train/val folds.

    Fit on the **train** fold, score on the **val** fold. UNSW-test is not opened (that is Phase 6's
    single scoring pass) and neither is TON_IoT.

    Two arms, because ``MLPClassifier`` has **no** ``class_weight`` parameter and so cannot be made
    to solve the objective this project's model actually minimizes:

    * the *unweighted control* against the reference, at :data:`PARITY_TOLERANCE` — the honest
      like-for-like implementation check;
    * the *locked, class-weighted* model against the same reference, at the wider
      :data:`WEIGHTED_PARITY_TOLERANCE` — plus an assertion that the extra distance is the
      weighting doing its job (higher balanced accuracy), not the implementation losing ground.

    Takes ~55 s: two 40-epoch scratch fits at ~13 s each plus the reference's ~145 Adam iterations.
    """
    skip_reason = _processed_artifacts_missing()
    if skip_reason:
        pytest.skip(skip_reason)

    from sklearn.neural_network import MLPClassifier

    from src.config import set_seeds
    from src.models.scratch_mlp import _load_folds

    set_seeds()
    folds = _load_folds()
    X_train, y_train = folds["train"]
    X_val, y_val = folds["val"]
    assert len(X_train) == 140_272 and len(X_val) == 35_069
    # The input width the locked architecture is dimensioned on. _load_folds() already refuses to
    # return anything else; restated here so the reason this test's net has 22 inputs is visible.
    assert X_train.shape[1] == INPUT_DIM == TUNED_PARAMS["layer_sizes"][0] == 22

    locked = make_scratch_mlp().fit(X_train, y_train)
    control = make_scratch_mlp(class_weight={0: 1.0, POSITIVE_LABEL: 1.0}).fit(X_train, y_train)

    # Same shape as the locked net, same batch size, same seed. What cannot be matched is the
    # optimizer (Adam vs fixed-step SGD) or the objective (unweighted, necessarily).
    reference = MLPClassifier(
        hidden_layer_sizes=TUNED_PARAMS["layer_sizes"][1:-1],
        batch_size=TUNED_PARAMS["batch_size"],
        random_state=RANDOM_SEED,
    ).fit(X_train, y_train)

    locked_scores = evaluate(locked, X_val, y_val)
    control_scores = evaluate(control, X_val, y_val)
    reference_scores = evaluate(reference, X_val, y_val)

    for metric in ("f1", "roc_auc"):
        gap = abs(control_scores[metric] - reference_scores[metric])
        assert gap <= PARITY_TOLERANCE, (
            f"{metric} (unweighted control): scratch {control_scores[metric]:.6f} vs sklearn "
            f"{reference_scores[metric]:.6f} -- gap {gap:.6f} exceeds {PARITY_TOLERANCE}"
        )
        gap = abs(locked_scores[metric] - reference_scores[metric])
        assert gap <= WEIGHTED_PARITY_TOLERANCE, (
            f"{metric} (locked, class-weighted): scratch {locked_scores[metric]:.6f} vs sklearn "
            f"{reference_scores[metric]:.6f} -- gap {gap:.6f} exceeds "
            f"{WEIGHTED_PARITY_TOLERANCE}"
        )

    # The weighted model's distance from the unweighted reference is the objective, not a defect:
    # it gives up attack-class F1 and gets prevalence-robust accuracy back.
    assert locked_scores["balanced_accuracy"] > control_scores["balanced_accuracy"]

    # Sanity floor, from the same argument baselines._assert_clears_floor makes: UNSW val is 68.06%
    # attack, so predicting "attack" everywhere scores F1 ~0.81 with ROC-AUC 0.50. A model that has
    # genuinely learned a ranking must clear the AUC floor by a wide margin.
    assert locked_scores["roc_auc"] > 0.85
    assert TUNED_PARAMS["class_weight"] == "balanced"
