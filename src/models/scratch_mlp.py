"""From-scratch multilayer perceptron in numpy (Phase 5).

Implements the neural axis *and* the base for RQ2 transfer learning (freeze-and-retrain needs
our own layers). Forward pass, backpropagation, mini-batch training loop, sigmoid/softmax
output. Must converge on a toy separable dataset (unit test) and land near the sklearn
equivalent in-distribution.

**Pure numpy, by requirement.** Nothing in this module imports sklearn, scipy, torch or any
autograd: the forward pass, the loss, every partial derivative and the optimizer are written out
below by hand. The only sklearn anywhere near this model is (a) ``evaluate.evaluate()``, which
*scores* its predictions, and (b) ``tests/test_scratch_mlp.py``, which compares it against
``MLPClassifier`` as a reference. Neither is part of the model. The sibling
``scratch_logreg`` this module imports from is itself pure numpy, so the import graph stays clean.

--------------------------------------------------------------------------------------------
The math (this is the derivation the report's Methods section typesets — rubric-bearing)
--------------------------------------------------------------------------------------------

Notation. Rows are examples: ``A^{(0)} = X`` is ``(n, d)``. Layer ``l = 1 .. L`` holds a weight
matrix ``W^{(l)}`` of shape ``(n_{l-1}, n_l)`` and a bias row ``b^{(l)}`` of shape ``(1, n_l)``.
``L`` is the output layer, which has exactly one unit.

Forward.

    Z^{(l)} = A^{(l-1)} W^{(l)} + b^{(l)}                      l = 1 .. L
    A^{(l)} = relu(Z^{(l)}) = max(Z^{(l)}, 0)                  l = 1 .. L-1   (hidden)
    p       = A^{(L)} = sigma(Z^{(L)}) = 1 / (1 + e^{-Z^{(L)}})               (output)

``p_i = P[y_i = 1 | x_i] = P[attack]``. The output is a **single sigmoid unit rather than a
2-way softmax**: for K = 2 the softmax is over-parameterized along the direction
``(w_0 + t, w_1 + t)``, so ``softmax([z_0, z_1])_1 == sigma(z_1 - z_0)`` and the two
parameterizations describe the identical model. The 1-unit form halves the output layer, has a
unique optimum, and gives ``predict_proba`` its attack column directly. ``layer_sizes[-1] != 1``
is therefore rejected in ``__init__`` rather than silently reinterpreted.

Loss. Class-weighted binary cross-entropy, averaged over the n rows of the mini-batch:

    J = (1/n) * sum_i c_{y_i} [ -y_i log p_i - (1 - y_i) log(1 - p_i) ]

``c_{y_i}`` is the weight of row i's *class*, derived once from the **full training labels** (see
``scratch_logreg._sample_weights``, which this module reuses verbatim). No L2 term: the tuning
surface the proposal promises for this model is width and depth, and at n/d ~ 6400 the net does
not need a ridge to generalize.

Backward. Write ``delta^{(l)} = dJ / dZ^{(l)}``, shape ``(n, n_l)``. Three facts:

  1. ``sigma'(z) = sigma(z)(1 - sigma(z)) = p(1 - p)``
  2. ``d/dp of -[y log p + (1-y) log(1-p)] = (p - y) / (p(1 - p))``
  3. ``relu'(z) = 1[z > 0]``

(1) and (2) multiply to cancel ``p(1-p)`` exactly — the same collapse that makes logistic
regression clean — so the output layer's error is the class-weighted residual and nothing else:

    delta^{(L)} = (1/n) * c ⊙ (p - y)                                        (n, 1)

That ``c`` is the *entire* reason the class weighting reaches the parameters. It multiplies the
error signal at the output before anything is propagated, so every gradient downstream of it
carries the weight. Weighting is in the objective, not in a threshold moved afterwards.

Then, layer by layer, from ``Z^{(l)} = A^{(l-1)} W^{(l)} + b^{(l)}``:

    dJ/dW^{(l)} = A^{(l-1)T} delta^{(l)}                                     (n_{l-1}, n_l)
    dJ/db^{(l)} = sum_i delta^{(l)}_i                                        (1, n_l)
    delta^{(l-1)} = ( delta^{(l)} W^{(l)T} ) ⊙ 1[Z^{(l-1)} > 0]              (n, n_{l-1})

The 1/n lives in ``delta^{(L)}`` and rides down the chain, so it appears exactly once. Those four
displayed expressions are what :meth:`ScratchMLP.backward` computes, line for line.

Update. Mini-batch stochastic gradient descent with a fixed step size, over shuffled batches:

    W^{(l)} <- W^{(l)} - lr * dJ/dW^{(l)}
    b^{(l)} <- b^{(l)} - lr * dJ/db^{(l)}

Initialization is **He (Kaiming) normal** — ``W^{(l)} ~ N(0, 2 / n_{l-1})``, biases zero — which
is the variance that keeps a ReLU stack's activations from collapsing or exploding with depth
(ReLU zeroes half its input, so the factor is 2 rather than 1). Unlike the convex logistic
regression, this objective is **non-convex**: the initialization matters, there are permutation
symmetries between hidden units, and a zero init would leave every hidden unit computing the same
function forever. That is why this class actually uses its ``seed``, and why the whole fit is
driven by one ``np.random.default_rng(seed)`` (init *and* batch shuffling) so a re-run is
bit-identical.

Because the objective is non-convex and the optimizer stochastic, there is no monotone-descent
guarantee to assert the way ``scratch_logreg`` does. Correctness of the derivation above is
established instead by ``tests/test_scratch_mlp.py::test_gradient_check_matches_finite_differences``,
which compares every entry of every ``dJ/dW^{(l)}`` and ``dJ/db^{(l)}`` against a central finite
difference of the weighted loss. That test is the load-bearing one: a sign error or a missing
transpose in backprop still runs, still trains to something plausible, and is invisible everywhere
else.

--------------------------------------------------------------------------------------------
Phase 7 (freeze-and-retrain) is designed for, not implemented here
--------------------------------------------------------------------------------------------

RQ2 adapts the MLP by freezing the early layer(s) and retraining the output head. Two properties
of this class exist for that and must survive refactoring: parameters are stored per layer in
:attr:`ScratchMLP.params` under ``"W1"/"b1" .. "WL"/"bL"``, so a layer can be addressed on its own;
and ``fit(..., freeze_hidden=True)`` keeps the fitted hidden layers fixed and updates only the head.
Phase 7 owns the experiment; this module only owns the capability.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from ..config import RANDOM_SEED, set_seeds

# Reused rather than re-implemented, deliberately. `_sample_weights` is the *definition* of this
# project's class-weighting convention -- including that `class_weight=None` means **balanced**,
# not unweighted (see its docstring). Importing it is what makes "identical to scratch_logreg" a
# structural fact instead of a claim two copies could drift apart on. `_sigmoid` is the same
# overflow-free tanh identity, and the two coercion helpers are the same input contract: a
# DataFrame or array-like becomes a finite float64 matrix, labels become a 0/1 int vector, and
# anything else raises rather than being guessed at. All four are pure numpy.
from .scratch_logreg import (  # noqa: F401 - BALANCED/CLASSES re-exported for callers
    BALANCED,
    CLASSES,
    ScratchLogReg,
    _sample_weights,
    _sigmoid,
)

_as_matrix = ScratchLogReg._as_matrix
_as_labels = ScratchLogReg._as_labels

#: Feature width the Phase 3 ``Preprocessor`` emits: 7 numeric (flow_duration, src_bytes,
#: dst_bytes, src_pkts, dst_pkts, bytes_per_sec, pkts_per_sec) + 14 one-hot (protocol x4,
#: service x6, conn_state x4) + 1 passthrough flag (zero_duration). Asserted at the entry point
#: rather than assumed, because ``layer_sizes[0]`` is dimensioned on it: a Phase 6 ablation that
#: drops ``proto`` changes this number, and a net silently built against the wrong input width
#: would fail as a shape error three calls away from the cause.
INPUT_DIM: int = 22

#: Locked hyperparameters for the real UNSW data, chosen on the **val** fold by
#: :func:`tune`. Same convention as ``baselines.TUNED_PARAMS`` and
#: ``scratch_logreg.TUNED_PARAMS``: the numbers live next to the model so Phases 6 and 7
#: re-instantiate an identical estimator with no search of their own.
#:
#: WHY THESE VALUES — all measured on the UNSW train/val folds, 2026-08-03, at d=22 z-scored
#: features and n=140,272 train rows. With these the fit takes **~13 s**, the training loss falls
#: 0.2841 -> 0.1609, and val F1 / ROC-AUC land at **0.9351 / 0.9832**.
#:
#: * ``layer_sizes=(22, 44, 22, 1)`` — two hidden layers, 2x then 1x the input width. Chosen by
#:   :func:`tune` over :data:`TUNING_GRID` (widths {11, 22, 44} at depth 1, plus two
#:   two-hidden-layer variants), scored by the **mean of val F1 and val ROC-AUC** (see
#:   :func:`selection_score`). Selecting on F1 alone is the mistake Phase 4 already made once and
#:   documented in ``deviations.md`` §3.6 -- it bought 0.6 F1 points at a cost of 4.4 ROC-AUC
#:   points on the decision tree -- so the same joint rule applies here. Measured composites:
#:   ``(22,11,1)`` 0.9484, ``(22,22,1)`` 0.9554, ``(22,22,11,1)`` 0.9461, ``(22,44,1)`` 0.9512,
#:   ``(22,44,22,1)`` **0.9592**. The winner clears the runner-up by 0.0038, which is outside
#:   :data:`SELECTION_TOLERANCE`, so the deeper net is a real win rather than a near-tie resolved
#:   the wrong way -- the second hidden layer buys 0.6 ROC-AUC points over the best single-layer
#:   net. Note this is *deeper* than the ``(22, 44, 1)`` the plan named as the starting point; the
#:   plan's tuning surface for this model is explicitly "MLP width **and depth**", the grid was run,
#:   and the depth-2 candidate won it. Full candidate table:
#:   ``python -m src.models.scratch_mlp --tune``.
#: * ``lr=0.05`` / ``batch_size=256`` / ``n_epochs=40`` — the schedule, held fixed across the grid
#:   so the comparison is width-and-depth only (the surface the proposal promises to tune for this
#:   model). Each was measured once, on the two strongest architectures, before being frozen:
#:   ``lr`` 0.02 / 0.05 / 0.1 give composites 0.9599 / 0.9592 / 0.9588 on the winner, i.e. the
#:   choice is worth ~0.1 points either way and 0.05 sits in the flat middle of that range.
#:   Doubling to ``n_epochs=80`` *lowers* the composite (0.9592 -> 0.9549: ROC-AUC still improves
#:   to 0.9846 but F1 at the 0.5 threshold falls 1.0 point as the net sharpens), so 40 is where
#:   the schedule stops paying rather than an arbitrary cap. A step size this large is usable only
#:   because the Phase 3 features are z-scored or 0/1 indicators; on raw byte counts spanning
#:   28 .. 1.3e7 no single step size works for every coordinate.
#: * ``class_weight="balanced"`` — stated explicitly even though it is this class's default, so a
#:   reader of Phase 6 never has to infer it. On the train fold (31.94% normal) the derived weights
#:   are ``{0: 1.5655, 1: 0.7346}``. It costs 1.5 F1 points against the unweighted control
#:   (0.9351 vs 0.9499) and buys 1.3 points of **balanced accuracy** (0.9204 vs 0.9071) plus 3.9
#:   points of precision -- which is the trade the project's metric choices exist to make, and the
#:   direction matters more cross-era, where the normal class is scarcer still.
TUNED_PARAMS: dict[str, Any] = {
    "layer_sizes": (INPUT_DIM, 44, 22, 1),
    "lr": 0.05,
    "n_epochs": 40,
    "batch_size": 256,
    "class_weight": BALANCED,
}

#: The width/depth grid :func:`tune` searched. Ordered **least-capacity first**, because
#: :data:`SELECTION_TOLERANCE` resolves near-ties by taking the first qualifying candidate. Do not
#: reorder. Widths are the {0.5x, 1x, 2x} multiples of the d=22 input the plan calls for; the two
#: deeper entries test whether a second hidden layer pays for itself at all.
TUNING_GRID: tuple[tuple[int, ...], ...] = (
    (INPUT_DIM, 11, 1),
    (INPUT_DIM, 22, 1),
    (INPUT_DIM, 22, 11, 1),
    (INPUT_DIM, 44, 1),
    (INPUT_DIM, 44, 22, 1),
)

#: Near-tie tolerance on the composite score, mirroring ``baselines.SELECTION_TOLERANCE`` (same
#: value, same argument): differences under 0.2 points are not signal on a val fold whose rows are
#: ~52% duplicate feature vectors, and when they are not signal the lower-capacity model is the
#: better pick -- fewer era-specific artifacts memorized, which is the failure mode RQ1 is about.
#: Defined here rather than imported because ``models.baselines`` pulls in sklearn and pandas at
#: import time, and this module's import graph is deliberately numpy-only.
SELECTION_TOLERANCE: float = 0.002


def selection_score(scores: dict[str, Any]) -> float:
    """Mean of F1 and ROC-AUC — the proposal's headline pair, weighted equally.

    Identical to ``baselines.selection_score``; see :data:`SELECTION_TOLERANCE` for why it is
    duplicated rather than imported.
    """
    return (float(scores["f1"]) + float(scores["roc_auc"])) / 2.0


def _relu(z: np.ndarray) -> np.ndarray:
    """``max(z, 0)``, elementwise."""
    return np.maximum(z, 0.0)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    """``1[z > 0]`` — the derivative of :func:`_relu`, with the kink at 0 resolved to 0.

    ReLU is not differentiable at exactly 0; any value in [0, 1] is a valid subgradient there and
    0 is the conventional choice. It matters for exactly one thing in this repo: a numerical
    gradient check straddling a kink would disagree with the analytic gradient for a real reason,
    so ``tests/test_scratch_mlp.py`` asserts its fixture keeps every pre-activation clear of 0
    before comparing.
    """
    return (z > 0.0).astype("float64")


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
    class_weight : dict[int, float] | str | None
        Appended to the stub's signature (keyword-compatible, last), because the class-weighted
        loss is a project constraint rather than an option and it has to be configurable to be
        *testable* -- the unweighted control arm in
        ``tests/test_scratch_mlp.py::test_class_weighting_recovers_minority_recall`` is spelled
        ``{0: 1.0, 1: 1.0}``. ``None`` (the default) and ``"balanced"`` both derive weights
        inversely proportional to class frequency from the training labels.

        **``None`` means balanced here, not unweighted** — the same deliberate inversion of
        sklearn's default that ``scratch_logreg`` makes, using literally the same
        :func:`~src.models.scratch_logreg._sample_weights` function. The two from-scratch models
        must not mean opposite things by the same argument, and defaulting to the safe thing means
        forgetting the argument cannot silently produce an all-normal net.
    """

    def __init__(
        self,
        layer_sizes: tuple[int, ...],
        lr: float = 0.01,
        n_epochs: int = 50,
        batch_size: int = 256,
        seed: int = RANDOM_SEED,
        class_weight: dict[int, float] | str | None = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError(
                f"layer_sizes={layer_sizes} needs at least an input and an output width"
            )
        if any(int(size) < 1 for size in layer_sizes):
            raise ValueError(f"layer_sizes={layer_sizes} holds a non-positive width")
        if int(layer_sizes[-1]) != 1:
            raise ValueError(
                f"layer_sizes={layer_sizes} ends in {layer_sizes[-1]} units; this is a binary "
                "classifier with a single sigmoid output. A 2-unit softmax is the same model "
                "(softmax([z0, z1])_1 == sigma(z1 - z0)) -- use one unit."
            )
        if int(batch_size) < 1:
            raise ValueError(f"batch_size={batch_size} must be >= 1")

        self.layer_sizes = tuple(int(size) for size in layer_sizes)
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = int(batch_size)
        self.seed = seed
        self.class_weight = class_weight

        #: Per-layer parameters, ``"W1"/"b1" .. "WL"/"bL"``. Addressed individually so Phase 7 can
        #: freeze a layer; ``W{l}`` has shape ``(layer_sizes[l-1], layer_sizes[l])`` and ``b{l}``
        #: shape ``(1, layer_sizes[l])``. Empty until :meth:`fit` (or :meth:`initialize_params`).
        self.params: dict[str, Any] = {}

        #: sklearn-compatible; read by ``evaluate.positive_scores`` to locate the attack column.
        self.classes_: np.ndarray = np.array(CLASSES, dtype=int)
        #: Full-training-set weighted loss after each epoch. Phase 9 can plot it for free.
        self.loss_history_: list[float] = []
        #: Epochs actually run.
        self.n_iter_: int = 0
        #: The resolved ``{class: weight}`` mapping, derived once from the *full* training labels
        #: in :meth:`fit`. Never re-derived per batch -- a batch's local class balance is a
        #: sampling artifact and weighting by it would make the objective depend on ``batch_size``.
        self.class_weight_: dict[int, float] = {}

    # --- Layer bookkeeping ----------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        """Number of *weight* layers L (one fewer than ``len(layer_sizes)``)."""
        return len(self.layer_sizes) - 1

    def initialize_params(self) -> dict[str, Any]:
        """He-normal weights, zero biases, from ``default_rng(seed)``. Replaces :attr:`params`.

        ``W^{(l)} ~ N(0, 2 / n_{l-1})``. The factor 2 is ReLU's: the unit zeroes half its input, so
        preserving activation variance across a layer needs twice the weight variance a linear
        stack would. Biases start at zero -- there is no symmetry for them to break, since the
        weights already broke it.
        """
        rng = np.random.default_rng(self.seed)
        params: dict[str, Any] = {}
        for layer in range(1, self.n_layers + 1):
            fan_in, fan_out = self.layer_sizes[layer - 1], self.layer_sizes[layer]
            params[f"W{layer}"] = rng.normal(
                loc=0.0, scale=np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)
            )
            params[f"b{layer}"] = np.zeros((1, fan_out), dtype="float64")
        self.params = params
        return params

    def head_keys(self) -> tuple[str, str]:
        """``("W{L}", "b{L}")`` — the output head, the only layer a frozen fit updates."""
        return f"W{self.n_layers}", f"b{self.n_layers}"

    def _check_fitted(self) -> None:
        if not self.params:
            raise RuntimeError(
                f"{type(self).__name__} used before fit -- no parameters exist yet"
            )

    # --- Class weights --------------------------------------------------------------------

    def _resolve_class_weights(self, y: np.ndarray) -> dict[int, float]:
        """Derive and store ``{class: weight}`` from the **full training labels**, once per fit.

        Delegates to :func:`~src.models.scratch_logreg._sample_weights`, so the mapping this model
        uses is by construction the same one the from-scratch logistic regression uses -- including
        ``None`` meaning *balanced*.
        """
        row_weights = _sample_weights(y, self.class_weight)
        self.class_weight_ = {
            int(label): float(row_weights[y == label][0])
            for label in CLASSES
            if (y == label).any()
        }
        return self.class_weight_

    def _row_weights(self, y: np.ndarray) -> np.ndarray:
        """Map the resolved per-class weights onto a batch's labels, as an ``(n, 1)`` column."""
        if not self.class_weight_:
            raise RuntimeError(
                "class weights have not been resolved; call _resolve_class_weights(y) with the "
                "full training labels before computing a loss or a gradient"
            )
        weights = np.empty((y.shape[0], 1), dtype="float64")
        for label, weight in self.class_weight_.items():
            weights[y == label, 0] = weight
        return weights

    # --- Forward / loss / backward --------------------------------------------------------

    def forward(self, X: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Forward pass; cache activations for backprop.

        Returns ``(p, cache)`` where ``p`` is the ``(n, 1)`` attack probability and ``cache`` holds
        every ``A^{(l)}`` and ``Z^{(l)}`` :meth:`backward` needs. The stub types the return as
        ``Any``; a 2-tuple is chosen over returning the bare cache so that a caller who only wants
        the prediction reads naturally (``p, _ = model.forward(X)``) while ``backward(y, cache)``
        still gets everything in one object, exactly as its signature requires.

        ``cache["Z"][-1]`` (the output pre-activation) is kept because :meth:`_loss` needs the
        logit, not the probability, to stay numerically stable -- see there.
        """
        self._check_fitted()
        matrix = _as_matrix(X)
        if matrix.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"input has {matrix.shape[1]} features but the net was built for "
                f"{self.layer_sizes[0]} (layer_sizes={self.layer_sizes})"
            )

        activations: list[np.ndarray] = [matrix]
        pre_activations: list[np.ndarray] = []
        for layer in range(1, self.n_layers + 1):
            z = activations[-1] @ self.params[f"W{layer}"] + self.params[f"b{layer}"]
            pre_activations.append(z)
            activations.append(_sigmoid(z) if layer == self.n_layers else _relu(z))

        return activations[-1], {"A": activations, "Z": pre_activations}

    def _loss(self, z_out: np.ndarray, y: np.ndarray, c: np.ndarray) -> float:
        """Class-weighted mean cross-entropy, computed from the **logit** for stability.

        Per row the cross-entropy is rewritten ``log(1 + e^z) - y*z`` (identical to
        ``-[y log p + (1-y) log(1-p)]`` under ``p = sigma(z)``) and evaluated with
        ``np.logaddexp(0, z)``, so nothing that could be a floating-point 0.0 is ever handed to
        ``log``. Computing it as written returns ``inf`` the moment a confident row saturates to
        ``p = 1.0``, which on the toy separable set happens within a few epochs.
        """
        targets = y.reshape(-1, 1).astype("float64")
        return float(np.mean(c * (np.logaddexp(0.0, z_out) - targets * z_out)))

    def backward(self, y_true: Any, cache: Any) -> dict[str, Any]:
        """Backpropagate loss, return parameter gradients.

        Returns ``{"W1": dJ/dW1, "b1": dJ/db1, ...}``, matching :attr:`params` key for key. This is
        the module docstring's four displayed expressions, in order:

        * ``delta^{(L)} = (1/n) c ⊙ (p - y)`` — the sigmoid/cross-entropy collapse, class-weighted.
          The weight enters *here*, at the output error, so it multiplies every gradient below.
        * ``dJ/dW^{(l)} = A^{(l-1)T} delta^{(l)}``
        * ``dJ/db^{(l)} = column sums of delta^{(l)}``
        * ``delta^{(l-1)} = (delta^{(l)} W^{(l)T}) ⊙ 1[Z^{(l-1)} > 0]``

        Gradients are returned for **every** layer even when :meth:`fit` was called with
        ``freeze_hidden=True``; freezing is applied at the update step, not by truncating the chain.
        Keeping the full chain here is what lets the gradient check verify the hidden layers at all.
        """
        activations: list[np.ndarray] = cache["A"]
        pre_activations: list[np.ndarray] = cache["Z"]
        p = activations[-1]

        labels = np.asarray(y_true).reshape(-1, 1).astype("float64")
        if labels.shape[0] != p.shape[0]:
            raise ValueError(
                f"cache holds {p.shape[0]} rows but y_true has {labels.shape[0]}"
            )
        c = self._row_weights(labels.ravel().astype("int64"))

        n = p.shape[0]
        delta = c * (p - labels) / n

        grads: dict[str, Any] = {}
        for layer in range(self.n_layers, 0, -1):
            grads[f"W{layer}"] = activations[layer - 1].T @ delta
            grads[f"b{layer}"] = delta.sum(axis=0, keepdims=True)
            if layer > 1:
                delta = (delta @ self.params[f"W{layer}"].T) * _relu_grad(
                    pre_activations[layer - 2]
                )
        return grads

    # --- Fit / predict --------------------------------------------------------------------

    def fit(self, X: Any, y: Any, freeze_hidden: bool = False) -> "ScratchMLP":
        """Mini-batch training loop.

        ``freeze_hidden=True`` retrains only the classifier head — used by RQ2 transfer (Phase 7).

        Shuffled mini-batch SGD for :attr:`n_epochs` passes at :attr:`lr`. One
        ``np.random.default_rng(seed)`` drives both the He init and every epoch's permutation, so
        two fits with the same seed are bit-identical -- which matters here in a way it does not for
        the convex ``scratch_logreg``: this objective is non-convex and the run genuinely depends on
        where it started.

        ``freeze_hidden=False`` (the default) **re-initializes** the parameters, so a re-fit is a
        fresh fit rather than a continuation -- sklearn's semantics, and the behavior every caller
        in Phases 5 and 6 wants.

        ``freeze_hidden=True`` requires an already-fitted model: the hidden layers keep their
        source-era weights and act as a fixed feature extractor while only ``W{L}``/``b{L}`` move.
        The head is *continued* from its fitted values rather than re-initialized, since Phase 7
        adapts on target slices as small as 1% and a fresh random head would throw away the source
        fit that is the entire point of the comparison. Class weights are always re-derived from
        the ``y`` of *this* call, because a target-era fine-tune has a different balance than the
        source fit did.

        :attr:`loss_history_` records the class-weighted loss over the **whole** training set after
        each epoch (one extra forward pass per epoch, negligible), not a running average of the
        mini-batch losses -- the latter mixes parameter values from across the epoch and is not a
        quantity any assertion can be written against.
        """
        matrix = _as_matrix(X)
        labels = _as_labels(y, matrix.shape[0])
        self._resolve_class_weights(labels)
        row_weights = self._row_weights(labels)

        if freeze_hidden:
            self._check_fitted()
            if self.params["W1"].shape[0] != matrix.shape[1]:
                raise ValueError(
                    f"freeze_hidden with a net built for {self.params['W1'].shape[0]} features "
                    f"against a {matrix.shape[1]}-column matrix; the schema changed under the model"
                )
        else:
            self.initialize_params()

        head_w, head_b = self.head_keys()
        trainable = (head_w, head_b) if freeze_hidden else tuple(self.params)

        rng = np.random.default_rng(self.seed)
        n = matrix.shape[0]
        self.loss_history_ = []
        self.n_iter_ = 0

        for _ in range(int(self.n_epochs)):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                rows = order[start : start + self.batch_size]
                _, cache = self.forward(matrix[rows])
                grads = self.backward(labels[rows], cache)
                for key in trainable:
                    self.params[key] = self.params[key] - self.lr * grads[key]

            _, cache = self.forward(matrix)
            self.loss_history_.append(self._loss(cache["Z"][-1], labels, row_weights))
            self.n_iter_ += 1

        return self

    def predict_proba(self, X: Any) -> Any:
        """Class probabilities as an ``(n, 2)`` array, columns ordered by :data:`CLASSES`.

        Two columns, sklearn-style, rather than the bare positive-class vector: that is the shape
        ``evaluate.positive_scores`` indexes with ``classes_.index(POSITIVE_LABEL)``, and column 1
        (attack) is the continuous score ROC-AUC is computed from.
        """
        positive = self.forward(X)[0].ravel()
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: Any, threshold: float = 0.5) -> Any:
        """Hard 0/1 labels: attack (1) where ``P[attack] >= threshold``.

        The default 0.5 is left alone on purpose, for the same reason as in ``scratch_logreg``:
        with the class-weighted loss the imbalance is handled *in the fit*, so the threshold has no
        correction left to make, and tuning it here would double-count the correction and make the
        Phase 6 delta un-attributable.
        """
        return (self.forward(X)[0].ravel() >= threshold).astype(int)


def make_scratch_mlp(**params: Any) -> ScratchMLP:
    """The locked from-scratch MLP: :data:`TUNED_PARAMS`, overridable.

    Mirrors ``models.baselines`` and ``scratch_logreg``'s factory convention so Phases 6 and 7 can
    instantiate the exact estimator this phase measured without repeating its hyperparameters --
    calling it bare gives the locked model, and ``**params`` is what a tuning or ablation run
    overrides.
    """
    return ScratchMLP(**{**TUNED_PARAMS, "seed": RANDOM_SEED, **params})


# --- Entry point ---------------------------------------------------------------------------
# Deliberately does NOT write to reports/metrics.csv, and is deliberately not wired into run.sh.
# Same argument as scratch_logreg's entry point: the scratch models' logged rows belong to Phase
# 6's regime run, whose `run_id` convention ("one run_id per experimental condition", see
# evaluate.log_metrics) owns both halves of the in-distribution/cross-era pair. Logging an
# in-distribution row from here would either invent a fifth run_id or collide with Phase 6's, and
# an upsert makes a collision silent. So this prints and returns; Phase 6 logs.


def _load_folds() -> dict[str, tuple[Any, Any]]:
    """The Phase 5a loader, plus the input-width assertion this model is dimensioned on.

    ``scratch_logreg.load_in_distribution_folds()`` is reused rather than duplicated: it is the
    function that encodes Phase 5's leakage contract (Preprocessor loaded and never refit, fold
    boundary reproduced from the seed, **UNSW-test and TON_IoT never opened**). A parallel loader
    here would be a second place for that contract to drift.
    """
    from .scratch_logreg import load_in_distribution_folds  # noqa: PLC0415 - keeps pandas out

    folds = load_in_distribution_folds()
    width = folds["train"][0].shape[1]
    if width != INPUT_DIM:
        raise RuntimeError(
            f"the Phase 3 Preprocessor emitted {width} features, not the expected {INPUT_DIM}. "
            "The MLP's input layer is dimensioned on that number, so this is stopped rather than "
            "guessed at -- re-derive INPUT_DIM (and layer_sizes) from the live artifact before "
            "training anything."
        )
    return folds


def tune(folds: dict[str, tuple[Any, Any]]) -> tuple[int, ...]:
    """Grid-search width/depth on the **val** fold and return the winner.

    Fit on ``train``, score on ``val``, select by :func:`selection_score` (the mean of F1 and
    ROC-AUC) with :data:`SELECTION_TOLERANCE` resolving near-ties toward the smaller net. Neither
    UNSW-test nor TON_IoT is opened.

    Prints a full candidate table to stdout and writes nothing to ``reports/metrics.csv``.
    """
    from ..evaluate import evaluate  # noqa: PLC0415 - keeps sklearn out of the import graph

    X_train, y_train = folds["train"]
    X_val, y_val = folds["val"]
    schedule = {key: TUNED_PARAMS[key] for key in ("lr", "n_epochs", "batch_size", "class_weight")}

    print(
        f"width/depth grid, fixed schedule "
        f"{', '.join(f'{k}={v}' for k, v in schedule.items())}"
    )
    graded: list[tuple[tuple[int, ...], dict[str, Any], float]] = []
    for layer_sizes in TUNING_GRID:
        started = time.perf_counter()
        model = ScratchMLP(layer_sizes=layer_sizes, seed=RANDOM_SEED, **schedule)
        model.fit(X_train, y_train)
        scores = evaluate(model, X_val, y_val)
        elapsed = time.perf_counter() - started
        composite = selection_score(scores)
        graded.append((layer_sizes, scores, composite))
        print(
            f"    {str(layer_sizes):<22} val f1={scores['f1']:.6f}  "
            f"roc_auc={scores['roc_auc']:.6f}  mean={composite:.6f}  ({elapsed:5.1f}s)"
        )

    best = max(composite for _, _, composite in graded)
    layer_sizes, scores, composite = next(
        entry for entry in graded if entry[2] >= best - SELECTION_TOLERANCE
    )
    margin = "the grid best" if composite >= best else f"{best - composite:.6f} below best"
    print(
        f"  -> chosen: {layer_sizes}  (val f1={scores['f1']:.6f}, "
        f"roc_auc={scores['roc_auc']:.6f}, mean={composite:.6f} -- {margin})"
    )
    marker = (
        ""
        if tuple(layer_sizes) == tuple(TUNED_PARAMS["layer_sizes"])
        else "   <-- DIFFERS FROM TUNED_PARAMS"
    )
    print(f"\nlocked layer_sizes -- copy into TUNED_PARAMS if it differs:\n    {layer_sizes}{marker}")
    return layer_sizes


def fit_in_distribution() -> tuple[ScratchMLP, dict[str, Any]]:
    """Fit the locked MLP on the UNSW **train** fold and score it on the **val** fold.

    Returns ``(model, val_metrics)``. **UNSW-test is not opened here, and neither is TON_IoT** --
    see ``scratch_logreg.load_in_distribution_folds``.
    """
    from ..evaluate import evaluate  # noqa: PLC0415 - keeps sklearn out of the import graph

    folds = _load_folds()
    X_train, y_train = folds["train"]
    X_val, y_val = folds["val"]

    started = time.perf_counter()
    model = make_scratch_mlp().fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    per_class = {label: round(weight, 4) for label, weight in model.class_weight_.items()}
    n_params = sum(int(np.asarray(value).size) for value in model.params.values())

    print(
        f"train fold n={len(X_train):,}  val fold n={len(X_val):,}  "
        f"features={X_train.shape[1]}\n"
        f"architecture {model.layer_sizes}  ({n_params:,} parameters, ReLU hidden, sigmoid out)\n"
        f"class weights (balanced, from the train labels): {per_class}\n"
        f"mini-batch SGD: lr={model.lr}  batch_size={model.batch_size}  "
        f"epochs={model.n_epochs}\n"
        f"    trained in {elapsed:.1f}s\n"
        f"    loss {model.loss_history_[0]:.6f} -> {model.loss_history_[-1]:.6f}"
    )

    metrics = evaluate(model, X_val, y_val)
    print(
        f"UNSW val   f1={metrics['f1']:.4f}  roc_auc={metrics['roc_auc']:.4f}  "
        f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}\n"
        f"           accuracy={metrics['accuracy']:.4f}  "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}  "
        f"macro_f1={metrics['macro_f1']:.4f}\n"
        f"           confusion [[tn, fp], [fn, tp]] = {metrics['confusion_matrix']}"
    )
    return model, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.models.scratch_mlp",
        description=(
            "Phase 5b: fit the from-scratch MLP on the UNSW train fold and report its val-fold "
            "scores. Nothing is logged to reports/metrics.csv (Phase 6 owns the logged rows) and "
            "neither UNSW-test nor TON_IoT is opened."
        ),
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help=(
            "re-run the val-fold width/depth grid and print the winner (stdout only -- nothing "
            "is logged to reports/metrics.csv)"
        ),
    )
    args = parser.parse_args(argv)

    set_seeds()

    if args.tune:
        tune(_load_folds())
        return 0

    fit_in_distribution()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
