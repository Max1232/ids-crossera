"""From-scratch logistic regression in numpy (Phase 5 — lower-risk fallback).

Clean math, guaranteed to converge on this imbalanced tabular data. Either this or the
from-scratch MLP satisfies the from-scratch requirement. Supports class weighting for
imbalance and warm-start for RQ2 transfer.

**Pure numpy, by requirement.** Nothing in this module imports sklearn, scipy, torch or any
autograd: the forward pass, the loss, the gradient and the optimizer are all written out below.
The only sklearn that appears anywhere near this model is (a) ``evaluate.evaluate()``, which
*scores* its predictions, and (b) ``tests/test_scratch_logreg.py``, which compares it against
``LogisticRegression`` as a reference. Neither is part of the model.

--------------------------------------------------------------------------------------------
The math (this is the derivation the report's Methods section typesets — rubric-bearing)
--------------------------------------------------------------------------------------------

Model. For a row :math:`x_i \\in \\mathbb{R}^d` with weights :math:`w` and bias :math:`b`:

    z_i = w^T x_i + b                          (the "margin" / logit)
    p_i = sigma(z_i) = 1 / (1 + e^{-z_i})       (P[y_i = 1 | x_i], i.e. P[attack])

Loss. Class-weighted binary cross-entropy with optional L2, averaged over the n rows:

    L(w, b) = -(1/n) * sum_i c_{y_i} [ y_i log p_i + (1 - y_i) log(1 - p_i) ]
              + (lambda / 2) * ||w||^2

where :math:`c_{y_i}` is the weight of row i's *class* (see ``_sample_weights``) and
:math:`\\lambda` is ``l2``. The bias is deliberately **not** regularized — penalizing it would
shrink the model's ability to represent the base rate, which is exactly the thing class
weighting is here to re-balance.

Gradient. Two facts do all the work:

  1. sigma'(z) = sigma(z) (1 - sigma(z)) = p (1 - p)
  2. d/dp of -[y log p + (1-y) log(1-p)] = -y/p + (1-y)/(1-p) = (p - y) / (p (1 - p))

Chaining them, the p(1-p) cancels and the per-row derivative w.r.t. the logit collapses to the
residual — which is the whole reason logistic regression is the clean from-scratch model:

    dL_i / dz_i = c_{y_i} (p_i - y_i)

Then, since dz_i/dw = x_i and dz_i/db = 1:

    grad_w = (1/n) X^T [ c * (p - y) ] + lambda * w         # shape (d,)
    grad_b = (1/n) sum_i c_{y_i} (p_i - y_i)                # scalar

``*`` is elementwise. That is exactly what :meth:`ScratchLogReg._gradient` computes, term for
term, and the class weight ``c`` enters *inside* the sum — in the loss and in the gradient, not
as a post-hoc threshold shift. See the note on :func:`_sample_weights`.

Update. Full-batch gradient descent with a fixed step size:

    w <- w - lr * grad_w
    b <- b - lr * grad_b

The objective is convex in (w, b) (strictly so when ``l2 > 0``), so full-batch GD from a zero
initialization converges to the global optimum for any step size below 2/L, where L is the
Lipschitz constant of the gradient. That is what makes this the *lower-risk* half of Phase 5:
there is no local minimum to land in and no initialization to get unlucky with.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from ..config import RANDOM_SEED, set_seeds

#: Both classes, in the order :meth:`ScratchLogReg.predict_proba` emits its columns. Exposed as
#: ``classes_`` on a fitted model because ``evaluate.positive_scores`` reads that attribute to find
#: which proba column is the attack class, and refuses to guess if it is absent. Positive = 1 =
#: attack, matching ``evaluate.POSITIVE_LABEL`` and ``schema_map``'s harmonized label encoding.
CLASSES: tuple[int, int] = (0, 1)

#: Locked hyperparameters for the real UNSW data, chosen on the **val** fold (see ``make_scratch_
#: logreg``). Same convention as ``baselines.TUNED_PARAMS``: the numbers live next to the model so
#: Phase 6 and Phase 7 re-instantiate an identical estimator with no search of their own.
#:
#: WHY THESE VALUES — all four measured on the UNSW train/val folds, 2026-08-03, at d=22 z-scored
#: features and n=140,272 train rows. With these the fit stops on the tolerance (not the cap) after
#: **4,346 iterations in ~32 s**, the loss decreases monotonically at every one of them, and val
#: F1 / ROC-AUC land **0.19 / 0.16 points** below sklearn's ``LogisticRegression`` (lbfgs).
#:
#: * ``lr=1.0`` — the features arriving from the Phase 3 ``Preprocessor`` are z-scored (unit
#:   variance) or 0/1 indicators, so every column contributes on the same scale and the curvature
#:   of the loss is O(1); that is what makes a step this large usable at all, and it is why
#:   standardization is a *precondition* of this model rather than a nicety (on raw byte counts
#:   spanning 28 .. 1.3e7 no single step size works for every coordinate). The stability limit was
#:   measured rather than guessed: **lr=2.0 still descends monotonically, lr=2.5 diverges on the
#:   second iteration** (the loss jumps by +9e-2 and never recovers). 1.0 therefore keeps a ~2.5x
#:   margin below the observed 2/L boundary, which matters because Phase 7 warm-starts this same
#:   model on small TON_IoT slices whose curvature is not the train fold's. Going to 2.0 would buy
#:   ~0.17 F1 points at the edge of divergence -- not a trade worth making for a number reported to
#:   three decimals.
#: * ``n_epochs=5000`` — the cap, a safety net rather than the stopping rule: ``tol`` fires first,
#:   at iteration 4,346. Left ~15% above the observed stop so the schedule is not silently truncated
#:   if a Phase 6 ablation changes the feature set slightly.
#: * ``tol=1e-6`` — stop when the weighted loss improves by less than this between consecutive
#:   iterations. Matched to ``evaluate.METRIC_DECIMALS`` (6): past this point the objective is
#:   moving by less per step than the precision any reported metric is written down at. Measured
#:   trajectory of the per-iteration improvement: below 1e-5 at iteration 1,598, below 1e-6 at
#:   4,346, below 1e-7 at 7,446. Running all the way to 1e-7 costs 2.3x the iterations and recovers
#:   only 0.14 F1 / 0.13 AUC points, so 1e-6 is where the curve stops paying.
#: * ``l2=1e-4`` — mild ridge. It is not needed for generalization here (n exceeds d by four orders
#:   of magnitude) but it makes the objective *strictly* convex, which pins down a unique optimum
#:   and stops ``||w||`` drifting along the near-separable directions of the one-hot block. For
#:   scale: sklearn's ``LogisticRegression(C=1.0)`` penalizes the *sum* of losses, an equivalent
#:   lambda of ``1/(C*n)`` ~ 7e-6 against this module's mean-loss formulation, so 1e-4 is the same
#:   order of magnitude as the reference model's default -- one reason the parity gap is 0.2 points
#:   and not 2. Raising it to 1e-3 costs 2.9 F1 points, so it is not free.
#: * ``class_weight="balanced"`` — stated explicitly even though it is this class's default (see
#:   :meth:`ScratchLogReg.__init__`), so a reader of Phase 6 never has to infer it. On the train
#:   fold (31.94% normal) the derived weights are ``{0: 1.5655, 1: 0.7346}``.
TUNED_PARAMS: dict[str, Any] = {
    "lr": 1.0,
    "n_epochs": 5000,
    "tol": 1e-6,
    "l2": 1e-4,
    "class_weight": "balanced",
}

#: Accepted string spelling of :data:`class_weight`, matching sklearn's vocabulary so the scratch
#: model and the library baselines read the same way at their call sites.
BALANCED: str = "balanced"


def _sample_weights(y: np.ndarray, class_weight: Any) -> np.ndarray:
    """Per-row loss weight ``c_i`` from the *training labels*, never from a target-set statistic.

    Three accepted forms:

    * ``None`` or ``"balanced"`` -> inversely proportional to class frequency,
      ``c_k = n / (n_classes * n_k)``, which is sklearn's ``class_weight="balanced"`` formula.
      The two class weights then average to exactly 1.0 over the training set, so the *scale* of
      the loss (and therefore the usable range of ``lr``) is unchanged by weighting — only its
      balance is. Under UNSW-train's ~55/45 attack/normal split the weights are mild; under
      TON_IoT's 76/24 (Phase 6/7) they are not, which is the point.
    * an explicit ``{0: w0, 1: w1}`` mapping -> those weights verbatim. ``{0: 1.0, 1: 1.0}`` is
      how the unweighted variant is spelled, and it is what
      ``tests/test_scratch_logreg.py::test_class_weighting_recovers_minority_recall`` fits as its
      control arm.

    **``None`` means balanced here, not unweighted** — a deliberate departure from sklearn's
    default. The project constraint is class weights on every model *from the start* (README,
    Non-negotiable constraints), and the failure mode it guards against is silent: an unweighted
    fit under this imbalance drifts toward the majority class and posts a deceptively high
    accuracy. Defaulting to the safe thing means forgetting the argument cannot produce that
    failure; asking for it explicitly is the only way to get it.

    The weighting is applied **inside** the loss and gradient (as the factor ``c`` in the sums
    derived in the module docstring), not by moving the decision threshold afterwards. Those are
    not equivalent: re-weighting changes which optimum gradient descent walks to and therefore the
    *ranking* the ROC curve is drawn from, while a post-hoc threshold shift slides along a ranking
    that was already fit to the majority class.
    """
    counts = np.array([float((y == label).sum()) for label in CLASSES])

    if class_weight is None or class_weight == BALANCED:
        if (counts == 0).any():
            missing = [label for label, count in zip(CLASSES, counts) if count == 0]
            raise ValueError(
                f"cannot derive balanced class weights: class(es) {missing} have no rows in y. "
                "A single-class training fold is a data-wiring bug, not a fit."
            )
        weights = {
            label: float(y.shape[0]) / (len(CLASSES) * count)
            for label, count in zip(CLASSES, counts)
        }
    elif isinstance(class_weight, dict):
        missing = [label for label in CLASSES if label not in class_weight]
        if missing:
            raise ValueError(
                f"class_weight mapping {class_weight} is missing class(es) {missing}; give a "
                f"weight for every label in {list(CLASSES)} or pass None/'balanced'."
            )
        weights = {label: float(class_weight[label]) for label in CLASSES}
    else:
        raise ValueError(
            f"class_weight must be None, {BALANCED!r}, or a {{class: weight}} mapping; "
            f"got {class_weight!r}"
        )

    out = np.empty(y.shape[0], dtype="float64")
    for label, weight in weights.items():
        out[y == label] = weight
    return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable ``sigma(z) = 1 / (1 + exp(-z))``, via the exact ``tanh`` identity.

        1 / (1 + e^{-z}) == (1/2) (1 + tanh(z/2))

    An algebraic identity, not an approximation (divide numerator and denominator of
    ``tanh(z/2) = (e^{z/2} - e^{-z/2}) / (e^{z/2} + e^{-z/2})`` by ``e^{z/2}``), and it is written
    this way for one reason: the naive form **overflows**. ``exp(-z)`` blows up for very negative z,
    and this model's logits do reach the tens once it has converged on near-separable data, so the
    direct expression emits overflow warnings and pins probabilities to exactly 0.0 or 1.0 -- which
    then poison a log. ``np.tanh`` saturates gracefully to +-1 over the whole real line with no
    branch, no clipping and no epsilon, so the result is exact where it matters and finite
    everywhere. It is also ~3.5x faster than the equivalent piecewise-``exp`` implementation
    (one vectorized C pass instead of two masked ones), which is why the full-batch fit over
    140,272 rows costs seconds rather than a minute.
    """
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class ScratchLogReg:
    """Binary logistic regression trained by gradient descent."""

    def __init__(
        self,
        lr: float = 0.01,
        n_epochs: int = 200,
        l2: float = 0.0,
        class_weight: dict[int, float] | str | None = None,
        seed: int = RANDOM_SEED,
        tol: float = 1e-6,
    ) -> None:
        """Full-batch gradient descent, class-weighted from the start.

        Parameters
        ----------
        lr:
            Fixed step size. Safe up to ~2/L on this convex objective; the z-scored feature space
            makes L an O(1) quantity. :data:`TUNED_PARAMS` locks 1.0 for the real folds — the
            0.01 default here is a deliberately conservative value that converges on anything.
        n_epochs:
            Maximum full-batch iterations. An upper bound, not the stopping rule: ``tol`` normally
            stops the loop first, and :attr:`converged_` records which one fired.
        l2:
            Ridge coefficient lambda on ``||w||^2 / 2``. The bias is never penalized.
        class_weight:
            ``None`` (the default) or ``"balanced"`` derives weights inversely proportional to
            class frequency **from the training labels**; a ``{0: w0, 1: w1}`` mapping is taken
            verbatim, so ``{0: 1.0, 1: 1.0}`` is the unweighted control. Note ``None`` means
            *balanced*, not *unweighted* -- see :func:`_sample_weights` for why the default is
            inverted relative to sklearn.
        seed:
            Kept for interface parity with the rest of ``src/models`` and with ``config``. It is
            unused *by construction*: the objective is convex, the initialization is exact zeros,
            and the optimizer is full-batch, so there is no stochasticity to seed. ``set_seeds()``
            is still called at this module's entry point, because the fold split upstream of it is
            seeded.
        tol:
            Convergence tolerance on the *improvement in the weighted loss* between consecutive
            iterations; the loop stops when it drops below this. Added to the stub's signature
            (keyword-compatible, appended last) because "converged" has to mean something checkable
            for the report -- ``n_iter_`` and ``converged_`` are what it means.
        """
        self.lr = lr
        self.n_epochs = n_epochs
        self.l2 = l2
        self.class_weight = class_weight
        self.seed = seed
        self.tol = tol
        self.weights: Any = None
        self.bias: float = 0.0
        #: sklearn-compatible; read by ``evaluate.positive_scores`` to locate the attack column.
        self.classes_: np.ndarray = np.array(CLASSES, dtype=int)
        #: Weighted loss after every iteration. The convergence unit test asserts this decreases
        #: monotonically, and Phase 9 can plot it as the training curve for free.
        self.loss_history_: list[float] = []
        self.n_iter_: int = 0
        self.converged_: bool = False

    # --- Internals ----------------------------------------------------------------------

    @staticmethod
    def _as_matrix(X: Any) -> np.ndarray:
        """Coerce a DataFrame or array-like to a float64 2-D matrix, non-finite values refused."""
        matrix = np.asarray(getattr(X, "to_numpy", lambda: X)(), dtype="float64")
        if matrix.ndim != 2:
            raise ValueError(f"expected a 2-D feature matrix, got shape {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError(
                "feature matrix holds non-finite values; the Phase 3 Preprocessor guarantees "
                "finite output, so this is a wiring bug upstream of the model"
            )
        return matrix

    @staticmethod
    def _as_labels(y: Any, n_rows: int) -> np.ndarray:
        """Coerce labels to a 0/1 int vector, refusing anything outside :data:`CLASSES`."""
        labels = np.asarray(getattr(y, "to_numpy", lambda: y)()).ravel()
        labels = labels.astype("int64")
        unexpected = sorted(set(np.unique(labels)) - set(CLASSES))
        if unexpected:
            raise ValueError(
                f"labels {unexpected} are outside {list(CLASSES)}; this is a binary "
                "attack(1)/normal(0) classifier and the harmonized frames carry that encoding"
            )
        if labels.shape[0] != n_rows:
            raise ValueError(f"X has {n_rows} rows but y has {labels.shape[0]}")
        return labels

    def _loss(self, z: np.ndarray, y: np.ndarray, c: np.ndarray) -> float:
        """Class-weighted mean cross-entropy plus the ridge term, computed in a stable form.

        The per-row cross-entropy is rewritten as ``log(1 + e^z) - y*z`` (identical to
        ``-[y log p + (1-y) log(1-p)]`` by substituting ``p = sigma(z)``) and evaluated with
        ``np.logaddexp(0, z)``, so nothing is fed to ``log`` that could be a floating-point 0.0.
        Computing it as written would return ``inf`` the moment a converged model assigns a
        confident row ``p = 1.0``, and the convergence test's monotone-decrease assertion would be
        comparing infinities.
        """
        cross_entropy = float(np.mean(c * (np.logaddexp(0.0, z) - y * z)))
        ridge = 0.5 * self.l2 * float(np.dot(self.weights, self.weights))
        return cross_entropy + ridge

    def _gradient(
        self, X: np.ndarray, y: np.ndarray, c: np.ndarray, p: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """``(grad_w, grad_b)`` — the two expressions derived in the module docstring, literally.

        ``residual = c * (p - y)`` is ``dL/dz`` per row (up to the 1/n), which is where the class
        weight enters the gradient. Everything else is the chain rule against ``dz/dw = x`` and
        ``dz/db = 1``.
        """
        n = X.shape[0]
        residual = c * (p - y)
        grad_w = (X.T @ residual) / n + self.l2 * self.weights
        grad_b = float(residual.sum() / n)
        return grad_w, grad_b

    # --- Fit / predict ------------------------------------------------------------------

    def fit(self, X: Any, y: Any, warm_start: bool = False) -> "ScratchLogReg":
        """Gradient-descent fit. ``warm_start=True`` keeps existing weights (RQ2 transfer).

        Runs full-batch gradient descent for at most :attr:`n_epochs` iterations, stopping early
        when the weighted loss improves by less than :attr:`tol`. Records
        :attr:`loss_history_`, :attr:`n_iter_` and :attr:`converged_`.

        Initialization is **exact zeros** (``w = 0``, ``b = 0``), not a random draw: the objective
        is convex, so there is no symmetry to break and nothing to gain from randomness, and zeros
        make the whole fit bit-reproducible without depending on RNG state at all. ``p_i = 0.5``
        everywhere on the first iteration, which is the correct uninformed prior.

        ``warm_start=True`` keeps whatever ``weights``/``bias`` are already fitted and continues
        descending from them -- which is how Phase 7 adapts a UNSW-trained model on a small slice
        of TON_IoT instead of starting over. The class weights are always re-derived from the ``y``
        passed to *this* call, since a warm-started fit on target data has a different balance than
        the source fit did.
        """
        matrix = self._as_matrix(X)
        labels = self._as_labels(y, matrix.shape[0])
        c = _sample_weights(labels, self.class_weight)

        if warm_start and self.weights is not None:
            if self.weights.shape[0] != matrix.shape[1]:
                raise ValueError(
                    f"warm_start with {self.weights.shape[0]} fitted weights against a "
                    f"{matrix.shape[1]}-column matrix; the feature schema changed under the model"
                )
            weights = np.asarray(self.weights, dtype="float64").copy()
            bias = float(self.bias)
        else:
            weights = np.zeros(matrix.shape[1], dtype="float64")
            bias = 0.0

        self.weights, self.bias = weights, bias
        self.loss_history_ = []
        self.converged_ = False
        previous = np.inf

        for iteration in range(1, self.n_epochs + 1):
            z = matrix @ self.weights + self.bias
            p = _sigmoid(z)
            loss = self._loss(z, labels, c)
            self.loss_history_.append(loss)

            grad_w, grad_b = self._gradient(matrix, labels, c, p)
            self.weights = self.weights - self.lr * grad_w
            self.bias = self.bias - self.lr * grad_b

            self.n_iter_ = iteration
            improvement = previous - loss
            if improvement < -self.tol:
                # The loss went UP. On a convex, smooth objective that means exactly one thing:
                # the step size overshot (lr > 2/L), so gradient descent is oscillating outward.
                # This is raised rather than tolerated because the naive stopping rule below --
                # "stop when the improvement is small" -- reads a *negative* improvement as
                # convergence and would return a diverged model flagged converged_=True. Measured
                # on the real train fold (2026-08-03): lr=2.0 descends monotonically, lr=2.5
                # diverges on the second iteration, which is why TUNED_PARAMS locks 1.0.
                raise RuntimeError(
                    f"gradient descent diverged at iteration {iteration}: weighted loss rose "
                    f"{-improvement:.3e} (from {previous:.6f} to {loss:.6f}). lr={self.lr} is "
                    "above the 2/L stability limit for this feature matrix -- lower it. (Are the "
                    "features standardized? This model assumes the Phase 3 Preprocessor's "
                    "z-scored output.)"
                )
            if improvement < self.tol:
                # The step that produced this loss was smaller than the tolerance, so the
                # parameters have stopped moving in any way a reported metric could see.
                self.converged_ = True
                break
            previous = loss

        return self

    def decision_function(self, X: Any) -> Any:
        """Signed margin ``z = Xw + b``. Positive means attack (class 1).

        A monotone transform of :meth:`predict_proba`'s attack column, so it ranks identically and
        would give the identical ROC-AUC. Provided because it is the quantity the math above is
        written in and because it costs nothing; ``evaluate.positive_scores`` prefers
        ``predict_proba`` and will not reach for this.
        """
        if self.weights is None:
            raise RuntimeError("ScratchLogReg.decision_function called before fit")
        return self._as_matrix(X) @ self.weights + self.bias

    def predict_proba(self, X: Any) -> Any:
        """Class probabilities as an ``(n, 2)`` array, columns ordered by :data:`CLASSES`.

        Two columns, sklearn-style, rather than the bare positive-class vector: that is the shape
        ``evaluate.positive_scores`` indexes with ``classes_.index(POSITIVE_LABEL)``, and column 1
        (attack) is the continuous score ROC-AUC is computed from.
        """
        positive = _sigmoid(np.asarray(self.decision_function(X), dtype="float64"))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: Any, threshold: float = 0.5) -> Any:
        """Hard 0/1 labels: attack (1) where ``P[attack] >= threshold``.

        The default 0.5 is left alone on purpose. With the class-weighted loss the imbalance is
        already handled *in the fit*, so the threshold has no correction to make -- and tuning it
        here would double-count the correction and make the Phase 6 delta un-attributable.
        """
        positive = _sigmoid(np.asarray(self.decision_function(X), dtype="float64"))
        return (positive >= threshold).astype(int)


def make_scratch_logreg(**params: Any) -> ScratchLogReg:
    """The locked from-scratch logistic regression: :data:`TUNED_PARAMS`, overridable.

    Mirrors ``models.baselines``' factory convention so Phases 6 and 7 can instantiate the exact
    estimator this phase measured without repeating its hyperparameters -- calling it bare gives
    the locked model, and ``**params`` is what a tuning or ablation run overrides.
    """
    return ScratchLogReg(**{**TUNED_PARAMS, "seed": RANDOM_SEED, **params})


# --- Entry point ---------------------------------------------------------------------------
# Deliberately does NOT write to reports/metrics.csv. The scratch models' logged rows belong to
# Phase 6's regime run, whose `run_id` convention ("one run_id per experimental condition", see
# evaluate.log_metrics) owns both halves of the in-distribution/cross-era pair. Logging an
# in-distribution row from here would either invent a fifth run_id or collide with Phase 6's, and
# an upsert makes a collision silent. So this prints and returns; Phase 6 logs.
#
# For the same reason it is not wired into run.sh yet -- the `# TODO Phase 5:` line there stays
# commented until Phase 5b (the MLP) lands and Phase 6 owns the logging.


def load_in_distribution_folds() -> dict[str, tuple[Any, Any]]:
    """Transform the UNSW **train** and **val** folds with the already-fitted Preprocessor.

    Returns ``{"train": (X, y), "val": (X, y)}``.

    Deliberately **not** ``baselines.load_folds()``, which is otherwise the same three lines: that
    function also materializes ``unsw_test``, and Phase 5a's contract is that it does not open the
    holdout at all. Phase 5a's question is "does the from-scratch model converge, and does it match
    sklearn in-distribution" -- both answerable on the val fold, which nothing trains on. UNSW-test
    is scored exactly once, by Phase 6, and TON_IoT not until then either.

    The two invariants it shares with ``baselines.load_folds()`` are the load-bearing ones:

    * **the Preprocessor is loaded, never refit** (``preprocess.fit_preprocessor()`` would refit
      it, making "was this the artifact Phase 6 used?" unanswerable);
    * **the fold boundary is reproduced, not re-drawn** -- ``split_source`` is stratified and seeded
      from ``config.RANDOM_SEED``, so it returns the same 140,272 / 35,069 partition the
      Preprocessor was fit on. That is why the seed is not a parameter here.

    Imports are function-local on purpose. ``preprocess`` pulls in pandas and sklearn
    transitively, and keeping that out of this module's import graph is what lets the toy-set unit
    tests import :class:`ScratchLogReg` on a fresh clone with no ``data/`` -- and keeps the model
    itself, top to bottom, a pure-numpy object.
    """
    from ..config import PREPROCESSOR  # noqa: PLC0415 - see docstring
    from ..preprocess import (  # noqa: PLC0415 - see docstring
        LABEL_COL,
        Preprocessor,
        load_source,
        split_source,
    )

    if not PREPROCESSOR.exists():
        raise FileNotFoundError(
            f"{PREPROCESSOR} not found -- Phase 3 has not run. Run `python -m src.preprocess` "
            "first; Phase 5 loads that artifact and must never fit its own."
        )
    preprocessor = Preprocessor.load(str(PREPROCESSOR))
    train_fold, val_fold = split_source(load_source(), seed=RANDOM_SEED)
    return {
        name: (preprocessor.transform(frame), frame[LABEL_COL].astype(int))
        for name, frame in (("train", train_fold), ("val", val_fold))
    }


def fit_in_distribution() -> tuple[ScratchLogReg, dict[str, Any]]:
    """Fit the locked model on the UNSW **train** fold and score it on the **val** fold.

    Returns ``(model, val_metrics)``. **UNSW-test is not opened here, and neither is TON_IoT** --
    see :func:`load_in_distribution_folds`.
    """
    from ..evaluate import evaluate  # noqa: PLC0415 - keeps sklearn out of the import graph

    folds = load_in_distribution_folds()
    X_train, y_train = folds["train"]
    X_val, y_val = folds["val"]

    started = time.perf_counter()
    model = make_scratch_logreg().fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    labels = np.asarray(y_train).ravel().astype(int)
    row_weights = _sample_weights(labels, TUNED_PARAMS["class_weight"])
    per_class = {
        int(label): round(float(row_weights[labels == label][0]), 4) for label in CLASSES
    }

    print(
        f"train fold n={len(X_train):,}  val fold n={len(X_val):,}  "
        f"features={X_train.shape[1]}\n"
        f"class weights (balanced, from the train labels): {per_class}\n"
        f"gradient descent: lr={model.lr}  tol={model.tol}  l2={model.l2}  "
        f"max_epochs={model.n_epochs}\n"
        f"    stopped after {model.n_iter_} iterations "
        f"(converged={model.converged_}) in {elapsed:.1f}s\n"
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
        prog="python -m src.models.scratch_logreg",
        description=(
            "Phase 5a: fit the from-scratch logistic regression on the UNSW train fold and "
            "report its val-fold scores. Nothing is logged to reports/metrics.csv (Phase 6 owns "
            "the logged rows) and neither UNSW-test nor TON_IoT is opened."
        ),
    )
    parser.parse_args(argv)

    set_seeds()
    fit_in_distribution()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
