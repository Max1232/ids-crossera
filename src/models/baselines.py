"""Library baseline models (Phase 4) — the classical axis and sanity floor.

Random Forest, linear SVM, Decision Tree (library implementations) plus a majority-class Dummy.
Class weights are used from the start to handle imbalance; hyperparameters are tuned on the
validation split *before* any cross-era run.

What this module establishes is the **"before" ceiling** for RQ1: how well a classical detector
does when the test set comes from the same era it was trained on. Phase 6 re-runs these same
estimators against TON_IoT and reports the delta, so the numbers here are one half of the headline
result and must be produced by models Phase 6 can rebuild *identically* — hence the tuned
hyperparameters are baked into the factory defaults below rather than living in a search that
would have to be re-run.

Data flow, and it is leakage-critical (see ``preprocess``):

* the ``Preprocessor`` is **loaded** from ``data/processed/preprocessor.joblib``, never refit;
* models are fit on the **UNSW train fold** (140,272 rows), the same fold the Preprocessor saw;
* hyperparameters are chosen on the **UNSW val fold** (35,069 rows), which no model trains on;
* the locked models are scored **once** on **UNSW-test** (82,332 rows), which nothing tunes on;
* **TON_IoT is not opened here at all.** That is Phase 6, and touching it early is how an
  in-distribution ceiling quietly becomes a number that saw the target.

Entry point (wired into ``run.sh``)::

    python -m src.models.baselines            # locked models -> UNSW-test -> metrics.csv
    python -m src.models.baselines --tune     # re-run the val-fold grid search, stdout only

``--tune`` prints and logs nothing: a grid search is a decision record, not a run, and dumping 17
candidate fits into ``reports/metrics.csv`` would bury the four rows that are actually the result.
The winners it prints are what ``TUNED_PARAMS`` holds.
"""

from __future__ import annotations

import argparse
import itertools
import time
from typing import Any, Callable, Iterator

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from ..config import PREPROCESSOR, RANDOM_SEED, set_seeds
from ..evaluate import evaluate, log_metrics, round_metrics
from ..preprocess import (
    LABEL_COL,
    Preprocessor,
    load_holdout,
    load_source,
    split_source,
)

# --- Tuned hyperparameters ---------------------------------------------------------------
# Chosen by `--tune` on the UNSW **val** fold and locked here so Phase 6 (and Phase 7) can
# instantiate byte-identical models straight from BASELINE_FACTORIES with no search of their own.
# Nothing here was chosen on UNSW-test.
#
# Only depth and regularization are tuned, which is exactly the surface the proposal promises
# ("we tune tree depth, regularization strength, and MLP width/depth"); MLP width/depth is
# Phase 5's. Everything else is left at the sklearn default deliberately -- an unbounded search
# over a 22-column feature space would be tuning noise, and ~52% of the train fold's rows are
# duplicate feature vectors in the shared subspace, so val scores already carry mild optimism.
#
# tuned on UNSW val fold, 2026-08-01
TUNED_PARAMS: dict[str, dict[str, Any]] = {
    "decision_tree": {"max_depth": None, "min_samples_leaf": 20},
    "random_forest": {"max_depth": 20, "n_estimators": 100},
    "svm": {"C": 0.1},
}

# The search that produced the above. Small on purpose: three grids, 8 + 6 + 3 = 17 fits.
#
# Candidates are ordered **most-constrained first** in every grid, because SELECTION_TOLERANCE
# below resolves near-ties by taking the first qualifying candidate. Do not reorder them.
TUNING_GRIDS: dict[str, dict[str, tuple[Any, ...]]] = {
    "decision_tree": {"max_depth": (6, 10, 16, None), "min_samples_leaf": (20, 1)},
    "random_forest": {"max_depth": (10, 20, None), "n_estimators": (100, 300)},
    # C is the regularization knob: small C = strong regularization. Log-spaced, three points.
    "svm": {"C": (0.01, 0.1, 1.0)},
}

# --- Selection rule ----------------------------------------------------------------------
# Score a candidate by the **mean of val F1 and val ROC-AUC**, then among everything within
# SELECTION_TOLERANCE of the best take the most constrained (first in grid order).
#
# The joint criterion is not decoration. Selecting on F1 alone picks the *fully grown, leaf-of-1*
# decision tree -- measured 2026-08-01: val F1 0.9550 against 0.9490 for `min_samples_leaf=20`, a
# 0.6-point gain, while val ROC-AUC collapses 0.9858 -> 0.9422. A saturated tree emits hard 0/1
# leaf probabilities, so its score *ranking* is coarse and AUC punishes it. Phase 6 leads the drift
# claim with ROC-AUC precisely because the two test sets do not share a class balance, so a tuning
# rule that trades 4.4 AUC points for 0.6 F1 points would sabotage the headline metric one phase
# before it is reported. F1 and ROC-AUC are the promised pair; weight them equally.
#
# The tolerance encodes the honest precision of a single val fold whose rows are ~52% duplicated
# feature vectors: differences under 0.2 points are not signal, and when they are not, the
# lower-capacity model is the better choice -- fewer era-specific artifacts memorized, which is
# exactly the failure mode RQ1 is about. It also broke two real near-ties here: Random Forest
# `max_depth=20, n_estimators=100` over `None/300` (composite 0.9726 vs 0.9730, at a third of the
# fit cost) and linear SVM `C=0.1` over `C=1.0` (0.9129 vs 0.9131).
SELECTION_TOLERANCE: float = 0.002


def selection_score(scores: dict[str, Any]) -> float:
    """Mean of F1 and ROC-AUC — the proposal's headline pair, weighted equally."""
    return (float(scores["f1"]) + float(scores["roc_auc"])) / 2.0


# --- Factories ---------------------------------------------------------------------------
# Every real model gets class_weight="balanced" and the project seed. `**params` overrides the
# tuned defaults, which is what --tune drives; calling a factory bare gives the locked model.


def make_dummy(**params: Any) -> DummyClassifier:
    """Majority-class baseline — the sanity floor.

    No class weighting and no tuning surface, by definition: this exists so "the Random Forest
    got 0.94" can be read against something. UNSW-test is ~55% attack, so a most-frequent
    classifier predicts "attack" everywhere and posts an accuracy near 0.55 with ROC-AUC 0.50 --
    which is the point. Any real model that lands near this row is broken, not modest.
    """
    return DummyClassifier(**{"strategy": "most_frequent", "random_state": RANDOM_SEED, **params})


def make_random_forest(**params: Any) -> RandomForestClassifier:
    """RandomForestClassifier with balanced class weights and the project seed."""
    return RandomForestClassifier(
        **{
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
            # Deterministic regardless of thread count -- each tree's RNG is drawn from
            # random_state before any parallel dispatch, so n_jobs buys wall-clock only.
            "n_jobs": -1,
            **TUNED_PARAMS["random_forest"],
            **params,
        }
    )


def make_svm(**params: Any) -> LinearSVC:
    """Linear SVM with balanced class weights — ``LinearSVC`` (liblinear, primal).

    Must be a *linear* SVM, not a kernel ``SVC``. UNSW-NB15's training split is 175,341 rows, so a
    kernel SVC's n x n Gram matrix is ~2.5e10 entries (hundreds of GB in float64) and libsvm's
    training cost is roughly quadratic-to-cubic in n — it does not finish at this data size.
    ``LinearSVC`` (liblinear, primal) and ``SGDClassifier(loss='hinge')`` (out-of-core,
    ``partial_fit``) both scale linearly and are the two supported options. ``LinearSVC`` is the
    one implemented: at 140,272 x 22 dense it fits in seconds, and it has no learning-rate
    schedule to tune, so the only knob is ``C`` — which is the regularization strength the
    proposal actually promises to tune.

    The features arriving here are already z-scored by the Phase 3 ``Preprocessor``, which is a
    precondition rather than a convenience: a margin-based loss on raw byte counts spanning
    28 .. 1.3e7 would be dominated by scale.

    ROC-AUC for this estimator comes from ``decision_function``, not ``predict_proba`` (which
    ``LinearSVC`` does not expose). ``evaluate.positive_scores`` handles that automatically.
    Deliberately **not** wrapped in ``CalibratedClassifierCV``: AUC scores a *ranking*, and any
    monotone calibration of the signed margin gives the identical ranking and therefore the
    identical AUC — the wrapper would cost a k-fold refit and change no reported number. Nothing
    in Phases 4–7 needs calibrated probabilities; if a later phase ever does, add the wrapper
    there and say so, rather than paying for it on every run here.

    RATIFIED 2026-08-03. Implemented 2026-08-01 and previously marked pending review; the division
    of labor that review depended on is no longer in effect, so the three open choices were checked
    directly instead and all three stand: ``LinearSVC`` over ``SGDClassifier``, ``decision_function``
    for AUC without ``CalibratedClassifierCV``, and the ``C`` grid. On the grid specifically —
    ``C=0.1`` was **not** the floor of the search. ``TUNING_GRIDS["svm"]`` spans
    ``(0.01, 0.1, 1.0)``, bracketing the winner on both sides, and ``C=0.01`` lost by 0.29 composite
    points (0.9100 vs 0.9129), far outside :data:`SELECTION_TOLERANCE`. Re-measured 2026-08-03 over
    a widened ``(0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 10.0)``: the composite rises monotonically to a
    plateau at ``C >= 0.05`` (0.9125, 0.9129, 0.9131, 0.9131) and falls away below it, so nothing
    beneath the committed floor competes and the selection is unchanged. No logged number moved.
    """
    return LinearSVC(
        **{
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
            # dual="auto" picks the primal for n_samples >> n_features, which is this shape.
            "dual": "auto",
            # liblinear's default 1000 is not always enough at 140k rows; raised so a
            # ConvergenceWarning means something real rather than "the budget was tight".
            "max_iter": 5000,
            **TUNED_PARAMS["svm"],
            **params,
        }
    )


def make_decision_tree(**params: Any) -> DecisionTreeClassifier:
    """DecisionTreeClassifier with balanced class weights and tuned depth."""
    return DecisionTreeClassifier(
        **{
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
            **TUNED_PARAMS["decision_tree"],
            **params,
        }
    )


# Registry so run.sh / evaluate.py can iterate over the classical models uniformly.
BASELINE_FACTORIES: dict[str, Callable[..., Any]] = {
    "dummy": make_dummy,
    "random_forest": make_random_forest,
    "svm": make_svm,
    "decision_tree": make_decision_tree,
}


# --- Fold loading ------------------------------------------------------------------------


def load_folds() -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Transform the three in-distribution folds with the **already-fitted** Preprocessor.

    Returns ``{"train": (X, y), "val": (X, y), "unsw_test": (X, y)}``.

    Two invariants this function exists to hold:

    * **The Preprocessor is loaded, never refit.** ``preprocess.fit_preprocessor()`` would refit
      it -- harmless here in principle (it fits on the same fold) but it makes "was this the
      artifact Phase 6 used?" unanswerable. Load the serialized one and fail loudly if it is
      missing.
    * **The fold boundary is reproduced, not re-drawn.** ``split_source`` is a stratified split
      seeded from ``config.RANDOM_SEED``, so calling it again returns the same 140,272 / 35,069
      partition the Preprocessor was fit on. That is why the seed may not be overridden here.

    ``toniot`` is conspicuously absent and stays absent: Phase 4 measures the in-distribution
    ceiling only.
    """
    if not PREPROCESSOR.exists():
        raise FileNotFoundError(
            f"{PREPROCESSOR} not found -- Phase 3 has not run. Run `python -m src.preprocess` "
            "first; Phase 4 loads that artifact and must never fit its own."
        )
    preprocessor = Preprocessor.load(str(PREPROCESSOR))

    train_fold, val_fold = split_source(load_source(), seed=RANDOM_SEED)
    raw = {"train": train_fold, "val": val_fold, "unsw_test": load_holdout()}
    return {
        name: (preprocessor.transform(frame), frame[LABEL_COL].astype(int))
        for name, frame in raw.items()
    }


# --- Tuning ------------------------------------------------------------------------------


def _grid(candidates: dict[str, tuple[Any, ...]]) -> Iterator[dict[str, Any]]:
    keys = list(candidates)
    for values in itertools.product(*(candidates[key] for key in keys)):
        yield dict(zip(keys, values))


def tune(folds: dict[str, tuple[pd.DataFrame, pd.Series]]) -> dict[str, dict[str, Any]]:
    """Grid-search depth / regularization on the **val** fold and return the winners.

    Fit on ``train``, score on ``val``, select by :func:`selection_score` with
    :data:`SELECTION_TOLERANCE` resolving near-ties toward the more constrained model. UNSW-test is
    never touched -- that is the whole reason a val fold exists, and it is the difference between
    reporting a ceiling and reporting a fit.

    Prints a full candidate table to stdout and writes nothing to ``reports/metrics.csv``.
    """
    X_train, y_train = folds["train"]
    X_val, y_val = folds["val"]
    winners: dict[str, dict[str, Any]] = {}

    for name, candidates in TUNING_GRIDS.items():
        factory = BASELINE_FACTORIES[name]
        print(f"\n{name}: grid over {', '.join(f'{k}={v}' for k, v in candidates.items())}")
        graded: list[tuple[dict[str, Any], dict[str, Any], float]] = []
        for params in _grid(candidates):
            started = time.perf_counter()
            model = factory(**params)
            model.fit(X_train, y_train)
            scores = evaluate(model, X_val, y_val)
            elapsed = time.perf_counter() - started
            composite = selection_score(scores)
            graded.append((params, scores, composite))
            spelled = "  ".join(f"{k}={v}" for k, v in params.items())
            print(
                f"    {spelled:<40} val f1={scores['f1']:.6f}  "
                f"roc_auc={scores['roc_auc']:.6f}  mean={composite:.6f}  ({elapsed:5.1f}s)"
            )

        best = max(composite for _, _, composite in graded)
        # First qualifying candidate wins, and the grids are ordered most-constrained-first, so
        # this is "prefer lower capacity when the val fold cannot tell the difference".
        params, scores, composite = next(
            entry for entry in graded if entry[2] >= best - SELECTION_TOLERANCE
        )
        winners[name] = params
        margin = "the grid best" if composite >= best else f"{best - composite:.6f} below best"
        print(
            f"  -> chosen: {params}  (val f1={scores['f1']:.6f}, "
            f"roc_auc={scores['roc_auc']:.6f}, mean={composite:.6f} -- {margin})"
        )

    print("\nlocked hyperparameters -- copy into TUNED_PARAMS if they differ:")
    for name, params in winners.items():
        marker = "" if params == TUNED_PARAMS.get(name) else "   <-- DIFFERS FROM TUNED_PARAMS"
        print(f"    {name}: {params}{marker}")
    return winners


# --- In-distribution evaluation ----------------------------------------------------------

#: A fixed label, not a timestamp, so a re-run upserts onto its own rows in the log and leaves the
#: file byte-identical — "did this reproduce?" is a `git diff`, which a pseudo-unique id would make
#: impossible. Phase 6 takes its own `run_id`, and per the convention in ``evaluate.log_metrics``
#: every distinct condition (ablation, transfer fraction) needs its own too.
RUN_ID: str = "phase4-baselines"

#: The regime label Phase 6's ``run_regimes()`` uses for this half of the comparison. Spelled
#: identically so the two phases' rows join.
REGIME: str = "in_distribution"


def run_baselines(log: bool = True) -> dict[str, dict[str, Any]]:
    """Fit the four locked baselines on the train fold, score them on UNSW-test, log one row each.

    Returns ``{model_name: metrics}``. One row per model is upserted into ``reports/metrics.csv``
    via :func:`evaluate.log_metrics` -- the locked models only, never a tuning candidate. Re-running
    replaces this phase's four rows rather than appending a second copy of them.

    Models are fit on the **train fold**, not on train+val. Refitting on the union after locking
    hyperparameters would be the textbook move, but here it would (a) make Phase 4's training set
    differ from the fold the ``Preprocessor`` was fit on, (b) burn the val fold that Phase 5 still
    needs for MLP width/depth and Phase 6 for the ``conn_state`` ablation, and (c) buy little --
    ~52% of the train fold's rows are duplicate feature vectors in the 11-column shared subspace,
    so the marginal 35,069 rows add less information than their count suggests. Every phase
    therefore trains on the same literal frame.
    """
    folds = load_folds()
    X_train, y_train = folds["train"]
    X_test, y_test = folds["unsw_test"]
    X_val, y_val = folds["val"]

    print(
        f"train fold n={len(X_train):,}  val fold n={len(X_val):,}  "
        f"UNSW-test n={len(X_test):,}  features={X_train.shape[1]}"
    )
    print(f"UNSW-test attack share (positive_rate) = {float((y_test == 1).mean()):.4f}")

    results: dict[str, dict[str, Any]] = {}
    for name, factory in BASELINE_FACTORIES.items():
        params = TUNED_PARAMS.get(name, {})
        started = time.perf_counter()
        model = factory()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        val_scores = evaluate(model, X_val, y_val)
        test_scores = evaluate(model, X_test, y_test)
        results[name] = test_scores

        spelled = ", ".join(f"{k}={v}" for k, v in params.items()) or "no tuning surface"
        print(
            f"\n{name}  [{type(model).__name__}: {spelled}]  fit {fit_seconds:.1f}s\n"
            f"    val       f1={val_scores['f1']:.4f}  roc_auc={val_scores['roc_auc']:.4f}\n"
            f"    UNSW-test f1={test_scores['f1']:.4f}  roc_auc={test_scores['roc_auc']:.4f}  "
            f"precision={test_scores['precision']:.4f}  recall={test_scores['recall']:.4f}\n"
            f"              accuracy={test_scores['accuracy']:.4f}  "
            f"balanced_accuracy={test_scores['balanced_accuracy']:.4f}  "
            f"macro_f1={test_scores['macro_f1']:.4f}\n"
            f"              confusion [[tn, fp], [fn, tp]] = {test_scores['confusion_matrix']}"
        )

        if log:
            note = "tuned on UNSW val fold 2026-08-01; " + (
                f"params: {spelled}" if params else "majority-class floor, no tuning surface"
            )
            log_metrics(
                round_metrics(
                    {
                        "run_id": RUN_ID,
                        "model": name,
                        "regime": REGIME,
                        "seed": RANDOM_SEED,
                        "notes": note,
                        # The confusion matrix has no column in the frozen 14-field header, so it
                        # goes to stdout only; Phase 6 renders it as a figure.
                        **{k: v for k, v in test_scores.items() if k != "confusion_matrix"},
                    }
                )
            )

    _assert_clears_floor(results)
    return results


def _assert_clears_floor(results: dict[str, dict[str, Any]]) -> None:
    """Refuse to call Phase 4 done if a real model is no better than the majority-class floor.

    A Random Forest scoring at the Dummy's level is not a modest result, it is a wiring bug --
    mismatched X/y ordering, an all-constant feature matrix, a label column that never got mapped.
    Phase 4's whole job is to establish a ceiling worth measuring drift *against*, so this is
    checked rather than eyeballed.
    """
    floor = results.get("dummy")
    if floor is None:
        return
    failures = [
        name
        for name, scores in results.items()
        if name != "dummy"
        and (scores["f1"] <= floor["f1"] or scores["roc_auc"] <= floor["roc_auc"])
    ]
    if failures:
        raise RuntimeError(
            f"model(s) {failures} did not clear the dummy floor "
            f"(f1={floor['f1']:.4f}, roc_auc={floor['roc_auc']:.4f}) on UNSW-test. That is a "
            "pipeline bug, not a weak baseline -- check the feature matrix and label alignment "
            "before logging anything as a result."
        )


# --- Entry point -------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.models.baselines",
        description=(
            "Phase 4: fit the classical baselines on the UNSW train fold and record the "
            "in-distribution ceiling on UNSW-test. TON_IoT is not opened."
        ),
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help=(
            "re-run the val-fold grid search over depth/regularization and print the winners "
            "(stdout only -- nothing is logged to reports/metrics.csv)"
        ),
    )
    args = parser.parse_args(argv)

    set_seeds()

    if args.tune:
        tune(load_folds())
        return 0

    run_baselines()
    print(f"\nlogged {len(BASELINE_FACTORIES)} in-distribution rows -> reports/metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
