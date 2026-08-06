# Tests

Run from the repo root:

```bash
./.venv/bin/python -m pytest            # or just `pytest` with the venv activated
```

`pytest==9.1.1` is pinned in `requirements.txt`. It is **not** wired into `run.sh` — `run.sh` is the
reproduction pipeline and must stay the thing a grader runs to regenerate results, so tests are a
separate, deliberate invocation.

## What lands here

**Phase 5 — the toy separable set (the reason this directory exists).** The from-scratch logistic
regression (`src/models/scratch_logreg.py`) and MLP (`src/models/scratch_mlp.py`) are graded on being
implemented from first principles in numpy, which means their correctness cannot be borrowed from
sklearn. The plan's done-when for Phase 5 is that both **converge on a toy linearly-separable set**
and land within a few points of their sklearn equivalents in-distribution. The first half of that is a
test, not a paragraph:

- a small synthetic separable 2-D set both models must drive to (near-)zero training error — if a
  from-scratch model cannot fit data that is trivially fittable, the gradient or the backprop is
  wrong, and no in-distribution score from it means anything;
- the **class-weighted** loss actually weighting: on a deliberately imbalanced toy set, the unweighted
  variant collapses to predicting the majority class and the weighted one does not. This is the single
  most important test in the directory. Under this project's imbalance a plain net predicts
  "normal" for everything and posts a *deceptively high accuracy*, which is a failure mode that looks
  like success in every metric except the ones we report;
- gradient checks (numerical vs analytic) on the MLP backward pass.

**Phase 9 — the per-attack-family data path** (`test_per_family.py`). The figures are rendered from
committed artifacts and are checked by eye, but the *numbers* behind the two per-family figures are a
new computation, so three of its contracts are pinned here: that a family is scored **one-vs-normal**
(the family's rows plus every normal row of the same set — an attack family is all-positive, so
scoring it alone would make "F1" a relabelling of recall and put it out of reach of the
majority-class floor); that a family label disagreeing with the binary label raises rather than
silently attributing one family's predictions to another; and that `family_set` is part of
`reports/per_family_metrics.csv`'s upsert key, because `dos`, `scanning` and `backdoor` exist in
*both* vocabularies over completely different row populations and would otherwise overwrite each
other.

**Also worth covering, cheap:** `evaluate.log_metrics()` idempotence — that logging the same row twice
leaves `reports/metrics.csv` byte-identical, and that two rows differing only in `run_id` coexist
rather than overwrite (the ablation/transfer-fraction convention documented under **Metrics log** in
the root `README.md`). Both are contracts Phases 6–7 depend on and both are one-liners against a
`tmp_path` fixture.

## Conventions

- `test_*.py`, one file per module under test (`test_scratch_logreg.py`, `test_evaluate.py`, …).
- Use pytest's `tmp_path` fixture for anything that writes. **No test may write to
  `reports/metrics.csv`** — it is a committed artifact and the run log the reproduction claim rests
  on.
- Tests must not need `data/raw/`. The real CSVs are git-ignored and absent from a fresh clone, so
  anything requiring them is not a unit test; synthesize fixtures instead.
- Seed from `config.RANDOM_SEED` so a failure is reproducible.
