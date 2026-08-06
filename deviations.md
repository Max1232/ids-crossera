# Deviations from the approved proposal

_Consolidated record of where the delivered project departs from the scope promised in
`Proposal-Final.md`, plus the substantive methodological decisions a reader could reasonably
question. Last updated: 2026-08-06 (content through **Phase 9**, and the code is complete as of that
date — §7 records the final audit. §§3.14–3.17 record the four Phase 9 decisions a grader could
question; §1.3's coverage percentage now names its basis, and §1.4 states Phase 8 as cut rather than
pending. See the two changelogs at the foot of this file.)_

**Δ convention, everywhere in this file:** `Δ = in_distribution − cross_era`, i.e. **what the model
lost**. Positive = degraded, negative = improved. This is `evaluate.metric_deltas()`, and it is what
`./run.sh` prints, so this file and a grader's own run cannot disagree on a sign. Δ-of-Δ for an
ablation is `Δ full − Δ ablated`.

## What this file is — and is not

This is a **stable diff against the approved proposal**, which is frozen. It exists so the
report's Methods and Data & Experiments sections can justify every departure without
reconstructing the reasoning in August — each entry names the evidence and the report section it
feeds.

It is **not** a source of authority and **not** a log of routine engineering fixes:

- Scope is owned by `Proposal-Final.md`. The build and its per-phase deviations are owned by
  `Implementation-Plan.md`. This file consolidates and points back; it never overrides them.
- Mid-build engineering course-corrections stay in the plan. Only decisions that change the
  **promised scope** or that a grader would question belong here.
- When an entry below needs to change, change it in the authoritative file first, then reflect it
  here. Do not let this file drift ahead of the plan — silent drift between copies is exactly what
  caused the late-July framing mess.

Report sections referenced: Methods · Data & Experiments · Results (the handout has no standalone
Limitations section, so caveats land in Data & Experiments or attach to the reported delta in
Results).

---

## 1. Scope deviations from the proposal

### 1.1 TTL statistics — promised, dropped
**Promised:** TTL-based features. **Delivered:** none. TTL is absent from the TON_IoT flow CSVs
(Zeek `conn.log`-derived; Zeek exports no per-flow IP TTL) and recoverable only by reprocessing the
67.7 GB raw captures, which is out of scope for a four-week timeline.
**Defensible framing:** state it as "absent from the flow CSVs and recoverable only from the raw
captures, out of scope for this timeline," **not** "structurally unavailable." Secondary argument
that turns the shortfall into a strength: UNSW attack traffic came from the IXIA PerfectStorm
appliance while normal traffic came from other hosts, so initial TTL is a **generator fingerprint,
not an attack signal** — including `sttl` (a near-perfect single-feature discriminator on
UNSW-NB15) would have inflated in-distribution scores and made the cross-era drop look
artificially catastrophic.
**Evidence:** 44-column TON_IoT header scan, zero substring matches for `ttl`/`hop`
(`schema_catalogue.md` §4.9). Cite the header, not the Zeek-schema inference.
**Report home:** Data & Experiments (feature set) + a Methods sentence.

### 1.2 Pre-computed rate features — promised, derived instead of taken directly
**Promised:** rate features. **Delivered:** `bytes_per_sec` and `pkts_per_sec` derived from
duration and counts on both sides. UNSW ships `sload`/`dload` in **bits** per second and TON_IoT
has no rate column at all, so no usable shared rate column exists to take directly. The UNSW rates
also carry an Argus `(spkts−1)/spkts` correction, so they cannot be cross-checked against a naive
`sbytes*8/dur`.
**Evidence:** `sbytes*8*(spkts−1)/(spkts*dur)` matches `sload` within 1% on 99.98% of `dur>0` rows.
`duration == 0` is guarded and carried as a named `zero_duration` flag (see 2.3-adjacent note in
the plan) because the zero case's **prevalence** inverts across eras (**1.52%** of UNSW rows vs
**28.44%** of TON_IoT's — a *rate* inversion, not §3.10's signal inversion).
**Report home:** Methods (feature derivation).

### 1.3 Per-attack-family analysis — restricted to three shared families
**Promised:** per-family comparison. **Delivered:** only `DoS`↔`dos`, `Reconnaissance`↔`scanning`,
and `Backdoor`↔`backdoor` align across the two label spaces. UNSW-NB15 has **no DDoS class**, so
the earlier shared-family list was wrong. Say that limit out loud.

**The coverage percentage depends on which UNSW frame you mean, and an earlier draft of this entry
mislabelled its own basis** (it cited the *training partition*'s counts under a "train-fold" label —
the two happen to agree to 0.03 points, so the error was invisible). All four bases, measured
2026-08-06:

| basis | shared-family attack rows | coverage |
|---|---|---|
| UNSW training partition (175,341 rows) | 24,501 / 119,341 | **20.53%** |
| UNSW seeded train fold (140,272 rows) | 19,569 / 95,472 | 20.50% |
| UNSW full concatenated frame (257,673 rows, train+test per §3.3) | 32,669 / 164,673 | 19.84% |
| **UNSW-test (82,332 rows)** — what the per-family figure scores | 8,168 / 45,332 | **18.02%** |
| **TON_IoT (211,043 rows)** | 60,000 / 161,043 | **37.26%** |

**Quote 18.02% / 37.26% beside the cross-era per-family figure.** That figure's in-distribution
regime is UNSW-**test**, not the train fold, and the caption computes both shares at render time from
named `metrics.csv` rows (`n_test × positive_rate`) rather than transcribing them, so it cannot drift
from this table. The three-family restriction applies **only** to the per-family breakdown — it never
narrows the binary label (§3.10).

**The per-family unit is F1 / ROC-AUC, not a confusion matrix.** The proposal's Metrics paragraph
offered "per-attack-family confusion matrices **if time allows**" — a conditional, so this is not a
scope breach, but a grader reading the two documents side by side will look for the matrices and
should find this note instead. Delivered in their place: `reports/per_family_metrics.csv` (369 rows,
its own key including `family_set` — §3.16) carrying precision, recall, F1, ROC-AUC, accuracy,
balanced accuracy and macro-F1 per family, plus the two figures those rows drive. Every family is
scored **one-vs-normal**, and that is what makes a matrix the weaker unit here: an attack family's own
rows are all-positive, so a 2×2 over them alone has one populated row and its F1 collapses to a
relabelling of recall. Aggregate confusion matrices *are* delivered, for all six models across both
regimes (`reports/confusion_matrices.json`, Fig 3) — the omission is per-family, not confusion
matrices as such.
**Report home:** Data & Experiments + Results (per-family figure caveat).

### 1.4 RQ3 live-malware probe — promised as an optional stretch, **not delivered**
**Promised:** optional stretch. **Delivered:** nothing — the phase was **cut on the timeline**, which
is the exit the proposal itself reserved ("if isolation cannot be safely guaranteed or time runs
short, we drop the live probe and still have a complete project"). This is a settled decision, not a
pending one: neither hard gate was ever attempted, so there is no instructor authorization on file
and no air-gapped environment, and no code exists for it anywhere in the repo — the `lab/` directory
that once held the plan was **removed 2026-08-06** rather than left as an empty invitation to start.
The public-dataset core (Phases 0–7 plus 9) is the complete project, and RQ1/RQ2 are answered without
it.
**Consequence for the two claims that leaned on RQ3:** the "decade-forward" test does not exist, so
the UNSW→TON_IoT gap of ~4–5 years is the *only* temporal span measured, and §2.1's upper-bound
caveat has no live-capture counterweight. Say both out loud rather than leaving RQ3 unmentioned.
**Report home:** one sentence stating RQ3 was scoped as optional and dropped for time — a reader
comparing the repo against the proposal will look for it, and silence reads worse than the cut.

### 1.5 EDA notebook — dropped in favor of scripted figures
**Delivered:** `reports/schema_catalogue.{md,csv}` covers the Phase 1 EDA, and graded figures come
from `src/plots.py` → `reports/figures/` via `run.sh`. No notebook, because the rubric grades the
Results **plots** directly (labels, legends, captions) and they must be files the report PDF embeds.
**Report home:** not a report claim; recorded so the missing notebook is not read as an omission.

### 1.6 Framing — concept-drift retained (record of a reverted deviation)
A "Cross-Dataset Generalization" retitling was drafted and **archived; it was never submitted**.
The concept-drift framing stands. Logged here so the retitling does not resurface — it briefly
propagated into the README and was mistakenly treated as settled.
**Report home:** none; internal guardrail.

---

## 2. Confounds the proposal's clean framing needs stated

Not scope cuts, but caveats that must ride alongside every reported delta.

### 2.1 Enterprise-vs-IoT domain shift
UNSW-NB15 is general enterprise traffic; TON_IoT is IoT/IIoT. The UNSW→TON_IoT delta therefore
bundles temporal drift with a domain shift and is treated as an **upper bound on pure temporal
drift**, not a clean time-only experiment. The true decade-forward test is only the optional RQ3
captures.
**Report home:** Results (attach to the headline Δ) + Data & Experiments.

### 2.2 Class-balance mismatch
UNSW-test is ~45% normal; TON_IoT is ~24% normal, so prevalence-sensitive metrics move on that
difference alone — every reported delta names both test sets' normal share. Separately, TON_IoT's
delivered `Train_Test_Network.csv` is **211,043 rows / 50,000 normal** against a documented
461,043 / 300,000 (250,000 normal short); two independent downloads are byte-identical, so this is
part of the measurement, not a re-download fix. This is why the drift claim leads with **ROC-AUC**
(prevalence-insensitive) and reports F1 with the balance caveat attached.
**Evidence:** `data/README.md`. **Report home:** Data & Experiments + Results.

### 2.3 The recovery generalizes to unseen flows — the limitation is feature-space determinism, not memorization
The RQ2 result is that **1% of the fine-tune pool (1,055 rows) recovers essentially all of the lost
ROC-AUC**: `random_forest` goes 0.2121 → 0.997939 on the frozen test half and every model clears
the dummy's 0.5000 (§3.13). The fine-tune rows are index-disjoint from that half by construction
(§3.12), but **TON_IoT repeats itself in the 22-column shared subspace**, so the first objection a
grader should raise is that the recovery might be a lookup table rather than a learned decision
boundary. **It is not, and that was measured rather than argued.**

**The redundancy is real.** Measured on the delivered frame and re-verified 2026-08-04:

| quantity | value |
|---|---|
| distinct feature vectors in the full target frame | **92,438** of 211,043 rows (43.80%) |
| distinct feature vectors in the frozen test half | **48,985** of 105,521 rows (46.42%) |
| frozen-test rows whose exact feature vector also appears in the **1%** fine-tune draw | **35.78%** |
| ... in the 5% / 10% / 25% / full-pool draw | 46.44% / 49.28% / 52.57% / 57.03% |
| distinct vectors carrying **both** labels | **15** (7,522 rows, 3.56%) |

**But it does not explain the recovery.** Score the *same* fitted model on only the frozen-test rows
whose exact 22-feature vector is **absent** from the fine-tune draw — the rows it provably cannot
have memorized — and the curve barely moves:

| model | budget | AUC, all rows | AUC, unseen only | F1, all rows | F1, unseen only | unseen rows | unseen share |
|---|---|---|---|---|---|---|---|
| `random_forest` | 1% (n_ft=1,055) | 0.997939 | **0.995852** | 0.992254 | **0.988248** | 67,769 | 64.22% |
| `random_forest` | 10% (n_ft=10,552) | 0.998816 | 0.998177 | 0.996027 | 0.993545 | 53,524 | 50.72% |
| `random_forest` | ceiling (n_ft=105,522) | 0.999759 | 0.999479 | 0.997746 | 0.996369 | 45,345 | 42.97% |
| `decision_tree` | 1% (n_ft=1,055) | 0.986216 | **0.976380** | 0.988179 | **0.981802** | 67,769 | 64.22% |
| `decision_tree` | 10% (n_ft=10,552) | 0.994068 | 0.988591 | 0.990607 | 0.985698 | 53,524 | 50.72% |
| `decision_tree` | ceiling (n_ft=105,522) | 0.998387 | 0.996403 | 0.995226 | 0.990850 | 45,345 | 42.97% |

At the 1% budget the unseen subset is **64.22% of the frozen half (67,769 rows) at attack share
0.7596** — within 0.004 of the half's own 0.76308, so it is not a rarer or an easier
subpopulation. Memorization is therefore worth **0.0021 AUC** to the random forest and **0.0098** to
the single tree, against recoveries of **0.786** and **0.634**. **The conclusion is generalization,
not memorization: on rows whose feature vector it has never seen, 1,055 labelled modern flows still
buy ROC-AUC 0.9959 / F1 0.9882, so the recovered model is not a lookup table.**

**Provenance — what was measured on which split.** The all-rows columns are the committed rows
`phase7-recovery-{f0.01,f0.10,ceiling},{random_forest,decision_tree},target_frozen_test` in
`reports/metrics.csv`. The unseen-only columns **are not in `reports/metrics.csv` and cannot be** —
the log has no per-row-subset breakdown — so they are a re-derivation: a read-only script loads the
Phase 3 preprocessor transform-only, reproduces the canonical frozen split by calling
`transfer.frozen_split_indices()` / `fraction_indices()` rather than re-implementing them, refits
through `transfer.finetune()`, and partitions the frozen half with `np.unique` over the 22 features
rounded to 9 decimals. Its all-rows column reproduces the committed rows to **4.9e-07**, which is
what licenses the unseen-only column standing beside them; the redundancy table above was
re-verified in the same pass and every figure held.

**The limitation is real, but it is a different one: given these 22 features the label is
near-deterministic, so few labels suffice.** Only **15 of the 92,438 distinct vectors carry both
labels** (7,522 rows, 3.56% of the frame), and a per-vector majority vote — a perfect lookup table
over the shared subspace — would score **99.90% accuracy** on the whole target frame, missing 221
rows of 211,043. The 22-column subspace nearly *determines* the TON_IoT label, and a model needs
very few examples to find a function that simple. That is a claim about **task difficulty in this
feature space** — a handful of IoT device types emitting near-identical flows (§2.1) — not about the
model cheating, and it leaves the disjointness invariant untouched: these are distinct records,
which is exactly why disjointness is asserted on **row indices** and never on values (the same ~52%
duplicate-vector rate is documented for the UNSW train fold in §3.6/§3.7).
**How to state it:** "1% of modern labels (1,055 rows) recovers the lost ROC-AUC, and it
generalizes — 0.9959 AUC on the 64% of test rows whose feature vector never appears in the
fine-tune draw. But the label is close to a deterministic function of these 22 features (15 of
92,438 distinct vectors are label-ambiguous; a lookup table would score 99.90%), so 1% is a **lower
bound** on the labelling effort a feature space with genuine class overlap would need." Report the
lower-bound clause with the recovery figure; do not report the recovery as memorization.
**Evidence:** `reports/metrics.csv` for the all-rows column; `transfer.duplicate_overlap()`, printed
by `python -m src.transfer` on every run, for the redundancy table; the unseen-vector re-derivation
above for the rest.
**Report home:** Results (attach to the RQ2 curve) + Data & Experiments (the determinism caveat).

---

## 3. Methodological decisions not in the original plan

Report-bearing engineering decisions a grader could question. Full detail lives in
`Implementation-Plan.md`; summarized here because they shape the reported numbers.

### 3.1 Byte features repointed to `*_ip_bytes` (Phase 2, commit cb593c9)
The original plan paired UNSW `sbytes`/`dbytes` with TON_IoT `src_bytes`/`dst_bytes` and dropped
`*_ip_bytes`; that was backwards. TON_IoT `src_bytes`/`dst_bytes` are Zeek **payload** bytes (0 on
65%/71% of rows); `*_ip_bytes` are **total IP** bytes, matching UNSW's IP-level `sbytes` (28 B
floor, never 0). `FEATURE_MAP` now uses `(sbytes, src_ip_bytes)` / `(dbytes, dst_ip_bytes)` and the
payload columns are dropped.
**Verified in output:** TON_IoT `src_bytes` zero-rate is **8.10%** (the IP-bytes signature);
payload would have been 65.46%. **Report home:** Methods (feature alignment).

### 3.2 Connection state collapsed to a coarse three-way set (plus an `other` bucket) — and ablated
UNSW `state` (Argus codes) and TON_IoT `conn_state` (Zeek codes) share **zero tokens**, so no
lexical alignment is possible. Both are hand-collapsed to **completed / reset / no-response**, and
`RARE_BUCKET` then adds **`other`** as a fourth *encoded* level for anything the collapse does not
claim — so the semantic decision is three-way while the fitted `Preprocessor` emits
`conn_state × 4`, which the d=22 feature width depends on. Both statements are true; neither
supersedes the other.

**Why this needed measuring rather than asserting.** Unlike the other shared-feature mappings, this
feature is *ours*: the bridge between the two vocabularies is a modelling decision we invented, not
a correspondence the datasets provide. It is also badly asymmetric across the eras — measured on the
UNSW train fold against TON_IoT:

| encoded level | UNSW train fold | TON_IoT | ratio |
|---|---|---|---|
| `reset` | 0.0421% | 23.6757% | ~560× |
| `other` (`RARE_BUCKET`) | 0.0057% | 11.0556% | ~1,900× |

The `other` mismatch is the *larger* of the two, and earlier drafts of this entry cited only `reset`.
So part of the RQ1 drop could be our own collapse carrying an **instrumentation** change rather than
attacker evolution.

**Delivered:** the ablation the proposal committed to, under `run_id = phase6-crossera-no_conn_state`,
using the identical retrain-without design as the `proto` ablation (**§3.9** — the feature is removed
from the hypothesis class and the models are retrained on the UNSW train fold at d=18, *not* masked
at test time). It is a matched in-distribution/cross-era pair for all six models, so the reported
quantity is again the **difference of the deltas**.

> **The two ablations share a width and are not the same experiment.** `protocol` and `conn_state`
> each encode to four one-hot levels, so both conditions run at d=18. `d` therefore does not identify
> a condition — only the `run_id` and the `notes` column do. Do not write "the d=18 ablation".

**Result: the objection does not survive, on the same pattern as `proto`.** Δ-of-Δ ROC-AUC spans
**−0.0427 to +0.0535** across the six models against a ~0.7 collapse — at most **~7.6%** — and for
three of them (`decision_tree` −0.0001, `svm` −0.0427, `scratch_logreg` −0.0167) the ablated model
degrades *further* without the feature. Only `scratch_mlp` (+0.0535) and `random_forest` (+0.0194)
lose meaningfully less when the state collapse is unavailable. **The hand-invented collapse is not
what produces the rank inversion**; §3.10's response-side mechanism is.

**Report home:** Methods (the collapse, and why it had to be ablated rather than argued) + Results
(the ablation, beside §3.9's).

### 3.3 UNSW parquet is train+test concatenated with a `split` column
`config` names one UNSW parquet but UNSW ships two CSVs, so Phase 2 concatenated them (257,673 rows
= 175,341 train + 82,332 test) with a `split` tag. This creates a **leakage dependency**: Phase 3
must filter `split == "train"` before fitting the `Preprocessor`. Fitting the unfiltered frame
leaks test statistics and raises no error.
**Report home:** none (correctness note); recorded so it is never re-fit unfiltered.

### 3.4 `flow_duration` upper-tail clip — asymmetric by design (Phase 3, commit c033d47)
Numeric order is impute → log1p → clip → z-score. The clip bound is fitted on the UNSW train fold
only and recovers `log1p(60.00) = 4.1109` (UNSW's 60 s cap) from the data. It is scoped to
`flow_duration`, the only column that overshoots meaningfully (TON_IoT reaches +16.93σ vs a train
max of +5.75σ; 6,857 rows / 3.25% above train support). The clip is a **no-op on the fit data**, so
in-distribution (Phase 4) numbers do not move — but cross-era (Phase 6) numbers do.
**The asymmetry is a judgment, and must be stated as one:** the upper clip treats UNSW's 60 s cap
as a collection artifact, while the **lower tail is deliberately preserved** — TON_IoT falls below
the train minimum on 50,872 `src_bytes` rows (24.1%) because UNSW `sbytes` has a 28 B floor and is
never 0 while TON_IoT has real zero-byte flows. That is a genuine cross-era difference RQ1 exists to
find; clipping it would delete a result. `clipped_fraction()` reports the bind rate per frame (0.00%
in-distribution, 3.25% TON_IoT), so the discarded tail is auditable.
**Report home:** Methods (preprocessing) — state both the clip and why the lower tail was kept.

### 3.5 SVM is linear, not kernel — DONE (Phase 4, commit `ffae5d7`), RATIFIED 2026-08-03
The proposal's unqualified "SVM" is satisfied by `LinearSVC`/`SGDClassifier`. A kernel SVM at
175,341 rows needs a kernel matrix in the hundreds of GB and will not finish.
**Delivered 2026-08-01:** `baselines.make_svm` is `LinearSVC(C=0.1, class_weight="balanced",
dual="auto", max_iter=5000)`; the stub docstring specifying `SVC(probability=True)` was corrected in
place, and no kernel path exists anywhere in `src/`. `LinearSVC` over `SGDClassifier` because it fits
the 140,272-row train fold in **0.8 s** with `C` as its only knob — which is the regularization
strength the proposal promises to tune anyway.
**ROC-AUC comes from `decision_function`, deliberately without `CalibratedClassifierCV`:** AUC scores
a *ranking*, and any monotone calibration of a signed margin yields the identical ranking and
therefore the identical AUC, so the wrapper would cost a k-fold refit and move no reported number.
Nothing in Phases 4–7 needs calibrated probabilities.
**Ratified 2026-08-03 — no longer provisional.** `LinearSVC` and the `decision_function`-for-AUC
choice were previously carried as pending a collaborator review that is not happening, so they were
checked directly and both stand. The `C` grid was checked with them, because the specific risk was
that `C=0.1` had won by being the *floor* of the search. **It was not.** `TUNING_GRIDS["svm"]` spans
`(0.01, 0.1, 1.0)`, bracketing the winner on both sides, and `C=0.01` lost by 0.29 composite points
(0.9100 vs 0.9129) — well outside the 0.002 selection tolerance. Re-measured on the val fold over a
widened `(0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 10.0)`: the composite climbs monotonically to a plateau
at `C ≥ 0.05` (0.9125 / 0.9129 / 0.9131 / 0.9131) and falls away below it. Nothing beneath the
committed floor competes, the selection is unchanged, and **no logged number moved.**
**Evidence:** in-distribution F1 0.7883 / ROC-AUC 0.8846 on UNSW-test, `reports/metrics.csv` row
`phase4-baselines,svm,in_distribution`. **Report home:** Methods.

### 3.6 Baseline tuning selects on F1 *and* ROC-AUC jointly, not F1 alone (Phase 4, commit `ffae5d7`)
Depth/regularization are chosen on the UNSW val fold by the **mean of val F1 and val ROC-AUC** — the
proposal's headline pair weighted equally — with near-ties inside 0.002 resolved toward the more
constrained model. This is worth stating because **F1 alone picks a materially worse model:** the
fully grown `min_samples_leaf=1` Decision Tree scores val F1 0.9550 against 0.9490, a 0.6-point gain,
while val ROC-AUC collapses **0.9858 → 0.9422**. A saturated tree emits hard 0/1 leaf probabilities,
so its score ranking is coarse and AUC punishes it. Since the drift claim leads with ROC-AUC (§2.2),
an F1-only rule would have degraded the headline metric one phase before it is reported.
The parsimony tolerance encodes the honest precision of a val fold whose rows are ~52% duplicate
feature vectors; it flipped RF to `max_depth=20, n_estimators=100` (from `None/300`, at a third of
the cost) and the SVM to `C=0.1` (from `C=1.0`).
**Locked params:** DT `max_depth=None, min_samples_leaf=20`; RF `max_depth=20, n_estimators=100`;
SVM `C=0.1` — baked into `baselines.TUNED_PARAMS` so Phase 6 re-instantiates identical models with no
search. **Report home:** Methods (model selection).

### 3.7 Baselines are fit on the train fold, not refit on train+val (Phase 4, commit `ffae5d7`)
The textbook move after locking hyperparameters is to refit on train+val. Not done, so that every
phase trains on the same literal frame: refitting on the union would make Phase 4's training set
differ from the fold the `Preprocessor` was fit on (§3.3), the val fold is still needed for Phase 5's
MLP width/depth tuning, and the marginal 35,069 rows carry less information than their count suggests
given the ~52% duplicate-vector rate in the 11-column subspace. (An earlier draft also cited Phase
6's ablations as needing the val fold. They do not: both are retrains from the locked `TUNED_PARAMS`
with no search — §3.9 — so they consume no val fold even in principle.)
**Report home:** Data & Experiments (state it as a choice, not an oversight).

### 3.8 The Dummy floor's F1 is high, and *rises* across eras — report it in both regimes
Not a deviation but a reporting obligation Phase 4 surfaced. UNSW-**test** is 55.06% *attack*, so
`most_frequent` predicts "attack" everywhere and posts **F1 = 0.7102 at recall 1.0 with ROC-AUC
exactly 0.5000** and macro-F1 0.3551 — a high-looking F1 from a model with zero discriminative power.
This is the cleanest single argument in the project for why the headline is neither accuracy nor raw
F1. And it *rises* across eras: TON_IoT is 76.31% attack, so the same trivial model scores **higher**
there. (Reserve "inverts" for §3.10's rank inversion — a different phenomenon. The Dummy's ROC-AUC
does not move at all; only its F1 rises, and purely on prevalence.)
Phase 6 must therefore log the Dummy in **both** regimes, or a reader cannot separate the prevalence
artifact from drift. **Discharged in Phase 6, and the number is now measured:** the Dummy's F1
**rises 0.7102 → 0.8656** cross-era (Δ = **−0.1554** on the `in − cross` convention of §3.10 — it
*gained* F1, so it lost a negative amount) at ROC-AUC exactly 0.5000 in both regimes —
that is the size of the prevalence artifact every real model's F1 delta has to be read against.
**Report home:** Results (beside the headline Δ) + Data & Experiments.

### 3.9 The `proto` ablation is a retrain-without, not a test-time mask (Phase 6)
The `proto` hazard is measured, not hypothetical: 18.31% of UNSW train rows use a protocol TON_IoT
never contains, those rows are **91% attack**, and both sides collapse to
`{tcp, udp, icmp, other}` — so a model can learn "`other` → attack" from a bucket that is 91% attack
in training and **0% of rows at test time**, and part of the RQ1 drop could be that signal going
inert rather than attacker evolution.

**The obvious implementation of the ablation is the wrong experiment.** Zeroing the `protocol`
one-hot columns at test time on a model *trained* with them evaluates that model on inputs neither
era produces — an all-zero one-hot block is off the training manifold for reasons unrelated to
drift — so the resulting drop confounds "the proto signal went inert" with "the model was
perturbed". **Delivered instead:** the feature is removed from the hypothesis class. The models are
**retrained on the UNSW train fold** with the four protocol one-hots excluded (d=18 rather than
d=22) and run through *both* regimes, giving the ablation its own matched
in-distribution/cross-era pair under `run_id = phase6-crossera-no_proto`. A Δ needs both halves, so
the reported quantity is the **difference of the deltas**. The retrain is train-fold-only and
therefore not leakage; nothing is refit on UNSW-test or TON_IoT, and the Phase 3 `Preprocessor` is
not refit at all (dropping the one-hot columns post-transform is exactly equivalent to refitting it
without `protocol`, since per-column encodings are independent).
**Result:** the objection does not survive. The Δ-of-Δ spans **−0.087 to +0.061** across the six
models, and for three of them the ablated model degrades *further* — so the `other`-bucket artifact
accounts for at most **~12%** of a ~0.7 collapse (0.0867 / 0.70; an earlier draft said "~8%", which
read the 0.087 as a percentage).
**Report home:** Methods (ablation design — state why a test-time mask was rejected) + Results.

### 3.10 Cross-era ROC-AUC lands *below* 0.5 — the ranking inverts (Phase 6)
Not a deviation but a finding that needs stating before someone reads it as a bug. **Every real
model inverts.** In-distribution ROC-AUC spans **0.8811–0.9788**; cross-era it spans
**0.1846–0.3534** — below the Dummy's exact 0.5000. The 2015-learned score ranking does not merely
stop working on 2019–20 traffic, it runs backwards, so the detector is *anti-correlated* with ground
truth on TON_IoT. This is a **rank inversion, not a decay**, and the report has to say so.

**Δ convention, used throughout this file and by the pipeline:** `Δ = in_distribution − cross_era`,
i.e. **what the model lost** — positive means it degraded, negative means it improved. This is
`evaluate.metric_deltas()`, and it is what `./run.sh`'s headline table prints, so the report and a
grader's own run agree on the sign.

| model | in-distribution | cross-era | AUC lost (in − cross) |
|---|---|---|---|
| `random_forest` | 0.9788 | 0.2106 | +0.7682 |
| `scratch_mlp` | 0.9621 | 0.1846 | +0.7775 |
| `decision_tree` | 0.9648 | 0.3534 | +0.6115 |
| `svm` | 0.8846 | 0.2114 | +0.6732 |
| `scratch_logreg` | 0.8811 | 0.2489 | +0.6321 |
| `dummy` | 0.5000 | 0.5000 | 0.0000 |

**Evidence:** `reports/metrics.csv`, rows `phase4-baselines,*,in_distribution` and
`phase6-crossera,*,cross_era`; the two scratch models' in-distribution halves are logged under
`phase6-crossera` because they have no Phase 4 entry. Sub-0.5 AUC is also the exact signature of a
flipped label, so polarity was verified directly against both parquets before the inversion was
accepted: `label = 0` ⟺ normal and `label = 1` ⟺ attack in both, with TON_IoT's `label = 1` covering
**all nine** of its attack `type` levels — `backdoor`, `ddos`, `dos`, `injection`, `password`,
`ransomware`, `scanning`, `xss` at 20,000 rows each plus `mitm` at 1,043, for 161,043 positives
against 50,000 normal. (The *three-family* restriction of §1.3 applies only to the per-family
breakdown; it does not narrow the binary label.) It is not a wiring fault.

**Mechanism: response-side feature inversion between eras.** An earlier draft of this entry blamed
`zero_duration`; **that attribution is withdrawn.** The flag fires on 2.5% of UNSW-test *normal* rows
against 0.08% of UNSW-test *attack* rows, but 25.4% vs 29.4% on TON_IoT normal vs attack — near-flat
across TON_IoT's classes — and its standalone ROC-AUC is 0.488 in-distribution and 0.520 cross-era.
A feature that close to random cannot drive a ~0.77 AUC swing. What does drive it is the **response
side of the flow flipping sign between the two datasets**, at the class median.

*Basis: the UNSW columns below are **UNSW-test**, the in-distribution evaluation set — not the train
fold, whose medians are 1,112 and 10. Every §3.10 figure is on that basis; §1.2/§1.3/§3.2 quote
train-fold figures instead, so the two groups are not interchangeable.*

| feature | UNSW-test normal | UNSW-test attack | TON_IoT normal | TON_IoT attack |
|---|---|---|---|---|
| `dst_bytes` | 354 | 0 | 0 | 40 |
| `dst_pkts` | 8 | 0 | 0 | 1 |

UNSW attacks are largely unanswered probes and scans against enterprise servers, so *absence of a
reply* marks an attack. TON_IoT normal traffic is silent IoT telemetry, so *absence of a reply*
marks normal. A UNSW-trained model has learned "silence is hostile"; on IoT traffic that rule is not
merely uninformative but **actively backwards**, which is why cross-era AUC lands *below* 0.5 rather
than degrading toward it. This is the enterprise-vs-IoT domain shift of **§2.1**, now made
quantitative at the feature level — not a second, separate confound.

**Consequence, and it is a limitation the report must state rather than bury:** the headline Δ is an
**upper bound on temporal drift bundled with domain shift, not a measurement of pure temporal
drift.** The two eras differ in deployment environment as well as in time, and this design cannot
separate them. It attaches to the reported number in Results, not to a footnote.

**Robustness — the three obvious deflations of the result are all measured, and none holds.**
- *"It's just the `proto` shortcut going inert."* It is not. The ablation is a **retrain-without at
  d=18, not a test-time mask** (design and the full argument: **§3.9** — masking the protocol
  one-hots on a model trained with them evaluates it on inputs no era produces, confounding "the
  learned shortcut went inert" with "the model was perturbed off its training manifold"). Matched
  pairs are refit on the UNSW train fold and run through **both** regimes, so the reported quantity
  is a **difference of deltas**: it spans **−0.087 to +0.061** against a ~0.7 collapse, and three of
  the six models degrade *further* without the protocol features.
- *"It's the hand-invented `conn_state` collapse, not the traffic."* Also measured, also no. The
  matched retrain-without at d=18 under `phase6-crossera-no_conn_state` (**§3.2**) gives a Δ-of-Δ
  spanning **−0.0427 to +0.0535** — at most ~7.6% of the collapse — and **three of the six models
  degrade *further*** without the state feature. The collapse is ours and is badly asymmetric across
  eras (`reset` 0.0421% vs 23.6757%, `other` 0.0057% vs 11.0556%), which is exactly why it was
  ablated rather than defended in prose; it is still not what drives the inversion.
- *"It's a prevalence artifact."* Not on ROC-AUC. The majority-class (`most_frequent`) Dummy holds ROC-AUC exactly
  **0.5000 in both regimes** while its cross-era F1 *rises* to **0.8656** (from 0.7102) purely on
  prevalence — UNSW-test is 55.06% attack against TON_IoT's 76.31%. Any F1 delta has to be read
  against that artifact, which is precisely why the drift claim leads with ROC-AUC (**§2.2**,
  **§3.8**) and why every logged row carries `n_test` and `positive_rate` for the set it was
  measured on.

**Report home:** Results — the claim is "the detector inverts, and here is the feature-level reason",
which is stronger than "it degrades" and must be argued rather than buried — plus the upper-bound
sentence in Data & Experiments alongside §2.1.

### 3.11 Gradient boosting excluded
`xgboost` was removed (640 MB of CUDA libs, zero references in `src/`, and boosting appears nowhere
in the approved proposal). If a boosting baseline is ever wanted, use sklearn's
`HistGradientBoostingClassifier`. **Report home:** none unless added.

### 3.12 The recovery curve's fraction-0 point is re-measured, not taken from Phase 6 (Phase 7)
**The obvious thing to do here is wrong.** Phase 6's `phase6-crossera` `cross_era` rows are the
same unadapted models measured on the same target era, so they look like the natural left endpoint
of the recovery curve. They are not: they were scored on **all 211,043 TON_IoT rows**, which
*includes* the 105,522-row fine-tune pool every later point draws from. Plotting them as fraction 0
would make the curve's first segment a mixture of two changes — "gained fine-tune data" and
"changed test set" — and the segment from 0 to 1% is precisely where the RQ2 claim lives.

**Delivered:** TON_IoT is split **once**, stratified on the binary label and seeded from
`config.RANDOM_SEED = 42`, into a **permanent 105,521-row test half** and a 105,522-row fine-tune
pool. Every point on the curve is scored on that identical half, including a **re-measured
zero-shot point** under its own `run_id` (`phase7-recovery-f0.00`) using the same
UNSW-train-fitted models, unadapted. Budgets are fractions **of the pool**, not of the full target:
1% = 1,055 rows, 5% = 5,276, 10% = 10,552, 25% = 26,380, ceiling = 105,522. Draws are nested
(each budget is a superset of the smaller ones), so a non-monotone point cannot be a different
*sample* rather than a different *budget*.

**The re-measurement changes almost nothing, which is itself the useful result** — the frozen half
is representative of the era it was cut from, so nothing downstream rests on which half was drawn:

| model | ROC-AUC, frozen half | ROC-AUC, full 211,043 | Δ |
|---|---|---|---|
| `decision_tree` | 0.3519 | 0.3534 | -0.0015 |
| `random_forest` | 0.2121 | 0.2106 | +0.0015 |
| `svm` | 0.2111 | 0.2114 | -0.0003 |
| `scratch_logreg` | 0.2490 | 0.2489 | +0.0001 |
| `scratch_mlp` | 0.1847 | 0.1846 | +0.0000 |
| `dummy` | 0.5000 | 0.5000 | 0.0000 |

Largest F1 difference is 0.0013 on the same rows. Phase 6's rows stay Phase 6's and are **not**
re-logged under a Phase 7 `run_id`; the contrast above is printed on every run.
Leakage is guarded structurally, but the seal has to sit differently than in Phase 6: Phase 7
legitimately fits models on target labels, so `evaluate.sealed()` wraps **only** each model's
evaluation span, while the `Preprocessor` stays sealed for the whole run (it must never refit at
all). An instrumented run recorded **37 fits — 6 at n=140,272 (the source fits), 30 across the five
budgets, 1 for the freeze control at n=105,522 — zero at n=105,521 and zero `Preprocessor` fits.**
**Report home:** Methods (why the zero-shot point is measured twice) + Results (the curve's left
endpoint).

### 3.13 Adaptation is per-model and target-only, and each ceiling is its own mechanism (Phase 7)
Three decisions a grader could question, all made so that the **data budget is the only variable**
along the x-axis. Nothing is re-tuned per fraction: every model is re-instantiated from its locked
`TUNED_PARAMS`, and class weights stay on at every point (the pool is 76.31% attack).

**1. The MLP freezes *both* hidden layers, not just the first.** `ScratchMLP.fit(freeze_hidden=True)`
as committed in Phase 5 marks only `head_keys()` trainable, so the whole `(22, 44, 22)` stack keeps
its UNSW weights as a fixed feature extractor and **23 of the net's 2,025 parameters (1.1%)** move;
the head is *continued* from its fitted values rather than re-initialized. There is no
freeze-the-first-layer-only mode and the interface was **not** widened to add one — the head-only
variant is the strongest form of the data-efficiency claim (the modern budget buys a new decision
boundary in a 2015 feature space and nothing else). **What the freeze costs is measured, not
assumed:** at the full budget, head-only reaches ROC-AUC 0.9808 / F1 0.9663 against **0.9980 /
0.9861** for the same architecture retrained end to end on the same pool
(`phase7-recovery-ceiling-no_freeze`, one extra row, its own `run_id`) — so freezing costs
**0.0172 AUC / 0.0198 F1**, and the MLP is the only model whose ceiling is set by its mechanism
rather than by the data.

**2. `scratch_logreg` warm-starts; the classical models retrain on the target sample alone.**
Warm-starting is what `ScratchLogReg.fit(warm_start=True)` exists for. `DecisionTreeClassifier`,
`LinearSVC` and `DummyClassifier` expose no incremental interface at all, and
`RandomForestClassifier.warm_start` grows more trees on the *same* data rather than carrying a fit
onto new data, so "retrain on the small sample" (the stub's wording) is the mechanism available.
**Pooling (source + small target) was considered and rejected on two grounds:** it is not a
data-budget curve — at the 1% budget the fit would be 140,272 source rows against 1,055 target rows,
so the source era would dominate every point a reader cares about — and it does not converge to the
ceiling, so "how close does 25% get?" would compare against an asymptote the curve never approaches.

**3. Each model's ceiling is its own adaptation mechanism at the full budget** (fraction 1.0 of the
pool), which is what makes the curve continuous into its own upper bound. For the four classical
models that *is* a full target retrain; for `scratch_logreg` it is equivalent to one, because the
objective is convex and strictly so at `l2 = 1e-4`, so a warm start and a cold start reach the same
optimum; only the MLP's ceiling is mechanism-limited, which is what the control in (1) quantifies.
**Result (ROC-AUC on the frozen half, zero-shot -> 1% -> ceiling):** `random_forest` 0.2121 ->
0.9979 -> 0.9998, `decision_tree` 0.3519 -> 0.9862 -> 0.9984, `svm` 0.2111 -> 0.9843 -> 0.9881,
`scratch_logreg` 0.2490 -> 0.9859 -> 0.9884, `scratch_mlp` 0.1847 -> 0.8845 -> 0.9808. **Every
model clears the dummy's 0.5000 at the smallest budget tested**, i.e. 1,055 labelled modern flows
undo the RQ1 rank inversion, and the 25% budget closes 99.5-100% of the gap to each ceiling. Read
it against **§2.3** (the recovery generalizes; why 1% suffices is feature-space determinism, not
memorization) and against the dummy floor, which is F1 0.8656 at ROC-AUC
0.5000 on this half — an F1 recovering toward 0.87 would mean nothing at all.
**Report home:** Methods (adaptation design, one paragraph per mechanism) + Results (the curve).

### 3.14 `RandomForestClassifier.predict_proba` is not bit-reproducible — and it only showed up in a committed artifact (Phase 9)
**A real threat to the graded reproducibility claim, found by diffing rather than by reasoning.**
`RandomForestClassifier(n_jobs=-1).predict_proba` accumulates per-tree probabilities in whatever
order the worker threads finish, so **the same fitted forest returns scores differing by up to
4.4e-16 between two calls in one process.**

Nothing already committed could see it: every scalar in `reports/metrics.csv` is rounded to six
decimals (§`METRIC_DECIMALS`) and the confusion matrices count hard predictions. **A ROC curve can**,
because it resolves individual scores — the last-bit noise splits and re-merges tied thresholds, so
the vertex list came out different on every run and `reports/roc_curves.json` showed a spurious
`git diff` after each `./run.sh`. That is exactly the signal the committed-artifact convention exists
to produce, and it would have been invisible had the curves not been persisted.

**Fix:** scores are snapped to `ROC_SCORE_DECIMALS = 12` before the curve is built. `roc_auc` is
still computed from the **raw** scores, so **no logged number moves** — the snap only ever changes
which vertices the curve is drawn through. Measured across three separate processes, all six forest
curves are identical at 13, 12 and 11 decimals and **still vary at 14**, so the snap clears the noise
by a wide margin rather than merely exceeding it.

**The snap is conditional, and that is a measurement rather than caution.** The decision tree's leaf
fractions contain **110 pairs that are mathematically equal but land ~7e-18 apart** in float64;
`roc_auc_score` already scored those as distinct thresholds, so the committed AUC reflects the split
and merging them would move the curve's area by **2.24e-05** — twenty times past what the sidecar
promises. The snap is therefore applied only where it demonstrably does not move the AUC past
`ROC_CURVE_TOLERANCE = 1e-6`, and each stored curve records `scores_snapped` (**33 of 36** snapped).
**Verified:** three consecutive Phase 6 runs → byte-identical `roc_curves.json`.
**Report home:** none required — it is an engineering fix, not a scope deviation. Worth one sentence
in Methods **only** if the report claims bit-level reproducibility, which it should, since `./run.sh`
now does reproduce all 11 artifacts byte-identically.

### 3.15 ROC curves are stored as thinned integer counts, area-preservingly (Phase 9)
The 36 raw curves come to **432,667 vertices** — several MB of committed JSON for figures a few
hundred pixels wide. Two decisions keep the sidecar small without moving the number it illustrates.

**Counts, not rates.** A vertex is the integer `(false positives, true positives)` pair with the two
class sizes recorded once per curve, so `fpr = fp/n_neg` and `tpr = tp/n_pos` reconstruct sklearn's
output exactly and no rounding error is introduced on top of the thinning.

**Area-preserving simplification**, budget 512 vertices/curve → **13,838 stored (3.20%)**. The
objective matters, and three schemes were measured on the real curves at a comparable budget:

| scheme | worst \|Δ AUC\| |
|---|---|
| uniform spacing along the curve | 6.5e-05 |
| Douglas–Peucker (perpendicular distance) | 5.8e-06 |
| **sign-aware area greedy (`simplify_roc`)** | **1.5e-07** |

Perpendicular distance is the **wrong objective**: it bounds how far the drawn line strays, not how
much area that costs, and on a curve that is concave (or, cross-era, convex) throughout, every
residual carries the same sign and they accumulate. `simplify_roc` drops the vertex whose *signed*
trapezoid change is smallest, preferring the sign that pulls the running total back toward zero, so
they cancel. `roc_points()` recomputes the area over the kept vertices and **refuses to write** past
`ROC_CURVE_TOLERANCE = 1e-6`. Worst observed across all 36 stored curves: **5.03e-07**, itself an
upper bound since it also carries the sidecar's own 6-dp rounding. Each figure annotates the
**committed** `metrics.csv` scalar, and all 12 rendered curves agree with it at **exactly 0.0e+00**.
**Report home:** none — presentation detail. Relevant only if a reader asks whether the plotted curve
is the one behind the printed AUC; it provably is.

### 3.16 Three sidecar artifacts, because `METRICS_HEADER` is frozen (Phase 9)
`reports/metrics.csv`'s 14 columns are frozen and its key is `(run_id, model, regime)`. None of a
2×2 integer matrix, a ROC vertex list, or a per-family breakdown is a scalar metric, and the last has
a dimension the key cannot express at all. Widening the header for a figure's sake is precisely what
its own comment forbids. **Delivered:** three files beside the log —
`reports/confusion_matrices.json`, `reports/roc_curves.json`, and `reports/per_family_metrics.csv`
(its own frozen 18-column header, upsert-keyed on `(run_id, model, regime, family_set, family)`).

Each is written by the phase that already computed the quantity, **gated on the same `log` flag** as
the metrics upserts, sorted with floats rounded to `METRIC_DECIMALS`, and therefore byte-identical
across re-runs — the same contract `log_metrics` gives `metrics.csv`. A missing file raises naming
the command that produces it; nothing downstream recomputes, because that would mean re-fitting
eighteen models to redraw one figure.

`family_set` is load-bearing rather than decorative: `dos`, `scanning` and `backdoor` exist in
**both** the shared and the native vocabularies over different populations, and without that field
they would silently overwrite each other. Every family is scored **one-vs-normal** — an attack family
is all-positive, so an F1 over its own rows alone is a relabelling of recall — and each subset
carries its own `n_family`/`n_normal`/`positive_rate`.
**`reports/metrics.csv` was not modified by any of this:** md5
`116def31c6bca1bb788d4a86f9cc976e` before and after every Phase 9 pass.
**Report home:** none — repo mechanics. Named in Methods only if the report points a reader at the
committed artifacts.

### 3.17 `family_native` added to the harmonized parquets (Phase 9)
`COMMON_COLUMNS` gained a second family column, so both parquets went **14 → 15 columns** (UNSW
257,673 × 15, TON_IoT 211,043 × 15). The existing `family` is the **shared** map: every level without
a counterpart in the other era becomes NA, which is right for the cross-era comparison and useless
for a per-era one — **101,043 of TON_IoT's 211,043 rows are NA there**, including all 20,000 each of
`ddos`, `injection`, `password`, `ransomware` and `xss`. `family_native` is each dataset's own
delivered level, lower-cased and stripped and nothing else, so Phase 7's recovery breakdown over
TON_IoT's own eight 20,000-row types is expressible at all.

**No model sees it.** It is a label column; `Preprocessor` builds its matrix from an explicit
allowlist (`NUMERIC_FEATURES` 7 + `CATEGORICAL_FEATURES` 3 → d=22) and `_required_columns()` never
names it. Verified after the change: the fitted feature schema is unchanged at 22 columns with zero
family-derived names, and all 73 `metrics.csv` rows reproduced byte-identically from rebuilt
parquets. The two columns disagree by design where a shared name won: UNSW `Reconnaissance` is
`scanning` in `family` (TON_IoT's spelling, so the eras join) and `reconnaissance` in `family_native`.
**Report home:** Methods (feature alignment) — one clause, if the per-family figures are discussed.

---

## 4. Open items against rubric claims

- **One-command reproducibility — RESOLVED 2026-08-02, commit `ffae5d7`.**
  `run.sh` used to default `PYTHON` to bare `python3`, so a fresh clone failed with "required
  packages missing" unless the venv was activated or `PYTHON=.venv/bin/python` was passed. The error
  was clear, but the graded "one-command end-to-end run" claim leaned on it. It now defaults to
  `./.venv/bin/python` when that file exists and `PYTHON` is unset; an explicit `PYTHON=` override
  is unchanged. Verified with a bare `./run.sh` from a shell with no venv activated and no `PYTHON`
  set. Nothing left open here. **Report home:** none — this was an engineering fix, not a scope
  deviation, so it needs no mention in the report.

  Two engineering changes landed alongside it and are recorded in `Implementation-Plan.md` rather
  than here, for the same reason: `evaluate.log_metrics()` is now an upsert keyed on
  `(run_id, model, regime)` so re-running `run.sh` no longer doubles every row of
  `reports/metrics.csv`, and `run.sh --help` no longer spills the first three lines of script body.

---

## 5. Corrections log — 2026-08-05

This file had drifted ahead of `Implementation-Plan.md`, which is the failure the preamble warns
about. Recorded here so the same claims are not "restored" later from an older draft.

| § | Was | Now | Basis |
|---|---|---|---|
| 3.2 | "the cross-era run is evaluated **with and without** the feature" — it was not; only `proto` had been ablated | the ablation is **delivered** under `phase6-crossera-no_conn_state`, with its measured Δ-of-Δ | `reports/metrics.csv` carried no `no_conn_state` run_id; it now carries 12 rows |
| 3.7 | cited "Phase 6's `conn_state` ablation" as a reason to keep the val fold | clause removed — an ablation retrains from locked `TUNED_PARAMS` with no search, so it consumes no val fold even in principle | §3.9's design |
| 3.8 / 3.10 | Δ signs disagreed: §3.8 used `in − cross`, §3.10's table used `cross − in`, and the pipeline printed `in − cross` — so the document and a grader's own run disagreed on the headline sign | **one convention, stated in the preamble**: `Δ = in − cross` ("what was lost"). §3.10's table now reads `+0.7682` etc.; §3.8's `−0.1554` was already correct | `evaluate.metric_deltas()`; a stray inline `cross − in` in `_print_headline` was fixed to match |
| 3.8 | heading said the Dummy's F1 "inverts" across eras | "**rises**" — its F1 rises monotonically and its ROC-AUC does not move. "Inverts" is reserved for §3.10's rank inversion | two different phenomena were sharing one word |
| 3.9 | "the `other`-bucket artifact accounts for at most **~8%**" | **~12%** (0.0867 / 0.70) | the 0.087 Δ-of-Δ had been read as a percentage |
| 3.10 | TON `label = 1` covers "only `backdoor`/`ddos`/`dos`/`injection`/`mitm`" (81,043 rows) | **all nine** attack `type` levels, 161,043 rows | measured on `Train_Test_Network.csv`; the three-family limit applies only to the per-family breakdown |
| 3.10 | "the **stratified/prior** Dummy" | "the majority-class (**`most_frequent`**) Dummy" | `baselines.make_dummy`; §3.8 already said this correctly |
| 1.3, 3.10 | percentages quoted with no denominator, and on *different* bases between sections | bases now labelled — §1.3 is the UNSW **train fold**, §3.10 is **UNSW-test** | both were arithmetically right and mutually inconsistent in presentation |

## 6. Corrections log — 2026-08-06 (Phase 9)

| § | Was | Now | Basis |
|---|---|---|---|
| 1.3 | "**20.5%** of UNSW _train-fold_ attack rows (24,501 of 119,341)" — the count is the training **partition**'s, not the seeded train fold's, and neither is the frame the per-family figure scores | all four bases tabled; **18.02%** (UNSW-**test**) named as the number to quote beside the figure | measured 2026-08-06; the figure's in-distribution regime is UNSW-test, n=82,332 |
| new §3.14 | — | `RandomForestClassifier.predict_proba` is not bit-reproducible; conditional score snap | three-process measurement; `roc_curves.json` diffed on every run before the fix |
| new §3.15 | — | ROC curves stored as area-preservingly thinned integer counts | three simplification schemes measured; worst stored drift 5.03e-07 |
| new §3.16 | — | three sidecar artifacts, and why they are not columns | `METRICS_HEADER` is frozen and its key has no family dimension |
| new §3.17 | — | `family_native` in `COMMON_COLUMNS`; parquets 14 → 15 columns | no model sees it — `Preprocessor` builds from an allowlist, d=22 unchanged |

**Also checked this pass:** the `SHARED_FAMILIES` `"Backdoors"`/`"Backdoor"` bug CLAUDE.md warned
about **is not present** — `schema_map.py` carries the singular key with a comment against
"restoring" the plural, and it fires on 2,329 UNSW rows / 20,000 TON_IoT rows. `plots.py` now
*derives* the expected family set from `SHARED_FAMILIES`, so a fourth family (or a vanished third)
raises rather than drawing silently.

**Finding worth the report, not a deviation:** cross-era, **14 of 15 (family, model) pairs** among
the five real models fall below chance. The one exception is `decision_tree` on `backdoor`
(AUC 0.6717); it is also near-chance on `dos` (0.4999). §3.10's inversion is present in all three
shared families and is not one family's artifact.

**Checked and deliberately NOT changed.** §3.4's "50,872 `src_bytes` rows (24.1%)" was queried as
contradicting §3.1's 8.10% zero-rate. It does not: the two measure different quantities. The seeded
**train fold** (140,272 rows — the frame the `Preprocessor` was fit on, and the basis §3.4 names) has
`src_bytes` minimum **46**, and 50,872 TON_IoT rows (24.105%) fall below it; §3.1's 8.10% is TON's
*zero*-rate. 33,778 TON rows sit strictly between 0 and 46, which is the whole difference. The full
`split == "train"` frame has minimum 28 — using that basis is what produces the spurious 17,094 /
8.10% figure. Both entries stand as written.

## 7. Corrections log — 2026-08-06 (final code audit)

Three staleness fixes from an end-of-build pass that checked the delivered repo against the course
handout, `Proposal-Final.md` and `Implementation-Plan.md`. **No delivered number changed and no code
was touched** — 38 tests green, `reports/metrics.csv` still md5
`116def31c6bca1bb788d4a86f9cc976e`.

| § | Was | Now | Basis |
|---|---|---|---|
| preamble | "`Implementation-Plan.md` (**mirrored by hand to HackMD**)" | clause dropped | HackMD was retired 2026-08-06 and its last state is stale; this was the final surviving pointer to it, and it sat in a committed collaborator-facing file |
| 1.4 | "**Status:** … cut first if time is short. **May not land**" — contingent | "**not delivered** — cut on the timeline", plus the two claims RQ3's absence weakens | Phase 8 is a settled cut, not a pending one: neither hard gate was attempted, and the `lab/` directory was removed outright in the same pass |
| 1.3 | silent on the proposal's conditional per-family confusion matrices | states what was delivered instead (per-family F1/ROC-AUC across 369 rows + Figs 5–6) and why a one-vs-normal 2×2 is the weaker unit | proposal Metrics: "per-attack-family confusion matrices **if time allows**" — conditional, so not a breach, but a grader reading both documents will look for it |

**Verified green in the same pass, against the handout's three code constraints.** ≥1 classifier from
first principles — two, and importing either loads no `sklearn`/`pandas`/`torch`/`matplotlib` (the
heavy imports are function-local to the CLI driver paths, annotated as such). No LLM or transformer
anywhere. One-command reproducibility: `./run.sh` rebuilds all 11 committed artifacts
byte-identically from the raw CSVs. Zero `NotImplementedError`/`TODO`/`FIXME`/`...` bodies remain in
`src/`, `tests/` or `run.sh`; the only bare `pass` (`config.py`'s pre-numpy `ImportError` arm) and the
only single-`raise` body (`evaluate.py`'s `sealed()` closure) are both intentional guards. The `lab/`
directory was deleted in the same pass, so **every tracked path in the repo is delivered work** —
there is no scaffold left standing for a phase that will not run.

**What is left is not code.** The 6-page report, the ≤6-min deck, the pre-recorded `./run.sh` demo,
and grader access to the repo. Every phase this file documents is delivered.
