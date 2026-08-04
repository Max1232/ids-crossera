# Deviations from the approved proposal

_Consolidated record of where the delivered project departs from the scope promised in
`Proposal-Final.md`, plus the substantive methodological decisions a reader could reasonably
question. Last updated: 2026-08-04 (content through Phase 7; §2.3, §3.12 and §3.13 added — the
RQ2 recovery curve, the frozen test set, and the unseen-vector check that shows the curve is
generalization rather than memorization)._

## What this file is — and is not

This is a **stable diff against the approved proposal**, which is frozen. It exists so the
report's Methods and Data & Experiments sections can justify every departure without
reconstructing the reasoning in August — each entry names the evidence and the report section it
feeds.

It is **not** a source of authority and **not** a log of routine engineering fixes:

- Scope is owned by `Proposal-Final.md`. The build and its per-phase deviations are owned by
  `Implementation-Plan.md` (mirrored by hand to HackMD). This file consolidates and points back;
  it never overrides them.
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
the earlier shared-family list was wrong. The three families cover **20.5%** of UNSW attack rows
and **37.3%** of TON_IoT's; say that limit out loud.
**Report home:** Data & Experiments + Results (per-family figure caveat).

### 1.4 RQ3 live-malware probe — in scope as optional, cut-first
**Promised:** optional stretch. **Status:** gated behind written instructor authorization + a
Northeastern policy check + a verified air-gap, and cut first if time is short. May not land; the
public-dataset core (Phases 0–7, 9) is a complete project without it.
**Report home:** only if executed; otherwise omit.

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

### 3.2 Connection state collapsed to a coarse three-way set
UNSW `state` (Argus codes) and TON_IoT `conn_state` (Zeek codes) share **zero tokens**, so no
lexical alignment is possible; both are hand-collapsed to completed / reset / no-response. `reset`
is 0.05% of UNSW vs 23.7% of TON_IoT, so the cross-era run is evaluated **with and without** the
feature. **Report home:** Methods + Results ablation.

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
MLP width/depth tuning and Phase 6's `conn_state` ablation, and the marginal 35,069 rows carry less
information than their count suggests given the ~52% duplicate-vector rate in the 11-column subspace.
**Report home:** Data & Experiments (state it as a choice, not an oversight).

### 3.8 The Dummy floor's F1 is high, and inverts across eras — report it in both regimes
Not a deviation but a reporting obligation Phase 4 surfaced. UNSW-**test** is 55.06% *attack*, so
`most_frequent` predicts "attack" everywhere and posts **F1 = 0.7102 at recall 1.0 with ROC-AUC
exactly 0.5000** and macro-F1 0.3551 — a high-looking F1 from a model with zero discriminative power.
This is the cleanest single argument in the project for why the headline is neither accuracy nor raw
F1. It also *inverts*: TON_IoT is 76.31% attack, so the same trivial model scores **higher** there.
Phase 6 must therefore log the Dummy in **both** regimes, or a reader cannot separate the prevalence
artifact from drift. **Discharged in Phase 6, and the number is now measured:** the Dummy's F1
**rises 0.7102 → 0.8656** cross-era (Δ = **−0.1554**) at ROC-AUC exactly 0.5000 in both regimes —
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
**Result:** the objection does not survive. Δ ROC-AUC moves by between **−0.087 and +0.061** across
the six models, and for three of them the ablated model degrades *further* — the `other`-bucket
artifact accounts for at most ~8% of a ~0.7 collapse.
**Report home:** Methods (ablation design — state why a test-time mask was rejected) + Results.

### 3.10 Cross-era ROC-AUC lands *below* 0.5 — the ranking inverts (Phase 6)
Not a deviation but a finding that needs stating before someone reads it as a bug. **Every real
model inverts.** In-distribution ROC-AUC spans **0.8811–0.9788**; cross-era it spans
**0.1846–0.3534** — below the Dummy's exact 0.5000. The 2015-learned score ranking does not merely
stop working on 2019–20 traffic, it runs backwards, so the detector is *anti-correlated* with ground
truth on TON_IoT. This is a **rank inversion, not a decay**, and the report has to say so.

| model | in-distribution | cross-era | Δ ROC-AUC |
|---|---|---|---|
| `random_forest` | 0.9788 | 0.2106 | −0.7682 |
| `scratch_mlp` | 0.9621 | 0.1846 | −0.7775 |
| `decision_tree` | 0.9648 | 0.3534 | −0.6115 |
| `svm` | 0.8846 | 0.2114 | −0.6732 |
| `scratch_logreg` | 0.8811 | 0.2489 | −0.6321 |
| `dummy` | 0.5000 | 0.5000 | 0.0000 |

**Evidence:** `reports/metrics.csv`, rows `phase4-baselines,*,in_distribution` and
`phase6-crossera,*,cross_era`; the two scratch models' in-distribution halves are logged under
`phase6-crossera` because they have no Phase 4 entry. Sub-0.5 AUC is also the exact signature of a
flipped label, so polarity was verified directly against both parquets before the inversion was
accepted: `label = 0` ⟺ normal and `label = 1` ⟺ attack in both, with TON_IoT's `label = 1` covering
only the `backdoor`/`ddos`/`dos`/`injection`/`mitm` rows. It is not a wiring fault.

**Mechanism: response-side feature inversion between eras.** An earlier draft of this entry blamed
`zero_duration`; **that attribution is withdrawn.** The flag fires on 2.5% of UNSW *normal* rows
against 0.08% of UNSW *attack* rows, but 25.4% vs 29.4% on TON_IoT normal vs attack — near-flat
across TON_IoT's classes — and its standalone ROC-AUC is 0.488 in-distribution and 0.520 cross-era.
A feature that close to random cannot drive a ~0.77 AUC swing. What does drive it is the **response
side of the flow flipping sign between the two datasets**, at the class median:

| feature | UNSW normal | UNSW attack | TON_IoT normal | TON_IoT attack |
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

**Robustness — the two obvious deflations of the result are both measured, and neither holds.**
- *"It's just the `proto` shortcut going inert."* It is not. The ablation is a **retrain-without at
  d=18, not a test-time mask** (design and the full argument: **§3.9** — masking the protocol
  one-hots on a model trained with them evaluates it on inputs no era produces, confounding "the
  learned shortcut went inert" with "the model was perturbed off its training manifold"). Matched
  pairs are refit on the UNSW train fold and run through **both** regimes, so the reported quantity
  is a **difference of deltas**: it spans **−0.087 to +0.061** against a ~0.7 collapse, and three of
  the six models degrade *further* without the protocol features.
- *"It's a prevalence artifact."* Not on ROC-AUC. The stratified/prior Dummy holds ROC-AUC exactly
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
