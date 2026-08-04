# Deviations from the approved proposal

_Consolidated record of where the delivered project departs from the scope promised in
`Proposal-Final.md`, plus the substantive methodological decisions a reader could reasonably
question. Last updated: 2026-08-03 (content through Phase 6; §3.5 ratified, §3.9–3.10 added)._

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
the plan) because the zero case inverts across eras.
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
Not a deviation but a finding that needs stating before someone reads it as a bug. Every real model
scores **ROC-AUC 0.13–0.35 on TON_IoT** against 0.88–0.98 in-distribution, i.e. below the Dummy's
exact 0.5000: the 2015-learned score ranking does not merely stop working on 2019–20 traffic, it
inverts. Sub-0.5 AUC is also the exact signature of a flipped label, so that was checked directly —
TON_IoT's harmonized `label = 1` covers only the `backdoor`/`ddos`/`dos`/`injection`/`mitm` rows and
`label = 0` only `normal` — and the polarity is correct. The mechanism to investigate in Phase 9 is
the set of features whose *meaning* inverts across eras (`zero_duration`: 1.52% of UNSW rows, 99% of
them normal, against 28.44% of TON_IoT rows, only 21% normal), not a wiring fault.
**Report home:** Results — the claim is "the detector inverts", which is stronger than "it degrades"
and must be argued rather than buried.

### 3.11 Gradient boosting excluded
`xgboost` was removed (640 MB of CUDA libs, zero references in `src/`, and boosting appears nowhere
in the approved proposal). If a boosting baseline is ever wanted, use sklearn's
`HistGradientBoostingClassifier`. **Report home:** none unless added.

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
