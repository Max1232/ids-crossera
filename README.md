# ids-crossera

**Measuring and Correcting Temporal Concept Drift in Network Intrusion Detection**
CS4100 (Foundations of AI), Summer B 2026 

_Group 10: Max Cossill & Enqi Zhang (Steven)_

Machine-learning intrusion detection systems are almost always trained and tested on a single
static dataset captured at one point in time. That means the accuracy numbers you see published
describe only *in-distribution* performance — how well a detector does on traffic that looks like
what it already saw. Real networks don't hold still: protocols change, services change, and attack
tooling changes. That gap has a name — **concept drift** — and most published NIDS work never
measures it.

This project measures it directly. We train a detector on **UNSW-NB15** ([2015](https://research.unsw.edu.au/projects/unsw-nb15-dataset)), test it
unchanged on the newer **TON_IoT Network** dataset ([~2019–2020](https://research.unsw.edu.au/projects/toniot-datasets)) to see how far it falls, and then
measure how little modern labeled data it takes to claw the performance back. The goal is two
honest numbers: **how much the detector degrades**, and **how cheaply that can be corrected**.

Neither dataset is ours. Both are the work of **Nour Moustafa and co-authors at UNSW Canberra at
ADFA**, used here under their academic-research terms — see
[Datasets, credit and license](#datasets-credit-and-license) for the citations those terms require.

## Research questions

- **RQ1 — drift:** How much does a 2015-trained detector degrade on newer traffic?
- **RQ2 — correction:** How little modern labeled data does transfer learning need to recover that
  performance?
- **RQ3 — stretch (optional):** Does the model flag traffic from live, in-house-detonated 2026
  malware — a full decade forward from the training data? **Not run** — scoped as optional and cut on
  the timeline; see [`deviations.md`](deviations.md) §1.4. RQ1 and RQ2 are answered without it.

## Limitations

UNSW-NB15 is general enterprise traffic; TON_IoT is IoT/IIoT. The UNSW→TON_IoT delta therefore
bundles *temporal drift* (attacks evolving over ~4–5 years) with an *enterprise-vs-IoT domain
shift* — it is not a clean time-only experiment, and we don't claim it is. We treat the measured
delta as an **upper bound on pure temporal drift**, and restrict the per-attack-family analysis to
families both datasets actually share so the comparison stays fair. The true decade-forward test
comes only from the optional RQ3 2026 captures.

Class balance is a second confound on any Δ. **UNSW-test is 45% normal; TON_IoT is 24% normal**, so
prevalence-sensitive metrics move on that difference alone — every reported delta names both test
sets' normal share. And TON_IoT's published `Train_Test_Network.csv` is not the file its own
documentation describes: it delivers **211,043 rows with 50,000 normal** against a documented
**461,043 rows with 300,000 normal**, i.e. 250,000 normal rows short. That is what upstream
publishes — two independent downloads are byte-identical — so this balance is part of the
measurement, not a cosmetic detail to be fixed by re-downloading.

Two features the proposal promised aren't in the shared subspace. **TTL statistics** are absent from
the flow CSVs and recoverable only from the raw captures — reprocessing those is out of scope. The
evidence is a header scan, not an inference: all 44 delivered TON_IoT columns were checked and not
one is a TTL field of any kind (zero substring matches for `ttl` or `hop`). The underlying reason is
that they are Zeek `conn.log`-derived and Zeek exports no per-flow IP TTL.
Dropping them arguably strengthens the result: UNSW-NB15's attack traffic came from an IXIA
PerfectStorm appliance while its normal traffic came from different hosts, which makes initial TTL a
*generator fingerprint* rather than an attack signal, so including it would have inflated
in-distribution scores. **Pre-computed rates** are derived from duration and counts instead, since
UNSW's `sload`/`dload` are in *bits* per second — and carry an Argus `(spkts-1)/spkts` correction, so
they cannot be cross-checked against a naive `sbytes*8/dur` — while TON_IoT has no rate column at
all. Both deviations are documented in the implementation plan.

The **`proto` collapse** invites the same objection, and this one is measured. 18.31% of UNSW
training rows (32,097) use a protocol TON_IoT never contains — `unas`, `arp`, `ospf`, `sctp` and a
long Argus tail — and those rows are **91% attack** (chiefly Exploits 13,080, DoS 9,625,
Reconnaissance 1,805, Fuzzers 1,478, Backdoor 1,446; only 2,942 Normal). So once both sides collapse
to `{tcp, udp, icmp, other}`, a model learns "`other` → attack" from a bucket that is 91% attack in
training and **0% of rows at test time**: part of the RQ1 drop would be a learned signal going inert
rather than attacker evolution. That is the same generator-fingerprint argument that justifies
dropping TTL, so `proto` gets an ablation — the cross-era evaluation run once with it and once
without, both halves retrained so the two conditions are comparable (see
[Status](#status) for the measured answer: the objection does not survive the test).
`icmp` has the mirror problem: **15 UNSW train rows** (0.01%) against 281 in TON_IoT, so
that level is effectively untrained yet active cross-era.

## Status

**Phases 0–7 are complete:** Phase 0 repo scaffold and pinned environment, Phase 1 data download and
first look, Phase 2 feature alignment, Phase 3 preprocessing, Phase 4 in-distribution baselines,
Phase 5 both from-scratch models, Phase 6 the zero-shot cross-era evaluation (RQ1, the primary
result), Phase 7 the transfer-learning recovery curve (RQ2, the secondary result), Phase 9 the
figures. **All the code is written**: 63 tests green, and `./run.sh` rebuilds all eleven committed
artifacts byte-identically from the raw CSVs in about five minutes. What remains of Phase 9 is the
write-up — report, deck and demo recording — see [Implementation steps](#implementation-steps) for
what it owes. **Phase 8 (RQ3) was cut on the timeline**; nothing for it was ever built.

Where the delivered build departs from the approved proposal, and why, is recorded in
[`deviations.md`](deviations.md).

All three CSVs are extracted and verified — MD5s and row counts in
[`data/README.md`](data/README.md) — and the schema comparison, measured from the delivered files
rather than quoted from documentation, is in `reports/schema_catalogue.md` with a machine-readable
`reports/schema_catalogue.csv` companion (one row per delivered column).

`src/schema_map.py`'s mapping constants are corrected against that catalogue, and
`build_common_frames()` emits both harmonized parquets (UNSW 257,673 × 15, TON_IoT 211,043 × 15).
`src/preprocess.py`'s `Preprocessor` is fit on the UNSW **train fold only** and serialized to
`data/processed/preprocessor.joblib`, giving the 22-column feature schema every later phase consumes;
`fit()` raises rather than silently accepting UNSW-test, TON_IoT, or the concatenated frame.
`src/models/baselines.py` trains the four classical baselines and records the in-distribution ceiling
on UNSW-test:

| Model | F1 | ROC-AUC |
| --- | --- | --- |
| Random Forest | **0.9145** | **0.9788** |
| Decision Tree | 0.9099 | 0.9648 |
| Linear SVM (`LinearSVC`) | 0.7883 | 0.8846 |
| Dummy (majority-class floor) | 0.7102 | 0.5000 |

`src/models/scratch_logreg.py` and `src/models/scratch_mlp.py` are the from-scratch pair — pure
numpy, hand-written gradients and backprop, no sklearn/torch/autograd in either model's import
graph. Scored on the UNSW **val** fold (not UNSW-test, which stays sealed until Phase 6):

| From-scratch model | F1 | ROC-AUC | vs. sklearn equivalent |
| --- | --- | --- | --- |
| Logistic regression (full-batch GD, 4,346 iters) | 0.8805 | 0.9337 | −0.19 F1 / −0.16 AUC points |
| MLP `(22, 44, 22, 1)` (mini-batch SGD, 40 epochs) | 0.9351 | 0.9832 | −1.84 F1 / −0.11 AUC points |

The MLP's wider F1 gap is the class weighting, not the implementation: `MLPClassifier` has no
`class_weight` parameter, so the reference is necessarily unweighted, and the unweighted control arm
of the same scratch model closes the gap to 0.52 F1 / 0.42 AUC points while giving up 1.3 points of
balanced accuracy. Both models are verified by `tests/` — including a finite-difference gradient
check on the MLP's backprop, worst relative error 2.0e-8.

> **Convention, and it is inverted relative to sklearn:** on **both** from-scratch models
> `class_weight=None` means **balanced** (inverse-frequency), not unweighted — the explicit
> unweighted control is spelled `{0: 1.0, 1: 1.0}`. Class weights on every model from the start is a
> project constraint, and the failure it guards against is silent, so forgetting the argument must
> not be able to produce it.

`src/evaluate.py`'s `run_regimes()` then carries each of those six models, unchanged, from UNSW-test
onto TON_IoT — **RQ1, the headline result.** Every model is fit once on the UNSW train fold and
sealed against refitting for the span of both evaluations, and TON_IoT goes through the frozen Phase
3 preprocessor transform-only:

| Model | ROC-AUC in-dist. | ROC-AUC cross-era | **Δ AUC** | F1 in-dist. | F1 cross-era | **Δ F1** |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.9788 | 0.2106 | **+0.7682** | 0.9145 | 0.3050 | +0.6095 |
| Scratch MLP | 0.9621 | 0.1846 | **+0.7775** | 0.8953 | 0.2821 | +0.6132 |
| Decision Tree | 0.9648 | 0.3534 | **+0.6115** | 0.9099 | 0.5112 | +0.3987 |
| Linear SVM | 0.8846 | 0.2114 | **+0.6732** | 0.7883 | 0.2287 | +0.5597 |
| Scratch logreg | 0.8811 | 0.2489 | **+0.6321** | 0.7832 | 0.3032 | +0.4800 |
| Dummy (majority-class) | 0.5000 | 0.5000 | +0.0000 | 0.7102 | **0.8656** | **−0.1554** |

_in-distribution = UNSW-test, n=82,332, **44.94% normal**; cross-era = TON_IoT, n=211,043, **23.69%
normal**._

Two things have to be read together there. Every real model lands **below the 0.5000 no-skill line
cross-era** — the learned ranking does not merely stop working, it *inverts*, which is a stronger
claim than "performance degrades" and is the finding Phase 9 has to explain rather than smooth over.
(Checked, because sub-0.5 AUC is also the exact signature of a flipped label: TON_IoT's harmonized
`label = 1` covers all nine attack `type` levels — `backdoor`, `ddos`, `dos`, `injection`,
`password`, `ransomware`, `scanning`, `xss` and `mitm`, 161,043 rows — and `label = 0` only
`normal`, so the polarity is right and the inversion is real.) And the Dummy moves the *other* way:
its F1 **rises 0.7102 → 0.8656** cross-era at ROC-AUC exactly 0.5000, purely because the target era
is 76.31% attack rather than 55.06%. That 0.1554 F1 gain is the size of the prevalence artifact, and
it is why the drift claim leads with ROC-AUC and never with F1 alone.

**Two ablations run as second and third conditions**, each under its own `run_id` and each a matched
in-distribution/cross-era pair **retrained on the train fold with one feature's one-hots removed**
(d=18) rather than masked at test time — so the comparable quantity is the difference of the deltas.
Both land at the same width and are *different experiments*: only the `run_id` and the `notes` column
tell them apart, never `d`.

- **`phase6-crossera-no_proto`** — the `proto` ablation. The with-protocol drop almost entirely
  survives: Δ-of-Δ AUC spans −0.087 to +0.061 across the six models, so the `other`-bucket artifact
  accounts for at most ~12% of a ~0.7 collapse, and for three models the ablated model drops
  *further*.
- **`phase6-crossera-no_conn_state`** — the `conn_state` ablation. This one matters because the
  Argus↔Zeek state collapse is *ours* rather than the datasets': the two vocabularies share zero
  tokens, and the collapse is badly asymmetric across eras (`reset` 0.0421% of UNSW train rows vs
  23.6757% of TON_IoT's; the `other` bucket 0.0057% vs 11.0556%). Same verdict: Δ-of-Δ AUC spans
  −0.0427 to +0.0535, at most ~7.6% of the collapse, and three of six degrade *further* without it.

RQ1's degradation is neither the protocol collapse nor our connection-state collapse.

`src/transfer.py` then answers **RQ2: how little modern labelled data undoes that.** TON_IoT is
split **once**, stratified on the binary label and seeded from `RANDOM_SEED`, into a **permanent
105,521-row test half** and a 105,522-row fine-tune pool; every point below — the re-measured
zero-shot point included — is scored on that identical half, so the data budget is the only thing
moving along the x-axis. Budgets are fractions **of the pool** (1% = 1,055 rows, 5% = 5,276,
10% = 10,552, 25% = 26,380, ceiling = the whole pool), the fine-tune rows are disjoint from the test
half on **row indices**, and nothing is re-tuned per fraction. Adaptation is per model: the MLP
retrains its output head with **both hidden layers frozen** (23 of 2,025 parameters), the
from-scratch logistic regression warm-starts from its UNSW weights, and the classical models are
refit on the target sample (none has an incremental interface that preserves a fit).

**ROC-AUC on the frozen TON_IoT test half** (n=105,521, **23.69% normal** — the same balance as the
full target frame):

| Model | 0% (zero-shot) | 1% | 5% | 10% | 25% | ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.2121 | **0.9979** | 0.9988 | 0.9988 | 0.9993 | 0.9998 |
| Decision Tree | 0.3519 | **0.9862** | 0.9920 | 0.9941 | 0.9969 | 0.9984 |
| Linear SVM | 0.2111 | **0.9843** | 0.9869 | 0.9877 | 0.9880 | 0.9881 |
| Scratch logreg | 0.2490 | **0.9859** | 0.9878 | 0.9884 | 0.9885 | 0.9884 |
| Scratch MLP (head only) | 0.1847 | **0.8845** | 0.9598 | 0.9689 | 0.9766 | 0.9808 |
| Dummy (majority-class) | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

F1 over the same budgets: RF 0.3061 → 0.9923 → 0.9977, DT 0.5100 → 0.9882 → 0.9952, SVM 0.2292 →
0.9500 → 0.9626, logreg 0.3035 → 0.9527 → 0.9637, MLP 0.2825 → 0.7701 → 0.9663, and the Dummy sits
flat at **0.8656** — which is the number every recovered F1 has to be read against, since an F1
climbing toward 0.87 on a 76.31%-attack set may have recovered nothing.

**The RQ1 inversion is undone by 1,055 labelled modern flows.** Every model clears the no-skill
0.5000 line at the smallest budget tested, and the 25% budget closes 99.5–100% of each model's gap
to its own ceiling, so the curve is flat after 1% rather than climbing. Two caveats travel with
that, both measured. **The target era is highly redundant, but the recovery generalizes:** only
46.4% of the frozen test half's rows are distinct feature vectors and 35.8% of them appear
*verbatim* inside the 1% draw, so the same 1%-budget forest was re-scored on only the 64.22% of the
half (67,769 rows, attack share 0.7596) whose exact 22-feature vector is **absent** from the draw —
0.9959 AUC / 0.9882 F1 there against 0.9979 / 0.9923 on all rows, i.e. memorization is worth 0.0021
AUC against a recovery of 0.786. What the redundancy measures instead is that the label is close to
a deterministic function of these 22 features: just 15 of 92,438 distinct vectors carry both labels,
and a per-vector majority-vote lookup over them would score 99.8953% accuracy on the whole target
frame (221 errors in 211,043). That is why ~1% of the target labels suffices, and why 1% is a lower
bound on the labelling effort a genuinely overlapping feature space needs. **And the MLP's freeze
costs something:** at the full budget, head-only reaches 0.9808 AUC against 0.9980 for the same
architecture retrained end to end on the same pool (logged as `phase7-recovery-ceiling-no_freeze`),
so 0.0172 AUC is the price of holding the 2015 feature space fixed. The zero-shot point is
re-measured on the frozen half rather than taken from Phase 6 — those rows were scored on all
211,043 rows, which *include* this phase's fine-tune pool — and the two agree to within 0.0015 AUC,
which is what makes the frozen half representative of the era it was cut from. Both decisions are
`deviations.md` §3.12 / §3.13, the feature-space determinism caveat is §2.3.

`./run.sh` reproduces Phases 2 → 4, 6, 7 and 9 from the raw CSVs in about five minutes, and
re-running leaves `reports/metrics.csv`, its three sidecars and all seven figures byte-identical.
Phase 5 is the one step deliberately not wired into it: the scratch models' `reports/metrics.csv`
rows belong to Phase 6's regime run under its `run_id` convention (and Phase 6 fits the same locked
models anyway), so `python -m src.models.scratch_logreg` / `scratch_mlp` print their val scores and
log nothing. `src/plots.py` closes the loop — it opens the four committed artifacts **read-only**,
cross-checks them against each other, and writes `reports/figures/` plus the caption-and-provenance
index beside them; it fits, transforms and re-derives nothing, so no figure can disagree with the
table it illustrates.

### Adjusting a figure

At half-page width, *where* a legend sits is a judgement an eye makes better than a hardcoded
`loc=`. But a hand-edited PNG cannot be the artifact the report embeds — the moment an image editor
touches it, `./run.sh` no longer reproduces what a grader sees. So placement is committed **data**
rather than a manual step:

```bash
python -m src.plots --tune roc_curves   # drag any legend or dashed-line label, press s to save
python -m src.plots                     # re-render; the tuned position is now what run.sh produces
```

`--tune` opens the figure in a real window with its legends and reference-line labels draggable and
writes what you moved to `reports/figures/layout.json`, which every later render reads — so the
tuning is captured once and reproduced deterministically forever after. Positions only: styling stays
in `src/plots.py`, where it is documented. Deleting an entry falls back to the coded default, and
deleting the file falls back to all of them. `python -m src.plots --list` names the seven figures.

For a change no position can express, `--vector` writes editable SVG and PDF (real text, not
outlines) into `reports/figures/editable/` for Inkscape or Illustrator. That directory is
git-ignored and is scratch: the committed figure is always the PNG the pipeline renders, so anything
done there has to come back as either a `layout.json` entry or a change to `src/plots.py`.

Read the Dummy row as the warning it is: UNSW-test is 55.06% *attack*, so a `most_frequent`
classifier predicts "attack" everywhere and posts F1 = 0.7102 at recall 1.0 with **ROC-AUC exactly
0.5000**. It has no discriminative power whatsoever. That is why this project's headline is F1 and
ROC-AUC together and never accuracy.

## Setup

Python 3.11 or 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

Datasets are **not** committed (large; git-ignored). Download them first — see
[`data/README.md`](data/README.md) — then run the full pipeline end to end:

```bash
./run.sh
```

Reproducibility: a single global `RANDOM_SEED = 42` (see `src/config.py`) is used for numpy,
stdlib random, and all splits; dependencies are pinned in `requirements.txt`.

`run.sh` uses `./.venv/bin/python` when that venv exists, so a fresh clone needs no activation step;
set `PYTHON=` to override it (`PYTHON=python3.12 ./run.sh`).

## Datasets, credit and license

Neither dataset is ours, and neither is redistributed here — `data/raw/` is git-ignored and every file
is fetched from the authors' own project pages ([`data/README.md`](data/README.md) records exactly
what was pulled). Both come out of **UNSW Canberra at ADFA**:

| Dataset | Role here | Credit |
| --- | --- | --- |
| [**UNSW-NB15**](https://research.unsw.edu.au/projects/unsw-nb15-dataset) (2015) | source era — every model is trained on it | Nour Moustafa and Jill Slay |
| [**TON_IoT Network**](https://research.unsw.edu.au/projects/toniot-datasets) (~2019–2020) | target era — the drift and the recovery are measured on it | Nour Moustafa, with the co-authors of the papers below |

Both are granted **free for academic research purposes in perpetuity**; commercial use must be agreed
with the authors, who assert copyright. The condition attached to that grant is **citation**, and in
neither case is it a single reference: the UNSW-NB15 page requires the **five** papers below, and the
TON_IoT page requires **all eight** of its own. Both lists are reproduced here in full, with
bibliographic details verified against Crossref rather than transcribed from a secondary source.

The report cannot carry all thirteen — its 6-page limit includes references — so it cites the
load-bearing subset inline, at minimum UNSW-NB15's dataset paper and evaluation paper (items 1 and 2
below, the two the license names), and points here for the rest. The presentation owes the same
credit on a slide.

### UNSW-NB15 — the five citations its license requires

1. N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection
   systems (UNSW-NB15 network data set)," *2015 Military Communications and Information Systems
   Conference (MilCIS)*, 2015, pp. 1–6.
   [doi:10.1109/MilCIS.2015.7348942](https://doi.org/10.1109/MilCIS.2015.7348942)
2. N. Moustafa and J. Slay, "The evaluation of Network Anomaly Detection Systems: Statistical
   analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set," *Information
   Security Journal: A Global Perspective*, vol. 25, no. 1–3, pp. 18–31, 2016.
   [doi:10.1080/19393555.2015.1125974](https://doi.org/10.1080/19393555.2015.1125974)
3. N. Moustafa, J. Slay and G. Creech, "Novel Geometric Area Analysis Technique for Anomaly Detection
   Using Trapezoidal Area Estimation on Large-Scale Networks," *IEEE Transactions on Big Data*,
   vol. 5, no. 4, pp. 481–494, 2019.
   [doi:10.1109/TBDATA.2017.2715166](https://doi.org/10.1109/TBDATA.2017.2715166)
4. N. Moustafa, G. Creech and J. Slay, "Big Data Analytics for Intrusion Detection System:
   Statistical Decision-Making Using Finite Dirichlet Mixture Models," in *Data Analytics and
   Decision Support for Cybersecurity*, Springer, Cham, 2017, pp. 127–156.
   [doi:10.1007/978-3-319-59439-2_5](https://doi.org/10.1007/978-3-319-59439-2_5)
5. M. Sarhan, S. Layeghy, N. Moustafa and M. Portmann, "NetFlow Datasets for Machine Learning-Based
   Network Intrusion Detection Systems," in *Big Data Technologies and Applications (BDTA/WiCON
   2020)*, LNICST vol. 371, Springer, Cham, 2021, pp. 117–135.
   [doi:10.1007/978-3-030-72802-1_9](https://doi.org/10.1007/978-3-030-72802-1_9)

### TON_IoT — the eight citations its license requires

1. N. Moustafa, "A new distributed architecture for evaluating AI-based security systems at the edge:
   Network TON_IoT datasets," *Sustainable Cities and Society*, vol. 72, art. 102994, 2021.
   [doi:10.1016/j.scs.2021.102994](https://doi.org/10.1016/j.scs.2021.102994)
2. T. M. Booij, I. Chiscop, E. Meeuwissen, N. Moustafa and F. T. H. den Hartog, "ToN_IoT: The Role of
   Heterogeneity and the Need for Standardization of Features and Attack Types in IoT Network
   Intrusion Data Sets," *IEEE Internet of Things Journal*, vol. 9, no. 1, pp. 485–496, 2022.
   [doi:10.1109/JIOT.2021.3085194](https://doi.org/10.1109/JIOT.2021.3085194)
3. A. Alsaedi, N. Moustafa, Z. Tari, A. Mahmood and A. Anwar, "TON_IoT Telemetry Dataset: A New
   Generation Dataset of IoT and IIoT for Data-Driven Intrusion Detection Systems," *IEEE Access*,
   vol. 8, pp. 165130–165150, 2020.
   [doi:10.1109/ACCESS.2020.3022862](https://doi.org/10.1109/ACCESS.2020.3022862)
4. N. Moustafa, M. Keshky, E. Debiez and H. Janicke, "Federated TON_IoT Windows Datasets for
   Evaluating AI-Based Security Applications," *2020 IEEE 19th International Conference on Trust,
   Security and Privacy in Computing and Communications (TrustCom)*, 2020, pp. 848–855.
   [doi:10.1109/TrustCom50675.2020.00114](https://doi.org/10.1109/TrustCom50675.2020.00114)
5. N. Moustafa, M. Ahmed and S. Ahmed, "Data Analytics-Enabled Intrusion Detection: Evaluations of
   ToN_IoT Linux Datasets," *2020 IEEE 19th International Conference on Trust, Security and Privacy
   in Computing and Communications (TrustCom)*, 2020, pp. 727–735.
   [doi:10.1109/TrustCom50675.2020.00100](https://doi.org/10.1109/TrustCom50675.2020.00100)
6. N. Moustafa, "New Generations of Internet of Things Datasets for Cybersecurity Applications based
   Machine Learning: TON_IoT Datasets," *Proceedings of the eResearch Australasia Conference*,
   Brisbane, Australia, 2019.
7. N. Moustafa, "A systemic IoT-Fog-Cloud architecture for big-data analytics and cyber security
   systems: a review of fog computing," arXiv:1906.01055, 2019.
   [arXiv:1906.01055](https://arxiv.org/abs/1906.01055)
8. J. Ashraf, M. Keshk, N. Moustafa, M. Abdel-Basset, H. Khurshid, A. D. Bakhshi and R. R. Mostafa,
   "IoTBoT-IDS: A novel statistical learning-enabled botnet detection framework for protecting
   networks of smart cities," *Sustainable Cities and Society*, vol. 72, art. 103041, 2021.
   [doi:10.1016/j.scs.2021.103041](https://doi.org/10.1016/j.scs.2021.103041)

Two of this project's findings are corrections to what the *documentation* around these datasets
says, not to the authors' work: TON_IoT's published `train_test_network.csv` delivers 211,043 rows
against a documented 461,043 (see [Limitations](#limitations)), and the flow CSVs carry no TTL field
because Zeek's `conn.log` exports none. Both are recorded as measured facts in
[`data/README.md`](data/README.md) so that neither reads as a complaint about the datasets, which are
the reason this experiment is possible at all.

## Metrics log

`reports/metrics.csv` is the committed run log, and it is **committed on purpose**: reproducing a
run is meant to be a `git diff`, not a judgement call. So `evaluate.log_metrics()` **upserts** rather
than appends — it keys each row on `(run_id, model, regime)`, replaces or inserts the incoming row,
and rewrites the whole file under the frozen 14-column header sorted by that key. Running `./run.sh`
twice therefore leaves the file byte-identical instead of doubling every row. The 14 columns are
frozen: adding one is safe (older rows carry forward by name), renaming or reordering one blanks that
field on every row already logged.

That key is also a **convention every later phase has to honour: one `run_id` per experimental
condition.** `model` and `regime` do not identify a run on their own, so anything that changes *what
was measured* without changing those two must encode itself into `run_id` — otherwise it overwrites
the row it should be sitting next to:

```
phase4-baselines                # in-distribution ceiling
phase6-crossera                 # zero-shot cross-era, full feature set
phase6-crossera-no_proto        # same model + regime, `proto` ablated (d=18)
phase6-crossera-no_conn_state   # same model + regime, `conn_state` ablated (also d=18)
phase7-recovery-f0.05           # same model + regime, 5% fine-tune budget
phase7-recovery-f0.25           # ... and 25%
```

Without those suffixes each ablation would land on top of its unablated row and the whole
recovery curve would collapse to whichever fraction ran last. `run_id` is deliberately a fixed label
and never a timestamp — a timestamp would make every re-run a spurious diff.

## Repo layout

```
ids-crossera/
├── README.md
├── requirements.txt
├── run.sh                    # end-to-end reproduction
├── data/
│   ├── README.md             # verified record: MD5s, counts, deviations, download caveats
│   ├── raw/                  # git-ignored; reference/ holds the official feature docs
│   └── processed/
├── src/
│   ├── config.py             # RANDOM_SEED, paths, set_seeds()
│   ├── schema_map.py         # UNSW ↔ TON_IoT feature/label mapping + drop-list
│   ├── preprocess.py         # fit-on-source Preprocessor
│   ├── models/
│   │   ├── scratch_logreg.py # from-scratch logistic regression (do first)
│   │   ├── scratch_mlp.py    # from-scratch MLP (upgrade)
│   │   └── baselines.py      # RF / Decision Tree / LinearSVC wrappers
│   ├── transfer.py           # RQ2 adaptation + recovery curve
│   ├── evaluate.py           # metrics + both regimes
│   ├── plots.py              # Phase 9: the seven figures, from committed artifacts only
    └── figure_layout.py      # Phase 9: committed artist placement + the `--tune` drag editor
└── reports/
    ├── metrics.csv           # the committed run log; frozen 14-column header
    ├── confusion_matrices.json  # Phase 6 sidecar: the 2x2 counts the log header has no room for
    ├── roc_curves.json          # Phase 6 sidecar: the ROC vertices behind each logged roc_auc
    ├── per_family_metrics.csv   # Phases 6 + 7, own header, keyed (run_id, model, regime,
    │                            #   family_set, family) — shared families and TON_IoT's own
    ├── schema_catalogue.md   # Phase 1: both schemas side by side, drives Phase 2
    ├── schema_catalogue.csv  # machine-readable companion
    └── figures/              # written by `python -m src.plots`; regenerated by run.sh
        ├── README.md                    # report-ready captions + the log row behind every mark
        ├── layout.json                  # human-tuned legend/label positions, re-applied every run
        ├── drift_indist_vs_crossera.png # Fig 1 — RQ1 in-distribution vs cross-era bars
        ├── recovery_curve.png           # Fig 2 — RQ2 recovery vs fine-tune budget
        ├── confusion_matrices.png       # Fig 3 — RQ1, all six models x both regimes
        ├── roc_curves.png               # Fig 4 — RQ1, both regimes, sub-chance region tinted
        ├── per_family_crossera.png      # Fig 5 — RQ1 over the three shared families
        ├── per_family_recovery.png      # Fig 6 — RQ2 over TON_IoT's own eight attack types
        └── pipeline.png                 # Fig 7 — the Methods pipeline diagram
```

## Constraints

Four rules shape every design decision below:

- **From scratch.** At least one classifier is implemented from first principles in numpy, not
  pulled from a library.
- **No LLM or transformer as the core technique.** Classical and from-scratch methods only.
- **Reproducible.** Fixed seed, pinned dependencies, one-command end-to-end run.
- **No leakage across eras.** The `Preprocessor` is fit on **UNSW-train only** and applied unchanged
  to UNSW-test and TON_IoT. Refitting or re-tuning anything on the target invalidates the entire
  drift measurement — the whole result is a comparison between a model that has seen 2015 traffic and
  one era it has not. For the same reason `DROP_COLUMNS` removes the identity columns — they let a
  model memorize rather than generalize. Neither delivered file has a timestamp column, so the live
  leakage vectors are just two: UNSW's row `id`, which is monotonically informative about
  `attack_cat` because the partition was built by concatenating per-class blocks, and TON_IoT's
  `src_ip`, which takes only **51 distinct values** across 211,043 rows — close to a label lookup
  table for the target era.

Scope is bounded: **Phases 0–7 plus 9** are the guaranteed core and make a complete project on their
own. Phase 8 was the one phase held as cut-first, and it was cut.

## Implementation steps

Condensed from the implementation plan, whose canonical copy is
`University/CS4100/Project/Implementation-Plan.md` in Max's vault. This section and
`deviations.md` are the authoritative in-repo record; where they disagree with the vault plan, the
vault file wins.

**Phase 0 — Repo & environment** *(complete)*
GitHub repo and collaborator access; Python venv with pinned `requirements.txt`; global
`RANDOM_SEED = 42`; `README.md` + `run.sh` for one-command reproduction.

**Phase 1 — Data & first look** *(complete)*
Download UNSW-NB15 (pre-partitioned `UNSW_NB15_training-set.csv`, 175,341 rows /
`UNSW_NB15_testing-set.csv`, 82,332 rows) and TON_IoT (`Train_Test_Network.csv`) from the authors'
OneDrive shares, plus the official feature-description files for both — those are the source of
truth for Phase 2. Store under `data/raw/` (git-ignored), record exactly what was pulled in
`data/README.md`. First look — shapes, dtypes, class balance (binary and multiclass), null/`-`
counts, value ranges — plus both schemas catalogued side by side, all of it landing in
`reports/schema_catalogue.md` (human) and `reports/schema_catalogue.csv` (one row per delivered
column, so Phase 2 can assert coverage). No notebook: the graded figures have to be files the
report PDF embeds, so they come from `src/plots.py` via `run.sh` in Phase 9 instead.

**Phase 2 — Feature alignment** *(the core engineering and the time sink)*

**Do this first — extend `METRICS_HEADER` in `src/evaluate.py`.** Add `balanced_accuracy`,
`macro_f1`, `n_test` and `positive_rate` to the existing ten columns: the first two are the
prevalence-robust cross-checks Phase 6 needs, the last two record the class balance each row was
measured against. It belongs here rather than in Phase 6 because `reports/metrics.csv` was
header-only at the time: extending the header before Phase 4 logged its first run was free, whereas
afterwards it means every already-logged row carries a blank in the new columns (see
[Metrics log](#metrics-log) for what the frozen header does and does not tolerate).

The two datasets don't share a schema. Build an explicit mapping to a shared subspace in
`src/schema_map.py`. Verdicts are what `reports/schema_catalogue.md` measured, not what the docs
claim:

| Concept | UNSW-NB15 | TON_IoT Network | Verdict |
| --- | --- | --- | --- |
| Flow duration | `dur` | `duration` | **normalize** — UNSW is hard-capped at 60 s, TON_IoT runs to 93,517 s (≈26 h); zero-duration rate 1.52% vs 28.44% |
| Protocol | `proto` | `proto` | **collapse** — 133 UNSW levels vs 3; ablate it, see the `proto` hazard under [Limitations](#limitations) |
| Service | `service` | `service` | **collapse** — 5 shared levels (`-`, `dns`, `ftp`, `http`, `ssl`) cover 93.3% of UNSW and 99.8% of TON_IoT; split TON_IoT's `;`-joined cells (`smb;gssapi`) first |
| Connection state | `state` | `conn_state` | **collapse or drop** — **zero** shared tokens; Argus codes vs Zeek codes |
| Src→dst bytes | `sbytes` | `src_ip_bytes` | normalize (log1p) — pairing repointed to the IP-level column |
| Dst→src bytes | `dbytes` | `dst_ip_bytes` | normalize (log1p) — same repointing |
| Src→dst packets | `spkts` | `src_pkts` | **keep as-is** (log1p for the tail) — the cleanest of the eight |
| Dst→src packets | `dpkts` | `dst_pkts` | **keep as-is** (log1p) |

Every mapping is grounded in the official feature docs *first*, then verified against the
distributions — the catalogue is that verification, and it settles both known traps. **Bytes:**
TON_IoT `src_bytes`/`dst_bytes` are Zeek *payload* bytes and are 0 on 65% / 71% of rows, while
`src_ip_bytes`/`dst_ip_bytes` are *total IP* bytes; UNSW `sbytes` is IP-level (its floor is 28 B — an
IP + UDP header — and it is never 0). So the `*_ip_bytes` columns are the mappable ones and the
payload columns get dropped. **State:** the coarse shared set is completed / reset / no-response,
hand-written on both sides, since there is nothing to align lexically. Derive shared rate features
(`bytes_per_sec`, `pkts_per_sec`) from duration and counts — required, since the proposal promised
rates and neither dataset exposes a usable shared rate column; guard `duration == 0`, and carry a
`zero_duration` flag, because that case is ~19× more common in the target era. Drop
identity/leakage columns (UNSW row `id`, TON_IoT IPs and ports, the unmapped payload-byte columns;
there are no timestamp columns to drop in either file) — they let a model memorize and destroy
cross-era transfer. Normalize categorical vocabularies,
bucketing rare values to `other`. Define the shared label space: binary `normal (0)` /
`attack (1)` for the headline, plus the shared families for per-family analysis — only `DoS`,
`Reconnaissance`↔`scanning`, and `Backdoor` align across the two era labels (UNSW-NB15 has **no**
DDoS class), covering **37.26%** of TON_IoT's attack rows and, on the UNSW side, a figure that
depends on which frame you mean: **20.53%** of the training partition's attack rows (24,501/119,341),
**20.50%** of the seeded train fold's (19,569/95,472), but only **18.02%** of **UNSW-test**'s
(8,168/45,332) — and UNSW-test is what the cross-era per-family figure actually scores, so 18.02% is
the number that belongs beside Fig 5. Emit `unsw_common.parquet` and `toniot_common.parquet`.

> ⚠️ **Ordering hazard — drop before renaming.** `src_bytes`/`dst_bytes` are *both* `FEATURE_MAP`
> concept keys (the harmonized output names, fed by UNSW `sbytes`/`dbytes` and TON_IoT
> `src_ip_bytes`/`dst_ip_bytes`) *and* entries on `DROP_COLUMNS`, where they mean TON_IoT's raw Zeek
> payload columns. So `build_common_frames()` must apply `DROP_COLUMNS` to the raw TON_IoT frame
> **first**, then rename `src_ip_bytes` → `src_bytes`. Reversed, the drop deletes the byte features it
> just built — silently, since every name involved is legitimate either way. Guard it:
> `assert {"src_bytes", "dst_bytes"} <= set(df.columns)` after the drop step. The collision only exists
> because two Phase 1 corrections landed together (repointing the pairing, and dropping the payload
> columns), so it isn't in the original scaffold.

We deliberately do **not** reproduce all 49 UNSW-NB15 features — they come from Argus + Bro/Zeek
plus ~12 custom algorithms, which is a reverse-engineering trap. The ~8–10 features both datasets
already expose are enough for a valid experiment.

**Phase 3 — Preprocessing**
One `Preprocessor` **fit on UNSW-train only**, then applied unchanged to UNSW-test and TON_IoT —
refitting on the target leaks. One-hot the small categorical set, z-score numerics using
UNSW-train statistics, log-transform heavy-tailed byte/packet counts, impute missing/`-`. Split
UNSW-train stratified ~80/20 into train/validation; UNSW-test stays held out; TON_IoT stays
untouched until eval. Serialize the fitted preprocessor.

**Phase 4 — In-distribution baselines** *(establishes the "before" ceiling)*
Dummy/majority-class sanity floor, then the three classifiers the proposal promises without
qualification — Random Forest, Decision Tree, and SVM — so all three are **required**. The SVM is
linear (`LinearSVC`/`SGDClassifier`), which satisfies an unqualified "SVM"; what's excluded is a
*kernel* SVM, which at 175,341 rows needs a kernel matrix in the hundreds of GB and won't finish.
Class weights throughout. Tune depth/regularization on the validation split *before* any cross-era
run. Log every run
(params, seed, metrics) to `reports/metrics.csv`.

**Phase 5 — From-scratch models** *(required for the grade; lock in the easy one first)*
**Logistic regression from scratch** in numpy (gradient descent, weighted loss) comes first: clean
math, guaranteed to converge, and it satisfies the from-scratch requirement on its own. Then the
**MLP from scratch** (forward pass, backprop, mini-batch loop, sigmoid/softmax) as the upgrade —
with **class weighting in the loss**, since a plain net on imbalanced tabular data will predict
"normal" for everything and post a deceptively high accuracy. Unit-test both on a toy separable
set; check in-distribution scores land within a few points of the sklearn equivalents. Keep the
math clean — it drops straight into the report's Methods section.

**Phase 6 — Zero-shot cross-era evaluation** *(RQ1, the primary result)*
Run every model from Phases 4–5 in two regimes side by side: **in-distribution** (train
UNSW-train → test UNSW-test) and **cross-era zero-shot** (train UNSW-train → test TON_IoT, no
retraining). The approved proposal promises **F1 and ROC-AUC** as the headline pair, so both stay —
but the emphasis inside that pair matters, because the two test sets don't share a class balance
(45% vs 24% normal). Lead the *drift claim* with **ROC-AUC**, which is insensitive to class
prevalence. Report **F1 alongside it with the prevalence caveat attached**, since some of any F1
delta is the balance change rather than drift. Add **balanced accuracy and macro-F1 as
supplementary** metrics, justified by the delivered balance. This is emphasis within the promised
pair, not a substitution for it — and plain accuracy still misleads under imbalance, so it is never
the headline. Plus precision/recall, confusion matrices, and per-shared-family breakdowns. The
headline is the **Δ (in-distribution − cross-era)** per metric, reported with both test sets' normal
share. A large drop is the *expected finding*, not a failure.

**Phase 7 — Recovery curve** *(RQ2, the secondary result)*
**Fix the TON_IoT test set once, up front** — freeze ~50% as the test set and draw fine-tune
fractions from the *other* 50%, so the data budget is the only thing changing along the x-axis.
Adapt each model on stratified **1%, 5%, 10%, 25%** of the fine-tune pool: the MLP freezes its
early layer(s) and retrains the output head; logistic regression and the classical models retrain
on (source + small target) or the small target alone. Plot post-adaptation F1/ROC-AUC vs. fraction
of modern data used, and report how close each model gets to a full-TON_IoT-trained ceiling and at
what data budget.

*Delivered* — the open choices above resolved, with the reasoning in `deviations.md` §3.12/§3.13:
the MLP freezes **both** hidden layers (that is what `fit(freeze_hidden=True)` does, and the freeze
cost is measured under its own `run_id` rather than assumed); the classical models retrain on the
**small target alone**, because a pooled fit would be 140,272 source rows against 1,055 target rows
at the 1% budget and would not converge to the ceiling the curve is read against; the zero-shot
point is **re-measured on the frozen test half** rather than reused from Phase 6, whose rows were
scored on the full target frame including this phase's fine-tune pool. Curve in
[Status](#status); the plot itself is Phase 9's.

**Phase 8 — Stretch: live 2026 malware probe** *(**CUT** — optional, and dropped for time)*
Not delivered. Neither hard gate below was ever attempted, so there is no authorization on file and
no air-gapped environment, and no code, capture scripts or `lab/` directory exist for it anywhere in
the repo. This is the exit the proposal reserved, and RQ1/RQ2 stand without it. What it *would* have
required, kept for the record:
two hard gates before anything is downloaded or run — **authorization** (instructor sign-off plus
Northeastern policy) and a **verified air-gap** (isolated virtual segment, no route to the internet
or campus network, disposable snapshots). Then a few catalogued samples from **MalwareBazaar** with
hashes recorded, traffic captured via tcpdump/Zeek, and only the ~8–10 aligned features extracted
and run through the trained and recovered models as a qualitative check. If either gate fails or
time is short, drop it — the public-dataset core is a complete project.

**Phase 9 — Results, report & demo** *(figures complete)*
Figures to `reports/figures/` — labels, legends, and captions are graded, so each one also ships a
report-ready caption and the exact `(run_id, model, regime)` row behind every mark in
`reports/figures/README.md`. Seven landed: in-distribution vs. cross-era bars, the recovery curve,
confusion matrices, ROC curves, **two** per-family figures, and the Methods pipeline diagram.
The two per-family figures are deliberately not one: the *cross-era* breakdown covers only the three
families both eras label, while the *recovery* breakdown covers TON_IoT's own eight 20,000-row
attack types on the frozen test half — a different population over a different regime, and five of
those eight have no UNSW-NB15 counterpart to compare against at all. Legend and label placement is
tunable without breaking reproducibility — see [Adjusting a figure](#adjusting-a-figure). Still owed:
the report
(Abstract, Intro + Related Work, Methods, Data & Experiments, Results, Conclusion) at ≤6 pages plus
a title page; demo video recorded, uploaded privately, embedded in the deck; 65–70 word team
contributions statement. The References section is **license-bearing, not just courtesy** — both
datasets are used under terms that require citation, so the report must cite UNSW-NB15's dataset and
evaluation papers inline at a minimum and the deck must credit both datasets on a slide; the full
required lists are in [Datasets, credit and license](#datasets-credit-and-license). Final check: the
repo is accessible to the graders — and `run.sh`
reproduces every figure from raw data, which it now does, since `python -m src.plots` is wired in as
its last step.
