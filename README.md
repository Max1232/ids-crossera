# ids-crossera

**Cross-Dataset Generalization for Network Intrusion Detection**
CS4100 (Foundations of AI), Summer B 2026 

_Group 10: Max Cossill & Enqi Zhang (Steven)_

Machine-learning intrusion detection systems are almost always trained and tested on a single
static dataset captured at one point in time. That means the accuracy numbers you see published
describe only *in-distribution* performance — how well a detector does on traffic that looks like
what it already saw. Real networks don't hold still: protocols change, services change, and attack
tooling changes.

This project measures what that costs. We train a detector on **UNSW-NB15** ([2015](https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fz5025758%5Fad%5Funsw%5Fedu%5Fau%2FDocuments%2FTON%5FIoT%20datasets&ga=1)), test it
unchanged on the newer **TON_IoT Network** dataset ([~2019–2020](https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fz5025758%5Fad%5Funsw%5Fedu%5Fau%2FDocuments%2FTON%5FIoT%20datasets&ga=1)) to see how far it falls, and then
measure how little modern labeled data it takes to claw the performance back. The goal is two
honest numbers: **how big the generalization gap is**, and **how cheaply it can be closed**.

## Research questions

- **RQ1 — the gap:** How much does a 2015-trained detector degrade on newer, unseen traffic?
- **RQ2 — recovery:** How little modern labeled data does it take to recover most of that
  performance?
- **RQ3 — stretch (optional):** Does the model flag traffic from live, in-house-detonated 2026
  malware?

## What we measure, stated honestly

UNSW-NB15 is general enterprise traffic; TON_IoT is IoT/IIoT. The gap we measure therefore bundles
*temporal drift* (attacks evolving over ~4–5 years) with an *IoT-vs-enterprise domain shift* — it
is not a clean time-only experiment, and we don't claim it is. We frame the result as
**cross-dataset generalization**, and we restrict the per-attack-family analysis to families both
datasets actually share so the comparison stays fair.

## Status

Phase 0 scaffold. The layout, config, and feature map are in place; the pipeline modules are
stubs that raise `NotImplementedError`, each tagged with the phase that fills it in
(see [Implementation steps](#implementation-steps)). The datasets have not been downloaded yet, so
`run.sh` currently walks the phase sequence without doing work.

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

## Repo layout

```
ids-crossera/
├── README.md
├── requirements.txt
├── run.sh                    # end-to-end reproduction
├── data/
│   ├── README.md             # download URLs (raw files git-ignored)
│   ├── raw/
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
│   └── plots.py
├── lab/                      # RQ3 stretch: capture → feature-map scripts (optional)
├── notebooks/
│   └── 01_eda.ipynb
└── reports/
    ├── metrics.csv
    └── figures/
```

## Constraints

Three rules shape every design decision below:

- **From scratch.** At least one classifier is implemented from first principles in numpy, not
  pulled from a library.
- **No LLM or transformer as the core technique.** Classical and from-scratch methods only.
- **Reproducible.** Fixed seed, pinned dependencies, one-command end-to-end run.

Scope is bounded to fit four weeks: Phases 0–4, 6, and 7 are the guaranteed core and make a
complete project on their own. Phase 8 is cut first if time runs short.

## Implementation steps

Condensed from `University/CS4100/Project/Proposal/Implementation Plan.md` in the Obsidian vault —
see that file for the full detail, dataset URLs, and risk table.

**Phase 0 — Repo & environment** *(½ day)*
GitHub repo and collaborator access; Python venv with pinned `requirements.txt`; global
`RANDOM_SEED = 42`; `README.md` + `run.sh` for one-command reproduction.

**Phase 1 — Data & first look** *(1–2 days)*
Download UNSW-NB15 (pre-partitioned `UNSW_NB15_training-set.csv`, 175,341 rows /
`UNSW_NB15_testing-set.csv`, 82,332 rows) and TON_IoT (`Train_Test_Network.csv`) from the authors'
OneDrive shares, plus the official feature-description files for both — those are the source of
truth for Phase 2. Store under `data/raw/` (git-ignored), record exactly what was pulled in
`data/README.md`. EDA notebook: shapes, dtypes, class balance (binary and multiclass), null/`-`
counts, value ranges. Catalogue both schemas side by side.

**Phase 2 — Feature alignment** *(4–5 days — the core engineering and the time sink)*
The two datasets don't share a schema. Build an explicit mapping to a shared subspace in
`src/schema_map.py`:

| Concept | UNSW-NB15 | TON_IoT Network |
| --- | --- | --- |
| Flow duration | `dur` | `duration` |
| Protocol | `proto` | `proto` |
| Service | `service` | `service` |
| Connection state | `state` | `conn_state` |
| Src→dst bytes | `sbytes` | `src_bytes` |
| Dst→src bytes | `dbytes` | `dst_bytes` |
| Src→dst packets | `spkts` | `src_pkts` |
| Dst→src packets | `dpkts` | `dst_pkts` |

Ground every mapping in the official feature docs *first*, then verify against the distributions.
Two known traps: **bytes** — TON_IoT `src_bytes`/`dst_bytes` are *payload* bytes while
`src_ip_bytes`/`dst_ip_bytes` are *total IP* bytes, and UNSW's transaction bytes sit closer to the
total, so pick one notion and apply it consistently; **state** — UNSW `state` codes are *not* Zeek
`conn_state` codes, so collapse both to a coarse shared set (completed / reset / no-response) or
drop the feature. Optionally derive shared rate features (bytes/sec, packets/sec). Drop
identity/leakage columns (IPs, ports, timestamps, row `id`, unmapped `*_ip_bytes`) — they let a
model memorize and destroy cross-dataset transfer. Normalize categorical vocabularies, bucketing
rare values to `other`. Define the shared label space: binary `normal (0)` / `attack (1)` for the
headline, plus shared families (`DoS`, `DDoS`, `backdoor`, `scanning/reconnaissance`) for per-family
analysis. Emit `unsw_common.parquet` and `toniot_common.parquet`.

We deliberately do **not** reproduce all 49 UNSW-NB15 features — they come from Argus + Bro/Zeek
plus ~12 custom algorithms, which is a reverse-engineering trap. The ~8–10 features both datasets
already expose are enough for a valid experiment.

**Phase 3 — Preprocessing** *(1–2 days)*
One `Preprocessor` **fit on UNSW-train only**, then applied unchanged to UNSW-test and TON_IoT —
refitting on the target leaks. One-hot the small categorical set, z-score numerics using
UNSW-train statistics, log-transform heavy-tailed byte/packet counts, impute missing/`-`. Split
UNSW-train stratified ~80/20 into train/validation; UNSW-test stays held out; TON_IoT stays
untouched until eval. Serialize the fitted preprocessor.

**Phase 4 — In-distribution baselines** *(1–2 days — establishes the "before" ceiling)*
Dummy/majority-class sanity floor; Random Forest and Decision Tree as the core classical
baselines; Linear SVM (`LinearSVC`/`SGDClassifier`) optional — **not** a kernel SVM, which at
175k rows needs a kernel matrix in the hundreds of GB and won't finish. Class weights throughout.
Tune depth/regularization on the validation split *before* any cross-dataset run. Log every run
(params, seed, metrics) to `reports/metrics.csv`.

**Phase 5 — From-scratch models** *(3–4 days — required for the grade; lock in the easy one first)*
**Logistic regression from scratch** in numpy (gradient descent, weighted loss) comes first: clean
math, guaranteed to converge, and it satisfies the from-scratch requirement on its own. Then the
**MLP from scratch** (forward pass, backprop, mini-batch loop, sigmoid/softmax) as the upgrade —
with **class weighting in the loss**, since a plain net on imbalanced tabular data will predict
"normal" for everything and post a deceptively high accuracy. Unit-test both on a toy separable
set; check in-distribution scores land within a few points of the sklearn equivalents. Keep the
math clean — it drops straight into the report's Methods section.

**Phase 6 — Zero-shot cross-dataset evaluation** *(1–2 days — RQ1, the primary result)*
Run every model from Phases 4–5 in two regimes side by side: **in-distribution** (train
UNSW-train → test UNSW-test) and **cross-dataset zero-shot** (train UNSW-train → test TON_IoT, no
retraining). Lead with F1 and ROC-AUC — accuracy misleads under imbalance — plus precision/recall,
confusion matrices, and per-shared-family breakdowns. The headline is the **Δ (in-distribution −
cross-dataset)** per metric. A large drop is the *expected finding*, not a failure.

**Phase 7 — Recovery curve** *(2–3 days — RQ2, the secondary result)*
**Fix the TON_IoT test set once, up front** — freeze ~50% as the test set and draw fine-tune
fractions from the *other* 50%, so the data budget is the only thing changing along the x-axis.
Adapt each model on stratified **1%, 5%, 10%, 25%** of the fine-tune pool: the MLP freezes its
early layer(s) and retrains the output head; logistic regression and the classical models retrain
on (source + small target) or the small target alone. Plot post-adaptation F1/ROC-AUC vs. fraction
of modern data used, and report how close each model gets to a full-TON_IoT-trained ceiling and at
what data budget.

**Phase 8 — Stretch: live 2026 malware probe** *(optional — cut first; do NOT start until 1–7 land)*
Two hard gates before anything is downloaded or run: **authorization** (instructor sign-off plus
Northeastern policy) and a **verified air-gap** (isolated virtual segment, no route to the internet
or campus network, disposable snapshots). Then a few catalogued samples from a reputable repo with
hashes recorded, traffic captured via tcpdump/Zeek, and only the ~8–10 aligned features extracted
and run through the trained and recovered models as a qualitative check. If either gate fails or
time is short, drop it — the public-dataset core is a complete project.

**Phase 9 — Results, report & demo** *(3–4 days, started in Week 3)*
Figures to `reports/figures/`: in-distribution vs. cross-dataset bars, confusion matrices, ROC
curves, per-family F1, and the recovery curve — labels, legends, and captions are graded. Report
(Abstract, Intro + Related Work, Methods, Data & Experiments, Results, Conclusion) at ≤6 pages plus
a title page. Demo video recorded, uploaded privately, embedded in the deck. 65–70 word team
contributions statement. Final check: the repo is accessible to the graders and `run.sh` reproduces
every figure from raw data.
