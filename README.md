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

## Research questions

- **RQ1 — drift:** How much does a 2015-trained detector degrade on newer traffic?
- **RQ2 — correction:** How little modern labeled data does transfer learning need to recover that
  performance?
- **RQ3 — stretch (optional):** Does the model flag traffic from live, in-house-detonated 2026
  malware — a full decade forward from the training data?

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
without. `icmp` has the mirror problem: **15 UNSW train rows** (0.01%) against 281 in TON_IoT, so
that level is effectively untrained yet active cross-era.

## Status

**Phases 0–4 are complete:** Phase 0 repo scaffold and pinned environment, Phase 1 data download and
first look, Phase 2 feature alignment, Phase 3 preprocessing, Phase 4 in-distribution baselines.
Phases 5–7 and 9 are the remaining core (Phase 8 is the optional stretch) — see
[Implementation steps](#implementation-steps) for what each one owes.

Where the delivered build departs from the approved proposal, and why, is recorded in
[`deviations.md`](deviations.md).

All three CSVs are extracted and verified — MD5s and row counts in
[`data/README.md`](data/README.md) — and the schema comparison, measured from the delivered files
rather than quoted from documentation, is in `reports/schema_catalogue.md` with a machine-readable
`reports/schema_catalogue.csv` companion (one row per delivered column).

`src/schema_map.py`'s mapping constants are corrected against that catalogue, and
`build_common_frames()` emits both harmonized parquets (UNSW 257,673 × 14, TON_IoT 211,043 × 14).
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

`./run.sh` reproduces Phases 2 → 4 from the raw CSVs in under 20 seconds. The remaining pipeline
modules are still stubs that raise `NotImplementedError`, each tagged with the phase that fills it in
(see [Implementation steps](#implementation-steps)), and `run.sh` walks past them printing banners.

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
phase4-baselines             # in-distribution ceiling
phase6-crossera              # zero-shot cross-era, full feature set
phase6-crossera-no_proto     # same model + regime, `proto` ablated
phase7-recovery-f0.05        # same model + regime, 5% fine-tune budget
phase7-recovery-f0.25        # ... and 25%
```

Without those suffixes each `proto` ablation would land on top of its unablated row and the whole
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
│   └── plots.py
├── lab/                      # RQ3 stretch: capture → feature-map scripts (optional)
└── reports/
    ├── metrics.csv
    ├── schema_catalogue.md   # Phase 1: both schemas side by side, drives Phase 2
    ├── schema_catalogue.csv  # machine-readable companion
    └── figures/
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
own. Phase 8 is cut first if anything has to go.

## Implementation steps

Condensed from the implementation plan. The canonical copy is
`University/CS4100/Project/Implementation-Plan.md` in Max's vault; the
[HackMD page](https://hackmd.io/@maxCoss/ImplementationPlan) is a hand-pasted mirror of it for
sharing, so the vault file wins if the two disagree.

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
DDoS class), covering **20.5%** of UNSW attack rows and **37.3%** of TON_IoT's. Emit `unsw_common.parquet`
and `toniot_common.parquet`.

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

**Phase 8 — Stretch: live 2026 malware probe** *(optional — cut first; do NOT start until 1–7 land)*
Two hard gates before anything is downloaded or run: **authorization** (instructor sign-off plus
Northeastern policy) and a **verified air-gap** (isolated virtual segment, no route to the internet
or campus network, disposable snapshots). Then a few catalogued samples from **MalwareBazaar** with
hashes recorded, traffic captured via tcpdump/Zeek, and only the ~8–10 aligned features extracted
and run through the trained and recovered models as a qualitative check. If either gate fails or
time is short, drop it — the public-dataset core is a complete project.

**Phase 9 — Results, report & demo**
Figures to `reports/figures/`: in-distribution vs. cross-era bars, confusion matrices, ROC
curves, per-family F1 (three families only), and the recovery curve — labels, legends, and captions
are graded. Report
(Abstract, Intro + Related Work, Methods, Data & Experiments, Results, Conclusion) at ≤6 pages plus
a title page. Demo video recorded, uploaded privately, embedded in the deck. 65–70 word team
contributions statement. Final check: the repo is accessible to the graders and `run.sh` reproduces
every figure from raw data.
