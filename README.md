# ids-crossera

**Measuring and correcting temporal concept drift in network intrusion detection.**

CS4100 (Foundations of AI), Summer B 2026 — Group 10: Max Cossill & Enqi Zhang (Steven).

Machine-learning intrusion detectors are almost always trained and tested on a single dataset captured at one point in time, so published accuracy describes *in-distribution* performance only — how well a detector does on traffic that looks like what it already saw. Real networks do not hold still: protocols, services and attack tooling all change. That gap has a name, **concept drift**, and most published NIDS work never measures it.

This project measures it in two halves. Six binary flow classifiers are fitted **once** on a 140,272-row fold of [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)'s 2015 training partition, then scored unchanged in two regimes: in-distribution on UNSW-NB15's 82,332-row test partition, and zero-shot on [TON_IoT Network](https://research.unsw.edu.au/projects/toniot-datasets), a Zeek-derived flow capture from roughly 2019–20. That is the drift measurement. The second half measures how little modern labelled data it takes to undo the damage.

The headline is not that the detector degrades. It is that **the learned score ranking inverts**: every real model's cross-era ROC-AUC lands *below* 0.5, in a band of 0.1846–0.3534, having spanned 0.8811–0.9788 in-distribution. A curve below the chance diagonal is not a weak detector but a backwards one — on 2019–20 traffic the 2015-learned score ranks attacks *below* benign flows. **1,055 labelled modern flows** undo it.

## Results

### Drift: the ranking inverts

| Model | ROC-AUC in-dist. | ROC-AUC cross-era | **Δ AUC** |
| --- | --- | --- | --- |
| Random Forest | 0.9788 | 0.2106 | **+0.7682** |
| Scratch MLP | 0.9621 | 0.1846 | **+0.7775** |
| Decision Tree | 0.9648 | 0.3534 | **+0.6115** |
| Linear SVM (`LinearSVC`) | 0.8846 | 0.2114 | **+0.6732** |
| Scratch logistic regression | 0.8811 | 0.2489 | **+0.6321** |
| Dummy (majority-class floor) | 0.5000 | 0.5000 | +0.0000 |

_Δ = in-distribution − cross-era, so positive means the model lost ground. In-distribution = UNSW-test, n = 82,332, 55.06% attack; cross-era = TON_IoT, n = 211,043, 76.31% attack._

All five real models land below the 0.5000 no-skill line cross-era, and balanced accuracy says the same thing without prevalence: averaged over the two classes, each one predicts the wrong label more often than the right one. The dummy moves the other way — its F1 **rises 0.7102 → 0.8656** cross-era at ROC-AUC exactly 0.5000, purely because the target era is 76.31% attack rather than 55.06%, which is why the drift claim leads with ROC-AUC and F1 never stands alone.

### Recovery: 1,055 labelled flows undo it

TON_IoT is split once, stratified and seeded, into a permanent 105,521-row test half and a 105,522-row fine-tune pool. Every point below is scored on that identical half, so the labelling budget is the only thing moving along the row. Budgets are fractions of the pool: 1% = 1,055 rows, 5% = 5,276, 10% = 10,552, 25% = 26,380, ceiling = the whole 105,522.

| Model | 0% (zero-shot) | 1% | 5% | 10% | 25% | ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.2121 | **0.9979** | 0.9988 | 0.9988 | 0.9993 | 0.9998 |
| Decision Tree | 0.3519 | **0.9862** | 0.9920 | 0.9941 | 0.9969 | 0.9984 |
| Linear SVM | 0.2111 | **0.9843** | 0.9869 | 0.9877 | 0.9880 | 0.9881 |
| Scratch logistic regression | 0.2490 | **0.9859** | 0.9878 | 0.9884 | 0.9885 | 0.9884 |
| Scratch MLP (head only) | 0.1847 | **0.8845** | 0.9598 | 0.9689 | 0.9766 | 0.9808 |
| Dummy (majority-class floor) | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

_ROC-AUC on the frozen TON_IoT test half, n = 105,521, 76.31% attack — the same balance as the full target frame._

Every model clears the 0.5000 no-skill line at the smallest budget tested, so the RQ1 inversion is undone by 1,055 labelled modern flows. The 25% budget closes 99.5–100% of each model's gap to its own ceiling, which means the curve is flat after 1% rather than climbing.

## Quickstart

Python 3.11 or 3.12. The datasets are not committed — download them first, see [`data/README.md`](data/README.md).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download the three CSVs into data/raw/
./run.sh
```

`run.sh` reproduces the whole pipeline from the raw CSVs in about five minutes and is idempotent: a second run leaves every committed artifact diff-clean. It defaults to `./.venv/bin/python` when that exists, so a fresh clone needs no activation step; `PYTHON=python3.12 ./run.sh` overrides.

### Development

```bash
pytest tests/ -q                        # 63 tests across 6 files
python -m src.plots --list              # name the seven figures
python -m src.plots --tune roc_curves   # drag a legend, press s; the position is saved to reports/figures/layout.json
```

The test suite is deliberately not wired into `run.sh`, which stays the single reproduction command. Legend and reference-label placement is committed **data** rather than a manual edit, so tuning a figure by hand cannot break reproducibility; `--vector` writes editable SVG/PDF into a git-ignored scratch directory for changes no position can express. Individual stages run standalone via `python -m src.schema_map --build`, `src.preprocess`, `src.models.baselines`, `src.evaluate --regimes` and `src.transfer`.

## Method

The two datasets share no schema. UNSW-NB15's delivered partitions are 45 columns of Argus + Bro/Zeek output plus about a dozen generated features; TON_IoT's is 44 columns of Zeek `conn.log`. `src/schema_map.py` is the single source for the shared subspace: it maps eight concepts across the two eras, each one a correction of something a naive name-match gets wrong (TON_IoT's `src_bytes`/`dst_bytes` are Zeek *payload* bytes and zero on most rows, so the mappable columns are `src_ip_bytes`/`dst_ip_bytes`, which are IP-level like UNSW's `sbytes`). Vocabularies that cannot align lexically are collapsed by hand — 133 UNSW protocol levels against TON_IoT's 3, and Argus connection-state codes against Zeek's, which share **zero** tokens. Rate features are derived identically on both sides from duration and counts, because UNSW's `sload`/`dload` are in bits per second with an Argus inter-packet correction and TON_IoT ships no rate column at all. Preprocessing lands a **22-column** feature schema: 7 numerics, 14 one-hot levels and one `zero_duration` passthrough flag.

Six models are scored, all carrying class weights from the start and all built from locked factories so nothing re-tunes itself downstream. Four are library baselines — Random Forest, decision tree, `LinearSVC` and a `most_frequent` dummy that fixes the majority-class floor. Two are written from scratch in pure numpy with hand-written gradients and backprop, and neither has sklearn, torch or any autograd package in its import graph: a convex logistic regression on full-batch gradient descent, and a `(22, 44, 22, 1)` MLP on mini-batch SGD. Both are verified against synthetic fixtures, including a finite-difference check on the MLP's backprop.

One `Preprocessor` is fit on the **UNSW train fold only** — a seeded, stratified 80/20 split of the 2015 training partition — and applied unchanged to UNSW-test and TON_IoT. Refitting anything on the target would invalidate the entire measurement, so leakage is enforced as a **runtime error rather than a convention**. Three mechanisms do it: `fit()` inspects the frame's `split` column and raises unless the rows are exactly source-train, so fitting the concatenated frame cannot silently succeed; `evaluate.sealed()` shadows `fit`, `partial_fit` and `fit_transform` on the live instance for the span of every evaluation and raises `LeakageError` if one is called, leaving `predict`/`transform` untouched; and frame-shape assertions check every row and attack count before a number is logged. The drop-list removes the identity columns for the same reason — UNSW's row `id` is monotonically informative about `attack_cat`, and TON_IoT's `src_ip` takes only **51 distinct values** across 211,043 rows.

Two obvious deflations of the drift result are measured rather than argued about, both as **retrain-without ablations**: `proto` and `conn_state` are each removed from the hypothesis class and the model refitted at d=18, giving a matched in-distribution/cross-era pair under its own `run_id` so the comparable quantity is the difference of the deltas. Masking a feature at test time would instead score the model on inputs no era produces. `proto` matters because the collapsed `other` bucket is 91% attack in training and 0% of rows at test time; `conn_state` matters because that collapse is the project's own invention rather than the datasets'. Neither survives: Δ-of-Δ AUC spans −0.087 to +0.061 for `proto` and −0.0427 to +0.0535 for `conn_state`, at most ~12% and ~7.6% of a ~0.7 collapse, and in both cases three of six models degrade *further* without the feature.

Reproducibility rests on a single global `RANDOM_SEED = 42` (`src/config.py`) seeding numpy, stdlib `random` and every split, seven pinned dependencies, and metric logs that upsert on `(run_id, model, regime)` under a frozen header instead of appending. `./run.sh` rebuilds all twelve committed artifacts — `reports/metrics.csv`, `reports/per_family_metrics.csv`, the two JSON sidecars, the seven figures and the caption index beside them. `src/plots.py` opens those artifacts **read-only** and re-derives nothing, so no figure can disagree with the table it illustrates.

## What this doesn't claim

**The headline Δ is an upper bound on temporal drift, not a measurement of it.** UNSW-NB15 is general enterprise traffic and TON_IoT is IoT/IIoT, so the delta bundles ~4–5 years of attack evolution with an enterprise-vs-IoT domain shift, and this design cannot separate the two. The feature-level mechanism makes that concrete: at the class median `dst_bytes` is 354 for normal and 0 for attack on UNSW-test, but 0 for normal and 40 for attack on TON_IoT. A UNSW-trained model has learned that silence is hostile, and on IoT telemetry that rule is not merely uninformative but backwards.

**The target era's class balance is part of the measurement, not a defect to normalize away.** TON_IoT's published `Train_Test_Network.csv` delivers 211,043 rows with 50,000 normal against a documented 461,043 with 300,000 — 250,000 normal rows short, with all nine attack classes matching their documented counts exactly. That is what upstream publishes, so the target era is **76.31% attack** and every reported delta names both sets' balance.

**The per-family view is narrow.** Only three families align across the two label vocabularies — `dos`, `scanning`/`Reconnaissance` and `backdoor`, since UNSW-NB15 has no DDoS class — covering **18.02%** of UNSW-test's attack rows and **37.26%** of TON_IoT's. The binary headline uses every row; only the per-family breakdown is restricted.

**The live-malware probe does not exist.** RQ3 — detonating 2026 malware in an air-gapped lab and testing a decade forward — was scoped as optional in the proposal and cut on the timeline; no code, captures or authorization exist for it. So the UNSW→TON_IoT span of ~4–5 years is the only temporal distance the project measures, and there is no decade-forward counterweight to the upper-bound caveat above.

**1% is a lower bound on the labelling budget, not a general result.** It holds because the label is close to a deterministic function of these 22 features: just 15 of 92,438 distinct feature vectors carry both labels, and a per-vector majority-vote lookup would score 99.8953% accuracy on the whole target frame. A genuinely overlapping feature space would need more. The related memorization objection was measured and failed — re-scoring the 1%-budget forest on only the 64.22% of the test half whose exact feature vector is absent from the draw gives 0.9959 AUC against 0.9979 on all rows, so memorization is worth 0.0021 AUC against a recovery of 0.786.

## Repo layout

```
ids-crossera/
├── README.md
├── deviations.md              # per-decision rationale
├── requirements.txt
├── run.sh                     # end-to-end reproduction
├── data/
│   ├── README.md              # MD5s, row counts, download provenance
│   ├── raw/                   # git-ignored; the three source CSVs
│   └── processed/             # build products
│       ├── unsw_common.parquet
│       ├── toniot_common.parquet
│       └── preprocessor.joblib
├── src/
│   ├── __init__.py
│   ├── config.py              # RANDOM_SEED, paths, set_seeds()
│   ├── schema_map.py          # UNSW ↔ TON_IoT mapping, collapses, drop-list
│   ├── preprocess.py          # fit-on-source Preprocessor
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baselines.py       # RF / decision tree / LinearSVC / dummy
│   │   ├── scratch_logreg.py  # pure-numpy logistic regression
│   │   └── scratch_mlp.py     # pure-numpy MLP, hand-written backprop
│   ├── evaluate.py            # metrics, both regimes, the leakage seal
│   ├── transfer.py            # recovery curve
│   ├── plots.py               # the seven figures
│   └── figure_layout.py       # committed artist placement + `--tune`
├── tests/
│   ├── README.md
│   ├── test_evaluate.py
│   ├── test_figure_layout.py
│   ├── test_per_family.py
│   ├── test_scratch_logreg.py
│   ├── test_scratch_mlp.py
│   └── test_transfer.py
└── reports/
    ├── metrics.csv            # the committed run log, frozen 14-column header
    ├── per_family_metrics.csv # per-family rows, own key and header
    ├── confusion_matrices.json
    ├── roc_curves.json        # thinned ROC vertices behind each logged AUC
    ├── schema_catalogue.md    # both delivered schemas side by side
    ├── schema_catalogue.csv   # machine-readable companion
    └── figures/               # seven PNGs, layout.json, and a caption index
```

## Datasets & citation

Neither dataset is ours and neither is redistributed here — `data/raw/` is git-ignored and every file is fetched from the authors' own project pages, with [`data/README.md`](data/README.md) recording exactly what was pulled. Both are the work of **Nour Moustafa and co-authors at UNSW Canberra at ADFA**.

| Dataset | Role here | Credit |
| --- | --- | --- |
| [**UNSW-NB15**](https://research.unsw.edu.au/projects/unsw-nb15-dataset) (2015) | source era — every model is trained on it | Nour Moustafa and Jill Slay |
| [**TON_IoT Network**](https://research.unsw.edu.au/projects/toniot-datasets) (~2019–2020) | target era — the drift and the recovery are measured on it | Nour Moustafa, with the co-authors of the papers below |

Both are granted free for academic research purposes in perpetuity; commercial use must be agreed with the authors, who assert copyright. The condition attached to that grant is **citation**, and in neither case is it a single reference: the UNSW-NB15 page requires five papers and the TON_IoT page requires eight. All thirteen are reproduced below, with bibliographic details verified against Crossref.

<details>
<summary><strong>The thirteen citations both licenses require</strong></summary>

**UNSW-NB15 — five:**

1. N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)," *2015 Military Communications and Information Systems Conference (MilCIS)*, 2015, pp. 1–6. [doi:10.1109/MilCIS.2015.7348942](https://doi.org/10.1109/MilCIS.2015.7348942)
2. N. Moustafa and J. Slay, "The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set," *Information Security Journal: A Global Perspective*, vol. 25, no. 1–3, pp. 18–31, 2016. [doi:10.1080/19393555.2015.1125974](https://doi.org/10.1080/19393555.2015.1125974)
3. N. Moustafa, J. Slay and G. Creech, "Novel Geometric Area Analysis Technique for Anomaly Detection Using Trapezoidal Area Estimation on Large-Scale Networks," *IEEE Transactions on Big Data*, vol. 5, no. 4, pp. 481–494, 2019. [doi:10.1109/TBDATA.2017.2715166](https://doi.org/10.1109/TBDATA.2017.2715166)
4. N. Moustafa, G. Creech and J. Slay, "Big Data Analytics for Intrusion Detection System: Statistical Decision-Making Using Finite Dirichlet Mixture Models," in *Data Analytics and Decision Support for Cybersecurity*, Springer, Cham, 2017, pp. 127–156. [doi:10.1007/978-3-319-59439-2_5](https://doi.org/10.1007/978-3-319-59439-2_5)
5. M. Sarhan, S. Layeghy, N. Moustafa and M. Portmann, "NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems," in *Big Data Technologies and Applications (BDTA/WiCON 2020)*, LNICST vol. 371, Springer, Cham, 2021, pp. 117–135. [doi:10.1007/978-3-030-72802-1_9](https://doi.org/10.1007/978-3-030-72802-1_9)

**TON_IoT — eight:**

1. N. Moustafa, "A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets," *Sustainable Cities and Society*, vol. 72, art. 102994, 2021. [doi:10.1016/j.scs.2021.102994](https://doi.org/10.1016/j.scs.2021.102994)
2. T. M. Booij, I. Chiscop, E. Meeuwissen, N. Moustafa and F. T. H. den Hartog, "ToN_IoT: The Role of Heterogeneity and the Need for Standardization of Features and Attack Types in IoT Network Intrusion Data Sets," *IEEE Internet of Things Journal*, vol. 9, no. 1, pp. 485–496, 2022. [doi:10.1109/JIOT.2021.3085194](https://doi.org/10.1109/JIOT.2021.3085194)
3. A. Alsaedi, N. Moustafa, Z. Tari, A. Mahmood and A. Anwar, "TON_IoT Telemetry Dataset: A New Generation Dataset of IoT and IIoT for Data-Driven Intrusion Detection Systems," *IEEE Access*, vol. 8, pp. 165130–165150, 2020. [doi:10.1109/ACCESS.2020.3022862](https://doi.org/10.1109/ACCESS.2020.3022862)
4. N. Moustafa, M. Keshky, E. Debiez and H. Janicke, "Federated TON_IoT Windows Datasets for Evaluating AI-Based Security Applications," *2020 IEEE 19th International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom)*, 2020, pp. 848–855. [doi:10.1109/TrustCom50675.2020.00114](https://doi.org/10.1109/TrustCom50675.2020.00114)
5. N. Moustafa, M. Ahmed and S. Ahmed, "Data Analytics-Enabled Intrusion Detection: Evaluations of ToN_IoT Linux Datasets," *2020 IEEE 19th International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom)*, 2020, pp. 727–735. [doi:10.1109/TrustCom50675.2020.00100](https://doi.org/10.1109/TrustCom50675.2020.00100)
6. N. Moustafa, "New Generations of Internet of Things Datasets for Cybersecurity Applications based Machine Learning: TON_IoT Datasets," *Proceedings of the eResearch Australasia Conference*, Brisbane, Australia, 2019.
7. N. Moustafa, "A systemic IoT-Fog-Cloud architecture for big-data analytics and cyber security systems: a review of fog computing," arXiv:1906.01055, 2019. [arXiv:1906.01055](https://arxiv.org/abs/1906.01055)
8. J. Ashraf, M. Keshk, N. Moustafa, M. Abdel-Basset, H. Khurshid, A. D. Bakhshi and R. R. Mostafa, "IoTBoT-IDS: A novel statistical learning-enabled botnet detection framework for protecting networks of smart cities," *Sustainable Cities and Society*, vol. 72, art. 103041, 2021. [doi:10.1016/j.scs.2021.103041](https://doi.org/10.1016/j.scs.2021.103041)

</details>

Two of this project's findings are corrections to the *documentation* around these datasets rather than to the authors' work: the row shortfall in `Train_Test_Network.csv` noted above, and the absence of any TTL field in the flow CSVs, because Zeek's `conn.log` exports none. Both are recorded as measured facts in [`data/README.md`](data/README.md).

## Further reading

- [`deviations.md`](deviations.md) — per-decision rationale: what departed from the approved proposal, and why.
