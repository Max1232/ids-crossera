"""UNSW-NB15 <-> TON_IoT Network feature/label alignment (Phase 2).

The two datasets do not share a schema. This module is the single documented source for the
shared feature subspace, categorical normalization, the shared label space, and the
identity/leakage drop-list. Keep the mappings here explicit and reviewed by both members.

We deliberately align on the shared ~8-10 flow features rather than reproducing all 49
UNSW-NB15 features (those come from Argus + Bro/Zeek plus ~12 custom algorithms).

Every pairing, collapse map, disposition and percentage below comes from
``reports/schema_catalogue.md``, which measured them from the delivered files themselves (UNSW
train 175,341 rows / test 82,332 rows; TON_IoT ``Train_Test_Network.csv`` 211,043 rows). Section
markers (§) point into that file. Do not "tidy" a mapping here without re-measuring -- several of
them look wrong and are not, and two of the entries this module used to carry were wrong in ways
that failed *silently*.

Two delivery facts that bite any code reading the raw files (§4.1):

* All three CSVs carry a UTF-8 BOM. Read every one with ``encoding="utf-8-sig"`` or the first
  column name arrives BOM-prefixed and every lookup of ``id`` / ``src_ip`` raises a ``KeyError``
  pointing at the wrong thing.
* The delivered UNSW header is entirely lower-case -- ``sload``, ``dload``, ``spkts``, ``dpkts``,
  ``sjit``, ``djit``, ``label`` -- while the upstream description file and this project's own
  prose write ``Sload``/``Dload``/``Spkts``/``Sjit``/``Label``. Code copied from the docs raises.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import (
    TONIOT_COMMON,
    TONIOT_CSV,
    UNSW_COMMON,
    UNSW_TEST_CSV,
    UNSW_TRAIN_CSV,
    set_seeds,
)

# --- Shared feature subspace -----------------------------------------------------------
# concept -> (unsw_column, toniot_column). Each pairing was checked against the delivered
# distributions; §1 records the per-pairing verdict. Summary of what Phase 2 owes each one:
#
#   flow_duration  NORMALIZE. Same unit (seconds), incomparable support: UNSW `dur` is hard-capped
#                  at 60 s by the capture design (max 59.999989), TON_IoT `duration` runs to
#                  93,516.93 s (~26 h). log1p, then clip/winsorize the TON_IoT tail before
#                  z-scoring, or a UNSW-train-fitted scaler puts almost all of TON_IoT in one bin.
#   protocol       COLLAPSE to {tcp, udp, icmp, other}. UNSW has 133 Argus protocol names (31 of
#                  them under 100 rows) and {tcp,udp,icmp} covers only 81.69% of its own train
#                  rows; TON_IoT has exactly 3 levels, all of them present in UNSW. Anything
#                  finer is a UNSW-only vocabulary the model cannot exercise cross-era. Caveat:
#                  UNSW `icmp` is 15 train rows and 0 test rows, yet fires on 281 TON_IoT rows.
#   service        COLLAPSE to {-, dns, http, ftp, ssl, other} -- those five shared levels cover
#                  93.3% / 95.2% / 99.8% of the three splits. First split TON_IoT's `;`-joined
#                  MULTI-VALUED cells (Zeek joins concurrently-detected services, e.g.
#                  "smb;gssapi", 18 rows) and take the first token, or the one-hot gains a phantom
#                  level. `-` IS A REAL CATEGORY meaning "Zeek detected no application protocol";
#                  it is the modal value on both sides (53.71% UNSW train / 62.56% TON_IoT) and
#                  must be kept as its own level, never imputed as a missing value (§3.2, §4.2).
#   conn_state     COLLAPSE via STATE_COLLAPSE below -- the two raw vocabularies share ZERO
#                  tokens (Argus vs Zeek), so the raw columns are never comparable (§3.3, §4.5).
#   src_bytes /    The TON_IoT side is `src_ip_bytes`/`dst_ip_bytes`, NOT `src_bytes`/`dst_bytes`.
#   dst_bytes      UNSW `sbytes`/`dbytes` are IP-level counts -- `sbytes` has a floor of 28 B
#                  (20 B IPv4 + 8 B UDP header) and is never 0 -- whereas TON_IoT
#                  `src_bytes`/`dst_bytes` are Zeek *payload* bytes reconstructed from TCP
#                  sequence numbers and are 0 on 65.46% / 70.55% of rows. `*_ip_bytes` are total
#                  IP bytes, 0 on only 8.10% / 39.44%, and that 39.44% matches UNSW `dbytes`'s
#                  48.07% zero rate because both mean "destination never replied". The payload
#                  columns are also numerically unusable: 67 rows exceed 1e8 bytes, topping out
#                  at 3.89e9 in a single flow, and 0.47% report more payload than total IP bytes.
#                  Then log1p both sides. (§4.6 -- this is a correction, see DROP_COLUMNS.)
#   src_pkts /     KEEP AS-IS (log1p for the heavy tail). The cleanest two pairings: same notion,
#   dst_pkts       same unit, comparable zero rates. Only edge difference is UNSW's floor of 1.
FEATURE_MAP: dict[str, tuple[str, str]] = {
    "flow_duration": ("dur", "duration"),
    "protocol": ("proto", "proto"),
    "service": ("service", "service"),
    "conn_state": ("state", "conn_state"),
    "src_bytes": ("sbytes", "src_ip_bytes"),   # NOT toniot `src_bytes` -- payload, not IP (§4.6)
    "dst_bytes": ("dbytes", "dst_ip_bytes"),   # NOT toniot `dst_bytes` -- payload, not IP (§4.6)
    "src_pkts": ("spkts", "src_pkts"),
    "dst_pkts": ("dpkts", "dst_pkts"),
}

# --- Derived shared rate features ------------------------------------------------------
# Computed identically on both sides from the FEATURE_MAP ingredients above (§1):
#   bytes_per_sec = (sbytes + dbytes) / dur   ==  (src_ip_bytes + dst_ip_bytes) / duration
#   pkts_per_sec  = (spkts  + dpkts)  / dur   ==  (src_pkts     + dst_pkts)     / duration
# Derived in `_harmonize()` step 5 below, guarded against duration == 0 (see ZERO_DURATION_FLAG).
#
# Do NOT "sanity-check" these against UNSW `rate`/`sload`/`dload` (§4.8). Those carry an Argus
# (n-1)/n inter-packet-interval correction, and `sload` is in *bits* per second: on the median
# UNSW flow (spkts = 2) a naive bytes-rate is 2x `sload`/8, which is neither a bug nor the
# bits/bytes factor. All three are drop-unmapped anyway -- TON_IoT has no rate column, so the
# rates must be derived independently on both sides regardless.
DERIVED_FEATURES: tuple[str, ...] = ("bytes_per_sec", "pkts_per_sec")

# Divide-by-zero guard for the two rates above: `np.where(dur > 0, x / dur, 0.0)`. An explicit
# 0.0 -- not NaN, and not a 1e-6 epsilon, which would produce astronomically large rates.
#
# ZERO_DURATION_FLAG is then carried as its OWN named feature rather than being left implicit
# inside that guard, and that separation is the point (§4.7). `duration == 0` holds on 1.52% of
# UNSW train rows but 28.44% of TON_IoT's -- a 19x difference -- and ITS MEANING INVERTS ACROSS
# ERAS: those rows are 99.0% normal in 2015 but only 21.2% normal in 2019-2020, against base rates
# of 31.9% and 23.7% normal. So whatever sentinel the rate columns take on becomes a learned
# normal-class marker in training and then fires on 60,013 target rows meaning the opposite.
# Naming it makes that a reportable (and ablatable) drift finding instead of a strong drift signal
# smuggled into the model inside a divide-by-zero fallback.
ZERO_DURATION_FLAG = "zero_duration"

# --- Leakage / identity drop-list ------------------------------------------------------
# Never feed these to a model: they let it cheat and destroy cross-era transfer.
#
# Trimmed against the real headers (§4.1b). Seven names this tuple used to carry exist in NEITHER
# delivered file and have been removed: the four abbreviated UNSW source/destination address and
# port names, the two abbreviated UNSW flow start/end time names, and TON_IoT's `ts`. (§4.1b lists
# all seven against both real headers, column by column; they are left unquoted here so that a
# grep for them over this file stays a clean check.) The address/port four are *full*-dataset UNSW
# names, never present in the 45-column partitioned set; the two time columns were dropped when
# that partition was built; and `ts` is documented as TON_IoT feature #1 of 46 but is absent from
# the delivered 44 columns.
#
# Consequence worth stating plainly: THERE ARE NO TIMESTAMP COLUMNS LEFT TO DROP. The old
# "# timestamps" comment was worse than useless -- it read as though a temporal leakage vector had
# been identified and handled, when neither dataset delivers one. The live leakage vectors are the
# two named below, and they are worse than the old list implied.
#
# Phase 2 must drop these by RAW column name, BEFORE renaming anything through FEATURE_MAP: the
# TON_IoT raw columns `src_bytes`/`dst_bytes` collide with the FEATURE_MAP *concept* keys of the
# same name, so dropping after the rename would delete the harmonized byte features instead.
DROP_COLUMNS: tuple[str, ...] = (
    # Leakage. UNSW `id` is the 1-based row index of the partitioned file, strictly increasing,
    # and -- because that partition was built by concatenating per-class blocks -- monotonically
    # informative about `attack_cat`. TON_IoT `src_ip` takes only 51 distinct values across
    # 211,043 rows (`dst_ip` only 753), so source IP alone is close to a label lookup table for
    # the entire target era (§4.1b).
    "id",                                         # UNSW row index, 1..N, unique per row
    "src_ip", "src_port", "dst_ip", "dst_port",   # TON_IoT identity columns
    # Not leakage -- superseded. These are TON_IoT's Zeek *payload* byte counts; the mapped byte
    # columns are `src_ip_bytes`/`dst_ip_bytes` (see FEATURE_MAP and §4.6). This is the exact
    # opposite of the note that used to sit here, which said to drop "any unmapped *_ip_bytes
    # columns" -- backwards. Total IP bytes is the only notion UNSW offers, so it is the notion
    # kept, and the payload columns are the unmapped ones.
    "src_bytes", "dst_bytes",                     # TON_IoT payload bytes, 0 on 65.46% / 70.55%
)

# --- Categorical normalization ---------------------------------------------------------
# Delivered in the `_collapse_*` helpers below: `proto`/`service` are lower-cased and stripped,
# TON_IoT `service` is split on ";" with the first token kept, `proto` collapses to
# {tcp, udp, icmp, other} and `service` to {-, dns, http, ftp, ssl, other}, `state`/`conn_state` map
# through STATE_COLLAPSE, and anything else buckets to RARE_BUCKET.
#
# RARE_BUCKET is required for the IN-DISTRIBUTION split too, not only for cross-era: UNSW train
# and test ship different `state` vocabularies (`ECO`/`PAR`/`URN`/`no` train-only, `ACC`/`CLO`
# test-only), so an encoder fitted on UNSW-train alone meets unseen levels on UNSW-test. Bucket
# before encoding, or fit with handle_unknown="infrequent_if_exist" (§4.10).
RARE_BUCKET = "other"

# The kept levels for the two lexically-alignable categoricals. Anything outside them becomes
# RARE_BUCKET. Both sets come straight from §3.1 / §3.2 and are deliberately small:
#   SHARED_PROTOCOLS -- TON_IoT's entire vocabulary, all three present in UNSW. Everything else is
#                       a UNSW-only Argus name the model cannot exercise cross-era. Note the
#                       hazard this creates and which README.md's Limitations section owns: the
#                       18.31% of UNSW train rows that land in `other` are 91% attack, and 0% of
#                       TON_IoT rows land there -- hence the `proto` ablation in Phase 6.
#   SHARED_SERVICES  -- the five levels common to both sides, covering 93.34% / 95.21% / 99.79% of
#                       the three splits. `-` IS A LEVEL ("Zeek detected no application protocol"),
#                       modal on both sides; it is never imputed as missing.
SHARED_PROTOCOLS: frozenset[str] = frozenset({"tcp", "udp", "icmp"})
SHARED_SERVICES: frozenset[str] = frozenset({"-", "dns", "http", "ftp", "ssl"})

# --- UNSW `state` <-> TON_IoT `conn_state` collapse ------------------------------------
# THE TWO RAW VOCABULARIES SHARE ZERO TOKENS: UNSW ships Argus transaction states, TON_IoT ships
# Zeek `conn_state` codes (§3.3). There is nothing to align lexically, so this hand-written
# 3-way-plus-other collapse -- taken verbatim from §4.5 of reports/schema_catalogue.md, grounded
# in the Argus state definitions in NUSW-NB15_features.csv and the Zeek definitions in
# bro_log_vars.pdf -- is the only way the concept survives at all. Dropping it is also defensible.
#
# Exhaustive over every DELIVERED level on both sides (11 UNSW across both splits, 13 TON_IoT), so
# no row falls through. Argus codes that the upstream description file lists but neither split
# delivers (`ECR`, `MAS`, `TST`, `TXD`, `URH`, `-`) are deliberately absent: an unexpected level
# here should raise, not be silently bucketed.
#
# THE COLLAPSE IS LOSSY AND ASYMMETRIC, which is the whole reason it is spelled out here:
#
#   coarse level    UNSW train coverage    TON_IoT coverage
#   completed             44.38%                32.30%
#   reset                  0.05%                23.68%     <-- ~470x
#   no_response           55.56%                32.97%
#   other                  0.01%                11.06%     <-- ~1100x
#
# A model that never saw a reset in training gets a one-hot column that fires on nearly a quarter
# of the target era. So: RUN THE CROSS-ERA EVALUATION ONCE WITH THIS FEATURE AND ONCE WITHOUT IT,
# and report both. If the delta is large, the feature is measuring the Argus->Zeek instrumentation
# change rather than concept drift -- exactly the confound this project exists not to bundle into
# its RQ1 number.
#
# `CON` -> `no_response` is the single judgement call in the table (13,152 UNSW train rows, 7.5%):
# Argus `CON` means "connected, no state transition observed", which sits closer to Zeek
# `S1`/`OTH` than to `FIN`. Revisit it if the with/without test above shows sensitivity.
#
# The `other` level happens to spell the same as RARE_BUCKET but is a different idea: these are
# named codes deliberately grouped, not infrequent values swept aside.
STATE_COLLAPSE: dict[str, dict[str, str]] = {
    "unsw": {      # Argus `state` -- all 11 levels delivered across the two splits
        "FIN": "completed",
        "CLO": "completed",
        "RST": "reset",
        "INT": "no_response",   # modal level, 46.9% -- means "no state recorded"
        "REQ": "no_response",
        "ACC": "no_response",
        "CON": "no_response",   # the judgement call; see above
        "ECO": "other",
        "PAR": "other",
        "URN": "other",
        "no": "other",          # junk level, 1 row
    },
    "toniot": {    # Zeek `conn_state` -- all 13 delivered levels
        "SF": "completed",
        "S1": "completed",
        "S2": "completed",
        "S3": "completed",
        "REJ": "reset",
        "RSTO": "reset",
        "RSTR": "reset",
        "RSTOS0": "reset",
        "RSTRH": "reset",
        "S0": "no_response",
        "SH": "no_response",
        "SHR": "no_response",
        "OTH": "other",
    },
}

# --- Label space -----------------------------------------------------------------------
# Binary headline label: normal (0) / attack (1). UNSW uses `label`; TON_IoT uses `label`.
# Verified correct as written in §3.4: the encoding agrees across both datasets and is a strict
# function of the multiclass column on both sides (cross-tabbed -- no row carries `Normal`/
# `normal` with label = 1, or an attack family with label = 0).
BINARY_LABEL_COL = {"unsw": "label", "toniot": "label"}

# The per-dataset multiclass column that SHARED_FAMILIES is keyed on. Named here rather than
# inlined in build_common_frames() so the two spellings live next to the map that consumes them:
# UNSW calls it `attack_cat` with capitalized values, TON_IoT calls it `type` with lower-case ones
# (§3.5). Both frames emit the harmonized result as FAMILY_LABEL_COL.
MULTICLASS_LABEL_COL = {"unsw": "attack_cat", "toniot": "type"}
FAMILY_LABEL_COL = "family"

# Which side of each FEATURE_MAP / STATE_COLLAPSE / SHARED_FAMILIES pair a dataset reads.
SIDES: tuple[str, ...] = ("unsw", "toniot")

# Shared attack-family map for per-family analysis. Maps each dataset's multiclass column
# (UNSW `attack_cat`, TON_IoT `type`) into a common family vocabulary. Taken from §4.4.
#
# Only THREE attack families plus the benign class have a genuine counterpart on both sides. Every
# delivered level is now named explicitly and the non-shared ones map to None, so an unmapped
# level raises a KeyError at build time instead of being silently dropped -- which is exactly how
# the previous version of this dict failed. It keyed the PLURAL spelling of `Backdoor`: that is the
# form used in NUSW-NB15_features.csv and UNSW-NB15_LIST_EVENTS.csv, but it NEVER appears in the
# partitioned CSVs, where the delivered `attack_cat` value is `Backdoor`, SINGULAR. So that entry
# matched zero rows, contributed nothing to the per-family analysis, and said nothing about it.
#
# TON_IoT's `ddos` is also gone from this map: UNSW-NB15 HAS NO DDoS CLASS AT ALL -- `attack_cat`
# has exactly the ten levels listed below, and UNSW-NB15_LIST_EVENTS.csv carries no DDoS
# subcategory either. It therefore is not a shared family and cannot appear in a cross-era
# per-family comparison, and folding its 20,000 rows into `dos` would silently redefine the DoS
# result. (README.md needs the same correction.)
#
# COVERAGE: the per-family breakdown reaches 24,501 of 119,341 UNSW attack rows (20.5%) and 60,000
# of 161,043 TON_IoT attack rows (37.3%). The headline binary result uses ALL rows; only the
# per-family table is restricted. The 20.5% figure belongs in the report so that the narrowness is
# stated rather than implied.
SHARED_FAMILIES: dict[str, dict[str, str | None]] = {
    "unsw": {      # `attack_cat` -- all 10 delivered levels
        "Normal": "normal",
        "DoS": "dos",
        "Backdoor": "backdoor",         # delivered vocabulary is SINGULAR; the docs' plural is a
                                        # zero-row key -- do not "restore" it
        "Reconnaissance": "scanning",   # UNSW Reconnaissance <-> TON_IoT scanning (synonyms)
        # No shared counterpart -> excluded from the per-family cross-era analysis (§4.4).
        "Exploits": None,     # broad IXIA class; TON_IoT splits it across injection/xss/password
        "Generic": None,      # 99% from one IXIA subcategory -- a 2015 generator artifact
        "Fuzzers": None,      # protocol fuzzing; absent from TON_IoT
        "Analysis": None,     # overlaps `scanning` only partially, so mapping it double-counts
        "Shellcode": None,    # OS-specific shellcode; absent from TON_IoT
        "Worms": None,        # absent from TON_IoT, and too small to score (130 train rows)
    },
    "toniot": {    # `type` -- all 10 delivered levels
        "normal": "normal",
        "dos": "dos",
        "backdoor": "backdoor",
        "scanning": "scanning",
        # No shared counterpart.
        "ddos": None,         # absent from UNSW entirely -- do NOT fold into `dos`; see above
        "injection": None,    # closest analogue is UNSW `Exploits`, but not defensibly
        "password": None,     # credential brute-force; no UNSW class
        "ransomware": None,   # no 2015 counterpart -- the class the drift story is really about
        "xss": None,          # no UNSW class
        "mitm": None,         # no UNSW class; also the one class short of its documented count
    },
}


# --- Harmonized output schema ----------------------------------------------------------
# The exact column list both parquets carry, in order. Identical on both sides by construction --
# that is the whole point of the shared subspace, and Phase 3's Preprocessor is fit on the UNSW
# frame and applied unchanged to the TON_IoT one, so any divergence here is a Phase 3 crash.
#
# `split` is carried on BOTH frames even though only UNSW needs it: UNSW_COMMON is the
# concatenation of the two delivered partitions (train 175,341 + test 82,332) and Phase 3 must be
# able to fit on train alone, while TON_IoT ships as one file and takes the constant "full".
# Keeping the column on both sides is what preserves the identical-schema guarantee above.
CATEGORICAL_FEATURES: tuple[str, ...] = ("protocol", "service", "conn_state")
NUMERIC_FEATURES: tuple[str, ...] = (
    "flow_duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", *DERIVED_FEATURES,
)
COMMON_COLUMNS: tuple[str, ...] = (
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
    ZERO_DURATION_FLAG,
    "label",
    FAMILY_LABEL_COL,
    "split",
)

TONIOT_SPLIT = "full"


def _read_raw(path) -> pd.DataFrame:
    """Read one delivered CSV. ``utf-8-sig`` is mandatory -- all three carry a BOM (§4.1)."""
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _collapse_protocol(values: pd.Series) -> pd.Series:
    """Collapse 133 Argus protocol names / 3 Zeek ones to {tcp, udp, icmp, other} (§3.1)."""
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.where(normalized.isin(SHARED_PROTOCOLS), RARE_BUCKET)


def _collapse_service(values: pd.Series) -> pd.Series:
    """Collapse `service` to {-, dns, http, ftp, ssl, other} (§3.2).

    Splits TON_IoT's ``;``-joined multi-valued cells (Zeek joins concurrently-detected services,
    e.g. ``smb;gssapi``) and keeps the first token, or the one-hot gains a phantom level. ``-`` is
    a real level meaning "no application protocol detected" and survives the collapse untouched --
    it is the modal value on both sides and must never be treated as missing.
    """
    normalized = (
        values.astype("string").str.strip().str.lower().str.split(";").str[0]
        .str.strip().astype("string")  # .str.split(...).str[0] falls back to object dtype
    )
    return normalized.where(normalized.isin(SHARED_SERVICES), RARE_BUCKET)


def _collapse_state(values: pd.Series, side: str) -> pd.Series:
    """Map Argus `state` / Zeek `conn_state` through STATE_COLLAPSE (§4.5).

    The two raw vocabularies share zero tokens, so this map is the only thing that makes the
    concept comparable at all. It is exhaustive over every delivered level on both sides, so an
    unseen level is a data-integrity failure and RAISES rather than falling into RARE_BUCKET --
    silently bucketing it would hide exactly the kind of vocabulary drift this project measures.
    """
    mapping = STATE_COLLAPSE[side]
    normalized = values.astype("string").str.strip()
    unknown = sorted(set(normalized.dropna().unique()) - set(mapping))
    if unknown:
        raise ValueError(
            f"{side}: STATE_COLLAPSE has no entry for delivered state level(s) {unknown}. "
            "Add them to schema_map.STATE_COLLAPSE deliberately -- do not bucket them."
        )
    return normalized.map(mapping).astype("string")


def _map_families(values: pd.Series, side: str) -> pd.Series:
    """Map the multiclass column into the shared family vocabulary (§4.4).

    Levels with a genuine counterpart become one of {normal, dos, scanning, backdoor}; the levels
    explicitly mapped to ``None`` become NA and are excluded from the per-family analysis. A level
    that is in neither group RAISES -- that is the whole reason SHARED_FAMILIES names every
    delivered level, since the previous version's plural ``Backdoors`` key failed silently.
    """
    mapping = SHARED_FAMILIES[side]
    normalized = values.astype("string").str.strip()
    unknown = sorted(set(normalized.dropna().unique()) - set(mapping))
    if unknown:
        raise ValueError(
            f"{side}: SHARED_FAMILIES has no entry for delivered "
            f"{MULTICLASS_LABEL_COL[side]} level(s) {unknown}. Map them explicitly (to a shared "
            "family, or to None to exclude them) -- do not bucket them."
        )
    return normalized.map(mapping).astype("string")


def _harmonize(raw: pd.DataFrame, side: str, split: str) -> pd.DataFrame:
    """Take one raw delivered frame to the shared subspace. Steps 2-6 of build_common_frames()."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    column_index = SIDES.index(side)

    label_col = BINARY_LABEL_COL[side]
    multiclass_col = MULTICLASS_LABEL_COL[side]

    # --- Step 2. Drop by RAW name, BEFORE any rename. ----------------------------------
    # This ordering is load-bearing, not stylistic: TON_IoT's raw payload-byte columns
    # `src_bytes`/`dst_bytes` are spelled identically to FEATURE_MAP's harmonized concept keys.
    # Renaming first would make this drop delete the byte features it had just built, silently,
    # because every name involved is legitimate on one reading or the other.
    df = raw.drop(columns=[col for col in DROP_COLUMNS if col in raw.columns])

    # --- Step 3. Rename each FEATURE_MAP pair to its shared concept name. ---------------
    renames = {pair[column_index]: concept for concept, pair in FEATURE_MAP.items()}
    missing = sorted((set(renames) | {label_col, multiclass_col}) - set(df.columns))
    if missing:
        raise ValueError(f"{side}: expected column(s) absent from the delivered file: {missing}")
    df = df.rename(columns=renames)

    # The guard the README asks for by name. If the drop and the rename ever swap places this is
    # the only thing standing between the pipeline and two silently-deleted byte features.
    assert {"src_bytes", "dst_bytes"} <= set(df.columns), (
        f"{side}: byte features destroyed -- DROP_COLUMNS must be applied before the "
        "FEATURE_MAP rename, not after"
    )

    out = pd.DataFrame(index=df.index)

    # --- Step 4. Categorical normalization. --------------------------------------------
    out["protocol"] = _collapse_protocol(df["protocol"])
    out["service"] = _collapse_service(df["service"])
    out["conn_state"] = _collapse_state(df["conn_state"], side)

    # --- Numerics carried through as-is (log1p / scaling is Phase 3's job, not Phase 2's). ---
    duration = pd.to_numeric(df["flow_duration"], errors="raise").astype("float64")
    out["flow_duration"] = duration
    for col in ("src_bytes", "dst_bytes", "src_pkts", "dst_pkts"):
        out[col] = pd.to_numeric(df[col], errors="raise").astype("int64")

    # --- Step 5. Derived rates + the zero-duration flag. --------------------------------
    # An explicit 0.0 on the divide-by-zero branch: not NaN (unrepresentable downstream) and not a
    # 1e-6 epsilon (which would manufacture astronomical rates). The flag is then carried as its
    # OWN named feature, because `duration == 0` inverts across eras -- 1.52% of UNSW rows, 99.0%
    # of them normal, against 28.44% of TON_IoT rows, only 21.2% normal (§4.7). Left implicit
    # inside the guard, that sentinel becomes a learned normal-class marker in 2015 that fires on
    # 60,013 target rows meaning the opposite. Named, it is a reportable and ablatable finding.
    nonzero = duration.to_numpy() > 0
    total_bytes = (out["src_bytes"] + out["dst_bytes"]).to_numpy(dtype="float64")
    total_pkts = (out["src_pkts"] + out["dst_pkts"]).to_numpy(dtype="float64")
    seconds = duration.to_numpy(dtype="float64")
    out["bytes_per_sec"] = np.where(nonzero, total_bytes / np.where(nonzero, seconds, 1.0), 0.0)
    out["pkts_per_sec"] = np.where(nonzero, total_pkts / np.where(nonzero, seconds, 1.0), 0.0)
    out[ZERO_DURATION_FLAG] = (~nonzero).astype("int8")

    # --- Step 6. Labels. ----------------------------------------------------------------
    label = pd.to_numeric(df[label_col], errors="raise").astype("int8")
    unexpected = sorted(set(label.unique()) - {0, 1})
    if unexpected:
        raise ValueError(f"{side}: `{label_col}` carries non-binary value(s) {unexpected}")
    out["label"] = label
    out[FAMILY_LABEL_COL] = _map_families(df[multiclass_col], side)

    out["split"] = pd.Series(split, index=df.index, dtype="string")
    return out.loc[:, list(COMMON_COLUMNS)].reset_index(drop=True)


def build_common_frames() -> None:
    """Load raw datasets, apply FEATURE_MAP/DROP_COLUMNS/label maps, emit harmonized parquet.

    Order of operations, so the two name collisions in this module cannot bite:

    1. Read each CSV with ``encoding="utf-8-sig"`` (all three carry a BOM).
    2. Drop ``DROP_COLUMNS`` by **raw** column name, *before* any rename -- TON_IoT's raw
       ``src_bytes``/``dst_bytes`` share their spelling with FEATURE_MAP's concept keys.
    3. Rename each FEATURE_MAP pair to the shared concept name.
    4. Normalize the categoricals: ``proto``/``service`` collapse, ``STATE_COLLAPSE``,
       ``RARE_BUCKET`` for the remainder.
    5. Derive ``DERIVED_FEATURES`` under the ``np.where(dur > 0, x / dur, 0.0)`` guard and emit
       ``ZERO_DURATION_FLAG`` as its own column.
    6. Map labels through ``BINARY_LABEL_COL`` and ``SHARED_FAMILIES`` (unmapped level -> raise).

    Writes ``data/processed/unsw_common.parquet`` and ``toniot_common.parquet``.

    The two UNSW partitions are concatenated into the single ``unsw_common.parquet`` that
    ``config`` names, tagged by the ``split`` column ("train" / "test"); the TON_IoT frame takes
    the constant "full".
    """
    set_seeds()

    unsw = pd.concat(
        [
            _harmonize(_read_raw(UNSW_TRAIN_CSV), "unsw", "train"),
            _harmonize(_read_raw(UNSW_TEST_CSV), "unsw", "test"),
        ],
        ignore_index=True,
    )
    toniot = _harmonize(_read_raw(TONIOT_CSV), "toniot", TONIOT_SPLIT)

    if list(unsw.columns) != list(toniot.columns):
        raise ValueError(
            "harmonized frames diverged: "
            f"unsw={list(unsw.columns)} toniot={list(toniot.columns)}"
        )

    UNSW_COMMON.parent.mkdir(parents=True, exist_ok=True)
    unsw.to_parquet(UNSW_COMMON, index=False)
    toniot.to_parquet(TONIOT_COMMON, index=False)


def _summarize(name: str, path) -> None:
    """One-screen build report -- shapes, balance, and the two cross-era hazards, per side.

    Reads the emitted file back rather than reusing the in-memory frame, so the summary doubles
    as a round-trip check that the parquet Phase 3 will open is actually loadable.
    """
    frame = pd.read_parquet(path)
    normal_share = float((frame["label"] == 0).mean())
    print(f"{name}: {frame.shape[0]:,} rows x {frame.shape[1]} cols -> {path}")
    for split, part in frame.groupby("split", observed=True):
        share = float((part["label"] == 0).mean())
        print(f"    split={split:<5} n={len(part):>9,}  normal={share:6.2%}")
    print(f"    normal share overall {normal_share:6.2%}   "
          f"{ZERO_DURATION_FLAG}={frame[ZERO_DURATION_FLAG].mean():6.2%}   "
          f"family-mapped={frame[FAMILY_LABEL_COL].notna().mean():6.2%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.schema_map",
        description="Phase 2: build the harmonized UNSW / TON_IoT parquets.",
    )
    parser.add_argument(
        "--build", action="store_true",
        help="build data/processed/{unsw,toniot}_common.parquet from data/raw/",
    )
    args = parser.parse_args(argv)
    if not args.build:
        parser.print_help()
        return 0

    build_common_frames()
    _summarize("unsw_common", UNSW_COMMON)
    _summarize("toniot_common", TONIOT_COMMON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
