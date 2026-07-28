"""UNSW-NB15 <-> TON_IoT Network feature/label alignment (Phase 2).

The two datasets do not share a schema. This module is the single documented source for the
shared feature subspace, categorical normalization, the shared label space, and the
identity/leakage drop-list. Keep the mappings here explicit and reviewed by both members.

We deliberately align on the shared ~8-10 flow features rather than reproducing all 49
UNSW-NB15 features (those come from Argus + Bro/Zeek plus ~12 custom algorithms).
"""

from __future__ import annotations

# --- Shared feature subspace -----------------------------------------------------------
# concept -> (unsw_column, toniot_column). Confirm each pairing by comparing distributions;
# units may differ (flow bytes vs IP-layer bytes) and must be reconciled before trusting.
FEATURE_MAP: dict[str, tuple[str, str]] = {
    "flow_duration": ("dur", "duration"),
    "protocol": ("proto", "proto"),
    "service": ("service", "service"),
    "conn_state": ("state", "conn_state"),
    "src_bytes": ("sbytes", "src_bytes"),
    "dst_bytes": ("dbytes", "dst_bytes"),
    "src_pkts": ("spkts", "src_pkts"),
    "dst_pkts": ("dpkts", "dst_pkts"),
}

# Derived shared rate features (bytes/sec, packets/sec) built from duration + counts.
# TODO Phase 2: define derivation from FEATURE_MAP ingredients.
DERIVED_FEATURES: tuple[str, ...] = ("bytes_per_sec", "pkts_per_sec")

# --- Leakage / identity drop-list ------------------------------------------------------
# Never feed these to a model: they let it cheat and destroy cross-era transfer.
DROP_COLUMNS: tuple[str, ...] = (
    "id",
    "srcip", "sport", "dstip", "dsport",   # UNSW identity columns
    "src_ip", "src_port", "dst_ip", "dst_port",  # TON_IoT identity columns
    "ts", "stime", "ltime",                # timestamps
    # plus any unmapped *_ip_bytes columns — dropped programmatically in Phase 2.
)

# --- Categorical normalization ---------------------------------------------------------
# TODO Phase 2: lower-case proto/service; map state <-> conn_state codes to a shared set;
# bucket rare/unknown categorical values to "other".
RARE_BUCKET = "other"

# --- Label space -----------------------------------------------------------------------
# Binary headline label: normal (0) / attack (1). UNSW uses `label`; TON_IoT uses `label`.
BINARY_LABEL_COL = {"unsw": "label", "toniot": "label"}

# Shared attack-family map for per-family analysis. Maps each dataset's multiclass column
# (UNSW `attack_cat`, TON_IoT `type`) into a common family vocabulary.
# TODO Phase 2: fill remaining family alignments; confirm against value counts.
SHARED_FAMILIES: dict[str, dict[str, str]] = {
    "unsw": {
        "DoS": "dos",
        "Reconnaissance": "scanning",   # UNSW Reconnaissance <-> TON_IoT scanning
        "Backdoors": "backdoor",
        # ... Fuzzers, Analysis, Exploits, Generic, Shellcode, Worms -> TODO
    },
    "toniot": {
        "dos": "dos",
        "ddos": "ddos",
        "scanning": "scanning",
        "backdoor": "backdoor",
        # ... remaining TON_IoT types -> TODO
    },
}


def build_common_frames() -> None:
    """Load raw datasets, apply FEATURE_MAP/DROP_COLUMNS/label maps, emit harmonized parquet.

    Writes ``data/processed/unsw_common.parquet`` and ``toniot_common.parquet``.
    """
    raise NotImplementedError("Phase 2: feature alignment")
