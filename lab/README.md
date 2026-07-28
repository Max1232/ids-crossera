# lab/ — RQ3 stretch: live 2026 malware probe (Phase 8, OPTIONAL)

The true decade-forward test. **Do not start until Phases 1–7 land.** This is cut first if
time runs short — the public-dataset core is a complete project on its own.

## Hard gates (both required before any download or detonation)

1. **Authorization.** Confirm with the instructor and check Northeastern policy before
   downloading or detonating any live samples. Do not skip this.
2. **Verified air-gap.** Dedicated VM/host with networking confined to an isolated virtual
   segment — no route to the internet or campus net. Use disposable snapshots.

If either gate cannot be met: **drop the probe** with no impact on the core project.

## Procedure (once gates pass)

- Pull a small number of **catalogued** samples from a reputable repo (e.g. MalwareBazaar);
  record hashes/provenance.
- Capture traffic with **tcpdump / Zeek**.
- Extract **only the ~10 aligned features** from `src/schema_map.py` (not all 49 — this is what
  keeps RQ3 tractable).
- Map captures into the shared subspace and run the trained + transfer-corrected models as a
  **qualitative** real-world check.
