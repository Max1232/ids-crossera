#!/usr/bin/env bash
#
# run.sh — end-to-end reproduction of the ids-crossera pipeline.
#
# Runs the whole CS4100 project pipeline from harmonized data through figures:
#   preprocess -> baselines -> from-scratch models -> cross-era eval -> transfer -> plots
#
# Raw datasets must already be downloaded into data/raw/ (see data/README.md).
# Phases 2-4, 6, 7 and 9 all run for real, so a bare `./run.sh` reproduces every figure
# in reports/figures/ from the raw CSVs. Phase 5 is the one banner without an invocation:
# its two entry points print val-fold scores and log nothing (see the note at that step).
# Phases 6 and 7 dominate the runtime (~3 min and ~2 min) because both re-fit every model
# from its factory rather than caching one -- Phase 6 runs three conditions; Phase 9 adds
# a few seconds because it only re-reads committed artifacts.
# Re-running is safe and idempotent: reports/metrics.csv, reports/per_family_metrics.csv
# and the two JSON sidecars are upserted or rewritten deterministically, and the figures
# are redrawn from them, so a second run leaves all of them byte-identical.
#
# Usage:
#   ./run.sh [-h]
#
# Options:
#   -h, --help   Show this help and exit.
#
# Environment:
#   PYTHON       Interpreter to run the pipeline with. Defaults to the project venv
#                (./.venv/bin/python) when it exists, otherwise python3. Override to
#                point at another interpreter, e.g. PYTHON=python3.12 ./run.sh

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default to the project venv so a fresh clone reproduces with a bare `./run.sh` -- the graded
# claim is a *one-command* end-to-end run, and bare `python3` is the system interpreter, which does
# not have numpy/pandas/sklearn and fails check_deps below. An explicit PYTHON= still wins, and the
# fallback stays `python3` for the case where no venv has been created yet (check_deps then reports
# the missing packages, which is the correct error at that point).
if [[ -z "${PYTHON:-}" && -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python"
fi
readonly PYTHON="${PYTHON:-python3}"

# Print the leading comment block (everything after the shebang, up to the first non-comment line)
# as the help text, so extending the header above cannot desynchronize `--help` from it.
usage() {
    awk 'NR == 1 { next } /^#/ { sub(/^#[[:space:]]?/, ""); print; next } { exit }' \
        "${BASH_SOURCE[0]}"
}

check_deps() {
    if ! command -v "${PYTHON}" >/dev/null 2>&1; then
        echo "error: '${PYTHON}' not found on PATH. Install Python 3.11/3.12 or set PYTHON=." >&2
        exit 1
    fi
    if ! "${PYTHON}" -c "import numpy, pandas, sklearn" >/dev/null 2>&1; then
        echo "error: required packages missing. Run: pip install -r requirements.txt" >&2
        exit 1
    fi
}

main() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help) usage; exit 0 ;;
            *) echo "error: unknown argument '$1'" >&2; usage; exit 1 ;;
        esac
    done

    check_deps
    cd "${SCRIPT_DIR}"

    echo "==> [Phase 2] Feature alignment + harmonized parquet"
    "${PYTHON}" -m src.schema_map --build

    echo "==> [Phase 3] Fit preprocessor on UNSW-train, apply to UNSW-test + TON_IoT"
    "${PYTHON}" -m src.preprocess

    echo "==> [Phase 4] Library baselines (Dummy / RF / SVM / Decision Tree)"
    # Hyperparameters are already locked in baselines.TUNED_PARAMS (tuned on the UNSW val fold),
    # so this trains the four locked models and upserts their in-distribution rows. The grid
    # search is deliberately NOT part of the pipeline -- `python -m src.models.baselines --tune`
    # re-runs it on demand and prints to stdout only.
    "${PYTHON}" -m src.models.baselines

    echo "==> [Phase 5] From-scratch models (logreg first, then MLP)"
    # Deliberately not invoked: both entry points print val-fold scores and log nothing, and the
    # scratch models' metrics.csv rows belong to Phase 6's run_id (which fits the same locked
    # models). Running them here would double their fit cost for no logged output.
    # Manual: "${PYTHON}" -m src.models.scratch_logreg / -m src.models.scratch_mlp

    echo "==> [Phase 6] Zero-shot cross-era evaluation (RQ1)"
    # The primary result. Fits every Phase 4-5 model on the UNSW train fold and scores it in both
    # regimes -- in-distribution on UNSW-test, then the SAME fitted model zero-shot on TON_IoT with
    # no retraining and no refit of the Preprocessor -- plus the `proto` and `conn_state` ablations
    # as second and third conditions, each under its own run_id. Both ablations land at d=18 (each
    # categorical encodes to four one-hots) and are DIFFERENT experiments: only the run_id and the
    # notes column tell them apart, never the width. ~3 minutes: the from-scratch models are re-fit
    # from their factories on every run, once per condition, because nothing is persisted. That is
    # deliberate -- a model cache would make this faster and stop it being a from-raw-data
    # reproduction.
    "${PYTHON}" -m src.evaluate --regimes

    echo "==> [Phase 7] Transfer-learning recovery curve (RQ2)"
    # The secondary result. Splits TON_IoT once into a permanent test half and a fine-tune pool,
    # then adapts every Phase 4-5 model on stratified 1/5/10/25% fractions of the pool -- MLP
    # head-only with both hidden layers frozen, scratch logreg warm-started, the classical models
    # refit on the target sample. Every point, including the re-measured zero-shot point and the
    # full-budget ceiling, is scored on the SAME frozen test half; one run_id per budget, so the
    # fractions cannot overwrite each other. ~2 minutes, dominated by the six source fits on the
    # UNSW train fold (the scratch logreg alone is ~32 s) which are re-done here rather than
    # cached, for the same from-raw-data reason as Phase 6. The Preprocessor is never refit.
    "${PYTHON}" -m src.transfer

    echo "==> [Phase 9] Figures -> reports/figures/"
    # Seven figures plus reports/figures/README.md (report-ready captions + per-row provenance).
    # Seconds, not minutes: this re-reads the four committed artifacts Phases 4-7 just wrote
    # (metrics.csv and the confusion / ROC / per-family sidecars) and re-renders from them. It
    # never fits, transforms or re-derives anything, which is what stops `./run.sh` running a
    # phase twice and what makes a figure unable to disagree with the table it illustrates.
    # Legend and label positions come from the committed reports/figures/layout.json, so a human
    # tuning (`python -m src.plots --tune`) is reproduced here rather than lost -- see
    # src/figure_layout.py. An absent or empty layout.json renders the coded defaults.
    "${PYTHON}" -m src.plots

    echo "==> done."
}

main "$@"
