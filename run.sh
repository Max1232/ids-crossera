#!/usr/bin/env bash
#
# run.sh — end-to-end reproduction of the ids-crossera pipeline.
#
# Runs the whole CS4100 project pipeline from harmonized data through figures:
#   preprocess -> baselines -> from-scratch models -> cross-era eval -> transfer -> plots
#
# Raw datasets must already be downloaded into data/raw/ (see data/README.md).
# Modules are currently Phase-0 stubs; each step below is a placeholder wired to the
# eventual entry point and will no-op until the corresponding phase is implemented.
#
# Usage:
#   ./run.sh [-h]
#
# Options:
#   -h, --help   Show this help and exit.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PYTHON="${PYTHON:-python3}"

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^#\s\?//'
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
    # TODO Phase 2: "${PYTHON}" -m src.schema_map --build

    echo "==> [Phase 3] Fit preprocessor on UNSW-train, apply to UNSW-test + TON_IoT"
    # TODO Phase 3: "${PYTHON}" -m src.preprocess

    echo "==> [Phase 4] Library baselines (Dummy / RF / SVM / Decision Tree)"
    # TODO Phase 4: "${PYTHON}" -m src.models.baselines

    echo "==> [Phase 5] From-scratch models (logreg first, then MLP)"
    # TODO Phase 5: "${PYTHON}" -m src.models.scratch_logreg
    # TODO Phase 5: "${PYTHON}" -m src.models.scratch_mlp

    echo "==> [Phase 6] Zero-shot cross-era evaluation (RQ1)"
    # TODO Phase 6: "${PYTHON}" -m src.evaluate --regimes

    echo "==> [Phase 7] Transfer-learning recovery curve (RQ2)"
    # TODO Phase 7: "${PYTHON}" -m src.transfer

    echo "==> [Phase 9] Figures -> reports/figures/"
    # TODO Phase 9: "${PYTHON}" -m src.plots

    echo "==> done."
}

main "$@"
