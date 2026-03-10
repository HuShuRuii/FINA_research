#!/usr/bin/env bash
set -euo pipefail

# Full-parameter Huggett JAX run (paper-aligned grids/hyperparams)
# Usage:
#   bash SRL/run_hugget_hpc_full.sh [OUT_DIR] [LOG_EVERY]

OUT_DIR="${1:-hugget_output_full}"
LOG_EVERY="${2:-10}"
LOG_FILE="${OUT_DIR}/train.log"

mkdir -p "$OUT_DIR"

# Prefer an existing venv if present.
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "$MPLCONFIGDIR"

python -u SRL/hugget_jax.py \
  --out_dir "$OUT_DIR" \
  --log_every "$LOG_EVERY" \
  2>&1 | tee "$LOG_FILE"
