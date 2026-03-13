#!/usr/bin/env bash
set -euo pipefail

# Full-parameter Huggett JAX run (paper-aligned grids/hyperparams)
# Usage:
#   bash SRL/run_hugget_hpc_full.sh [OUT_DIR] [LOG_EVERY] [extra args...]

OUT_DIR="${1:-hugget_output_full}"
LOG_EVERY="${2:-10}"
shift "$(( $# >= 1 ? 1 : 0 ))" || true
shift "$(( $# >= 1 ? 1 : 0 ))" || true
EXTRA_ARGS=("$@")
LOG_FILE="${OUT_DIR}/train.log"

mkdir -p "$OUT_DIR"

# Prefer an existing venv if present.
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "$MPLCONFIGDIR"

SCRIPT_PATH="SRL/hugget_jax.py"
if [[ -f "hugget_jax.py" ]]; then
  SCRIPT_PATH="hugget_jax.py"
elif [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Could not find hugget_jax.py from $(pwd)" >&2
  exit 1
fi

python -u "$SCRIPT_PATH" \
  --out_dir "$OUT_DIR" \
  --log_every "$LOG_EVERY" \
  --solve_steady_state \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
