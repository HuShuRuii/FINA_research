#!/usr/bin/env bash
set -euo pipefail

# One-command helper for HPC4 workflow with automatic proxy handling.
# Usage examples:
#   bash SRL/hpc4_auto.sh doctor
#   bash SRL/hpc4_auto.sh connect
#   bash SRL/hpc4_auto.sh start hugget_output_full_v4 10
#   bash SRL/hpc4_auto.sh logs hugget_output_full_v4
#   bash SRL/hpc4_auto.sh fetch hugget_output_full_v4

HPC_HOST="${HPC_HOST:-hpc4}"
REMOTE_DIR="${REMOTE_DIR:-SRL}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ensure_hpc_network() {
  # Load user network helpers if available.
  if [[ -f "${HOME}/.zshrc" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.zshrc" >/dev/null 2>&1 || true
  fi
  if command -v hpc_on >/dev/null 2>&1; then
    hpc_on >/dev/null 2>&1 || true
  else
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  fi
}

ssh_retry() {
  local cmd="$1"
  local n=0
  local max_try=4
  until ssh -o BatchMode=yes -o ConnectTimeout=10 "${HPC_HOST}" "${cmd}"; do
    n=$((n + 1))
    if [[ ${n} -ge ${max_try} ]]; then
      echo "[hpc4_auto] SSH failed after ${max_try} attempts." >&2
      return 1
    fi
    echo "[hpc4_auto] SSH retry ${n}/${max_try}..." >&2
    sleep 2
  done
}

scp_retry() {
  local src="$1"
  local dst="$2"
  local n=0
  local max_try=4
  until scp "${src}" "${dst}"; do
    n=$((n + 1))
    if [[ ${n} -ge ${max_try} ]]; then
      echo "[hpc4_auto] SCP failed after ${max_try} attempts: ${src}" >&2
      return 1
    fi
    echo "[hpc4_auto] SCP retry ${n}/${max_try}: ${src}" >&2
    sleep 2
  done
}

remote_home() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${HPC_HOST}" "printf %s \"\$HOME\""
}

remote_workdir() {
  local home
  home="$(remote_home)"
  printf "%s/%s" "${home}" "${REMOTE_DIR}"
}

doctor() {
  ensure_hpc_network
  echo "[hpc4_auto] proxy env:"
  echo "HTTP_PROXY=${HTTP_PROXY:-<empty>}"
  echo "HTTPS_PROXY=${HTTPS_PROXY:-<empty>}"
  echo "ALL_PROXY=${ALL_PROXY:-<empty>}"
  echo "NO_PROXY=${NO_PROXY:-<empty>}"
  echo "[hpc4_auto] testing ssh..."
  ssh_retry "echo CONNECT_OK && hostname && pwd"
}

connect() {
  ensure_hpc_network
  ssh_retry "echo CONNECT_OK && hostname && pwd"
}

start_train() {
  local out_dir="${1:-hugget_output_full}"
  local log_every="${2:-10}"
  local remote_dir
  ensure_hpc_network
  remote_dir="$(remote_workdir)"
  ssh_retry "mkdir -p ${remote_dir}"
  scp_retry "${LOCAL_DIR}/SRL/hugget_jax.py" "${HPC_HOST}:${remote_dir}/"
  scp_retry "${LOCAL_DIR}/SRL/run_hugget_hpc_full.sh" "${HPC_HOST}:${remote_dir}/"
  ssh_retry "cd ${remote_dir} && pwd && ls -l && chmod +x run_hugget_hpc_full.sh && nohup bash ./run_hugget_hpc_full.sh ${out_dir} ${log_every} > launch.log 2>&1 & echo STARTED && sleep 2 && tail -n 30 launch.log"
}

logs() {
  local out_dir="${1:-hugget_output_full}"
  local remote_dir
  ensure_hpc_network
  remote_dir="$(remote_workdir)"
  ssh -t "${HPC_HOST}" "tail -f ${remote_dir}/${out_dir}/train.log"
}

fetch() {
  local out_dir="${1:-hugget_output_full}"
  local dst="${2:-${LOCAL_DIR}/${out_dir}}"
  local remote_dir
  ensure_hpc_network
  remote_dir="$(remote_workdir)"
  mkdir -p "${dst}"
  scp -r "${HPC_HOST}:${remote_dir}/${out_dir}/*" "${dst}/"
  echo "[hpc4_auto] fetched to ${dst}"
}

stop_all() {
  ensure_hpc_network
  ssh_retry "pkill -f 'python -u .*hugget_jax.py' || true; echo STOPPED"
}

usage() {
  cat <<'EOF'
Usage:
  bash SRL/hpc4_auto.sh doctor
  bash SRL/hpc4_auto.sh connect
  bash SRL/hpc4_auto.sh start [OUT_DIR] [LOG_EVERY]
  bash SRL/hpc4_auto.sh logs [OUT_DIR]
  bash SRL/hpc4_auto.sh fetch [OUT_DIR] [LOCAL_DST]
  bash SRL/hpc4_auto.sh stop
EOF
}

cmd="${1:-}"
case "${cmd}" in
  doctor) shift; doctor "$@" ;;
  connect) shift; connect "$@" ;;
  start) shift; start_train "$@" ;;
  logs) shift; logs "$@" ;;
  fetch) shift; fetch "$@" ;;
  stop) shift; stop_all "$@" ;;
  *) usage ;;
esac
