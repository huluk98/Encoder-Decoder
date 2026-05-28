#!/usr/bin/env bash
set -euo pipefail

# Show likely torchrun/SFT training processes, the distributed master port, and GPU usage.
#
# Usage:
#   ./scripts/check_training_process.sh
#   MASTER_PORT=29601 ./scripts/check_training_process.sh

MASTER_PORT="${MASTER_PORT:-29573}"
PATTERN="${PATTERN:-run_4gpu_train_eval|train_sft_8gpu|train_contrastive_8gpu|torch.distributed.run|sft.py|evaluate_exact.py}"

echo "== Matching training/eval processes =="
if command -v ps >/dev/null 2>&1; then
  ps -eo pid,ppid,user,stat,etime,command 2>/dev/null \
    | grep -E "${PATTERN}" \
    | grep -v grep || true
else
  echo "ps is not available on this system."
fi

echo
echo "== Master port ${MASTER_PORT} =="
if command -v lsof >/dev/null 2>&1; then
  lsof -nP -iTCP:"${MASTER_PORT}" || true
elif command -v ss >/dev/null 2>&1; then
  ss -lptn "sport = :${MASTER_PORT}" || true
else
  echo "Neither lsof nor ss is available."
fi

echo
echo "== GPU usage =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi is not available here. Run this script on the GPU server."
fi
