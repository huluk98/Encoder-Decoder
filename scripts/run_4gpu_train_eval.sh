#!/usr/bin/env bash
set -euo pipefail

# Run training plus exact-match evaluation on physical GPUs 4,5,6,7.
#
# Usage:
#   ./scripts/run_4gpu_train_eval.sh sft
#   ./scripts/run_4gpu_train_eval.sh contrastive
#   ./scripts/run_4gpu_train_eval.sh both
#
# Common overrides without editing this file:
#   DRY_RUN=1 ./scripts/run_4gpu_train_eval.sh both
#   EPOCHS=1 TRAIN_SOURCE=data/my_sft.jsonl ./scripts/run_4gpu_train_eval.sh sft
#   MASTER_PORT=29601 ./scripts/run_4gpu_train_eval.sh sft

# ---------------------------------------------------------------------------
# EDIT THIS BLOCK
# ---------------------------------------------------------------------------
# Change these values directly when you want to point at different files.
DEFAULT_MODE="sft"  # sft, contrastive, or both
DEFAULT_ENV_NAME="DPO"
DEFAULT_CUDA_DEVICES="4,5,6,7"
DEFAULT_NPROC_PER_NODE="4"
DEFAULT_MASTER_PORT="29573"
DEFAULT_MODEL_PATH="charent/ChatLM-mini-Chinese"
DEFAULT_PRECISION="bf16"  # bf16, fp16, or fp32
DEFAULT_EPOCHS="3"

# Regular SFT paths.
DEFAULT_TRAIN_SOURCE="data/sft.jsonl"
DEFAULT_TRAIN_EVAL_SOURCE="${DEFAULT_TRAIN_SOURCE}"
DEFAULT_BENCHMARK_SOURCE="data/benchmark.jsonl"
DEFAULT_SFT_OUTPUT_DIR="runs/chatlm-mini-4gpu-sft"

# Contrastive SFT paths.
DEFAULT_CONTRASTIVE_TRAIN_SOURCE="/Users/luke/Documents/SCENIC agent/data/SCENIC_full_anchor_positive_negative.json"
DEFAULT_SFT_TRAIN_EVAL_SOURCE="data/sft.jsonl"
DEFAULT_CONTRASTIVE_BENCHMARK_SOURCE="${DEFAULT_BENCHMARK_SOURCE}"
DEFAULT_CONTRASTIVE_OUTPUT_DIR="runs/chatlm-mini-4gpu-contrastive"
DEFAULT_NEGATIVE_FIELD="invalid_negative"
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-${MODE:-${DEFAULT_MODE}}}"
ENV_NAME="${ENV_NAME:-${DEFAULT_ENV_NAME}}"
CUDA_DEVICES="${CUDA_DEVICES:-${DEFAULT_CUDA_DEVICES}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICES}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-${DEFAULT_NPROC_PER_NODE}}"
MASTER_PORT="${MASTER_PORT:-${DEFAULT_MASTER_PORT}}"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
PRECISION="${PRECISION:-${DEFAULT_PRECISION}}"
EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS}}"

TRAIN_SOURCE="${TRAIN_SOURCE:-${DEFAULT_TRAIN_SOURCE}}"
TRAIN_EVAL_SOURCE="${TRAIN_EVAL_SOURCE:-${DEFAULT_TRAIN_EVAL_SOURCE}}"
BENCHMARK_SOURCE="${BENCHMARK_SOURCE:-${DEFAULT_BENCHMARK_SOURCE}}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${DEFAULT_SFT_OUTPUT_DIR}}"

CONTRASTIVE_TRAIN_SOURCE="${CONTRASTIVE_TRAIN_SOURCE:-${DEFAULT_CONTRASTIVE_TRAIN_SOURCE}}"
SFT_TRAIN_EVAL_SOURCE="${SFT_TRAIN_EVAL_SOURCE:-${DEFAULT_SFT_TRAIN_EVAL_SOURCE}}"
CONTRASTIVE_BENCHMARK_SOURCE="${CONTRASTIVE_BENCHMARK_SOURCE:-${DEFAULT_CONTRASTIVE_BENCHMARK_SOURCE}}"
CONTRASTIVE_OUTPUT_DIR="${CONTRASTIVE_OUTPUT_DIR:-${DEFAULT_CONTRASTIVE_OUTPUT_DIR}}"
NEGATIVE_FIELD="${NEGATIVE_FIELD:-${DEFAULT_NEGATIVE_FIELD}}"

dry_run_args=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  dry_run_args+=(--dry_run)
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "Physical GPUs are remapped inside the process as cuda:0..cuda:$((NPROC_PER_NODE - 1))."

run_sft() {
  conda run -n "${ENV_NAME}" \
    python "${REPO_ROOT}/scripts/train_sft_8gpu.py" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --cuda_visible_devices "${CUDA_VISIBLE_DEVICES}" \
    --master_port "${MASTER_PORT}" \
    --model_path "${MODEL_PATH}" \
    --train_source "${TRAIN_SOURCE}" \
    --train_eval_source "${TRAIN_EVAL_SOURCE}" \
    --benchmark_source "${BENCHMARK_SOURCE}" \
    --output_dir "${SFT_OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --precision "${PRECISION}" \
    "${dry_run_args[@]}"
}

run_contrastive() {
  conda run -n "${ENV_NAME}" \
    python "${REPO_ROOT}/scripts/train_contrastive_8gpu.py" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --cuda_visible_devices "${CUDA_VISIBLE_DEVICES}" \
    --master_port "${MASTER_PORT}" \
    --model_path "${MODEL_PATH}" \
    --contrastive_train_source "${CONTRASTIVE_TRAIN_SOURCE}" \
    --sft_train_eval_source "${SFT_TRAIN_EVAL_SOURCE}" \
    --benchmark_source "${CONTRASTIVE_BENCHMARK_SOURCE}" \
    --output_dir "${CONTRASTIVE_OUTPUT_DIR}" \
    --negative_field "${NEGATIVE_FIELD}" \
    --epochs "${EPOCHS}" \
    --precision "${PRECISION}" \
    "${dry_run_args[@]}"
}

case "${MODE}" in
  sft)
    run_sft
    ;;
  contrastive)
    run_contrastive
    ;;
  both)
    run_sft
    run_contrastive
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use sft, contrastive, or both." >&2
    exit 2
    ;;
esac
