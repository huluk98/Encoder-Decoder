#!/usr/bin/env bash
set -euo pipefail

# Run training plus exact-match evaluation on physical GPUs 4,5,6,7.
#
# Usage:
#   ./scripts/run_4gpu_train_eval.sh sft
#   ./scripts/run_4gpu_train_eval.sh contrastive
#   ./scripts/run_4gpu_train_eval.sh both
#
# Common overrides:
#   DRY_RUN=1 ./scripts/run_4gpu_train_eval.sh both
#   EPOCHS=1 TRAIN_SOURCE=data/my_sft.jsonl ./scripts/run_4gpu_train_eval.sh sft
#   MASTER_PORT=29601 ./scripts/run_4gpu_train_eval.sh sft

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-${MODE:-sft}}"
ENV_NAME="${ENV_NAME:-encoder-decoder-prune}"
CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICES}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29573}"
MODEL_PATH="${MODEL_PATH:-charent/ChatLM-mini-Chinese}"
PRECISION="${PRECISION:-bf16}"
EPOCHS="${EPOCHS:-3}"

TRAIN_SOURCE="${TRAIN_SOURCE:-data/sft.jsonl}"
TRAIN_EVAL_SOURCE="${TRAIN_EVAL_SOURCE:-${TRAIN_SOURCE}}"
BENCHMARK_SOURCE="${BENCHMARK_SOURCE:-data/benchmark.jsonl}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-runs/chatlm-mini-4gpu-sft}"

CONTRASTIVE_TRAIN_SOURCE="${CONTRASTIVE_TRAIN_SOURCE:-/Users/luke/Documents/SCENIC agent/data/SCENIC_full_anchor_positive_negative.json}"
SFT_TRAIN_EVAL_SOURCE="${SFT_TRAIN_EVAL_SOURCE:-data/sft.jsonl}"
CONTRASTIVE_BENCHMARK_SOURCE="${CONTRASTIVE_BENCHMARK_SOURCE:-${BENCHMARK_SOURCE}}"
CONTRASTIVE_OUTPUT_DIR="${CONTRASTIVE_OUTPUT_DIR:-runs/chatlm-mini-4gpu-contrastive}"
NEGATIVE_FIELD="${NEGATIVE_FIELD:-invalid_negative}"

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
