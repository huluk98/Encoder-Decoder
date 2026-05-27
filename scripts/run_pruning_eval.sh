#!/usr/bin/env bash
set -euo pipefail

# Edit these paths or override them as environment variables.
MODEL_PATH="${MODEL_PATH:-runs/chatlm-mini-8gpu-sft}"
CALIBRATION_SOURCE="${CALIBRATION_SOURCE:-data/sft.jsonl}"
EVAL_SOURCE="${EVAL_SOURCE:-data/eval.jsonl}"
BENCHMARK_SOURCE="${BENCHMARK_SOURCE:-data/benchmark.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/pruning_eval}"

MODEL_FAMILY="${MODEL_FAMILY:-seq2seq}"
PRECISION="${PRECISION:-bf16}"
SPARSITY="${SPARSITY:-0.5}"
TOP_K="${TOP_K:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-256}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
SOURCE_MAX_LENGTH="${SOURCE_MAX_LENGTH:-768}"
TARGET_MAX_LENGTH="${TARGET_MAX_LENGTH:-256}"
CALIBRATION_LIMIT="${CALIBRATION_LIMIT:-128}"
PROMPT_FIELD="${PROMPT_FIELD:-prompt}"
RESPONSE_FIELD="${RESPONSE_FIELD:-response}"
CALIBRATION_PROMPT_FIELD="${CALIBRATION_PROMPT_FIELD:-${PROMPT_FIELD}}"
CALIBRATION_RESPONSE_FIELD="${CALIBRATION_RESPONSE_FIELD:-${RESPONSE_FIELD}}"
EVAL_PROMPT_FIELD="${EVAL_PROMPT_FIELD:-${PROMPT_FIELD}}"
EVAL_RESPONSE_FIELD="${EVAL_RESPONSE_FIELD:-${RESPONSE_FIELD}}"
BENCHMARK_PROMPT_FIELD="${BENCHMARK_PROMPT_FIELD:-${EVAL_PROMPT_FIELD}}"
BENCHMARK_RESPONSE_FIELD="${BENCHMARK_RESPONSE_FIELD:-${EVAL_RESPONSE_FIELD}}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
METHODS="${METHODS:-magnitude gradient nvidia wanda}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

trust_args=()
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  trust_args+=(--trust_remote_code)
fi

case "${PRECISION}" in
  bf16)
    TORCH_DTYPE="bfloat16"
    ;;
  fp16)
    TORCH_DTYPE="float16"
    ;;
  fp32)
    TORCH_DTYPE="float32"
    ;;
  auto)
    TORCH_DTYPE="auto"
    ;;
  *)
    echo "Unknown PRECISION='${PRECISION}'. Use bf16, fp16, fp32, or auto." >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_ROOT}"

run_prune() {
  local method="$1"
  local method_dir="${OUTPUT_ROOT}/${method}"
  local pruned_dir="${method_dir}/model"
  mkdir -p "${method_dir}"

  echo "==> Pruning method: ${method}"
  case "${method}" in
    magnitude)
      # Same as magnitude (1).py: per-layer abs(weight), prune the lowest 50%.
      python "${SCRIPT_DIR}/prune.py" \
        --model_name_or_path "${MODEL_PATH}" \
        --model_family "${MODEL_FAMILY}" \
        --torch_dtype "${TORCH_DTYPE}" \
        "${trust_args[@]}" \
        --method magnitude \
        --sparsity "${SPARSITY}" \
        --output_dir "${pruned_dir}"
      ;;
    gradient)
      # Same as gradient (1).py: Taylor score abs(weight * gradient) on calibration records.
      python "${SCRIPT_DIR}/prune.py" \
        --model_name_or_path "${MODEL_PATH}" \
        --model_family "${MODEL_FAMILY}" \
        --torch_dtype "${TORCH_DTYPE}" \
        "${trust_args[@]}" \
        --method gradient \
        --sparsity "${SPARSITY}" \
        --calibration_source "${CALIBRATION_SOURCE}" \
        --calibration_limit "${CALIBRATION_LIMIT}" \
        --prompt_field "${CALIBRATION_PROMPT_FIELD}" \
        --response_field "${CALIBRATION_RESPONSE_FIELD}" \
        --max_seq_length "${MAX_SEQ_LENGTH}" \
        --source_max_length "${SOURCE_MAX_LENGTH}" \
        --target_max_length "${TARGET_MAX_LENGTH}" \
        --output_dir "${pruned_dir}"
      ;;
    nvidia)
      # Same as nvidia (1).py: strict 2:4, zero the two smallest values per group of four.
      python "${SCRIPT_DIR}/prune.py" \
        --model_name_or_path "${MODEL_PATH}" \
        --model_family "${MODEL_FAMILY}" \
        --torch_dtype "${TORCH_DTYPE}" \
        "${trust_args[@]}" \
        --method nvidia \
        --nvidia_keep_n 2 \
        --nvidia_group_m 4 \
        --output_dir "${pruned_dir}"
      ;;
    wanda)
      # Same as wanda.py: abs(weight) * activation_norm, pruned row-wise at 50%.
      python "${SCRIPT_DIR}/prune.py" \
        --model_name_or_path "${MODEL_PATH}" \
        --model_family "${MODEL_FAMILY}" \
        --torch_dtype "${TORCH_DTYPE}" \
        "${trust_args[@]}" \
        --method wanda \
        --sparsity "${SPARSITY}" \
        --calibration_source "${CALIBRATION_SOURCE}" \
        --calibration_limit "${CALIBRATION_LIMIT}" \
        --prompt_field "${CALIBRATION_PROMPT_FIELD}" \
        --response_field "${CALIBRATION_RESPONSE_FIELD}" \
        --max_seq_length "${MAX_SEQ_LENGTH}" \
        --source_max_length "${SOURCE_MAX_LENGTH}" \
        --target_max_length "${TARGET_MAX_LENGTH}" \
        --output_dir "${pruned_dir}"
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      return 2
      ;;
  esac
}

run_eval() {
  local method="$1"
  local split_name="$2"
  local source_path="$3"
  local prompt_field="$4"
  local response_field="$5"
  local method_dir="${OUTPUT_ROOT}/${method}"
  local pruned_dir="${method_dir}/model"

  echo "==> Evaluating ${method} on ${split_name}"
  python "${SCRIPT_DIR}/evaluate_exact.py" \
    --model_name_or_path "${pruned_dir}" \
    --eval_source "${source_path}" \
    --output_path "${method_dir}/${split_name}_predictions.jsonl" \
    --metrics_path "${method_dir}/${split_name}_metrics.json" \
    --model_family "${MODEL_FAMILY}" \
    --torch_dtype "${TORCH_DTYPE}" \
    "${trust_args[@]}" \
    --prompt_field "${prompt_field}" \
    --response_field "${response_field}" \
    --max_input_length "${MAX_INPUT_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --top_k "${TOP_K}" \
    --num_beams "${TOP_K}"
}

for method in ${METHODS}; do
  run_prune "${method}"
  if [[ -n "${EVAL_SOURCE}" ]]; then
    run_eval "${method}" "eval" "${EVAL_SOURCE}" "${EVAL_PROMPT_FIELD}" "${EVAL_RESPONSE_FIELD}"
  fi
  if [[ -n "${BENCHMARK_SOURCE}" ]]; then
    run_eval "${method}" "benchmark" "${BENCHMARK_SOURCE}" "${BENCHMARK_PROMPT_FIELD}" "${BENCHMARK_RESPONSE_FIELD}"
  fi
done

python - "${OUTPUT_ROOT}" "${TOP_K}" ${METHODS} <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
top_k = sys.argv[2]
methods = sys.argv[3:]

print("\nPruning exact-match summary")
print(f"method\t split\t top1_exact\t exact@{top_k}\t total")
for method in methods:
    for split in ("eval", "benchmark"):
        metrics_path = root / method / f"{split}_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        topk_key = next(
            key
            for key in metrics
            if key.startswith("top_")
            and key.endswith("_accuracy")
            and key != "top_1_accuracy"
        )
        print(
            f"{method}\t {split}\t "
            f"{metrics['top_1_accuracy']:.6f}\t "
            f"{metrics[topk_key]:.6f}\t "
            f"{metrics['total']}"
        )
PY

echo "Done. Full outputs are under ${OUTPUT_ROOT}/"
