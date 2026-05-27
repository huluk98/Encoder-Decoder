#!/usr/bin/env bash
set -euo pipefail

# Edit these paths or override them as environment variables.
MODEL_PATH="${MODEL_PATH:-runs/chatlm-mini-8gpu-sft}"
CALIBRATION_SOURCE="${CALIBRATION_SOURCE:-data/sft.jsonl}"
EVAL_SOURCE="${EVAL_SOURCE:-data/eval.jsonl}"
BENCHMARK_SOURCE="${BENCHMARK_SOURCE:-data/benchmark.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/pruning_eval}"

MODEL_FAMILY="${MODEL_FAMILY:-seq2seq}"
SPARSITY="${SPARSITY:-0.5}"
TOP_K="${TOP_K:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-256}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
METHODS="${METHODS:-magnitude gradient nvidia wanda}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

trust_args=()
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  trust_args+=(--trust_remote_code)
fi

mkdir -p "${OUTPUT_ROOT}"

run_prune() {
  local method="$1"
  local method_dir="${OUTPUT_ROOT}/${method}"
  local pruned_dir="${method_dir}/model"
  mkdir -p "${method_dir}"

  echo "==> Pruning method: ${method}"
  case "${method}" in
    magnitude)
      python "${SCRIPT_DIR}/prune.py" \
        --model_name_or_path "${MODEL_PATH}" \
        --model_family "${MODEL_FAMILY}" \
        "${trust_args[@]}" \
        --method magnitude \
        --sparsity "${SPARSITY}" \
        --output_dir "${pruned_dir}"
      ;;
    gradient)
      python "${SCRIPT_DIR}/prune.py" \
        --model_name_or_path "${MODEL_PATH}" \
        --model_family "${MODEL_FAMILY}" \
        "${trust_args[@]}" \
        --method gradient \
        --sparsity "${SPARSITY}" \
        --calibration_source "${CALIBRATION_SOURCE}" \
        --output_dir "${pruned_dir}"
      ;;
    nvidia)
      python "${SCRIPT_DIR}/prune.py" \
        --model_name_or_path "${MODEL_PATH}" \
        --model_family "${MODEL_FAMILY}" \
        "${trust_args[@]}" \
        --method nvidia \
        --nvidia_keep_n 2 \
        --nvidia_group_m 4 \
        --output_dir "${pruned_dir}"
      ;;
    wanda)
      python "${SCRIPT_DIR}/prune.py" \
        --model_name_or_path "${MODEL_PATH}" \
        --model_family "${MODEL_FAMILY}" \
        "${trust_args[@]}" \
        --method wanda \
        --sparsity "${SPARSITY}" \
        --calibration_source "${CALIBRATION_SOURCE}" \
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
  local method_dir="${OUTPUT_ROOT}/${method}"
  local pruned_dir="${method_dir}/model"

  echo "==> Evaluating ${method} on ${split_name}"
  python "${SCRIPT_DIR}/evaluate_exact.py" \
    --model_name_or_path "${pruned_dir}" \
    --eval_source "${source_path}" \
    --output_path "${method_dir}/${split_name}_predictions.jsonl" \
    --metrics_path "${method_dir}/${split_name}_metrics.json" \
    --model_family "${MODEL_FAMILY}" \
    "${trust_args[@]}" \
    --max_input_length "${MAX_INPUT_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --top_k "${TOP_K}" \
    --num_beams "${TOP_K}"
}

for method in ${METHODS}; do
  run_prune "${method}"
  run_eval "${method}" "eval" "${EVAL_SOURCE}"
  run_eval "${method}" "benchmark" "${BENCHMARK_SOURCE}"
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
