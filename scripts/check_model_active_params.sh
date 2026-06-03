#!/usr/bin/env bash
set -euo pipefail

# Count active/nonzero and zero parameters by model scope.
#
# Usage:
#   ./scripts/check_model_active_params.sh
#   ./scripts/check_model_active_params.sh runs/my-pruned-model
#   ENV_NAME=DPO ./scripts/check_model_active_params.sh prune_eval_outputs/.../pruned_model
#
# Defaults:
#   SEARCH_ROOTS="prune_eval_outputs runs models outputs output"
#   MAX_DEPTH=8
#   TRUST_REMOTE_CODE=1
#   MODEL_FAMILY=auto
#   OUTPUT_JSON=""
#   LIST_ONLY=0

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Count active/nonzero and zero parameters by model scope.

Usage:
  ./scripts/check_model_active_params.sh
  ./scripts/check_model_active_params.sh runs/my-pruned-model
  ENV_NAME=DPO ./scripts/check_model_active_params.sh prune_eval_outputs/.../pruned_model

Defaults:
  SEARCH_ROOTS="prune_eval_outputs runs models outputs output"
  MAX_DEPTH=8
  TRUST_REMOTE_CODE=1
  MODEL_FAMILY=auto
  OUTPUT_JSON=""
  LIST_ONLY=0
EOF
  exit 0
fi

if [[ -n "${ENV_NAME:-}" && "${_CHECK_ACTIVE_PARAMS_IN_CONDA:-0}" != "1" ]]; then
  export _CHECK_ACTIVE_PARAMS_IN_CONDA=1
  exec conda run -n "${ENV_NAME}" "$0" "$@"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" - "$@" <<'PY'
from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path


WEIGHT_FILE_NAMES = {
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "model.safetensors",
    "model.safetensors.index.json",
}


def main(argv: list[str]) -> int:
    model_paths = [Path(arg).expanduser() for arg in argv]
    if model_paths:
        model_paths = [path.resolve() for path in model_paths]
    else:
        model_paths = discover_model_dirs()

    if not model_paths:
        roots = os.environ.get("SEARCH_ROOTS", "prune_eval_outputs runs models outputs output")
        print(f"No local model directories found under: {roots}", file=sys.stderr)
        print("Pass one or more model directories explicitly.", file=sys.stderr)
        return 2

    print("Detected model directories:")
    for path in model_paths:
        print(f"  {path}")
    print()

    if truthy(os.environ.get("LIST_ONLY")):
        return 0

    results = []
    for path in model_paths:
        try:
            result = inspect_model(path)
        except Exception as exc:  # noqa: BLE001 - show all per-model load failures.
            print(f"ERROR loading {path}: {exc}", file=sys.stderr)
            results.append({"model_path": str(path), "error": str(exc)})
            continue
        print_model_result(result)
        results.append(result)

    output_json = os.environ.get("OUTPUT_JSON", "").strip()
    if output_json:
        output_path = Path(output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {output_path}")
    return 0


def discover_model_dirs() -> list[Path]:
    roots = [
        Path(item).expanduser()
        for item in shlex.split(os.environ.get("SEARCH_ROOTS", "prune_eval_outputs runs models outputs output"))
    ]
    max_depth = int(os.environ.get("MAX_DEPTH", "8"))
    discovered: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        root = root.resolve()
        for path in [root, *root.rglob("*")]:
            if not path.is_dir():
                continue
            try:
                depth = len(path.relative_to(root).parts)
            except ValueError:
                continue
            if depth > max_depth:
                continue
            if is_model_dir(path) and path not in discovered:
                discovered.append(path)
    return sorted(discovered, key=lambda path: str(path))


def is_model_dir(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    if any((path / name).exists() for name in WEIGHT_FILE_NAMES):
        return True
    if any(path.glob("pytorch_model-*.bin")):
        return True
    if any(path.glob("model-*.safetensors")):
        return True
    return False


def inspect_model(path: Path) -> dict[str, object]:
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM

    trust_remote_code = truthy(os.environ.get("TRUST_REMOTE_CODE", "1"))
    family = os.environ.get("MODEL_FAMILY", "auto").lower()
    config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
    if family == "auto":
        family = "seq2seq" if getattr(config, "is_encoder_decoder", False) else "causal"

    model_class = AutoModelForSeq2SeqLM if family == "seq2seq" else AutoModelForCausalLM
    model = model_class.from_pretrained(
        path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    linear_modules = {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    embedding_modules = {
        name for name, module in model.named_modules() if isinstance(module, nn.Embedding)
    }
    norm_modules = {
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.LayerNorm) or "norm" in module.__class__.__name__.lower()
    }

    buckets = new_buckets()
    for name, parameter in model.named_parameters():
        tensor = parameter.detach()
        counts = tensor_counts(tensor)
        add_counts(buckets["all_params"], counts)

        module_name = name.rsplit(".", 1)[0]
        is_linear_weight = name.endswith(".weight") and module_name in linear_modules
        if is_linear_weight:
            add_counts(buckets["all_linear_weights"], counts)
            scope = linear_scope(module_name)
            add_counts(buckets[f"{scope}_linear_weights"], counts)
        else:
            add_counts(buckets["non_linear_params"], counts)

        if name.endswith(".bias"):
            add_counts(buckets["bias_params"], counts)
        if module_name in embedding_modules:
            add_counts(buckets["embedding_params"], counts)
        if module_name in norm_modules or "norm" in module_name.lower():
            add_counts(buckets["norm_params"], counts)

    return {
        "model_path": str(path),
        "model_family": family,
        "is_encoder_decoder": bool(getattr(config, "is_encoder_decoder", False)),
        "linear_module_count": len(linear_modules),
        "buckets": finalize_buckets(buckets),
    }


def new_buckets() -> dict[str, dict[str, int]]:
    names = [
        "all_params",
        "all_linear_weights",
        "encoder_linear_weights",
        "decoder_linear_weights",
        "lm_head_linear_weights",
        "other_linear_weights",
        "non_linear_params",
        "embedding_params",
        "norm_params",
        "bias_params",
    ]
    return {name: {"total": 0, "zero": 0, "active": 0} for name in names}


def tensor_counts(tensor) -> dict[str, int]:
    total = int(tensor.numel())
    zero = int((tensor == 0).sum().item())
    return {"total": total, "zero": zero, "active": total - zero}


def add_counts(bucket: dict[str, int], counts: dict[str, int]) -> None:
    bucket["total"] += counts["total"]
    bucket["zero"] += counts["zero"]
    bucket["active"] += counts["active"]


def linear_scope(module_name: str) -> str:
    parts = {part.lower() for part in module_name.replace("/", ".").split(".")}
    lowered = module_name.lower()
    if "lm_head" in parts or lowered.endswith("lm_head"):
        return "lm_head"
    if "encoder" in parts or ".encoder." in f".{lowered}.":
        return "encoder"
    if "decoder" in parts or ".decoder." in f".{lowered}.":
        return "decoder"
    return "other"


def finalize_buckets(buckets: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    finalized = {}
    for name, counts in buckets.items():
        total = counts["total"]
        zero = counts["zero"]
        active = counts["active"]
        finalized[name] = {
            "total": total,
            "active": active,
            "zero": zero,
            "active_percent": percentage(active, total),
            "zero_percent": percentage(zero, total),
        }
    return finalized


def print_model_result(result: dict[str, object]) -> None:
    print(f"== {result['model_path']} ==")
    print(f"family={result['model_family']} linear_modules={result['linear_module_count']}")
    print(f"{'scope':<28} {'total':>15} {'active':>15} {'zero':>15} {'active%':>9} {'zero%':>9}")
    buckets = result["buckets"]
    for name in [
        "all_params",
        "all_linear_weights",
        "encoder_linear_weights",
        "decoder_linear_weights",
        "lm_head_linear_weights",
        "other_linear_weights",
        "non_linear_params",
        "embedding_params",
        "norm_params",
        "bias_params",
    ]:
        bucket = buckets[name]
        print(
            f"{name:<28} {bucket['total']:>15} {bucket['active']:>15} "
            f"{bucket['zero']:>15} {bucket['active_percent']:>8.4f}% "
            f"{bucket['zero_percent']:>8.4f}%"
        )
    print()


def percentage(value: int, total: int) -> float:
    return (value / total * 100.0) if total else 0.0


def truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
PY
