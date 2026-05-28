#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encoder_decoder.evaluate_exact import (  # noqa: E402
    DEFAULT_CAUSAL_PROMPT_TEMPLATE,
    ExactEvalConfig,
    ExactEvalResult,
    evaluate_exact_with_model,
    metrics_payload,
)
from encoder_decoder.modeling import load_tokenizer_and_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# EDIT THIS BLOCK
# ---------------------------------------------------------------------------
MODEL_NAME_OR_PATH = "runs/chatlm-mini-8gpu-sft"
EVAL_SOURCE = "data/sft.jsonl"
BENCHMARK_SOURCE = "data/benchmark.jsonl"
OUTPUT_DIR = "runs/eval/chatlm-mini-8gpu-sft"

PROMPT_FIELD = "prompt"
RESPONSE_FIELD = "response"
EVAL_PROMPT_FIELD = None  # None uses PROMPT_FIELD.
EVAL_RESPONSE_FIELD = None  # None uses RESPONSE_FIELD.
BENCHMARK_PROMPT_FIELD = None  # None uses PROMPT_FIELD.
BENCHMARK_RESPONSE_FIELD = None  # None uses RESPONSE_FIELD.

MODEL_FAMILY = "seq2seq"  # use "causal" for decoder-only models, or "auto"
TRUST_REMOTE_CODE = True
PRECISION = "bf16"  # bf16, fp16, fp32, or auto

MAX_INPUT_LENGTH = 256
MAX_NEW_TOKENS = 256
TOP_K = 5
NUM_BEAMS = 5
LIMIT = None

REQUIRED_MODULES = ("datasets", "torch", "transformers")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one model on an eval JSON/JSONL file and a benchmark file."
    )
    parser.add_argument("--model_path", "--model_name_or_path", default=MODEL_NAME_OR_PATH)
    parser.add_argument("--eval_source", default=EVAL_SOURCE)
    parser.add_argument("--benchmark_source", default=BENCHMARK_SOURCE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--prompt_field", default=PROMPT_FIELD)
    parser.add_argument("--response_field", default=RESPONSE_FIELD)
    parser.add_argument("--eval_prompt_field", default=EVAL_PROMPT_FIELD)
    parser.add_argument("--eval_response_field", default=EVAL_RESPONSE_FIELD)
    parser.add_argument("--benchmark_prompt_field", default=BENCHMARK_PROMPT_FIELD)
    parser.add_argument("--benchmark_response_field", default=BENCHMARK_RESPONSE_FIELD)
    parser.add_argument("--eval_split")
    parser.add_argument("--benchmark_split")
    parser.add_argument("--model_family", choices=["auto", "causal", "seq2seq"], default=MODEL_FAMILY)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default=PRECISION)
    parser.add_argument("--device")
    parser.add_argument("--max_input_length", type=int, default=MAX_INPUT_LENGTH)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--num_beams", type=int, default=NUM_BEAMS)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--causal_prompt_template", default=DEFAULT_CAUSAL_PROMPT_TEMPLATE)
    parser.add_argument("--no_strip", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--collapse_whitespace", action="store_true")
    parser.add_argument("--no_trust_remote_code", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def build_eval_config(
    args: argparse.Namespace,
    *,
    name: str,
    source: str,
    prompt_field: str,
    response_field: str,
    split: str | None,
) -> ExactEvalConfig:
    output_dir = Path(args.output_dir)
    return ExactEvalConfig(
        model_name_or_path=resolve_local_arg(args.model_path),
        eval_source=resolve_local_arg(source),
        output_path=str(output_dir / f"{name}_predictions.jsonl"),
        metrics_path=str(output_dir / f"{name}_metrics.json"),
        split=split,
        prompt_field=prompt_field,
        response_field=response_field,
        model_family=args.model_family,
        trust_remote_code=trust_remote_code(args),
        device=args.device,
        torch_dtype=torch_dtype_for_precision(args.precision),
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        num_beams=args.num_beams,
        causal_prompt_template=args.causal_prompt_template,
        strip=not args.no_strip,
        lowercase=args.lowercase,
        collapse_whitespace=args.collapse_whitespace,
        limit=args.limit,
    )


def eval_jobs(args: argparse.Namespace) -> list[tuple[str, ExactEvalConfig]]:
    return [
        (
            "eval",
            build_eval_config(
                args,
                name="eval",
                source=args.eval_source,
                prompt_field=args.eval_prompt_field or args.prompt_field,
                response_field=args.eval_response_field or args.response_field,
                split=args.eval_split,
            ),
        ),
        (
            "benchmark",
            build_eval_config(
                args,
                name="benchmark",
                source=args.benchmark_source,
                prompt_field=args.benchmark_prompt_field or args.prompt_field,
                response_field=args.benchmark_response_field or args.response_field,
                split=args.benchmark_split,
            ),
        ),
    ]


def torch_dtype_for_precision(precision: str) -> str:
    if precision == "bf16":
        return "bfloat16"
    if precision == "fp16":
        return "float16"
    if precision == "fp32":
        return "float32"
    return "auto"


def trust_remote_code(args: argparse.Namespace) -> bool:
    return TRUST_REMOTE_CODE and not args.no_trust_remote_code


def missing_dependencies() -> list[str]:
    return [module for module in REQUIRED_MODULES if importlib.util.find_spec(module) is None]


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_dry_run(args: argparse.Namespace) -> None:
    print(f"model_path: {resolve_local_arg(args.model_path)}")
    print(f"model_family: {args.model_family}")
    print(f"precision: {args.precision}")
    print(f"output_dir: {args.output_dir}")
    for name, config in eval_jobs(args):
        print(
            f"{name}: source={config.eval_source} "
            f"prompt_field={config.prompt_field} response_field={config.response_field} "
            f"predictions={config.output_path} metrics={config.metrics_path}"
        )


def write_summary(output_dir: Path, results: dict[str, ExactEvalResult]) -> None:
    summary = {name: metrics_payload(result) for name, result in results.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "top1_top5_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {summary_path}")


def print_table(results: dict[str, ExactEvalResult], top_k: int) -> None:
    print("\nExact-match summary")
    print(f"split\t top1_exact\t exact@{top_k}\t total")
    top_k_key = f"top_{top_k}_accuracy"
    for name, result in results.items():
        payload = metrics_payload(result)
        print(
            f"{name}\t "
            f"{payload['top_1_accuracy']:.6f}\t "
            f"{payload[top_k_key]:.6f}\t "
            f"{payload['total']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.dry_run:
        print_dry_run(args)
        return 0

    missing = missing_dependencies()
    if missing:
        print(f"Missing packages: {', '.join(missing)}", file=sys.stderr)
        return 2

    import torch

    device = torch.device(args.device or default_device())
    model, tokenizer, resolved_family = load_tokenizer_and_model(
        resolve_local_arg(args.model_path),
        model_family=args.model_family,
        torch_dtype=torch_dtype_for_precision(args.precision),
        trust_remote_code=trust_remote_code(args),
    )
    model.to(device)
    model.eval()

    results = {}
    for name, config in eval_jobs(args):
        print(f"Evaluating {name}: {config.eval_source}")
        results[name] = evaluate_exact_with_model(
            config,
            model=model,
            tokenizer=tokenizer,
            resolved_family=resolved_family,
            device=device,
        )

    write_summary(Path(args.output_dir), results)
    print_table(results, args.top_k)
    return 0


def resolve_local_arg(value: str) -> str:
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    if path.is_absolute():
        return str(path)
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return str(repo_path.resolve())
    return value


if __name__ == "__main__":
    raise SystemExit(main())
