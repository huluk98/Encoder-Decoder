#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# EDIT THIS BLOCK
# ---------------------------------------------------------------------------
MODEL_NAME_OR_PATH = "charent/ChatLM-mini-Chinese"

# Contrastive training data must contain anchor/positive/negative/response fields.
CONTRASTIVE_TRAIN_SOURCE = "/Users/luke/Documents/SCENIC agent/data/SCENIC_full_anchor_positive_negative.json"

# Eval the contrastive-trained model on the regular SFT training data and benchmark.
SFT_TRAIN_EVAL_SOURCE = "data/sft.jsonl"
BENCHMARK_SOURCE = "data/benchmark.jsonl"
OUTPUT_DIR = "runs/chatlm-mini-8gpu-contrastive"

ANCHOR_FIELD = "anchor"
POSITIVE_FIELD = "positive"
NEGATIVE_FIELD = "invalid_negative"  # use "negative" for valid hard negatives
CONTRASTIVE_RESPONSE_FIELD = "response"

SFT_EVAL_PROMPT_FIELD = "prompt"
SFT_EVAL_RESPONSE_FIELD = "response"
BENCHMARK_PROMPT_FIELD = "prompt"
BENCHMARK_RESPONSE_FIELD = "response"

CONTRASTIVE_LOSS_WEIGHT = 0.2
CONTRASTIVE_MARGIN = 0.2

MODEL_FAMILY = "seq2seq"
TRUST_REMOTE_CODE = True
PRECISION = "bf16"  # bf16, fp16, or fp32

NPROC_PER_NODE = 8
CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
NUM_TRAIN_EPOCHS = 3.0
LEARNING_RATE = 5e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
MAX_STEPS = -1
GRADIENT_CHECKPOINTING = True

MAX_SEQ_LENGTH = 512
SOURCE_MAX_LENGTH = 256
TARGET_MAX_LENGTH = 256
MAX_NEW_TOKENS = 256
TOP_K = 5
NUM_BEAMS = 5
GENERATION_EVAL_LIMIT = None

REQUIRED_MODULES = ("accelerate", "datasets", "torch", "transformers")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train contrastive SFT on 8 GPUs, then exact-eval."
    )
    parser.add_argument("--model_path", "--model_name_or_path", default=MODEL_NAME_OR_PATH)
    parser.add_argument("--contrastive_train_source", default=CONTRASTIVE_TRAIN_SOURCE)
    parser.add_argument("--sft_train_eval_source", default=SFT_TRAIN_EVAL_SOURCE)
    parser.add_argument("--benchmark_source", default=BENCHMARK_SOURCE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--anchor_field", default=ANCHOR_FIELD)
    parser.add_argument("--positive_field", default=POSITIVE_FIELD)
    parser.add_argument("--negative_field", default=NEGATIVE_FIELD)
    parser.add_argument("--contrastive_response_field", default=CONTRASTIVE_RESPONSE_FIELD)
    parser.add_argument("--sft_eval_prompt_field", default=SFT_EVAL_PROMPT_FIELD)
    parser.add_argument("--sft_eval_response_field", default=SFT_EVAL_RESPONSE_FIELD)
    parser.add_argument("--benchmark_prompt_field", default=BENCHMARK_PROMPT_FIELD)
    parser.add_argument("--benchmark_response_field", default=BENCHMARK_RESPONSE_FIELD)
    parser.add_argument("--contrastive_loss_weight", type=float, default=CONTRASTIVE_LOSS_WEIGHT)
    parser.add_argument("--contrastive_margin", type=float, default=CONTRASTIVE_MARGIN)
    parser.add_argument("--model_family", choices=["auto", "causal", "seq2seq"], default=MODEL_FAMILY)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default=PRECISION)
    parser.add_argument("--nproc_per_node", type=int, default=NPROC_PER_NODE)
    parser.add_argument("--epochs", "--num_train_epochs", type=float, default=NUM_TRAIN_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--per_device_train_batch_size", type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=PER_DEVICE_EVAL_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--source_max_length", type=int, default=SOURCE_MAX_LENGTH)
    parser.add_argument("--target_max_length", type=int, default=TARGET_MAX_LENGTH)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--num_beams", type=int, default=NUM_BEAMS)
    parser.add_argument("--generation_eval_limit", type=int, default=GENERATION_EVAL_LIMIT)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_trust_remote_code", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    return parser.parse_args(argv)


def build_train_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(args.nproc_per_node),
        str(script_path("sft.py")),
        "--model_name_or_path",
        args.model_path,
        "--train_source",
        args.contrastive_train_source,
        "--output_dir",
        args.output_dir,
        "--training_mode",
        "contrastive",
        "--model_family",
        args.model_family,
        "--torch_dtype",
        torch_dtype_for_precision(args.precision),
        "--contrastive_anchor_field",
        args.anchor_field,
        "--contrastive_positive_field",
        args.positive_field,
        "--contrastive_negative_field",
        args.negative_field,
        "--contrastive_response_field",
        args.contrastive_response_field,
        "--contrastive_loss_weight",
        str(args.contrastive_loss_weight),
        "--contrastive_margin",
        str(args.contrastive_margin),
        "--max_seq_length",
        str(args.max_seq_length),
        "--source_max_length",
        str(args.source_max_length),
        "--target_max_length",
        str(args.target_max_length),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--num_train_epochs",
        str(args.epochs),
        "--max_steps",
        str(args.max_steps),
        "--report_to",
        "none",
    ]
    add_common_model_flags(command, args)
    return command


def build_eval_command(
    args: argparse.Namespace,
    *,
    name: str,
    source: str,
    prompt_field: str,
    response_field: str,
) -> list[str]:
    generation_dir = Path(args.output_dir) / "generation_eval"
    command = [
        sys.executable,
        str(script_path("evaluate_exact.py")),
        "--model_name_or_path",
        args.output_dir,
        "--eval_source",
        source,
        "--output_path",
        str(generation_dir / f"{name}_predictions.jsonl"),
        "--metrics_path",
        str(generation_dir / f"{name}_metrics.json"),
        "--prompt_field",
        prompt_field,
        "--response_field",
        response_field,
        "--model_family",
        args.model_family,
        "--torch_dtype",
        torch_dtype_for_precision(args.precision),
        "--max_input_length",
        str(args.source_max_length),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--top_k",
        str(args.top_k),
        "--num_beams",
        str(args.num_beams),
    ]
    if TRUST_REMOTE_CODE and not args.no_trust_remote_code:
        command.append("--trust_remote_code")
    if args.generation_eval_limit is not None:
        command.extend(["--limit", str(args.generation_eval_limit)])
    return command


def build_eval_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    return [
        (
            "sft_train",
            build_eval_command(
                args,
                name="sft_train",
                source=args.sft_train_eval_source,
                prompt_field=args.sft_eval_prompt_field,
                response_field=args.sft_eval_response_field,
            ),
        ),
        (
            "benchmark",
            build_eval_command(
                args,
                name="benchmark",
                source=args.benchmark_source,
                prompt_field=args.benchmark_prompt_field,
                response_field=args.benchmark_response_field,
            ),
        ),
    ]


def add_common_model_flags(command: list[str], args: argparse.Namespace) -> None:
    if TRUST_REMOTE_CODE and not args.no_trust_remote_code:
        command.append("--trust_remote_code")
    if args.precision == "bf16":
        command.append("--bf16")
    elif args.precision == "fp16":
        command.append("--fp16")
    if GRADIENT_CHECKPOINTING and not args.no_gradient_checkpointing:
        command.append("--gradient_checkpointing")


def torch_dtype_for_precision(precision: str) -> str:
    if precision == "bf16":
        return "bfloat16"
    if precision == "fp16":
        return "float16"
    return "float32"


def script_path(name: str) -> Path:
    return Path(__file__).resolve().with_name(name)


def missing_dependencies() -> list[str]:
    return [module for module in REQUIRED_MODULES if importlib.util.find_spec(module) is None]


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", CUDA_VISIBLE_DEVICES)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("OMP_NUM_THREADS", "8")
    return env


def run(command: Sequence[str], *, env: dict[str, str], capture_json: bool = False):
    print("Running:")
    print(" ".join(shlex.quote(part) for part in command))
    try:
        if capture_json:
            completed = subprocess.run(command, check=True, text=True, capture_output=True, env=env)
        else:
            subprocess.run(command, check=True, env=env)
            return None
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit status {exc.returncode}:", file=sys.stderr)
        if getattr(exc, "stdout", None):
            print("\n--- stdout ---", file=sys.stderr)
            print(exc.stdout, file=sys.stderr, end="" if exc.stdout.endswith("\n") else "\n")
        if getattr(exc, "stderr", None):
            print("\n--- stderr ---", file=sys.stderr)
            print(exc.stderr, file=sys.stderr, end="" if exc.stderr.endswith("\n") else "\n")
        raise

    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    print(completed.stdout, end="")
    return parse_json_output(completed.stdout)


def parse_json_output(text: str) -> dict[str, object]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def write_summary(args: argparse.Namespace, results: dict[str, dict[str, object]]) -> None:
    summary = {}
    for name, metrics in results.items():
        summary[name] = {
            "total": metrics["total"],
            "top_1_correct": metrics["top_1_correct"],
            "top_1_accuracy": metrics["top_1_accuracy"],
            f"top_{args.top_k}_correct": metrics[f"top_{args.top_k}_correct"],
            f"top_{args.top_k}_accuracy": metrics[f"top_{args.top_k}_accuracy"],
        }
    path = Path(args.output_dir) / "generation_eval" / summary_metrics_filename(args.top_k)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {path}")


def summary_metrics_filename(top_k: int) -> str:
    return f"top1_top{top_k}_metrics.json"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    env = runtime_env()
    commands: list[Sequence[str]] = []
    if not args.skip_train:
        commands.append(build_train_command(args))
    if not args.skip_eval:
        commands.extend(command for _name, command in build_eval_commands(args))

    if args.dry_run:
        for command in commands:
            print(" ".join(shlex.quote(part) for part in command))
        return 0

    missing = missing_dependencies()
    if missing:
        print(f"Missing packages: {', '.join(missing)}", file=sys.stderr)
        return 2

    if not args.skip_train:
        run(build_train_command(args), env=env)

    if not args.skip_eval:
        results = {
            name: run(command, env=env, capture_json=True)
            for name, command in build_eval_commands(args)
        }
        write_summary(args, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
