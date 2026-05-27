#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# EDIT THIS BLOCK FIRST
# ---------------------------------------------------------------------------
# You can use the Hugging Face model ID below or replace it with a local T5 /
# ChatLM checkpoint path, for example:
#   MODEL_NAME_OR_PATH = "/nvme1/home/luke/PycharmProjects/iot_t5/sft"
MODEL_NAME_OR_PATH = "charent/ChatLM-mini-Chinese"

TRAIN_SOURCE = "data/sft.jsonl"
TRAINING_MODE = "sft"  # Use "sft" or "contrastive".
EVAL_SOURCE = "data/eval.jsonl"
BENCHMARK_SOURCE = "data/benchmark.jsonl"
ANCHOR_EVAL_SOURCE = ""
OUTPUT_DIR = "runs/chatlm-mini-8gpu-sft"

# Change epochs here for the normal run.
NUM_TRAIN_EPOCHS = 3.0

# ChatLM-mini/T5-style defaults.
MODEL_FAMILY = "seq2seq"
TRUST_REMOTE_CODE = True
PRECISION = "bf16"  # Use "bf16", "fp16", or "fp32".

# 8-GPU and training defaults.
NPROC_PER_NODE = 8
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-5
MAX_STEPS = -1
GRADIENT_CHECKPOINTING = True

# Sequence and generation defaults.
MAX_SEQ_LENGTH = 512
SOURCE_MAX_LENGTH = 256
TARGET_MAX_LENGTH = 256
MAX_NEW_TOKENS = 256
TOP_K = 5
NUM_BEAMS = 5

# Data columns.
PROMPT_FIELD = "prompt"
RESPONSE_FIELD = "response"
ANCHOR_FIELD = "anchor"
POSITIVE_FIELD = "positive"
NEGATIVE_FIELD = "negative"
CONTRASTIVE_LOSS_WEIGHT = 0.2
CONTRASTIVE_MARGIN = 0.2

REQUIRED_MODULES = ("accelerate", "datasets", "torch", "transformers")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ChatLM-mini SFT on 8 GPUs, then compute top-1 and top-5 exact "
            "match on eval and benchmark files."
        )
    )
    parser.add_argument("--train_source", default=TRAIN_SOURCE)
    parser.add_argument("--training_mode", choices=["sft", "contrastive"], default=TRAINING_MODE)
    parser.add_argument("--eval_source", default=EVAL_SOURCE)
    parser.add_argument("--benchmark_source", default=BENCHMARK_SOURCE)
    parser.add_argument("--anchor_eval_source", default=ANCHOR_EVAL_SOURCE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--model_name_or_path", "--model_path", default=MODEL_NAME_OR_PATH)
    parser.add_argument("--model_family", choices=["auto", "causal", "seq2seq"], default=MODEL_FAMILY)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default=PRECISION)
    parser.add_argument("--nproc_per_node", type=int, default=NPROC_PER_NODE)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--num_beams", type=int, default=NUM_BEAMS)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--source_max_length", type=int, default=SOURCE_MAX_LENGTH)
    parser.add_argument("--target_max_length", type=int, default=TARGET_MAX_LENGTH)
    parser.add_argument("--per_device_train_batch_size", type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=PER_DEVICE_EVAL_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num_train_epochs", "--epochs", type=float, default=NUM_TRAIN_EPOCHS)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--prompt_field", default=PROMPT_FIELD)
    parser.add_argument("--response_field", default=RESPONSE_FIELD)
    parser.add_argument("--eval_prompt_field")
    parser.add_argument("--eval_response_field")
    parser.add_argument("--benchmark_prompt_field")
    parser.add_argument("--benchmark_response_field")
    parser.add_argument("--anchor_field", default=ANCHOR_FIELD)
    parser.add_argument("--anchor_response_field")
    parser.add_argument("--contrastive_positive_field", default=POSITIVE_FIELD)
    parser.add_argument("--contrastive_negative_field", default=NEGATIVE_FIELD)
    parser.add_argument("--contrastive_loss_weight", type=float, default=CONTRASTIVE_LOSS_WEIGHT)
    parser.add_argument("--contrastive_margin", type=float, default=CONTRASTIVE_MARGIN)
    parser.add_argument("--generation_eval_limit", type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_bf16", action="store_true", help="Compatibility alias for --precision fp32.")
    parser.add_argument("--fp16", action="store_true", help="Compatibility alias for --precision fp16.")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_trust_remote_code", action="store_true")
    return parser.parse_args(argv)


def missing_dependencies() -> list[str]:
    return [
        module
        for module in REQUIRED_MODULES
        if importlib.util.find_spec(module) is None
    ]


def print_dependency_help(missing: Sequence[str]) -> None:
    print(
        "Missing Python packages: "
        + ", ".join(missing)
        + "\n\nInstall them first:\n\n"
        "  conda env create -f environment.yml\n"
        "  conda activate encoder-decoder-prune\n"
        "  pip install -e \".[dev]\"\n\n"
        "or:\n\n"
        "  python -m pip install -r requirements.txt\n"
        "  python -m pip install -e \".[dev]\"\n",
        file=sys.stderr,
    )


def build_train_command(args: argparse.Namespace) -> list[str]:
    script_path = Path(__file__).resolve().with_name("sft.py")
    precision = resolve_precision(args)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(args.nproc_per_node),
        str(script_path),
        "--model_name_or_path",
        args.model_name_or_path,
        "--train_source",
        args.train_source,
        "--eval_source",
        args.eval_source,
        "--output_dir",
        args.output_dir,
        "--training_mode",
        args.training_mode,
        "--model_family",
        args.model_family,
        "--torch_dtype",
        torch_dtype_for_precision(precision),
        "--prompt_field",
        args.prompt_field,
        "--response_field",
        args.response_field,
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
        str(args.num_train_epochs),
        "--max_steps",
        str(args.max_steps),
        "--report_to",
        "none",
    ]
    if TRUST_REMOTE_CODE and not args.no_trust_remote_code:
        command.append("--trust_remote_code")
    if args.training_mode == "contrastive":
        command.extend(
            [
                "--contrastive_anchor_field",
                args.anchor_field,
                "--contrastive_positive_field",
                args.contrastive_positive_field,
                "--contrastive_negative_field",
                args.contrastive_negative_field,
                "--contrastive_response_field",
                contrastive_response_field(args),
                "--contrastive_loss_weight",
                str(args.contrastive_loss_weight),
                "--contrastive_margin",
                str(args.contrastive_margin),
            ]
        )
    if precision == "bf16":
        command.append("--bf16")
    elif precision == "fp16":
        command.append("--fp16")
    if GRADIENT_CHECKPOINTING and not args.no_gradient_checkpointing:
        command.append("--gradient_checkpointing")
    return command


def resolve_precision(args: argparse.Namespace) -> str:
    if args.fp16:
        return "fp16"
    if args.no_bf16 and args.precision == "bf16":
        return "fp32"
    return args.precision


def torch_dtype_for_precision(precision: str) -> str:
    if precision == "bf16":
        return "bfloat16"
    if precision == "fp16":
        return "float16"
    return "float32"


def build_eval_command(
    args: argparse.Namespace,
    *,
    name: str,
    source: str,
    prompt_field: str,
    response_field: str,
) -> list[str]:
    generation_dir = Path(args.output_dir) / "generation_eval"
    output_path = generation_dir / f"{name}_predictions.jsonl"
    metrics_path = generation_dir / f"{name}_metrics.json"
    script_path = Path(__file__).resolve().with_name("evaluate_exact.py")
    precision = resolve_precision(args)
    command = [
        sys.executable,
        str(script_path),
        "--model_name_or_path",
        args.output_dir,
        "--eval_source",
        source,
        "--output_path",
        str(output_path),
        "--metrics_path",
        str(metrics_path),
        "--prompt_field",
        prompt_field,
        "--response_field",
        response_field,
        "--model_family",
        args.model_family,
        "--torch_dtype",
        torch_dtype_for_precision(precision),
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


def build_eval_commands(args: argparse.Namespace) -> list[list[str]]:
    if args.training_mode == "contrastive":
        return [
            build_eval_command(
                args,
                name="anchor",
                source=args.anchor_eval_source or args.train_source,
                prompt_field=args.anchor_field,
                response_field=contrastive_response_field(args),
            ),
            build_eval_command(
                args,
                name="benchmark",
                source=args.benchmark_source,
                prompt_field=benchmark_prompt_field(args),
                response_field=benchmark_response_field(args),
            ),
        ]
    return [
        build_eval_command(
            args,
            name="eval",
            source=args.eval_source,
            prompt_field=eval_prompt_field(args),
            response_field=eval_response_field(args),
        ),
        build_eval_command(
            args,
            name="benchmark",
            source=args.benchmark_source,
            prompt_field=benchmark_prompt_field(args),
            response_field=benchmark_response_field(args),
        ),
    ]


def eval_prompt_field(args: argparse.Namespace) -> str:
    return args.eval_prompt_field or args.prompt_field


def eval_response_field(args: argparse.Namespace) -> str:
    return args.eval_response_field or args.response_field


def benchmark_prompt_field(args: argparse.Namespace) -> str:
    return args.benchmark_prompt_field or args.prompt_field


def benchmark_response_field(args: argparse.Namespace) -> str:
    return args.benchmark_response_field or args.response_field


def contrastive_response_field(args: argparse.Namespace) -> str:
    return args.anchor_response_field or args.response_field


def run(command: Sequence[str], *, capture_json: bool = False) -> dict[str, object] | None:
    print("Running:")
    print(" ".join(shlex.quote(part) for part in command))
    try:
        if not capture_json:
            subprocess.run(command, check=True)
            return None

        completed = subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit status {exc.returncode}:", file=sys.stderr)
        print(" ".join(shlex.quote(part) for part in command), file=sys.stderr)
        if exc.stdout:
            print("\n--- stdout ---", file=sys.stderr)
            print(exc.stdout, file=sys.stderr, end="" if exc.stdout.endswith("\n") else "\n")
        if exc.stderr:
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

    output_path = Path(args.output_dir) / "generation_eval" / "top1_top5_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    commands = []
    if not args.skip_train:
        commands.append(build_train_command(args))
    if not args.skip_eval:
        commands.extend(build_eval_commands(args))

    if args.dry_run:
        for command in commands:
            print(" ".join(shlex.quote(part) for part in command))
        return 0

    missing = missing_dependencies()
    if missing:
        print_dependency_help(missing)
        return 2

    if not args.skip_train:
        run(build_train_command(args))

    results = {}
    if not args.skip_eval:
        eval_names = ["anchor", "benchmark"] if args.training_mode == "contrastive" else ["eval", "benchmark"]
        for name, command in zip(eval_names, build_eval_commands(args)):
            results[name] = run(command, capture_json=True)
        write_summary(args, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
