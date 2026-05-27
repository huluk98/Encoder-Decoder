#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

DEFAULT_MODEL = "charent/ChatLM-mini-Chinese"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quick SFT runner for charent/ChatLM-mini-Chinese with exact and top-k "
            "generation evaluation."
        )
    )
    parser.add_argument("--train_source", required=True, help="SFT training data path or HF dataset.")
    parser.add_argument(
        "--benchmark_source",
        help="Benchmark eval data path or HF dataset. Uses prompt/response fields.",
    )
    parser.add_argument(
        "--mode",
        choices=["sft", "contrastive"],
        default="sft",
        help="sft evaluates the SFT data plus benchmark; contrastive evaluates anchor plus benchmark.",
    )
    parser.add_argument(
        "--sft_eval_source",
        help="Regular SFT eval data. Defaults to --train_source in sft mode.",
    )
    parser.add_argument(
        "--anchor_source",
        help="Contrastive eval data containing the anchor field. Required in contrastive mode.",
    )
    parser.add_argument("--output_dir", help="Output directory. Defaults under runs/.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--model_family", choices=["auto", "causal", "seq2seq"], default="auto")
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--response_field", default="response")
    parser.add_argument("--anchor_field", default="anchor")
    parser.add_argument("--anchor_response_field", default="response")
    parser.add_argument("--train_split")
    parser.add_argument("--sft_eval_split")
    parser.add_argument("--benchmark_eval_split")
    parser.add_argument("--anchor_eval_split")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--source_max_length", type=int, default=256)
    parser.add_argument("--target_max_length", type=int, default=256)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Use a small value like 20 for a quick smoke run.",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--generation_eval_limit", type=int)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--no_trust_remote_code",
        action="store_true",
        help="Disable trust_remote_code. The default is enabled for ChatLM-mini-Chinese.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print the command without running it.")
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    args.extra = extra
    if args.mode == "contrastive" and not args.anchor_source:
        parser.error("--anchor_source is required when --mode contrastive")
    return args


def build_command(args: argparse.Namespace) -> list[str]:
    script_path = Path(__file__).resolve().with_name("sft.py")
    output_dir = args.output_dir or f"runs/chatlm-mini-{args.mode}"

    command = [
        sys.executable,
        str(script_path),
        "--model_name_or_path",
        args.model_name_or_path,
        "--train_source",
        args.train_source,
        "--output_dir",
        output_dir,
        "--model_family",
        args.model_family,
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
        "--generation_eval_top_k",
        str(args.top_k),
        "--generation_eval_max_new_tokens",
        str(args.max_new_tokens),
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
        "--logging_steps",
        str(args.logging_steps),
        "--save_steps",
        str(args.save_steps),
        "--eval_steps",
        str(args.eval_steps),
    ]

    if not args.no_trust_remote_code:
        command.append("--trust_remote_code")
    if args.train_split:
        command.extend(["--train_split", args.train_split])
    if args.bf16:
        command.append("--bf16")
    if args.fp16:
        command.append("--fp16")
    if args.gradient_checkpointing:
        command.append("--gradient_checkpointing")
    if args.generation_eval_limit is not None:
        command.extend(["--generation_eval_limit", str(args.generation_eval_limit)])

    if args.mode == "sft":
        command.extend(["--sft_eval_source", args.sft_eval_source or args.train_source])
        if args.sft_eval_split:
            command.extend(["--sft_eval_split", args.sft_eval_split])
    else:
        command.extend(
            [
                "--anchor_eval_source",
                args.anchor_source,
                "--anchor_field",
                args.anchor_field,
                "--anchor_response_field",
                args.anchor_response_field,
            ]
        )
        if args.anchor_eval_split:
            command.extend(["--anchor_eval_split", args.anchor_eval_split])

    if args.benchmark_source:
        command.extend(["--benchmark_eval_source", args.benchmark_source])
        if args.benchmark_eval_split:
            command.extend(["--benchmark_eval_split", args.benchmark_eval_split])

    command.extend(args.extra)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_command(args)
    print("Running:")
    print(" ".join(shlex.quote(part) for part in command))
    if args.dry_run:
        return 0
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
