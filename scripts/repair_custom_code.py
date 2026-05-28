#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encoder_decoder.modeling import copy_custom_code_files, missing_custom_code_files  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy missing Hugging Face custom-code files into a local checkpoint."
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument(
        "--base_model_name_or_path",
        default="charent/ChatLM-mini-Chinese",
        help="Local base model path or Hugging Face repo id that contains the custom .py files.",
    )
    return parser.parse_args(argv)


def resolve_source(base_model_name_or_path: str) -> Path:
    path = Path(base_model_name_or_path).expanduser()
    if path.exists():
        return path.resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install huggingface_hub or pass a local --base_model_name_or_path containing "
            "the custom .py files."
        ) from exc

    return Path(
        snapshot_download(
            base_model_name_or_path,
            allow_patterns=["*.py", "config.json", "tokenizer_config.json"],
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    before = missing_custom_code_files(checkpoint_dir)
    if not before:
        print(f"No missing custom-code files detected in {checkpoint_dir}")
        return 0

    source_dir = resolve_source(args.base_model_name_or_path)
    missing = copy_custom_code_files(checkpoint_dir, source_paths=[source_dir])
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Could not find required custom-code files in {source_dir}: {missing_text}"
        )

    print(f"Repaired {checkpoint_dir}")
    print(f"Copied files from {source_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
