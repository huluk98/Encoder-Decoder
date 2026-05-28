from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_chatlm_quick.py"
SPEC = importlib.util.spec_from_file_location("run_chatlm_quick", SCRIPT_PATH)
assert SPEC is not None
run_chatlm_quick = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_chatlm_quick)


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_contrastive_quick_uses_separate_training_and_anchor_response_fields() -> None:
    args = run_chatlm_quick.parse_args(
        [
            "--train_source",
            "contrastive.jsonl",
            "--mode",
            "contrastive",
            "--anchor_source",
            "anchor_eval.jsonl",
            "--contrastive_response_field",
            "train_response",
            "--anchor_response_field",
            "eval_response",
            "--dry_run",
        ]
    )

    command = run_chatlm_quick.build_command(args)

    assert option_value(command, "--contrastive_response_field") == "train_response"
    assert option_value(command, "--anchor_response_field") == "eval_response"


def test_contrastive_quick_can_forward_regular_sft_eval_source() -> None:
    args = run_chatlm_quick.parse_args(
        [
            "--train_source",
            "contrastive.jsonl",
            "--mode",
            "contrastive",
            "--anchor_source",
            "anchor_eval.jsonl",
            "--sft_eval_source",
            "sft_eval.jsonl",
            "--sft_eval_prompt_field",
            "instruction",
            "--sft_eval_response_field",
            "answer",
            "--dry_run",
        ]
    )

    command = run_chatlm_quick.build_command(args)

    assert option_value(command, "--sft_eval_source") == "sft_eval.jsonl"
    assert option_value(command, "--sft_eval_prompt_field") == "instruction"
    assert option_value(command, "--sft_eval_response_field") == "answer"
