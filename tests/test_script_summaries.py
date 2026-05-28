from __future__ import annotations

import importlib.util
from pathlib import Path


def load_script(name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_summary_metric_filenames_follow_configured_top_k() -> None:
    assert load_script("train_sft_8gpu.py").summary_metrics_filename(3) == "top1_top3_metrics.json"
    assert (
        load_script("train_contrastive_8gpu.py").summary_metrics_filename(5)
        == "top1_top5_metrics.json"
    )
    assert (
        load_script("eval_model_and_benchmark.py").summary_metrics_filename(10)
        == "top1_top10_metrics.json"
    )


def test_distributed_training_scripts_forward_master_port() -> None:
    sft = load_script("train_sft_8gpu.py")
    sft_args = sft.parse_args(
        [
            "--train_source",
            "train.jsonl",
            "--train_eval_source",
            "train.jsonl",
            "--benchmark_source",
            "bench.jsonl",
            "--master_port",
            "29601",
        ]
    )
    assert option_value(sft.build_train_command(sft_args), "--master_port") == "29601"

    contrastive = load_script("train_contrastive_8gpu.py")
    contrastive_args = contrastive.parse_args(
        [
            "--contrastive_train_source",
            "contrastive.jsonl",
            "--sft_train_eval_source",
            "train.jsonl",
            "--benchmark_source",
            "bench.jsonl",
            "--master_port",
            "29602",
        ]
    )
    assert (
        option_value(contrastive.build_train_command(contrastive_args), "--master_port")
        == "29602"
    )
