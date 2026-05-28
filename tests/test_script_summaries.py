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
