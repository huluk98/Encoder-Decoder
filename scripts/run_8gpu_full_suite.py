#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

DEFAULT_CONFIG = "configs/experiment_8gpu.yaml"
DEFAULT_METHODS = ["magnitude", "gradient", "nvidia", "wanda"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SFT, contrastive SFT, and all pruning/eval jobs from one 8-GPU config."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["all", "sft", "contrastive"], default="all")
    parser.add_argument("--stage", choices=["all", "train", "prune", "summary"], default="all")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    config_path = resolve_repo_path(args.config, root)
    config = load_config(config_path)
    modes = selected_modes(args.mode)
    env = runtime_env(config)

    if args.stage in {"all", "train"}:
        for mode in modes:
            if stage_enabled(config, f"train_{mode}", default=True):
                run(build_train_command(config, mode), root=root, env=env, dry_run=args.dry_run)

    if args.stage in {"all", "prune"}:
        for mode in modes:
            if stage_enabled(config, f"prune_{mode}", default=True):
                run(build_prune_command(config, mode), root=root, env=env, dry_run=args.dry_run)

    if args.stage in {"all", "train", "prune", "summary"} and not args.dry_run:
        summary_path = write_combined_summary(config, modes)
        print(f"Wrote combined summary: {summary_path}")
    elif args.dry_run:
        print("Dry run only. No commands were executed and no summary was written.")
    return 0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml
    except ImportError:
        return load_simple_yaml(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Small fallback parser for this repo's two-level YAML config."""
    config: dict[str, Any] = {}
    current_section: str | None = None
    current_list_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = strip_yaml_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()

        if indent == 0 and content.endswith(":"):
            current_section = content[:-1].strip()
            config[current_section] = {}
            current_list_key = None
            continue

        if current_section is None:
            raise ValueError(f"Expected a top-level section before line: {raw_line}")
        section_data = config[current_section]

        if indent == 2 and content.startswith("- "):
            if current_list_key is None:
                raise ValueError(f"Unexpected list item without a key: {raw_line}")
            section_data[current_list_key].append(parse_scalar(content[2:].strip()))
            continue

        if indent == 2 and ":" in content:
            key, value = content.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                section_data[key] = parse_scalar(value)
                current_list_key = None
            else:
                if key in {"methods"}:
                    section_data[key] = []
                    current_list_key = key
                else:
                    section_data[key] = None
                    current_list_key = None
            continue

        if indent == 4 and content.startswith("- "):
            if current_list_key is None:
                raise ValueError(f"Unexpected nested list item without a key: {raw_line}")
            section_data[current_list_key].append(parse_scalar(content[2:].strip()))
            continue

        raise ValueError(f"Unsupported YAML line for fallback parser: {raw_line}")

    return config


def strip_yaml_comment(line: str) -> str:
    quote: str | None = None
    for index, char in enumerate(line):
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
        elif char == "#" and quote is None:
            return line[:index]
    return line


def parse_scalar(value: str) -> Any:
    if value == "":
        return None
    if value in {"''", '""'}:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def selected_modes(mode: str) -> list[str]:
    if mode == "all":
        return ["sft", "contrastive"]
    return [mode]


def stage_enabled(config: dict[str, Any], key: str, *, default: bool) -> bool:
    return bool(section(config, "stages").get(key, default))


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def config_value(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return default if current is None else current


def resolve_repo_path(value: str | Path, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def data_path(config: dict[str, Any], key: str, default: str = "") -> str:
    value = str(config_value(config, f"paths.{key}", default) or "")
    if not value:
        return ""
    return str(resolve_repo_path(value, repo_root()))


def output_root(config: dict[str, Any]) -> Path:
    return resolve_repo_path(
        str(config_value(config, "paths.output_root", "runs/full_8gpu_suite")),
        repo_root(),
    )


def mode_output_dir(config: dict[str, Any], mode: str) -> Path:
    explicit_key = "sft_model_path" if mode == "sft" else "contrastive_model_path"
    explicit = str(config_value(config, f"paths.{explicit_key}", "") or "")
    if explicit:
        return resolve_repo_path(explicit, repo_root())
    return output_root(config) / mode


def methods(config: dict[str, Any]) -> list[str]:
    configured = config_value(config, "pruning.methods", DEFAULT_METHODS)
    if isinstance(configured, str):
        return configured.split()
    return [str(item) for item in configured]


def precision(config: dict[str, Any]) -> str:
    return str(config_value(config, "model.precision", "bf16"))


def trust_remote_code(config: dict[str, Any]) -> bool:
    return bool(config_value(config, "model.trust_remote_code", True))


def field_value(config: dict[str, Any], key: str, fallback: str) -> str:
    fields = section(config, "fields")
    return str(fields.get(key) or fields.get(fallback) or fallback)


def sft_prompt_field(config: dict[str, Any]) -> str:
    return field_value(config, "sft_prompt", "prompt")


def sft_response_field(config: dict[str, Any]) -> str:
    return field_value(config, "sft_response", "response")


def sft_eval_prompt_field(config: dict[str, Any]) -> str:
    return str(section(config, "fields").get("sft_eval_prompt") or sft_prompt_field(config))


def sft_eval_response_field(config: dict[str, Any]) -> str:
    return str(section(config, "fields").get("sft_eval_response") or sft_response_field(config))


def benchmark_prompt_field(config: dict[str, Any]) -> str:
    return field_value(config, "benchmark_prompt", "prompt")


def benchmark_response_field(config: dict[str, Any]) -> str:
    return field_value(config, "benchmark_response", "response")


def contrastive_response_field(config: dict[str, Any]) -> str:
    return str(section(config, "fields").get("contrastive_response") or sft_response_field(config))


def runtime_env(config: dict[str, Any]) -> dict[str, str]:
    runtime = section(config, "runtime")
    env = os.environ.copy()
    runtime_map = {
        "cuda_visible_devices": "CUDA_VISIBLE_DEVICES",
        "tokenizers_parallelism": "TOKENIZERS_PARALLELISM",
        "nccl_debug": "NCCL_DEBUG",
        "pytorch_cuda_alloc_conf": "PYTORCH_CUDA_ALLOC_CONF",
        "omp_num_threads": "OMP_NUM_THREADS",
    }
    for config_key, env_key in runtime_map.items():
        value = runtime.get(config_key)
        if value is not None:
            env[env_key] = str(value)
    return env


def build_train_command(config: dict[str, Any], mode: str) -> list[str]:
    root = repo_root()
    script = root / "scripts" / "run_chatlm_8gpu_sft.py"
    generation = section(config, "generation")
    training = section(config, "training")
    contrastive = section(config, "contrastive")

    train_source = (
        data_path(config, "sft_train")
        if mode == "sft"
        else data_path(config, "contrastive_train")
    )
    eval_source = data_path(config, "sft_eval", data_path(config, "sft_train"))
    benchmark_source = data_path(config, "benchmark")

    command = [
        sys.executable,
        str(script),
        "--training_mode",
        mode,
        "--model_path",
        str(config_value(config, "model.name_or_path", "charent/ChatLM-mini-Chinese")),
        "--model_family",
        str(config_value(config, "model.family", "seq2seq")),
        "--precision",
        precision(config),
        "--nproc_per_node",
        str(training.get("nproc_per_node", 8)),
        "--train_source",
        train_source,
        "--eval_source",
        eval_source,
        "--benchmark_source",
        benchmark_source,
        "--output_dir",
        str(mode_output_dir(config, mode)),
        "--epochs",
        str(training.get("num_train_epochs", 3.0)),
        "--learning_rate",
        str(training.get("learning_rate", 5e-5)),
        "--per_device_train_batch_size",
        str(training.get("per_device_train_batch_size", 1)),
        "--per_device_eval_batch_size",
        str(training.get("per_device_eval_batch_size", 1)),
        "--gradient_accumulation_steps",
        str(training.get("gradient_accumulation_steps", 8)),
        "--max_steps",
        str(training.get("max_steps", -1)),
        "--top_k",
        str(generation.get("top_k", 5)),
        "--num_beams",
        str(generation.get("num_beams", 5)),
        "--max_new_tokens",
        str(generation.get("max_new_tokens", 256)),
        "--max_seq_length",
        str(generation.get("max_seq_length", 512)),
        "--source_max_length",
        str(generation.get("source_max_length", 256)),
        "--target_max_length",
        str(generation.get("target_max_length", 256)),
        "--prompt_field",
        sft_prompt_field(config) if mode == "sft" else benchmark_prompt_field(config),
        "--response_field",
        sft_response_field(config) if mode == "sft" else contrastive_response_field(config),
        "--eval_prompt_field",
        sft_eval_prompt_field(config),
        "--eval_response_field",
        sft_eval_response_field(config),
        "--benchmark_prompt_field",
        benchmark_prompt_field(config),
        "--benchmark_response_field",
        benchmark_response_field(config),
    ]
    if generation.get("generation_eval_limit") is not None:
        command.extend(["--generation_eval_limit", str(generation["generation_eval_limit"])])
    if not trust_remote_code(config):
        command.append("--no_trust_remote_code")
    if not bool(training.get("gradient_checkpointing", True)):
        command.append("--no_gradient_checkpointing")

    if mode == "contrastive":
        command.extend(
            [
                "--anchor_eval_source",
                data_path(config, "anchor_eval", data_path(config, "contrastive_train")),
                "--anchor_field",
                field_value(config, "anchor", "anchor"),
                "--anchor_response_field",
                contrastive_response_field(config),
                "--contrastive_positive_field",
                field_value(config, "positive", "positive"),
                "--contrastive_negative_field",
                field_value(config, "negative", "negative"),
                "--contrastive_loss_weight",
                str(contrastive.get("loss_weight", 0.2)),
                "--contrastive_margin",
                str(contrastive.get("margin", 0.2)),
            ]
        )
    return command


def build_prune_command(config: dict[str, Any], mode: str) -> tuple[list[str], dict[str, str]]:
    root = repo_root()
    generation = section(config, "generation")
    pruning = section(config, "pruning")
    output_dir = output_root(config) / "pruning" / mode
    calibration_source = (
        data_path(config, "sft_train") if mode == "sft" else data_path(config, "contrastive_train")
    )
    eval_source = data_path(config, "sft_eval") if mode == "sft" else data_path(config, "anchor_eval")

    env = runtime_env(config)
    env.update(
        {
            "MODEL_PATH": str(mode_output_dir(config, mode)),
            "CALIBRATION_SOURCE": calibration_source,
            "EVAL_SOURCE": eval_source,
            "BENCHMARK_SOURCE": data_path(config, "benchmark"),
            "OUTPUT_ROOT": str(output_dir),
            "MODEL_FAMILY": str(config_value(config, "model.family", "seq2seq")),
            "PRECISION": precision(config),
            "SPARSITY": str(pruning.get("sparsity", 0.5)),
            "TOP_K": str(generation.get("top_k", 5)),
            "MAX_NEW_TOKENS": str(generation.get("max_new_tokens", 256)),
            "MAX_INPUT_LENGTH": str(generation.get("source_max_length", 256)),
            "MAX_SEQ_LENGTH": str(generation.get("max_seq_length", 512)),
            "SOURCE_MAX_LENGTH": str(generation.get("source_max_length", 256)),
            "TARGET_MAX_LENGTH": str(generation.get("target_max_length", 256)),
            "CALIBRATION_LIMIT": str(pruning.get("calibration_limit", 128)),
            "CALIBRATION_PROMPT_FIELD": str(
                sft_prompt_field(config) if mode == "sft" else field_value(config, "anchor", "anchor")
            ),
            "CALIBRATION_RESPONSE_FIELD": str(
                sft_response_field(config) if mode == "sft" else contrastive_response_field(config)
            ),
            "EVAL_PROMPT_FIELD": str(
                sft_eval_prompt_field(config)
                if mode == "sft"
                else field_value(config, "anchor", "anchor")
            ),
            "EVAL_RESPONSE_FIELD": str(
                sft_eval_response_field(config)
                if mode == "sft"
                else contrastive_response_field(config)
            ),
            "BENCHMARK_PROMPT_FIELD": benchmark_prompt_field(config),
            "BENCHMARK_RESPONSE_FIELD": benchmark_response_field(config),
            "TRUST_REMOTE_CODE": "1" if trust_remote_code(config) else "0",
            "METHODS": " ".join(methods(config)),
        }
    )
    return ["bash", str(root / "scripts" / "run_pruning_eval.sh")], env


def run(
    command_spec: list[str] | tuple[list[str], dict[str, str]],
    *,
    root: Path,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    command = command_spec[0] if isinstance(command_spec, tuple) else command_spec
    command_env = command_spec[1] if isinstance(command_spec, tuple) else env
    print("Running:")
    if isinstance(command_spec, tuple):
        interesting_env = {
            key: command_env[key]
            for key in sorted(command_env)
            if key
            in {
                "MODEL_PATH",
                "CALIBRATION_SOURCE",
                "EVAL_SOURCE",
                "BENCHMARK_SOURCE",
                "OUTPUT_ROOT",
                "MODEL_FAMILY",
                "PRECISION",
                "METHODS",
                "CALIBRATION_PROMPT_FIELD",
                "EVAL_PROMPT_FIELD",
                "BENCHMARK_PROMPT_FIELD",
            }
        }
        for key, value in interesting_env.items():
            print(f"  {key}={shlex.quote(str(value))}")
    print("  " + " ".join(shlex.quote(part) for part in command))
    if not dry_run:
        subprocess.run(command, cwd=root, env=command_env, check=True)


def write_combined_summary(config: dict[str, Any], modes: list[str]) -> Path:
    root = output_root(config)
    top_k = int(config_value(config, "generation.top_k", 5))
    summary: dict[str, Any] = {
        "output_root": str(root),
        "model_outputs": {mode: str(mode_output_dir(config, mode)) for mode in modes},
        "generation": {},
        "pruning": {},
    }
    rows: list[dict[str, Any]] = []

    for mode in modes:
        generation_path = mode_output_dir(config, mode) / "generation_eval" / "top1_top5_metrics.json"
        if generation_path.exists():
            metrics = json.loads(generation_path.read_text(encoding="utf-8"))
            summary["generation"][mode] = metrics
            for split, split_metrics in metrics.items():
                rows.append(summary_row(mode, "generation", "", split, split_metrics, top_k))

        mode_pruning: dict[str, Any] = {}
        for method in methods(config):
            method_metrics: dict[str, Any] = {}
            for split in ("eval", "benchmark"):
                metrics_path = root / "pruning" / mode / method / f"{split}_metrics.json"
                if not metrics_path.exists():
                    continue
                split_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                method_metrics[split] = split_metrics
                rows.append(summary_row(mode, "pruning", method, split, split_metrics, top_k))
            if method_metrics:
                mode_pruning[method] = method_metrics
        if mode_pruning:
            summary["pruning"][mode] = mode_pruning

    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "combined_summary.json"
    csv_path = root / "combined_summary.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_summary_csv(csv_path, rows, top_k)
    return summary_path


def summary_row(
    mode: str,
    stage: str,
    method: str,
    split: str,
    metrics: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    top_k_key = f"top_{top_k}_accuracy"
    return {
        "mode": mode,
        "stage": stage,
        "method": method,
        "split": split,
        "total": metrics.get("total", ""),
        "top_1_accuracy": metrics.get("top_1_accuracy", metrics.get("accuracy", "")),
        top_k_key: metrics.get(top_k_key, ""),
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]], top_k: int) -> None:
    fieldnames = ["mode", "stage", "method", "split", "total", "top_1_accuracy", f"top_{top_k}_accuracy"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
