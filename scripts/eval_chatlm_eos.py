#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


EOS_TEXT = "[EOS]"
PROMPT_KEYS = ("prompt", "input", "question", "instruction")
RESPONSE_KEYS = ("response", "reponse", "answer", "output", "target")
TOKENIZER_FILE_NAMES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "vocab.txt",
)

# ---------------------------------------------------------------------------
# EDIT THIS BLOCK, THEN RUN:
#   python scripts/eval_chatlm_eos.py
# ---------------------------------------------------------------------------
MODEL_PATH = "/Users/luke/Documents/Encoder-Decoder/runs/chatlm-mini-8gpu-sft"
BASE_MODEL_PATH = "charent/ChatLM-mini-Chinese"
EVAL_FILE = "/Users/luke/Documents/SCENIC agent/data/SCENIC_full_training_dataset.json"
BENCHMARK_FILE = "/Users/luke/Documents/SCENIC agent/generated/iot_instruction_benchmark_200.json"
OUTPUT_DIR = "/Users/luke/Documents/Encoder-Decoder/runs/eval/chatlm-eos"

# Leave these as None for prompt/response files. For SCENIC anchor eval, use:
# PROMPT_KEY = "anchor"
# RESPONSE_KEY = "response"
PROMPT_KEY = None
RESPONSE_KEY = None
BENCHMARK_PROMPT_KEY = None
BENCHMARK_RESPONSE_KEY = None

BATCH_SIZE = 8
TOP_K = 5
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 256
NO_REPEAT_NGRAM_SIZE = 0
NORMALIZATION = "strip_eos"  # raw, strip, or strip_eos
ADD_EOS_TO_PROMPT = True
DEVICE = None  # None uses cuda if available, otherwise cpu.
FP16 = True
MODEL_LOAD_MODE = "auto"  # auto, direct, or base_then_weights
USE_8_GPUS = True
NPROC_PER_NODE = 8
MASTER_PORT = "29573"


def resolve_model_path(model_path: str | Path) -> Path:
    """
    Accept either:
      /path/to/model_save
    or:
      /path/to/ChatLM-mini-Chinese   # repo root containing model_save/
    or:
      /path/to/output_dir            # directory containing checkpoint-*/
    """
    p = Path(model_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model path does not exist: {p}")
    if p.is_file():
        raise FileNotFoundError(f"Model path points to a file, not a directory: {p}")
    if (p / "model_save").is_dir() and not (p / "config.json").exists():
        return p / "model_save"
    if (p / "config.json").exists():
        return p

    latest_checkpoint = latest_checkpoint_with_config(p)
    if latest_checkpoint is not None:
        return latest_checkpoint

    return p


def latest_checkpoint_with_config(path: Path) -> Path | None:
    checkpoints = []
    for child in path.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", child.name)
        if match and (child / "config.json").exists():
            checkpoints.append((int(match.group(1)), child))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def resolve_tokenizer_path(model_path: str | Path, resolved_model_path: Path) -> Path:
    original = Path(model_path).expanduser().resolve()
    for candidate in (resolved_model_path, original, resolved_model_path.parent):
        if candidate.exists() and any((candidate / name).exists() for name in TOKENIZER_FILE_NAMES):
            return candidate
    return resolved_model_path


def resolve_base_model_reference(base_model_path: str | Path) -> str:
    path = Path(base_model_path).expanduser()
    if path.exists():
        return str(resolve_model_path(path))
    if looks_like_local_path(str(base_model_path)):
        raise FileNotFoundError(f"Base model path does not exist: {path}")
    return str(base_model_path)


def looks_like_local_path(value: str) -> bool:
    path = Path(value).expanduser()
    if path.is_absolute():
        return True
    if value.startswith(("~", "./", "../")):
        return True
    if len(path.parts) > 2:
        return True
    if any(part.startswith("checkpoint-") for part in path.parts):
        return True
    return path.parts[:1] in {
        ("checkpoint",),
        ("checkpoints",),
        ("model",),
        ("models",),
        ("output",),
        ("outputs",),
        ("run",),
        ("runs",),
    }


def missing_custom_code_files(model_path: Path) -> list[str]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return []
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    auto_map = config.get("auto_map")
    if not isinstance(auto_map, dict):
        return []

    missing = []
    for value in auto_map.values():
        references = value if isinstance(value, list) else [value]
        for reference in references:
            if not isinstance(reference, str):
                continue
            module_reference = reference.split("--", 1)[-1]
            module_name = module_reference.split(".", 1)[0]
            if not module_name or module_name.startswith("transformers"):
                continue
            file_name = f"{module_name}.py"
            if not (model_path / file_name).exists() and file_name not in missing:
                missing.append(file_name)
    return missing


def checkpoint_weight_files(model_path: Path) -> list[Path]:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model_path / index_name
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_names = index.get("weight_map", {}).values()
            files = []
            seen = set()
            for name in weight_names:
                if name not in seen:
                    seen.add(name)
                    files.append(model_path / name)
            return files

    patterns = (
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
    )
    files = []
    for pattern in patterns:
        files.extend(sorted(model_path.glob(pattern)))
    return files


def load_weight_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                f"{path.name} requires safetensors. Install it with: pip install safetensors"
            ) from exc
        return dict(load_file(str(path), device="cpu"))

    import torch

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict"):
            if isinstance(payload.get(key), dict):
                return payload[key]
    return payload


def remap_checkpoint_key(key: str, target_keys: set[str]) -> str | None:
    candidates = [
        key,
        key.removeprefix("module."),
        key.removeprefix("model."),
        key.removeprefix("base_model.model."),
    ]
    for candidate in candidates:
        if candidate in target_keys:
            return candidate
    return None


def load_local_checkpoint_weights(model, model_path: Path) -> dict[str, Any]:
    files = checkpoint_weight_files(model_path)
    if not files:
        adapter_path = model_path / "adapter_model.safetensors"
        if adapter_path.exists():
            raise RuntimeError(
                f"Found only LoRA/PEFT adapter weights at {adapter_path}. "
                "This standalone script expects a full merged model checkpoint."
            )
        raise FileNotFoundError(
            f"No model weight files found in {model_path}. Expected model.safetensors, "
            "model-*.safetensors, pytorch_model.bin, or pytorch_model-*.bin."
        )

    target_state = model.state_dict()
    target_keys = set(target_state)
    loaded_keys = set()
    skipped = []

    for file_path in files:
        if not file_path.exists():
            raise FileNotFoundError(f"Checkpoint shard listed in index is missing: {file_path}")
        state_dict = load_weight_file(file_path)
        filtered = {}
        for key, tensor in state_dict.items():
            mapped_key = remap_checkpoint_key(key, target_keys)
            if mapped_key is None:
                skipped.append(key)
                continue
            target_tensor = target_state[mapped_key]
            if getattr(tensor, "shape", None) != target_tensor.shape:
                skipped.append(key)
                continue
            filtered[mapped_key] = tensor
            loaded_keys.add(mapped_key)
        model.load_state_dict(filtered, strict=False)
        del state_dict
        del filtered

    missing = sorted(target_keys - loaded_keys)
    return {
        "weight_files": [str(path) for path in files],
        "loaded_key_count": len(loaded_keys),
        "missing_key_count": len(missing),
        "skipped_key_count": len(skipped),
        "missing_key_examples": missing[:10],
        "skipped_key_examples": skipped[:10],
    }


def distributed_env() -> dict[str, int | bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_distributed": world_size > 1,
        "is_main": rank == 0,
    }


def setup_distributed(device: str | None):
    import torch

    ctx = distributed_env()
    if ctx["is_distributed"]:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(ctx["local_rank"]))
            torch_device = torch.device(f"cuda:{ctx['local_rank']}")
            backend = "nccl"
        else:
            torch_device = torch.device(device or "cpu")
            backend = "gloo"

        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        return torch_device, ctx

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return torch_device, ctx


def distributed_barrier() -> None:
    ctx = distributed_env()
    if not ctx["is_distributed"]:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    ctx = distributed_env()
    if not ctx["is_distributed"]:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def cuda_device_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def should_relaunch_distributed(args: argparse.Namespace) -> bool:
    if args.distributed_worker or distributed_env()["is_distributed"]:
        return False
    if not args.use_8_gpus or args.nproc_per_node <= 1:
        return False
    return cuda_device_count() >= 2


def relaunch_with_torchrun(args: argparse.Namespace) -> int:
    available_gpus = cuda_device_count()
    nproc = min(args.nproc_per_node, available_gpus)
    script_path = Path(__file__).resolve()
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(nproc),
        "--master_port",
        str(args.master_port),
        str(script_path),
        "--distributed_worker",
        *sys.argv[1:],
    ]
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"Launching distributed eval on {nproc} GPUs:")
    print(" ".join(str(part) for part in cmd))
    return subprocess.run(cmd, env=env, check=False).returncode


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Bad JSONL at {path}:{line_no}: {e}") from e
        return rows

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ("data", "examples", "records", "items"):
            if isinstance(obj.get(key), list):
                return obj[key]
        return [obj]

    raise ValueError(f"Expected JSON list/dict or JSONL records in {path}")


def find_key(row: dict[str, Any], explicit: str | None, candidates: tuple[str, ...], idx: int) -> str:
    if explicit:
        if explicit not in row:
            raise KeyError(f"Record {idx} is missing key {explicit!r}")
        return explicit

    for key in candidates:
        if key in row:
            return key

    raise KeyError(f"Record {idx} is missing one of: {candidates}")


def normalize_text(value: Any, mode: str = "strip_eos") -> str:
    """
    mode:
      raw       = byte-for-byte string compare
      strip     = strip surrounding whitespace
      strip_eos = strip whitespace and trailing textual [EOS]
    """
    s = "" if value is None else str(value)

    if mode in {"strip", "strip_eos"}:
        s = s.strip()

    if mode == "strip_eos":
        while s.endswith(EOS_TEXT):
            s = s[: -len(EOS_TEXT)].strip()

    if mode not in {"raw", "strip", "strip_eos"}:
        raise ValueError("normalization must be raw, strip, or strip_eos")

    return s


def add_eos(prompt: str, enabled: bool = True) -> str:
    prompt = str(prompt)
    if enabled and not prompt.rstrip().endswith(EOS_TEXT):
        return prompt + EOS_TEXT
    return prompt


def gold_values(value: Any, normalization: str) -> list[str]:
    """
    Supports either:
      "response": "answer"
    or:
      "response": ["valid answer 1", "valid answer 2"]
    """
    if isinstance(value, list):
        return [normalize_text(v, normalization) for v in value]
    return [normalize_text(value, normalization)]


def load_model_and_tokenizer(
    model_path: str | Path,
    device,
    fp16: bool = True,
    *,
    base_model_path: str | Path = BASE_MODEL_PATH,
    model_load_mode: str = MODEL_LOAD_MODE,
):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    resolved_model_path = resolve_model_path(model_path)
    tokenizer_path = resolve_tokenizer_path(model_path, resolved_model_path)
    resolved_base_model = resolve_base_model_reference(base_model_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            trust_remote_code=True,
        )
    except Exception as exc:
        print(
            f"Tokenizer load from local path failed: {exc}\n"
            f"Falling back to tokenizer from {resolved_base_model}",
            file=sys.stderr,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_base_model,
            trust_remote_code=True,
        )

    model_kwargs = {"trust_remote_code": True}
    if fp16 and device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    missing_code = missing_custom_code_files(resolved_model_path)
    load_report = {"mode": model_load_mode}
    model = None

    if model_load_mode not in {"auto", "direct", "base_then_weights"}:
        raise ValueError("model_load_mode must be auto, direct, or base_then_weights")

    if model_load_mode in {"auto", "direct"} and not missing_code:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                str(resolved_model_path),
                **model_kwargs,
            )
            load_report["mode"] = "direct"
        except Exception as exc:
            if model_load_mode == "direct":
                raise
            print(
                f"Direct local model load failed: {exc}\n"
                f"Falling back to architecture from {resolved_base_model} and local weights.",
                file=sys.stderr,
            )
    elif model_load_mode == "direct" and missing_code:
        raise FileNotFoundError(
            f"Local model is missing custom code files: {missing_code}. "
            "Use model_load_mode='base_then_weights' or copy those files into the checkpoint."
        )

    if model is None:
        if missing_code:
            print(
                f"Local checkpoint is missing custom code files {missing_code}; "
                f"loading architecture from {resolved_base_model} and applying local weights.",
                file=sys.stderr,
            )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            resolved_base_model,
            **model_kwargs,
        )
        weight_report = load_local_checkpoint_weights(model, resolved_model_path)
        load_report = {
            "mode": "base_then_weights",
            "base_model_path": resolved_base_model,
            **weight_report,
        }

    model._chatlm_eval_load_report = load_report
    model.to(device)
    model.eval()

    return model, tokenizer


def generate_topk(
    model,
    tokenizer,
    prompts: list[str],
    *,
    device,
    top_k: int = 5,
    max_input_tokens: int = 512,
    max_new_tokens: int = 256,
    no_repeat_ngram_size: int = 0,
) -> list[list[str]]:
    import torch

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    encoded.pop("token_type_ids", None)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.config.pad_token_id

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = model.config.eos_token_id

    decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
    if decoder_start_token_id is None:
        decoder_start_token_id = pad_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=top_k,
            num_return_sequences=top_k,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            remove_invalid_values=True,
        )

    decoded = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return [decoded[i : i + top_k] for i in range(0, len(decoded), top_k)]


def evaluate_file(
    *,
    model,
    tokenizer,
    file_path: str | Path,
    device,
    prompt_key: str | None = None,
    response_key: str | None = None,
    batch_size: int = 8,
    top_k: int = 5,
    max_input_tokens: int = 512,
    max_new_tokens: int = 256,
    normalization: str = "strip_eos",
    add_eos_to_prompt: bool = True,
    no_repeat_ngram_size: int = 0,
    predictions_path: str | Path | None = None,
    rank: int = 0,
    world_size: int = 1,
    show_progress: bool = True,
) -> dict[str, Any]:
    rows = read_json_or_jsonl(file_path)

    all_examples = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Record {i} is not a JSON object")

        p_key = find_key(row, prompt_key, PROMPT_KEYS, i)
        r_key = find_key(row, response_key, RESPONSE_KEYS, i)

        all_examples.append(
            {
                "idx": i,
                "prompt": str(row[p_key]),
                "gold": gold_values(row[r_key], normalization),
            }
        )

    examples = all_examples[rank::world_size] if world_size > 1 else all_examples
    correct_1 = 0
    correct_k = 0
    pred_rows = []

    progress = tqdm(
        range(0, len(examples), batch_size),
        desc=f"rank {rank} eval {Path(file_path).name}" if world_size > 1 else f"eval {Path(file_path).name}",
        disable=not show_progress,
    )
    for start in progress:
        batch = examples[start : start + batch_size]
        prompts = [add_eos(ex["prompt"], add_eos_to_prompt) for ex in batch]

        topk_outputs = generate_topk(
            model,
            tokenizer,
            prompts,
            device=device,
            top_k=top_k,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        for ex, candidates in zip(batch, topk_outputs):
            candidates = [normalize_text(c, normalization) for c in candidates]
            gold_set = set(ex["gold"])

            em1 = int(candidates[0] in gold_set)
            emk = int(any(c in gold_set for c in candidates[:top_k]))

            correct_1 += em1
            correct_k += emk

            pred_rows.append(
                {
                    "idx": ex["idx"],
                    "prompt": ex["prompt"],
                    "gold": ex["gold"],
                    "predictions": candidates,
                    "em1": em1,
                    f"em{top_k}": emk,
                }
            )

    n = len(examples)
    result = {
        "file": str(Path(file_path).expanduser().resolve()),
        "source_n": len(all_examples),
        "n": n,
        "correct@1": correct_1,
        f"correct@{top_k}": correct_k,
        "em@1": correct_1 / n if n else 0.0,
        f"em@{top_k}": correct_k / n if n else 0.0,
    }

    if predictions_path:
        predictions_path = Path(predictions_path).expanduser().resolve()
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8") as f:
            for row in pred_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        result["predictions_path"] = str(predictions_path)

    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prediction_part_path(parts_dir: Path, split_name: str, rank: int) -> Path:
    return parts_dir / f"{split_name}.rank{rank}.predictions.jsonl"


def rank_summary_path(parts_dir: Path, rank: int) -> Path:
    return parts_dir / f"rank{rank}.summary.json"


def merge_prediction_parts(part_paths: list[Path], output_path: Path) -> None:
    rows = []
    for path in part_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    rows.sort(key=lambda row: row.get("idx", 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_split_results(
    *,
    split_name: str,
    rank_summaries: list[dict[str, Any]],
    output_dir: Path,
    parts_dir: Path,
    top_k: int,
) -> dict[str, Any] | None:
    split_results = [summary[split_name] for summary in rank_summaries if split_name in summary]
    if not split_results:
        return None

    n = sum(int(result["n"]) for result in split_results)
    correct_1 = sum(int(result["correct@1"]) for result in split_results)
    correct_k = sum(int(result[f"correct@{top_k}"]) for result in split_results)
    output_path = output_dir / f"{split_name}.predictions.jsonl"
    merge_prediction_parts(
        [
            prediction_part_path(parts_dir, split_name, int(summary["distributed"]["rank"]))
            for summary in rank_summaries
            if split_name in summary
        ],
        output_path,
    )

    return {
        "file": split_results[0]["file"],
        "source_n": split_results[0].get("source_n", n),
        "n": n,
        "correct@1": correct_1,
        f"correct@{top_k}": correct_k,
        "em@1": correct_1 / n if n else 0.0,
        f"em@{top_k}": correct_k / n if n else 0.0,
        "predictions_path": str(output_path),
    }


def merge_distributed_summaries(
    *,
    output_dir: Path,
    parts_dir: Path,
    top_k: int,
    world_size: int,
) -> dict[str, Any]:
    rank_summaries = []
    for rank in range(world_size):
        path = rank_summary_path(parts_dir, rank)
        if not path.exists():
            raise FileNotFoundError(f"Missing distributed rank summary: {path}")
        rank_summaries.append(json.loads(path.read_text(encoding="utf-8")))

    merged = {
        "model_path": rank_summaries[0]["model_path"],
        "base_model_path": rank_summaries[0].get("base_model_path"),
        "model_load_report": rank_summaries[0].get("model_load_report", {}),
        "device": f"{world_size} GPUs",
        "distributed": {
            "world_size": world_size,
            "merged_from": [str(rank_summary_path(parts_dir, rank)) for rank in range(world_size)],
        },
    }

    for split_name in ("eval", "benchmark"):
        split_result = merge_split_results(
            split_name=split_name,
            rank_summaries=rank_summaries,
            output_dir=output_dir,
            parts_dir=parts_dir,
            top_k=top_k,
        )
        if split_result is not None:
            merged[split_name] = split_result

    merged["summary_path"] = str(output_dir / "summary.json")
    write_json(output_dir / "summary.json", merged)
    return merged


def run_chatlm_eval(
    *,
    model_path: str,
    eval_file: str,
    benchmark_file: str | None = None,
    prompt_key: str | None = None,
    response_key: str | None = None,
    benchmark_prompt_key: str | None = None,
    benchmark_response_key: str | None = None,
    batch_size: int = 8,
    top_k: int = 5,
    max_input_tokens: int = 512,
    max_new_tokens: int = 256,
    normalization: str = "strip_eos",
    add_eos_to_prompt: bool = True,
    no_repeat_ngram_size: int = 0,
    device: str | None = None,
    fp16: bool = True,
    base_model_path: str = BASE_MODEL_PATH,
    model_load_mode: str = MODEL_LOAD_MODE,
    output_dir: str | None = None,
) -> dict[str, Any]:
    torch_device, dist_ctx = setup_distributed(device)
    rank = int(dist_ctx["rank"])
    world_size = int(dist_ctx["world_size"])
    is_distributed = bool(dist_ctx["is_distributed"])
    is_main = bool(dist_ctx["is_main"])

    output_root = Path(output_dir).expanduser().resolve() if output_dir else None
    parts_dir = output_root / ".distributed_parts" if output_root and is_distributed else None
    if parts_dir:
        parts_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        model_path,
        torch_device,
        fp16=fp16,
        base_model_path=base_model_path,
        model_load_mode=model_load_mode,
    )

    summary = {
        "model_path": str(resolve_model_path(model_path)),
        "base_model_path": base_model_path,
        "model_load_report": getattr(model, "_chatlm_eval_load_report", {}),
        "device": str(torch_device),
        "distributed": {
            "rank": rank,
            "local_rank": int(dist_ctx["local_rank"]),
            "world_size": world_size,
        },
    }

    eval_pred_path = None
    if output_root:
        if parts_dir:
            eval_pred_path = prediction_part_path(parts_dir, "eval", rank)
        else:
            eval_pred_path = output_root / f"{Path(eval_file).stem}.predictions.jsonl"

    summary["eval"] = evaluate_file(
        model=model,
        tokenizer=tokenizer,
        file_path=eval_file,
        device=torch_device,
        prompt_key=prompt_key,
        response_key=response_key,
        batch_size=batch_size,
        top_k=top_k,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        normalization=normalization,
        add_eos_to_prompt=add_eos_to_prompt,
        no_repeat_ngram_size=no_repeat_ngram_size,
        predictions_path=eval_pred_path,
        rank=rank,
        world_size=world_size,
        show_progress=is_main,
    )

    if benchmark_file:
        bench_pred_path = None
        if output_root:
            if parts_dir:
                bench_pred_path = prediction_part_path(parts_dir, "benchmark", rank)
            else:
                bench_pred_path = output_root / f"{Path(benchmark_file).stem}.predictions.jsonl"

        summary["benchmark"] = evaluate_file(
            model=model,
            tokenizer=tokenizer,
            file_path=benchmark_file,
            device=torch_device,
            prompt_key=benchmark_prompt_key if benchmark_prompt_key is not None else prompt_key,
            response_key=benchmark_response_key if benchmark_response_key is not None else response_key,
            batch_size=batch_size,
            top_k=top_k,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            normalization=normalization,
            add_eos_to_prompt=add_eos_to_prompt,
            no_repeat_ngram_size=no_repeat_ngram_size,
            predictions_path=bench_pred_path,
            rank=rank,
            world_size=world_size,
            show_progress=is_main,
        )

    if output_root and not is_distributed:
        output_path = output_root / "summary.json"
        summary["summary_path"] = str(output_path)
        write_json(output_path, summary)
    elif output_root and parts_dir:
        write_json(rank_summary_path(parts_dir, rank), summary)
        distributed_barrier()
        if is_main:
            summary = merge_distributed_summaries(
                output_dir=output_root,
                parts_dir=parts_dir,
                top_k=top_k,
                world_size=world_size,
            )
        distributed_barrier()

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a ChatLM/T5-style seq2seq model with [EOS] prompt handling.")
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--base_model_path", default=BASE_MODEL_PATH)
    parser.add_argument(
        "--model_load_mode",
        choices=["auto", "direct", "base_then_weights"],
        default=MODEL_LOAD_MODE,
    )
    parser.add_argument("--eval_file", default=EVAL_FILE)
    parser.add_argument("--benchmark_file", default=BENCHMARK_FILE)

    parser.add_argument("--prompt_key", default=PROMPT_KEY)
    parser.add_argument("--response_key", default=RESPONSE_KEY)
    parser.add_argument("--benchmark_prompt_key", default=BENCHMARK_PROMPT_KEY)
    parser.add_argument("--benchmark_response_key", default=BENCHMARK_RESPONSE_KEY)

    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--max_input_tokens", type=int, default=MAX_INPUT_TOKENS)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=NO_REPEAT_NGRAM_SIZE)

    parser.add_argument("--normalization", choices=["raw", "strip", "strip_eos"], default=NORMALIZATION)
    parser.add_argument("--add_eos_to_prompt", dest="add_eos_to_prompt", action="store_true")
    parser.add_argument("--no_add_eos_to_prompt", dest="add_eos_to_prompt", action="store_false")
    parser.set_defaults(add_eos_to_prompt=ADD_EOS_TO_PROMPT)

    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")
    parser.set_defaults(fp16=FP16)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--use_8_gpus", dest="use_8_gpus", action="store_true")
    parser.add_argument("--no_8_gpus", dest="use_8_gpus", action="store_false")
    parser.set_defaults(use_8_gpus=USE_8_GPUS)
    parser.add_argument("--nproc_per_node", type=int, default=NPROC_PER_NODE)
    parser.add_argument("--master_port", default=MASTER_PORT)
    parser.add_argument("--distributed_worker", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if should_relaunch_distributed(args):
        return relaunch_with_torchrun(args)

    try:
        summary = run_chatlm_eval(
            model_path=args.model_path,
            eval_file=args.eval_file,
            benchmark_file=args.benchmark_file,
            prompt_key=args.prompt_key,
            response_key=args.response_key,
            benchmark_prompt_key=args.benchmark_prompt_key,
            benchmark_response_key=args.benchmark_response_key,
            batch_size=args.batch_size,
            top_k=args.top_k,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            normalization=args.normalization,
            add_eos_to_prompt=args.add_eos_to_prompt,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            device=args.device,
            fp16=args.fp16,
            base_model_path=args.base_model_path,
            model_load_mode=args.model_load_mode,
            output_dir=args.output_dir,
        )

        if bool(distributed_env()["is_main"]):
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
