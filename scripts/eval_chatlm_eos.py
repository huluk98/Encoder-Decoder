#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
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


def load_model_and_tokenizer(model_path: str | Path, device, fp16: bool = True):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    resolved_model_path = resolve_model_path(model_path)
    tokenizer_path = resolve_tokenizer_path(model_path, resolved_model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
    )

    model_kwargs = {"trust_remote_code": True}
    if fp16 and device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(resolved_model_path),
        **model_kwargs,
    )
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
) -> dict[str, Any]:
    rows = read_json_or_jsonl(file_path)

    examples = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Record {i} is not a JSON object")

        p_key = find_key(row, prompt_key, PROMPT_KEYS, i)
        r_key = find_key(row, response_key, RESPONSE_KEYS, i)

        examples.append(
            {
                "idx": i,
                "prompt": str(row[p_key]),
                "gold": gold_values(row[r_key], normalization),
            }
        )

    correct_1 = 0
    correct_k = 0
    pred_rows = []

    for start in tqdm(range(0, len(examples), batch_size), desc=f"eval {Path(file_path).name}"):
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
    output_dir: str | None = None,
) -> dict[str, Any]:
    import torch

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer = load_model_and_tokenizer(model_path, torch_device, fp16=fp16)

    summary = {
        "model_path": str(resolve_model_path(model_path)),
        "device": str(torch_device),
    }

    eval_pred_path = None
    if output_dir:
        eval_pred_path = Path(output_dir) / f"{Path(eval_file).stem}.predictions.jsonl"

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
    )

    if benchmark_file:
        bench_pred_path = None
        if output_dir:
            bench_pred_path = Path(output_dir) / f"{Path(benchmark_file).stem}.predictions.jsonl"

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
        )

    if output_dir:
        output_path = Path(output_dir).expanduser().resolve() / "summary.json"
        summary["summary_path"] = str(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a ChatLM/T5-style seq2seq model with [EOS] prompt handling.")
    parser.add_argument("--model_path", default=MODEL_PATH)
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

    args = parser.parse_args()

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
        output_dir=args.output_dir,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
