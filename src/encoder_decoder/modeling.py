from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

ModelFamily = Literal["auto", "causal", "seq2seq"]
ResolvedModelFamily = Literal["causal", "seq2seq"]

TOKENIZER_FILE_NAMES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "vocab.txt",
)


def infer_model_family(model_name_or_path: str, *, trust_remote_code: bool = False) -> ResolvedModelFamily:
    from transformers import AutoConfig

    resolved_path = resolve_model_name_or_path(model_name_or_path)
    config = AutoConfig.from_pretrained(resolved_path, trust_remote_code=trust_remote_code)
    return "seq2seq" if getattr(config, "is_encoder_decoder", False) else "causal"


def resolve_model_name_or_path(model_name_or_path: str) -> str:
    """Return an absolute local model path or a Hugging Face repo id.

    Transformers gives confusing Hub-style errors when a local path is misspelled
    or relative to the wrong working directory. This helper fails early for paths
    that are clearly local and also accepts an output directory that only contains
    `checkpoint-*` subdirectories.
    """
    path = Path(model_name_or_path).expanduser()
    if path.exists():
        if path.is_file():
            raise FileNotFoundError(
                f"Model path points to a file, not a directory: {path}. "
                "Pass the saved model directory or a checkpoint-* directory."
            )
        return str(_resolve_local_model_dir(path))

    if _looks_like_local_path(model_name_or_path):
        raise FileNotFoundError(
            f"Local model path does not exist: {path}. "
            "Check the path, quote it if it contains spaces, and pass either the "
            "training output directory or a checkpoint-* directory."
        )
    return model_name_or_path


def resolve_tokenizer_name_or_path(model_name_or_path: str, model_path: str) -> str:
    model_dir = Path(model_path)
    original_path = Path(model_name_or_path).expanduser()
    if model_dir.exists() and _has_tokenizer_files(model_dir):
        return str(model_dir)
    if original_path.exists() and _has_tokenizer_files(original_path):
        return str(original_path.resolve())
    if model_dir.exists() and _has_tokenizer_files(model_dir.parent):
        return str(model_dir.parent)
    return model_path


def _resolve_local_model_dir(path: Path) -> Path:
    resolved = path.resolve()
    if (resolved / "config.json").exists():
        return resolved

    checkpoint = _latest_checkpoint_with_config(resolved)
    if checkpoint is not None:
        return checkpoint

    raise FileNotFoundError(
        f"Local model directory does not contain config.json: {resolved}. "
        "Pass a Hugging Face saved model directory, or a checkpoint-* directory. "
        "If training stopped before final save, try the latest checkpoint under this directory."
    )


def _latest_checkpoint_with_config(path: Path) -> Path | None:
    checkpoints = []
    for child in path.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", child.name)
        if match and (child / "config.json").exists():
            checkpoints.append((int(match.group(1)), child.resolve()))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def _has_tokenizer_files(path: Path) -> bool:
    return any((path / name).exists() for name in TOKENIZER_FILE_NAMES)


def _looks_like_local_path(value: str) -> bool:
    path = Path(value).expanduser()
    if path.is_absolute():
        return True
    if value.startswith(("~", "./", "../")):
        return True
    if len(path.parts) > 2:
        return True
    if any(part.startswith("checkpoint-") for part in path.parts):
        return True
    first_part = path.parts[0] if path.parts else ""
    return first_part in {
        "checkpoint",
        "checkpoints",
        "model",
        "models",
        "output",
        "outputs",
        "run",
        "runs",
    }


def resolve_torch_dtype(dtype: str | None):
    if dtype in {None, "", "auto"}:
        return "auto"

    import torch

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype {dtype!r}. Use auto, float16, bfloat16, or float32.")
    return dtype_map[dtype]


def load_tokenizer_and_model(
    model_name_or_path: str,
    *,
    model_family: ModelFamily = "auto",
    torch_dtype: str | None = "auto",
    device_map: str | None = None,
    trust_remote_code: bool = False,
):
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    resolved_model_path = resolve_model_name_or_path(model_name_or_path)
    resolved_tokenizer_path = resolve_tokenizer_name_or_path(
        model_name_or_path,
        resolved_model_path,
    )
    resolved_family = (
        infer_model_family(resolved_model_path, trust_remote_code=trust_remote_code)
        if model_family == "auto"
        else model_family
    )
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_tokenizer_path,
        trust_remote_code=trust_remote_code,
    )
    dtype = resolve_torch_dtype(torch_dtype)

    load_kwargs = {"trust_remote_code": trust_remote_code}
    if dtype != "auto":
        load_kwargs["torch_dtype"] = dtype
    if device_map:
        load_kwargs["device_map"] = device_map

    if resolved_family == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(resolved_model_path, **load_kwargs)
    elif resolved_family == "causal":
        model = AutoModelForCausalLM.from_pretrained(resolved_model_path, **load_kwargs)
    else:
        raise ValueError(f"Unsupported model family: {resolved_family}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, resolved_family
