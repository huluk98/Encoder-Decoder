from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping
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


def custom_code_filenames_from_config(config_or_path) -> list[str]:
    """Return custom remote-code Python files referenced by a config auto_map."""
    auto_map = _auto_map_from_config(config_or_path)
    if not isinstance(auto_map, dict):
        return []

    filenames = []
    for value in auto_map.values():
        references = value if isinstance(value, list) else [value]
        for reference in references:
            if not isinstance(reference, str):
                continue
            module_reference = reference.split("--", 1)[-1]
            module_name = module_reference.split(".", 1)[0]
            if not module_name or module_name.startswith("transformers"):
                continue
            filename = f"{module_name}.py"
            if filename not in filenames:
                filenames.append(filename)
    return filenames


def copy_custom_code_files(
    output_dir: str | Path,
    *,
    config=None,
    source_paths: list[str | Path | None] | tuple[str | Path | None, ...] = (),
    objects: list[object | None] | tuple[object | None, ...] = (),
) -> list[str]:
    """Copy custom Hugging Face remote-code files into a local checkpoint directory.

    This keeps checkpoints with `auto_map` entries loadable after training/pruning.
    Returns the required filenames that could not be found in any source location.
    """
    output_path = Path(output_dir).expanduser().resolve()
    required = custom_code_filenames_from_config(config)
    if not required:
        required = custom_code_filenames_from_config(output_path)
    if not required:
        return []

    source_dirs = _custom_code_source_dirs(source_paths=source_paths, objects=objects)
    missing = []
    copied_from_dirs: set[Path] = set()
    output_path.mkdir(parents=True, exist_ok=True)

    for filename in required:
        if (output_path / filename).exists():
            continue
        source_dir = next((path for path in source_dirs if (path / filename).exists()), None)
        if source_dir is None:
            missing.append(filename)
            continue
        if source_dir in copied_from_dirs:
            continue
        copied_from_dirs.add(source_dir)
        for py_file in sorted(source_dir.glob("*.py")):
            shutil.copy2(py_file, output_path / py_file.name)

    return [filename for filename in required if not (output_path / filename).exists()]


def missing_custom_code_files(model_dir: str | Path) -> list[str]:
    path = Path(model_dir).expanduser()
    return [
        filename
        for filename in custom_code_filenames_from_config(path)
        if not (path / filename).exists()
    ]


def save_tokenizer_pretrained_safely(tokenizer, output_dir: str | Path):
    """Save a tokenizer after making its save config JSON-serializable."""
    sanitize_tokenizer_for_save(tokenizer)
    return tokenizer.save_pretrained(output_dir)


def sanitize_tokenizer_for_save(tokenizer):
    """Remove non-JSON Python objects from tokenizer metadata before saving."""
    for attr in ("init_kwargs", "init_inputs"):
        if hasattr(tokenizer, attr):
            setattr(tokenizer, attr, _json_safe_tokenizer_value(getattr(tokenizer, attr)))

    name_or_path = getattr(tokenizer, "name_or_path", None)
    if name_or_path is not None and not isinstance(name_or_path, str):
        tokenizer.name_or_path = str(name_or_path)
    return tokenizer


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


def _json_safe_tokenizer_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    torch_dtype_name = _torch_dtype_name(value)
    if torch_dtype_name is not None:
        return torch_dtype_name

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            _json_safe_tokenizer_key(key): _json_safe_tokenizer_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe_tokenizer_value(item) for item in value]

    if isinstance(value, set):
        return [_json_safe_tokenizer_value(item) for item in sorted(value, key=str)]

    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _json_safe_tokenizer_key(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _torch_dtype_name(value) -> str | None:
    value_type = type(value)
    if value_type.__module__ == "torch" and value_type.__name__ == "dtype":
        return str(value).removeprefix("torch.")
    return None


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
    if trust_remote_code:
        copy_custom_code_files(
            resolved_model_path,
            source_paths=[Path(resolved_model_path).parent],
        )
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
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(resolved_model_path, **load_kwargs)
        except (ImportError, OSError, ValueError) as exc:
            _raise_custom_code_error_if_missing(resolved_model_path, exc)
            raise
    elif resolved_family == "causal":
        try:
            model = AutoModelForCausalLM.from_pretrained(resolved_model_path, **load_kwargs)
        except (ImportError, OSError, ValueError) as exc:
            _raise_custom_code_error_if_missing(resolved_model_path, exc)
            raise
    else:
        raise ValueError(f"Unsupported model family: {resolved_family}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, resolved_family


def _auto_map_from_config(config_or_path):
    if config_or_path is None:
        return None
    if isinstance(config_or_path, (str, Path)):
        path = Path(config_or_path).expanduser()
        config_path = path / "config.json" if path.is_dir() else path
        if not config_path.exists():
            return None
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return payload.get("auto_map")

    auto_map = getattr(config_or_path, "auto_map", None)
    if auto_map is not None:
        return auto_map
    if hasattr(config_or_path, "to_dict"):
        return config_or_path.to_dict().get("auto_map")
    return None


def _custom_code_source_dirs(
    *,
    source_paths: list[str | Path | None] | tuple[str | Path | None, ...],
    objects: list[object | None] | tuple[object | None, ...],
) -> list[Path]:
    source_dirs: list[Path] = []
    for value in source_paths:
        if value is None:
            continue
        path = Path(value).expanduser()
        if not path.exists():
            continue
        path = path.resolve()
        if path.is_file():
            path = path.parent
        if path not in source_dirs:
            source_dirs.append(path)

    for obj in objects:
        source_dir = _object_module_dir(obj)
        if source_dir is not None and source_dir not in source_dirs:
            source_dirs.append(source_dir)
    return source_dirs


def _object_module_dir(obj) -> Path | None:
    if obj is None:
        return None
    try:
        import inspect

        path = Path(inspect.getfile(obj.__class__)).expanduser().resolve()
    except (OSError, TypeError):
        return None
    if path.name == "__init__.py":
        return path.parent
    return path.parent if path.is_file() else path


def _raise_custom_code_error_if_missing(model_path: str, exc: Exception) -> None:
    path = Path(model_path)
    if not path.exists():
        return
    missing = missing_custom_code_files(path)
    if not missing:
        return
    missing_text = ", ".join(missing)
    raise FileNotFoundError(
        f"Local checkpoint {path} is missing custom Hugging Face code files: {missing_text}. "
        "Copy those files from the base model into the checkpoint directory, or run "
        "`python scripts/repair_custom_code.py --checkpoint_dir "
        f"{path} --base_model_name_or_path charent/ChatLM-mini-Chinese`."
    ) from exc
