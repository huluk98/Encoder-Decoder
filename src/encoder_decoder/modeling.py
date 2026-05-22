from __future__ import annotations

from typing import Literal

ModelFamily = Literal["auto", "causal", "seq2seq"]
ResolvedModelFamily = Literal["causal", "seq2seq"]


def infer_model_family(model_name_or_path: str, *, trust_remote_code: bool = False) -> ResolvedModelFamily:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    return "seq2seq" if getattr(config, "is_encoder_decoder", False) else "causal"


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

    resolved_family = (
        infer_model_family(model_name_or_path, trust_remote_code=trust_remote_code)
        if model_family == "auto"
        else model_family
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    dtype = resolve_torch_dtype(torch_dtype)

    load_kwargs = {"trust_remote_code": trust_remote_code}
    if dtype != "auto":
        load_kwargs["torch_dtype"] = dtype
    if device_map:
        load_kwargs["device_map"] = device_map

    if resolved_family == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **load_kwargs)
    elif resolved_family == "causal":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    else:
        raise ValueError(f"Unsupported model family: {resolved_family}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, resolved_family

