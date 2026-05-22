from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from encoder_decoder.data import PromptResponseRecord, load_prompt_response_records
from encoder_decoder.modeling import ModelFamily, load_tokenizer_and_model
from encoder_decoder.tokenization import (
    DEFAULT_CAUSAL_PROMPT_TEMPLATE,
    DEFAULT_CAUSAL_RESPONSE_TEMPLATE,
    tokenize_causal_prompt_response,
    tokenize_seq2seq_prompt_response,
)


@dataclass
class LayerSparsity:
    name: str
    total: int
    zeros: int
    sparsity: float


@dataclass
class PruningReport:
    method: str
    total: int
    zeros: int
    sparsity: float
    layers: list[LayerSparsity]


@dataclass
class PruningConfig:
    model_name_or_path: str
    output_dir: str
    method: str
    sparsity: float = 0.5
    model_family: ModelFamily = "auto"
    trust_remote_code: bool = False
    torch_dtype: str | None = "auto"
    device: str | None = None
    layer_regex: str | None = None
    include_lm_head: bool = False
    global_pruning: bool = False
    calibration_source: str | None = None
    calibration_split: str | None = None
    calibration_limit: int = 128
    prompt_field: str = "prompt"
    response_field: str = "response"
    max_seq_length: int = 1024
    source_max_length: int = 768
    target_max_length: int = 256
    causal_prompt_template: str = DEFAULT_CAUSAL_PROMPT_TEMPLATE
    causal_response_template: str = DEFAULT_CAUSAL_RESPONSE_TEMPLATE
    nvidia_keep_n: int = 2
    nvidia_group_m: int = 4


def iter_linear_layers(
    model,
    *,
    layer_regex: str | None = None,
    include_lm_head: bool = False,
):
    import torch.nn as nn

    pattern = re.compile(layer_regex) if layer_regex else None
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not include_lm_head and (name == "lm_head" or name.endswith(".lm_head")):
            continue
        if pattern and not pattern.search(name):
            continue
        yield name, module


def magnitude_prune_model(
    model,
    *,
    sparsity: float,
    layer_regex: str | None = None,
    include_lm_head: bool = False,
    global_pruning: bool = False,
) -> dict[str, object]:
    layers = dict(
        iter_linear_layers(model, layer_regex=layer_regex, include_lm_head=include_lm_head)
    )
    scores = {
        name: module.weight.detach().abs().float().cpu()
        for name, module in layers.items()
    }
    return apply_score_pruning(layers, scores, sparsity=sparsity, global_pruning=global_pruning)


def gradient_prune_model(
    model,
    tokenizer,
    records: Sequence[PromptResponseRecord],
    *,
    resolved_family: str,
    sparsity: float,
    device,
    layer_regex: str | None = None,
    include_lm_head: bool = False,
    global_pruning: bool = False,
    max_seq_length: int = 1024,
    source_max_length: int = 768,
    target_max_length: int = 256,
    causal_prompt_template: str = DEFAULT_CAUSAL_PROMPT_TEMPLATE,
    causal_response_template: str = DEFAULT_CAUSAL_RESPONSE_TEMPLATE,
) -> dict[str, object]:
    layers = dict(
        iter_linear_layers(model, layer_regex=layer_regex, include_lm_head=include_lm_head)
    )
    scores = collect_gradient_scores(
        model,
        tokenizer,
        records,
        layers=layers,
        resolved_family=resolved_family,
        device=device,
        max_seq_length=max_seq_length,
        source_max_length=source_max_length,
        target_max_length=target_max_length,
        causal_prompt_template=causal_prompt_template,
        causal_response_template=causal_response_template,
    )
    return apply_score_pruning(layers, scores, sparsity=sparsity, global_pruning=global_pruning)


def wanda_prune_model(
    model,
    tokenizer,
    records: Sequence[PromptResponseRecord],
    *,
    resolved_family: str,
    sparsity: float,
    device,
    layer_regex: str | None = None,
    include_lm_head: bool = False,
    max_seq_length: int = 1024,
    source_max_length: int = 768,
    target_max_length: int = 256,
    causal_prompt_template: str = DEFAULT_CAUSAL_PROMPT_TEMPLATE,
    causal_response_template: str = DEFAULT_CAUSAL_RESPONSE_TEMPLATE,
) -> dict[str, object]:
    layers = dict(
        iter_linear_layers(model, layer_regex=layer_regex, include_lm_head=include_lm_head)
    )
    scores = collect_wanda_scores(
        model,
        tokenizer,
        records,
        layers=layers,
        resolved_family=resolved_family,
        device=device,
        max_seq_length=max_seq_length,
        source_max_length=source_max_length,
        target_max_length=target_max_length,
        causal_prompt_template=causal_prompt_template,
        causal_response_template=causal_response_template,
    )
    return apply_score_pruning(layers, scores, sparsity=sparsity, rowwise=True)


def nvidia_nm_prune_model(
    model,
    *,
    keep_n: int = 2,
    group_m: int = 4,
    layer_regex: str | None = None,
    include_lm_head: bool = False,
) -> dict[str, object]:
    import torch

    if keep_n <= 0 or group_m <= 0 or keep_n >= group_m:
        raise ValueError("NVIDIA N:M pruning expects 0 < keep_n < group_m.")
    layers = dict(
        iter_linear_layers(model, layer_regex=layer_regex, include_lm_head=include_lm_head)
    )
    masks: dict[str, object] = {}
    for name, module in layers.items():
        weight = module.weight.detach()
        if weight.dim() != 2:
            continue
        rows, cols = weight.shape
        usable_cols = cols - (cols % group_m)
        mask = torch.ones_like(weight, dtype=torch.bool, device=weight.device)
        if usable_cols:
            grouped_scores = weight[:, :usable_cols].abs().reshape(rows, -1, group_m)
            group_mask = torch.ones_like(grouped_scores, dtype=torch.bool)
            prune_count = group_m - keep_n
            prune_idx = torch.topk(
                grouped_scores,
                prune_count,
                dim=2,
                largest=False,
                sorted=False,
            ).indices
            group_mask.scatter_(2, prune_idx, False)
            mask[:, :usable_cols] = group_mask.reshape(rows, usable_cols)
        with torch.no_grad():
            module.weight.mul_(mask.to(dtype=module.weight.dtype))
        masks[name] = mask.detach().cpu()
    return masks


def apply_score_pruning(
    layers: dict[str, object],
    scores: dict[str, object],
    *,
    sparsity: float,
    global_pruning: bool = False,
    rowwise: bool = False,
) -> dict[str, object]:
    _validate_sparsity(sparsity)
    if not layers:
        raise ValueError("No linear layers matched the pruning filters.")

    if rowwise:
        masks = {
            name: _rowwise_lowest_score_mask(score.float().cpu(), sparsity)
            for name, score in scores.items()
        }
    elif global_pruning:
        masks = _global_lowest_score_masks(scores, sparsity)
    else:
        masks = {
            name: _lowest_score_mask(score.float().cpu(), sparsity)
            for name, score in scores.items()
        }

    for name, mask in masks.items():
        module = layers[name]
        with _no_grad():
            module.weight.mul_(mask.to(device=module.weight.device, dtype=module.weight.dtype))
    return masks


def collect_gradient_scores(
    model,
    tokenizer,
    records: Sequence[PromptResponseRecord],
    *,
    layers: dict[str, object],
    resolved_family: str,
    device,
    max_seq_length: int,
    source_max_length: int,
    target_max_length: int,
    causal_prompt_template: str,
    causal_response_template: str,
) -> dict[str, object]:
    import torch

    if not records:
        raise ValueError("Gradient pruning requires at least one calibration record.")

    scores = {
        name: torch.zeros_like(module.weight.detach(), dtype=torch.float32, device="cpu")
        for name, module in layers.items()
    }
    was_training = model.training
    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        for record in _progress(records, desc="Gradient scores"):
            batch = _record_to_batch(
                tokenizer,
                record,
                resolved_family=resolved_family,
                device=device,
                max_seq_length=max_seq_length,
                source_max_length=source_max_length,
                target_max_length=target_max_length,
                causal_prompt_template=causal_prompt_template,
                causal_response_template=causal_response_template,
            )
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None:
                raise RuntimeError("Model forward pass did not return a loss for calibration data.")
            loss.backward()
            for name, module in layers.items():
                if module.weight.grad is None:
                    continue
                scores[name] += (
                    module.weight.detach().abs().float().cpu()
                    * module.weight.grad.detach().abs().float().cpu()
                )
            model.zero_grad(set_to_none=True)
    finally:
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
        if was_training:
            model.train()
        else:
            model.eval()
    return scores


def collect_wanda_scores(
    model,
    tokenizer,
    records: Sequence[PromptResponseRecord],
    *,
    layers: dict[str, object],
    resolved_family: str,
    device,
    max_seq_length: int,
    source_max_length: int,
    target_max_length: int,
    causal_prompt_template: str,
    causal_response_template: str,
    eps: float = 1e-8,
) -> dict[str, object]:
    import torch

    if not records:
        raise ValueError("WANDA pruning requires at least one calibration record.")

    activation_sums = {
        name: torch.zeros(module.in_features, dtype=torch.float32, device="cpu")
        for name, module in layers.items()
    }
    activation_counts = {name: 0 for name in layers}
    handles = []

    def make_hook(name: str):
        def hook(module, inputs, _output):
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            acts = inputs[0].detach()
            if acts.shape[-1] != module.in_features:
                return
            acts = acts.reshape(-1, acts.shape[-1]).float()
            activation_sums[name] += acts.pow(2).sum(dim=0).cpu()
            activation_counts[name] += acts.shape[0]

        return hook

    for name, module in layers.items():
        handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False
    model.eval()
    try:
        with torch.no_grad():
            for record in _progress(records, desc="WANDA activations"):
                batch = _record_to_batch(
                    tokenizer,
                    record,
                    resolved_family=resolved_family,
                    device=device,
                    max_seq_length=max_seq_length,
                    source_max_length=source_max_length,
                    target_max_length=target_max_length,
                    causal_prompt_template=causal_prompt_template,
                    causal_response_template=causal_response_template,
                )
                model(**batch)
    finally:
        for handle in handles:
            handle.remove()
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
        if was_training:
            model.train()

    scores = {}
    for name, module in layers.items():
        count = max(1, activation_counts[name])
        scale = torch.sqrt(activation_sums[name] / count + eps)
        scores[name] = module.weight.detach().abs().float().cpu() * scale.view(1, -1)
    return scores


def summarize_linear_sparsity(
    model,
    *,
    layer_regex: str | None = None,
    include_lm_head: bool = False,
    method: str = "unknown",
) -> PruningReport:
    layers = []
    total = 0
    zeros = 0
    for name, module in iter_linear_layers(
        model,
        layer_regex=layer_regex,
        include_lm_head=include_lm_head,
    ):
        weight = module.weight.detach()
        layer_total = weight.numel()
        layer_zeros = int((weight == 0).sum().item())
        total += layer_total
        zeros += layer_zeros
        layers.append(
            LayerSparsity(
                name=name,
                total=layer_total,
                zeros=layer_zeros,
                sparsity=layer_zeros / layer_total if layer_total else 0.0,
            )
        )
    return PruningReport(
        method=method,
        total=total,
        zeros=zeros,
        sparsity=zeros / total if total else 0.0,
        layers=layers,
    )


def run_pruning(config: PruningConfig) -> PruningReport:
    import torch

    model, tokenizer, resolved_family = load_tokenizer_and_model(
        config.model_name_or_path,
        model_family=config.model_family,
        torch_dtype=config.torch_dtype,
        trust_remote_code=config.trust_remote_code,
    )
    device = torch.device(config.device or _default_device())
    model.to(device)

    method = config.method.lower()
    if method == "magnitude":
        magnitude_prune_model(
            model,
            sparsity=config.sparsity,
            layer_regex=config.layer_regex,
            include_lm_head=config.include_lm_head,
            global_pruning=config.global_pruning,
        )
    elif method == "nvidia":
        nvidia_nm_prune_model(
            model,
            keep_n=config.nvidia_keep_n,
            group_m=config.nvidia_group_m,
            layer_regex=config.layer_regex,
            include_lm_head=config.include_lm_head,
        )
    elif method in {"gradient", "wanda"}:
        records = _load_calibration_records(config)
        if method == "gradient":
            gradient_prune_model(
                model,
                tokenizer,
                records,
                resolved_family=resolved_family,
                sparsity=config.sparsity,
                device=device,
                layer_regex=config.layer_regex,
                include_lm_head=config.include_lm_head,
                global_pruning=config.global_pruning,
                max_seq_length=config.max_seq_length,
                source_max_length=config.source_max_length,
                target_max_length=config.target_max_length,
                causal_prompt_template=config.causal_prompt_template,
                causal_response_template=config.causal_response_template,
            )
        else:
            wanda_prune_model(
                model,
                tokenizer,
                records,
                resolved_family=resolved_family,
                sparsity=config.sparsity,
                device=device,
                layer_regex=config.layer_regex,
                include_lm_head=config.include_lm_head,
                max_seq_length=config.max_seq_length,
                source_max_length=config.source_max_length,
                target_max_length=config.target_max_length,
                causal_prompt_template=config.causal_prompt_template,
                causal_response_template=config.causal_response_template,
            )
    else:
        raise ValueError(f"Unsupported pruning method: {config.method}")

    report = summarize_linear_sparsity(
        model,
        layer_regex=config.layer_regex,
        include_lm_head=config.include_lm_head,
        method=method,
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with (output_dir / "pruning_report.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(report), handle, indent=2)
    return report


def _load_calibration_records(config: PruningConfig) -> list[PromptResponseRecord]:
    if not config.calibration_source:
        raise ValueError(f"{config.method} pruning requires --calibration_source.")
    return load_prompt_response_records(
        config.calibration_source,
        split=config.calibration_split,
        prompt_field=config.prompt_field,
        response_field=config.response_field,
        max_records=config.calibration_limit,
    )


def _record_to_batch(
    tokenizer,
    record: PromptResponseRecord,
    *,
    resolved_family: str,
    device,
    max_seq_length: int,
    source_max_length: int,
    target_max_length: int,
    causal_prompt_template: str,
    causal_response_template: str,
) -> dict[str, object]:
    import torch

    if resolved_family == "seq2seq":
        features = tokenize_seq2seq_prompt_response(
            tokenizer,
            prompt=record.prompt,
            response=record.response,
            source_max_length=source_max_length,
            target_max_length=target_max_length,
        )
    else:
        features = tokenize_causal_prompt_response(
            tokenizer,
            prompt=record.prompt,
            response=record.response,
            max_seq_length=max_seq_length,
            target_max_length=target_max_length,
            prompt_template=causal_prompt_template,
            response_template=causal_response_template,
        )
    return {
        key: torch.tensor([value], dtype=torch.long, device=device)
        for key, value in features.items()
    }


def _lowest_score_mask(score, sparsity: float):
    import torch

    flat_score = score.reshape(-1)
    prune_count = int(flat_score.numel() * sparsity)
    mask = torch.ones(flat_score.numel(), dtype=torch.bool, device=flat_score.device)
    if prune_count <= 0:
        return mask.view_as(score)
    if prune_count >= flat_score.numel():
        return torch.zeros_like(score, dtype=torch.bool)
    prune_idx = torch.topk(flat_score, prune_count, largest=False, sorted=False).indices
    mask[prune_idx] = False
    return mask.view_as(score)


def _rowwise_lowest_score_mask(score, sparsity: float):
    import torch

    if score.dim() != 2:
        return _lowest_score_mask(score, sparsity)
    prune_count = int(score.shape[1] * sparsity)
    mask = torch.ones_like(score, dtype=torch.bool)
    if prune_count <= 0:
        return mask
    if prune_count >= score.shape[1]:
        return torch.zeros_like(score, dtype=torch.bool)
    prune_idx = torch.topk(score, prune_count, dim=1, largest=False, sorted=False).indices
    mask.scatter_(1, prune_idx, False)
    return mask


def _global_lowest_score_masks(scores: dict[str, object], sparsity: float) -> dict[str, object]:
    import torch

    flat_scores = [score.float().cpu().reshape(-1) for score in scores.values()]
    all_scores = torch.cat(flat_scores)
    prune_count = int(all_scores.numel() * sparsity)
    global_mask = torch.ones(all_scores.numel(), dtype=torch.bool)
    if prune_count >= all_scores.numel():
        global_mask[:] = False
    elif prune_count > 0:
        prune_idx = torch.topk(all_scores, prune_count, largest=False, sorted=False).indices
        global_mask[prune_idx] = False

    masks = {}
    offset = 0
    for name, score in scores.items():
        size = score.numel()
        masks[name] = global_mask[offset : offset + size].view_as(score)
        offset += size
    return masks


def _validate_sparsity(sparsity: float) -> None:
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must be between 0.0 and 1.0")


def _no_grad():
    import torch

    return torch.no_grad()


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _progress(iterable, *, desc: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc)


def parse_args(argv: Sequence[str] | None = None) -> PruningConfig:
    parser = argparse.ArgumentParser(description="Apply vanilla pruning methods to HF models.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--method",
        choices=["gradient", "magnitude", "nvidia", "wanda"],
        required=True,
    )
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--model_family", choices=["auto", "causal", "seq2seq"], default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--device")
    parser.add_argument("--layer_regex")
    parser.add_argument("--include_lm_head", action="store_true")
    parser.add_argument("--global_pruning", action="store_true")
    parser.add_argument("--calibration_source")
    parser.add_argument("--calibration_split")
    parser.add_argument("--calibration_limit", type=int, default=128)
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--response_field", default="response")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--source_max_length", type=int, default=768)
    parser.add_argument("--target_max_length", type=int, default=256)
    parser.add_argument("--causal_prompt_template", default=DEFAULT_CAUSAL_PROMPT_TEMPLATE)
    parser.add_argument("--causal_response_template", default=DEFAULT_CAUSAL_RESPONSE_TEMPLATE)
    parser.add_argument("--nvidia_keep_n", type=int, default=2)
    parser.add_argument("--nvidia_group_m", type=int, default=4)
    return PruningConfig(**vars(parser.parse_args(argv)))


def main(argv: Sequence[str] | None = None) -> int:
    report = run_pruning(parse_args(argv))
    print(
        json.dumps(
            {
                "method": report.method,
                "total": report.total,
                "zeros": report.zeros,
                "sparsity": report.sparsity,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
