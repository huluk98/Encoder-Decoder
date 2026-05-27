from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from encoder_decoder.data import load_prompt_response_records
from encoder_decoder.modeling import ModelFamily, load_tokenizer_and_model
from encoder_decoder.tokenization import DEFAULT_CAUSAL_PROMPT_TEMPLATE, build_causal_prompt


@dataclass
class ExactEvalConfig:
    model_name_or_path: str
    eval_source: str
    output_path: str | None = None
    metrics_path: str | None = None
    split: str | None = None
    prompt_field: str = "prompt"
    response_field: str = "response"
    model_family: ModelFamily = "auto"
    trust_remote_code: bool = False
    device: str | None = None
    torch_dtype: str | None = "auto"
    max_input_length: int = 768
    max_new_tokens: int = 256
    top_k: int = 1
    num_beams: int | None = None
    causal_prompt_template: str = DEFAULT_CAUSAL_PROMPT_TEMPLATE
    strip: bool = True
    lowercase: bool = False
    collapse_whitespace: bool = False
    limit: int | None = None


@dataclass
class Prediction:
    prompt: str
    response: str
    prediction: str
    predictions: list[str]
    exact: bool
    top_k_exact: bool


@dataclass
class ExactEvalResult:
    total: int
    correct: int
    accuracy: float
    top_k: int
    top_k_correct: int
    top_k_accuracy: float
    predictions: list[Prediction]


def normalize_for_exact(
    text: str,
    *,
    strip: bool = True,
    lowercase: bool = False,
    collapse_whitespace: bool = False,
) -> str:
    if strip:
        text = text.strip()
    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text)
    if lowercase:
        text = text.lower()
    return text


def exact_match(prediction: str, response: str, *, config: ExactEvalConfig) -> bool:
    return normalize_for_exact(
        prediction,
        strip=config.strip,
        lowercase=config.lowercase,
        collapse_whitespace=config.collapse_whitespace,
    ) == normalize_for_exact(
        response,
        strip=config.strip,
        lowercase=config.lowercase,
        collapse_whitespace=config.collapse_whitespace,
    )


def evaluate_exact(config: ExactEvalConfig) -> ExactEvalResult:
    import torch

    model, tokenizer, resolved_family = load_tokenizer_and_model(
        config.model_name_or_path,
        model_family=config.model_family,
        torch_dtype=config.torch_dtype,
        trust_remote_code=config.trust_remote_code,
    )
    device = torch.device(config.device or _default_device())
    model.to(device)
    model.eval()

    records = load_prompt_response_records(
        config.eval_source,
        split=config.split,
        prompt_field=config.prompt_field,
        response_field=config.response_field,
        max_records=config.limit,
    )

    predictions: list[Prediction] = []
    for record in _progress(records, desc="Exact eval"):
        candidates = _generate_candidates(
            model,
            tokenizer,
            resolved_family=resolved_family,
            prompt=record.prompt,
            device=device,
            max_input_length=config.max_input_length,
            max_new_tokens=config.max_new_tokens,
            top_k=config.top_k,
            num_beams=config.num_beams,
            causal_prompt_template=config.causal_prompt_template,
        )
        prediction = candidates[0] if candidates else ""
        is_exact = exact_match(prediction, record.response, config=config)
        is_top_k_exact = any(
            exact_match(candidate, record.response, config=config)
            for candidate in candidates
        )
        predictions.append(
            Prediction(
                prompt=record.prompt,
                response=record.response,
                prediction=prediction,
                predictions=candidates,
                exact=is_exact,
                top_k_exact=is_top_k_exact,
            )
        )

    correct = sum(item.exact for item in predictions)
    top_k_correct = sum(item.top_k_exact for item in predictions)
    total = len(predictions)
    result = ExactEvalResult(
        total=total,
        correct=correct,
        accuracy=correct / total if total else 0.0,
        top_k=max(1, config.top_k),
        top_k_correct=top_k_correct,
        top_k_accuracy=top_k_correct / total if total else 0.0,
        predictions=predictions,
    )
    if config.output_path:
        _write_predictions(result, Path(config.output_path))
    if config.metrics_path:
        _write_metrics(result, Path(config.metrics_path))
    return result


def _generate_candidates(
    model,
    tokenizer,
    *,
    resolved_family: str,
    prompt: str,
    device,
    max_input_length: int,
    max_new_tokens: int,
    top_k: int,
    num_beams: int | None,
    causal_prompt_template: str,
) -> list[str]:
    import torch

    top_k = max(1, top_k)
    beams = max(top_k, num_beams or top_k)
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": beams,
        "num_return_sequences": top_k,
    }

    if resolved_family == "seq2seq":
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generate_kwargs)
        return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

    tokenizer.padding_side = "left"
    prompt_text = build_causal_prompt(prompt, causal_prompt_template)
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_length = inputs["input_ids"].shape[-1]
    if tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)
    return [
        tokenizer.decode(ids[input_length:], skip_special_tokens=True)
        for ids in output_ids
    ]


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _write_predictions(result: ExactEvalResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in result.predictions:
            handle.write(json.dumps(asdict(prediction), ensure_ascii=False) + "\n")


def metrics_payload(result: ExactEvalResult) -> dict[str, float | int]:
    return {
        "total": result.total,
        "top_1_correct": result.correct,
        "top_1_accuracy": result.accuracy,
        "correct": result.correct,
        "accuracy": result.accuracy,
        f"top_{result.top_k}_correct": result.top_k_correct,
        f"top_{result.top_k}_accuracy": result.top_k_accuracy,
    }


def _write_metrics(result: ExactEvalResult, metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics_payload(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _progress(iterable, *, desc: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc)


def parse_args(argv: Sequence[str] | None = None) -> ExactEvalConfig:
    parser = argparse.ArgumentParser(description="Run exact-match generation evaluation.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--eval_source", required=True)
    parser.add_argument("--output_path")
    parser.add_argument("--metrics_path")
    parser.add_argument("--split")
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--response_field", default="response")
    parser.add_argument("--model_family", choices=["auto", "causal", "seq2seq"], default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--max_input_length", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--num_beams", type=int)
    parser.add_argument("--causal_prompt_template", default=DEFAULT_CAUSAL_PROMPT_TEMPLATE)
    parser.add_argument("--no_strip", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--collapse_whitespace", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    values = vars(args)
    values["strip"] = not values.pop("no_strip")
    return ExactEvalConfig(**values)


def main(argv: Sequence[str] | None = None) -> int:
    result = evaluate_exact(parse_args(argv))
    print(json.dumps(metrics_payload(result), indent=2, sort_keys=True))
    return 0
