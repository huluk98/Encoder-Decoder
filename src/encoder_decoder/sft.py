from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from encoder_decoder.data import load_prompt_response_records
from encoder_decoder.modeling import ModelFamily, load_tokenizer_and_model
from encoder_decoder.tokenization import (
    DEFAULT_CAUSAL_PROMPT_TEMPLATE,
    DEFAULT_CAUSAL_RESPONSE_TEMPLATE,
    tokenize_causal_prompt_response,
    tokenize_seq2seq_prompt_response,
)


@dataclass
class SFTConfig:
    model_name_or_path: str
    train_source: str
    output_dir: str
    eval_source: str | None = None
    sft_eval_source: str | None = None
    benchmark_eval_source: str | None = None
    anchor_eval_source: str | None = None
    train_split: str | None = None
    eval_split: str | None = None
    sft_eval_split: str | None = None
    benchmark_eval_split: str | None = None
    anchor_eval_split: str | None = None
    prompt_field: str = "prompt"
    response_field: str = "response"
    anchor_field: str = "anchor"
    anchor_response_field: str = "response"
    model_family: ModelFamily = "auto"
    trust_remote_code: bool = False
    max_seq_length: int = 1024
    source_max_length: int = 768
    target_max_length: int = 256
    causal_prompt_template: str = DEFAULT_CAUSAL_PROMPT_TEMPLATE
    causal_response_template: str = DEFAULT_CAUSAL_RESPONSE_TEMPLATE
    eval_train_source_generation: bool = False
    generation_eval_top_k: int = 5
    generation_eval_num_beams: int | None = None
    generation_eval_max_new_tokens: int = 256
    generation_eval_limit: int | None = None
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    num_train_epochs: float = 3.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 2
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    report_to: list[str] = field(default_factory=list)
    resume_from_checkpoint: str | None = None


def train_sft(config: SFTConfig) -> dict[str, float]:
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    model, tokenizer, resolved_family = load_tokenizer_and_model(
        config.model_name_or_path,
        model_family=config.model_family,
        trust_remote_code=config.trust_remote_code,
    )
    tokenizer.padding_side = "right"

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    train_records = load_prompt_response_records(
        config.train_source,
        split=config.train_split,
        prompt_field=config.prompt_field,
        response_field=config.response_field,
    )
    eval_records = (
        load_prompt_response_records(
            config.eval_source,
            split=config.eval_split,
            prompt_field=config.prompt_field,
            response_field=config.response_field,
        )
        if config.eval_source
        else None
    )

    train_dataset = Dataset.from_list([record.__dict__ for record in train_records])
    eval_dataset = (
        Dataset.from_list([record.__dict__ for record in eval_records])
        if eval_records
        else None
    )

    def preprocess(example: dict[str, str]) -> dict[str, list[int]]:
        if resolved_family == "seq2seq":
            return tokenize_seq2seq_prompt_response(
                tokenizer,
                prompt=example["prompt"],
                response=example["response"],
                source_max_length=config.source_max_length,
                target_max_length=config.target_max_length,
            )
        return tokenize_causal_prompt_response(
            tokenizer,
            prompt=example["prompt"],
            response=example["response"],
            max_seq_length=config.max_seq_length,
            target_max_length=config.target_max_length,
            prompt_template=config.causal_prompt_template,
            response_template=config.causal_response_template,
        )

    tokenized_train = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
    tokenized_eval = (
        eval_dataset.map(preprocess, remove_columns=eval_dataset.column_names)
        if eval_dataset is not None
        else None
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if config.fp16 or config.bf16 else None,
    )
    training_args = _build_training_args(config, eval_enabled=tokenized_eval is not None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(config.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config.output_dir)
    trainer.save_state()

    metrics = dict(train_result.metrics)
    if trainer.is_world_process_zero():
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
    if tokenized_eval is not None:
        eval_metrics = trainer.evaluate()
        if trainer.is_world_process_zero():
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
        metrics.update({f"eval_{key}": value for key, value in eval_metrics.items()})
    if _has_generation_evals(config):
        if hasattr(trainer, "accelerator"):
            trainer.accelerator.wait_for_everyone()
        if trainer.is_world_process_zero():
            metrics.update(_run_generation_evals(config))
    return metrics


def _has_generation_evals(config: SFTConfig) -> bool:
    return any(
        (
            config.eval_train_source_generation,
            config.sft_eval_source,
            config.benchmark_eval_source,
            config.anchor_eval_source,
        )
    )


def _run_generation_evals(config: SFTConfig) -> dict[str, float]:
    from encoder_decoder.evaluate_exact import ExactEvalConfig, evaluate_exact

    jobs = []
    if config.eval_train_source_generation:
        jobs.append(
            (
                "sft_train",
                config.train_source,
                config.train_split,
                config.prompt_field,
                config.response_field,
            )
        )
    if config.sft_eval_source:
        jobs.append(
            (
                "sft",
                config.sft_eval_source,
                config.sft_eval_split,
                config.prompt_field,
                config.response_field,
            )
        )
    if config.benchmark_eval_source:
        jobs.append(
            (
                "benchmark",
                config.benchmark_eval_source,
                config.benchmark_eval_split,
                config.prompt_field,
                config.response_field,
            )
        )
    if config.anchor_eval_source:
        jobs.append(
            (
                "anchor",
                config.anchor_eval_source,
                config.anchor_eval_split,
                config.anchor_field,
                config.anchor_response_field,
            )
        )

    if not jobs:
        return {}

    metrics: dict[str, float] = {}
    prediction_dir = Path(config.output_dir) / "generation_eval"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for name, source, split, prompt_field, response_field in jobs:
        result = evaluate_exact(
            ExactEvalConfig(
                model_name_or_path=config.output_dir,
                eval_source=source,
                output_path=str(prediction_dir / f"{name}_predictions.jsonl"),
                split=split,
                prompt_field=prompt_field,
                response_field=response_field,
                model_family=config.model_family,
                trust_remote_code=config.trust_remote_code,
                max_input_length=config.source_max_length,
                max_new_tokens=config.generation_eval_max_new_tokens,
                top_k=config.generation_eval_top_k,
                num_beams=config.generation_eval_num_beams,
                causal_prompt_template=config.causal_prompt_template,
                limit=config.generation_eval_limit,
            )
        )
        metrics[f"{name}_exact_accuracy"] = result.accuracy
        metrics[f"{name}_exact_correct"] = result.correct
        metrics[f"{name}_top_1_accuracy"] = result.accuracy
        metrics[f"{name}_top_1_correct"] = result.correct
        metrics[f"{name}_total"] = result.total
        metrics[f"{name}_top_{result.top_k}_accuracy"] = result.top_k_accuracy
        metrics[f"{name}_top_{result.top_k}_correct"] = result.top_k_correct
    with (prediction_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return metrics


def _build_training_args(config: SFTConfig, *, eval_enabled: bool):
    from transformers import TrainingArguments

    eval_strategy = "steps" if eval_enabled else "no"
    kwargs = {
        "output_dir": config.output_dir,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "save_strategy": "steps",
        "save_total_limit": config.save_total_limit,
        "fp16": config.fp16,
        "bf16": config.bf16,
        "report_to": config.report_to,
        "remove_unused_columns": False,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = eval_strategy
    else:
        kwargs["evaluation_strategy"] = eval_strategy
    return TrainingArguments(**kwargs)


def parse_args(argv: Sequence[str] | None = None) -> SFTConfig:
    parser = argparse.ArgumentParser(description="Supervised fine-tune a causal or seq2seq model.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_source", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_source")
    parser.add_argument("--sft_eval_source")
    parser.add_argument("--benchmark_eval_source")
    parser.add_argument("--anchor_eval_source")
    parser.add_argument("--train_split")
    parser.add_argument("--eval_split")
    parser.add_argument("--sft_eval_split")
    parser.add_argument("--benchmark_eval_split")
    parser.add_argument("--anchor_eval_split")
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--response_field", default="response")
    parser.add_argument("--anchor_field", default="anchor")
    parser.add_argument("--anchor_response_field", default="response")
    parser.add_argument("--model_family", choices=["auto", "causal", "seq2seq"], default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--source_max_length", type=int, default=768)
    parser.add_argument("--target_max_length", type=int, default=256)
    parser.add_argument("--causal_prompt_template", default=DEFAULT_CAUSAL_PROMPT_TEMPLATE)
    parser.add_argument("--causal_response_template", default=DEFAULT_CAUSAL_RESPONSE_TEMPLATE)
    parser.add_argument("--eval_train_source_generation", action="store_true")
    parser.add_argument("--generation_eval_top_k", type=int, default=5)
    parser.add_argument("--generation_eval_num_beams", type=int)
    parser.add_argument("--generation_eval_max_new_tokens", type=int, default=256)
    parser.add_argument("--generation_eval_limit", type=int)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--resume_from_checkpoint")
    args = parser.parse_args(argv)
    values = vars(args)
    values["report_to"] = _parse_report_to(values["report_to"])
    return SFTConfig(**values)


def _parse_report_to(value: str) -> list[str]:
    if value.lower() in {"", "none", "false", "off"}:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    metrics = train_sft(parse_args(argv))
    if _is_main_process():
        print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0
