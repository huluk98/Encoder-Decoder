from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PromptResponseRecord:
    prompt: str
    response: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContrastiveRecord:
    anchor: str
    positive: str
    negative: str
    response: str
    meta: dict[str, Any] = field(default_factory=dict)


def load_prompt_response_records(
    source: str,
    *,
    split: str | None = None,
    prompt_field: str = "prompt",
    response_field: str = "response",
    max_records: int | None = None,
) -> list[PromptResponseRecord]:
    """Load prompt-response records from local files, HF datasets, or HF dataset dirs."""
    path = Path(source)
    if path.exists():
        if path.is_dir():
            rows = _load_dataset_from_disk(path, split)
        else:
            rows = _load_local_file(path)
    else:
        rows = _load_huggingface_dataset(source, split)

    return list(
        _coerce_records(
            rows,
            prompt_field=prompt_field,
            response_field=response_field,
            max_records=max_records,
        )
    )


def load_contrastive_records(
    source: str,
    *,
    split: str | None = None,
    anchor_field: str = "anchor",
    positive_field: str = "positive",
    negative_field: str = "negative",
    response_field: str = "response",
    max_records: int | None = None,
) -> list[ContrastiveRecord]:
    """Load anchor-positive-negative-response records for contrastive SFT."""
    path = Path(source)
    if path.exists():
        if path.is_dir():
            rows = _load_dataset_from_disk(path, split)
        else:
            rows = _load_local_file(path)
    else:
        rows = _load_huggingface_dataset(source, split)

    return list(
        _coerce_contrastive_records(
            rows,
            anchor_field=anchor_field,
            positive_field=positive_field,
            negative_field=negative_field,
            response_field=response_field,
            max_records=max_records,
        )
    )


def _load_local_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], list):
                return payload["data"]
            if "records" in payload and isinstance(payload["records"], list):
                return payload["records"]
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Unsupported JSON shape in {path}")
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise ValueError(f"Unsupported local dataset format: {path}")


def _load_dataset_from_disk(path: Path, split: str | None) -> Iterable[dict[str, Any]]:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise RuntimeError("Install datasets to load Hugging Face dataset directories.") from exc

    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        selected_split = split or "train"
        if selected_split not in dataset:
            available = ", ".join(dataset.keys())
            raise ValueError(f"Split {selected_split!r} not found. Available splits: {available}")
        dataset = dataset[selected_split]
    return dataset


def _load_huggingface_dataset(source: str, split: str | None) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to load datasets from the Hugging Face Hub.") from exc

    return load_dataset(source, split=split or "train")


def _coerce_records(
    rows: Iterable[dict[str, Any]],
    *,
    prompt_field: str,
    response_field: str,
    max_records: int | None,
) -> Iterable[PromptResponseRecord]:
    for index, row in enumerate(rows):
        if max_records is not None and index >= max_records:
            break
        if prompt_field not in row or response_field not in row:
            columns = ", ".join(sorted(str(key) for key in row.keys()))
            raise KeyError(
                f"Expected fields {prompt_field!r} and {response_field!r}; got columns: {columns}"
            )
        meta = {key: value for key, value in row.items() if key not in {prompt_field, response_field}}
        yield PromptResponseRecord(
            prompt="" if row[prompt_field] is None else str(row[prompt_field]),
            response="" if row[response_field] is None else str(row[response_field]),
            meta=meta,
        )


def _coerce_contrastive_records(
    rows: Iterable[dict[str, Any]],
    *,
    anchor_field: str,
    positive_field: str,
    negative_field: str,
    response_field: str,
    max_records: int | None,
) -> Iterable[ContrastiveRecord]:
    required_fields = {anchor_field, positive_field, negative_field, response_field}
    for index, row in enumerate(rows):
        if max_records is not None and index >= max_records:
            break
        missing = [field_name for field_name in required_fields if field_name not in row]
        if missing:
            columns = ", ".join(sorted(str(key) for key in row.keys()))
            missing_text = ", ".join(repr(field_name) for field_name in missing)
            raise KeyError(f"Expected fields {missing_text}; got columns: {columns}")
        meta = {key: value for key, value in row.items() if key not in required_fields}
        yield ContrastiveRecord(
            anchor="" if row[anchor_field] is None else str(row[anchor_field]),
            positive="" if row[positive_field] is None else str(row[positive_field]),
            negative="" if row[negative_field] is None else str(row[negative_field]),
            response="" if row[response_field] is None else str(row[response_field]),
            meta=meta,
        )
