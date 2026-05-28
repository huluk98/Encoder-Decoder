from __future__ import annotations

import pytest

from encoder_decoder.modeling import (
    resolve_model_name_or_path,
    resolve_tokenizer_name_or_path,
)


def test_resolve_model_name_or_path_expands_existing_local_dir(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    assert resolve_model_name_or_path(str(model_dir)) == str(model_dir.resolve())


def test_resolve_model_name_or_path_uses_latest_checkpoint(tmp_path) -> None:
    output_dir = tmp_path / "run"
    checkpoint_2 = output_dir / "checkpoint-2"
    checkpoint_10 = output_dir / "checkpoint-10"
    checkpoint_2.mkdir(parents=True)
    checkpoint_10.mkdir(parents=True)
    (checkpoint_2 / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint_10 / "config.json").write_text("{}", encoding="utf-8")

    assert resolve_model_name_or_path(str(output_dir)) == str(checkpoint_10.resolve())


def test_resolve_model_name_or_path_missing_local_path_fails_early(tmp_path) -> None:
    missing_path = tmp_path / "missing-model"

    with pytest.raises(FileNotFoundError, match="Local model path does not exist"):
        resolve_model_name_or_path(str(missing_path))


def test_resolve_model_name_or_path_keeps_hf_repo_id() -> None:
    assert resolve_model_name_or_path("charent/ChatLM-mini-Chinese") == "charent/ChatLM-mini-Chinese"


def test_resolve_tokenizer_name_or_path_falls_back_to_parent(tmp_path) -> None:
    output_dir = tmp_path / "run"
    checkpoint = output_dir / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (output_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    tokenizer_path = resolve_tokenizer_name_or_path(str(output_dir), str(checkpoint))

    assert tokenizer_path == str(output_dir.resolve())
