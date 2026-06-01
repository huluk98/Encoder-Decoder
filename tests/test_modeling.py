from __future__ import annotations

import json

import pytest

from encoder_decoder.modeling import (
    copy_custom_code_files,
    custom_code_filenames_from_config,
    missing_custom_code_files,
    resolve_model_name_or_path,
    resolve_tokenizer_name_or_path,
    save_tokenizer_pretrained_safely,
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


def test_custom_code_filenames_from_config_reads_auto_map(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"auto_map": {"AutoModelForSeq2SeqLM": "modeling_chat_model.ChatModel"}}',
        encoding="utf-8",
    )

    assert custom_code_filenames_from_config(model_dir) == ["modeling_chat_model.py"]
    assert missing_custom_code_files(model_dir) == ["modeling_chat_model.py"]


def test_copy_custom_code_files_copies_remote_code_siblings(tmp_path) -> None:
    source_dir = tmp_path / "base"
    output_dir = tmp_path / "checkpoint"
    source_dir.mkdir()
    output_dir.mkdir()
    (source_dir / "modeling_chat_model.py").write_text("# model\n", encoding="utf-8")
    (source_dir / "configuration_chat_model.py").write_text("# config\n", encoding="utf-8")
    (output_dir / "config.json").write_text(
        '{"auto_map": {"AutoConfig": "configuration_chat_model.ChatConfig", '
        '"AutoModelForSeq2SeqLM": "modeling_chat_model.ChatModel"}}',
        encoding="utf-8",
    )

    missing = copy_custom_code_files(output_dir, source_paths=[source_dir])

    assert missing == []
    assert (output_dir / "modeling_chat_model.py").read_text(encoding="utf-8") == "# model\n"
    assert (output_dir / "configuration_chat_model.py").read_text(encoding="utf-8") == "# config\n"


def test_save_tokenizer_pretrained_safely_sanitizes_dtype_metadata(tmp_path) -> None:
    fake_dtype = type("dtype", (), {"__module__": "torch", "__str__": lambda self: "torch.float16"})()

    class FakeTokenizer:
        def __init__(self) -> None:
            self.init_kwargs = {
                "torch_dtype": fake_dtype,
                "nested": {"dtype": fake_dtype},
                "path": tmp_path / "base-tokenizer",
            }
            self.init_inputs = (fake_dtype,)
            self.name_or_path = tmp_path / "base-tokenizer"

        def save_pretrained(self, output_dir):
            json.dumps(
                {
                    "init_kwargs": self.init_kwargs,
                    "init_inputs": self.init_inputs,
                    "name_or_path": self.name_or_path,
                }
            )
            return (str(output_dir),)

    tokenizer = FakeTokenizer()

    assert save_tokenizer_pretrained_safely(tokenizer, tmp_path / "out") == (str(tmp_path / "out"),)
    assert tokenizer.init_kwargs["torch_dtype"] == "float16"
    assert tokenizer.init_kwargs["nested"]["dtype"] == "float16"
    assert tokenizer.init_kwargs["path"] == str(tmp_path / "base-tokenizer")
    assert tokenizer.init_inputs == ["float16"]
    assert tokenizer.name_or_path == str(tmp_path / "base-tokenizer")
