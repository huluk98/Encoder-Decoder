from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_chatlm_eos.py"
SPEC = importlib.util.spec_from_file_location("eval_chatlm_eos", SCRIPT_PATH)
assert SPEC is not None
eval_chatlm_eos = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(eval_chatlm_eos)


def test_resolve_model_path_prefers_model_save(tmp_path) -> None:
    root = tmp_path / "ChatLM-mini-Chinese"
    model_save = root / "model_save"
    model_save.mkdir(parents=True)

    assert eval_chatlm_eos.resolve_model_path(root) == model_save.resolve()


def test_resolve_model_path_uses_latest_checkpoint(tmp_path) -> None:
    output_dir = tmp_path / "run"
    checkpoint_1 = output_dir / "checkpoint-1"
    checkpoint_20 = output_dir / "checkpoint-20"
    checkpoint_1.mkdir(parents=True)
    checkpoint_20.mkdir(parents=True)
    (checkpoint_1 / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint_20 / "config.json").write_text("{}", encoding="utf-8")

    assert eval_chatlm_eos.resolve_model_path(output_dir) == checkpoint_20.resolve()


def test_missing_custom_code_files_detects_auto_map_module(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"auto_map": {"AutoModelForSeq2SeqLM": "modeling_chat_model.ChatModel"}}),
        encoding="utf-8",
    )

    assert eval_chatlm_eos.missing_custom_code_files(model_dir) == ["modeling_chat_model.py"]


def test_checkpoint_weight_files_reads_safetensors_index(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "encoder.weight": "model-00001-of-00002.safetensors",
                    "decoder.weight": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    assert eval_chatlm_eos.checkpoint_weight_files(model_dir) == [
        model_dir / "model-00001-of-00002.safetensors",
        model_dir / "model-00002-of-00002.safetensors",
    ]


def test_read_json_or_jsonl_supports_records_wrapper(tmp_path) -> None:
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps({"records": [{"prompt": "p", "response": "r"}]}),
        encoding="utf-8",
    )

    assert eval_chatlm_eos.read_json_or_jsonl(path) == [{"prompt": "p", "response": "r"}]


def test_normalize_text_strips_trailing_textual_eos() -> None:
    assert eval_chatlm_eos.normalize_text(" answer [EOS] [EOS] ") == "answer"
