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


def test_parse_cuda_visible_devices() -> None:
    assert eval_chatlm_eos.parse_cuda_visible_devices("4,5, 6,7") == ["4", "5", "6", "7"]


def test_evaluate_file_shards_examples_by_rank(tmp_path, monkeypatch) -> None:
    path = tmp_path / "eval.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(5):
            handle.write(json.dumps({"prompt": f"p{index}", "response": f"p{index}"}) + "\n")

    def fake_generate_topk(_model, _tokenizer, prompts, **_kwargs):
        return [[prompt] for prompt in prompts]

    monkeypatch.setattr(eval_chatlm_eos, "generate_topk", fake_generate_topk)
    predictions_path = tmp_path / "rank1.predictions.jsonl"

    result = eval_chatlm_eos.evaluate_file(
        model=None,
        tokenizer=None,
        file_path=path,
        device="cpu",
        top_k=1,
        add_eos_to_prompt=False,
        predictions_path=predictions_path,
        rank=1,
        world_size=2,
        show_progress=False,
    )

    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    assert result["source_n"] == 5
    assert result["n"] == 2
    assert [row["idx"] for row in rows] == [1, 3]


def test_evaluate_file_halves_batch_size_after_cuda_oom(tmp_path, monkeypatch) -> None:
    path = tmp_path / "eval.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(4):
            handle.write(json.dumps({"prompt": f"p{index}", "response": f"p{index}"}) + "\n")

    calls = {"count": 0}

    def fake_generate_topk(_model, _tokenizer, prompts, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("CUDA out of memory")
        return [[prompt] for prompt in prompts]

    monkeypatch.setattr(eval_chatlm_eos, "generate_topk", fake_generate_topk)
    monkeypatch.setattr(eval_chatlm_eos, "clear_cuda_cache", lambda: None)

    result = eval_chatlm_eos.evaluate_file(
        model=None,
        tokenizer=None,
        file_path=path,
        device="cpu",
        batch_size=4,
        min_batch_size=1,
        top_k=1,
        add_eos_to_prompt=False,
        rank=0,
        world_size=1,
        show_progress=False,
    )

    assert result["n"] == 4
    assert result["final_batch_size_per_gpu"] == 2
    assert result["correct@1"] == 4


def test_merge_distributed_summaries_sums_metrics_and_predictions(tmp_path) -> None:
    output_dir = tmp_path / "out"
    parts_dir = output_dir / ".distributed_parts"
    parts_dir.mkdir(parents=True)

    for rank, indexes in [(0, [0, 2]), (1, [1, 3])]:
        pred_path = eval_chatlm_eos.prediction_part_path(parts_dir, "eval", rank)
        with pred_path.open("w", encoding="utf-8") as handle:
            for index in indexes:
                handle.write(json.dumps({"idx": index, "prediction": f"p{index}"}) + "\n")
        eval_chatlm_eos.write_json(
            eval_chatlm_eos.rank_summary_path(parts_dir, rank),
            {
                "model_path": "/model",
                "distributed": {"rank": rank, "world_size": 2},
                "eval": {
                    "file": "/eval.json",
                    "source_n": 4,
                    "n": 2,
                    "correct@1": 1,
                    "correct@5": 2,
                    "em@1": 0.5,
                    "em@5": 1.0,
                },
            },
        )

    summary = eval_chatlm_eos.merge_distributed_summaries(
        output_dir=output_dir,
        parts_dir=parts_dir,
        top_k=5,
        world_size=2,
    )

    merged_rows = [
        json.loads(line)
        for line in (output_dir / "eval.predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["eval"]["n"] == 4
    assert summary["eval"]["correct@1"] == 2
    assert summary["eval"]["correct@5"] == 4
    assert [row["idx"] for row in merged_rows] == [0, 1, 2, 3]
