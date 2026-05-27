from __future__ import annotations

import json

from encoder_decoder.data import load_contrastive_records


def test_load_contrastive_records_uses_configured_negative_field(tmp_path) -> None:
    path = tmp_path / "contrastive.json"
    path.write_text(
        json.dumps(
            [
                {
                    "anchor": "turn off the AC at 10",
                    "positive": "please turn off the AC at 10",
                    "negative": "turn off the AC at 11",
                    "invalid_negative": "make the AC brighter",
                    "response": "AC off at 10",
                    "source_id": "row-1",
                }
            ]
        ),
        encoding="utf-8",
    )

    records = load_contrastive_records(
        str(path),
        negative_field="invalid_negative",
    )

    assert len(records) == 1
    assert records[0].anchor == "turn off the AC at 10"
    assert records[0].positive == "please turn off the AC at 10"
    assert records[0].negative == "make the AC brighter"
    assert records[0].response == "AC off at 10"
    assert records[0].meta["source_id"] == "row-1"
