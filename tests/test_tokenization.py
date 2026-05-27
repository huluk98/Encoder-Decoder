from __future__ import annotations

from encoder_decoder.tokenization import tokenize_seq2seq_prompt_response


class TokenizerWithTokenTypeIds:
    def __call__(self, text=None, **kwargs):
        if "text_target" in kwargs:
            return {
                "input_ids": [4, 5],
                "attention_mask": [1, 1],
                "token_type_ids": [0, 0],
            }
        return {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "token_type_ids": [0, 0, 0],
        }


def test_seq2seq_tokenization_drops_token_type_ids() -> None:
    features = tokenize_seq2seq_prompt_response(
        TokenizerWithTokenTypeIds(),
        prompt="prompt",
        response="response",
        source_max_length=8,
        target_max_length=8,
    )

    assert "token_type_ids" not in features
    assert features["input_ids"] == [1, 2, 3]
    assert features["labels"] == [4, 5]
