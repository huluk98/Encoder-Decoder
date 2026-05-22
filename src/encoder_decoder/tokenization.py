from __future__ import annotations

DEFAULT_CAUSAL_PROMPT_TEMPLATE = "{prompt}\n"
DEFAULT_CAUSAL_RESPONSE_TEMPLATE = "{response}"


def build_causal_prompt(prompt: str, prompt_template: str = DEFAULT_CAUSAL_PROMPT_TEMPLATE) -> str:
    return prompt_template.format(prompt=prompt)


def build_causal_response(
    response: str,
    response_template: str = DEFAULT_CAUSAL_RESPONSE_TEMPLATE,
    eos_token: str | None = None,
) -> str:
    text = response_template.format(response=response)
    return text + (eos_token or "")


def tokenize_causal_prompt_response(
    tokenizer,
    *,
    prompt: str,
    response: str,
    max_seq_length: int,
    target_max_length: int,
    prompt_template: str = DEFAULT_CAUSAL_PROMPT_TEMPLATE,
    response_template: str = DEFAULT_CAUSAL_RESPONSE_TEMPLATE,
) -> dict[str, list[int]]:
    source_budget = max(1, max_seq_length - target_max_length)
    source_text = build_causal_prompt(prompt, prompt_template)
    target_text = build_causal_response(response, response_template, tokenizer.eos_token)

    source = tokenizer(
        source_text,
        add_special_tokens=False,
        truncation=True,
        max_length=source_budget,
    )
    target = tokenizer(
        target_text,
        add_special_tokens=False,
        truncation=True,
        max_length=target_max_length,
    )

    input_ids = (source["input_ids"] + target["input_ids"])[:max_seq_length]
    labels = ([-100] * len(source["input_ids"]) + target["input_ids"])[:max_seq_length]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def tokenize_seq2seq_prompt_response(
    tokenizer,
    *,
    prompt: str,
    response: str,
    source_max_length: int,
    target_max_length: int,
) -> dict[str, list[int]]:
    model_inputs = tokenizer(
        prompt,
        truncation=True,
        max_length=source_max_length,
    )

    try:
        labels = tokenizer(
            text_target=response,
            truncation=True,
            max_length=target_max_length,
        )
    except TypeError:
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                response,
                truncation=True,
                max_length=target_max_length,
            )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

