from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from encoder_decoder.data import PromptResponseRecord
from encoder_decoder.pruning import (
    magnitude_prune_model,
    nvidia_nm_prune_model,
    summarize_linear_sparsity,
    wanda_prune_model,
)


def test_magnitude_pruning_zeros_lowest_weights() -> None:
    model = nn.Sequential(nn.Linear(4, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    magnitude_prune_model(model, sparsity=0.5)

    assert model[0].weight.tolist() == [[0.0, 0.0, 3.0, 4.0]]
    report = summarize_linear_sparsity(model, method="magnitude")
    assert report.zeros == 2
    assert report.total == 4
    assert report.sparsity == 0.5


def test_nvidia_two_of_four_kept_per_group() -> None:
    model = nn.Sequential(nn.Linear(8, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                ]
            )
        )

    nvidia_nm_prune_model(model, keep_n=2, group_m=4)
    weight = model[0].weight

    assert torch.count_nonzero(weight[0, :4]).item() == 2
    assert torch.count_nonzero(weight[0, 4:]).item() == 2
    assert torch.count_nonzero(weight[1, :4]).item() == 2
    assert torch.count_nonzero(weight[1, 4:]).item() == 2
    assert weight[0].tolist() == [0.0, 0.0, 3.0, 4.0, 4.0, 3.0, 0.0, 0.0]


def test_nvidia_skips_layers_not_divisible_by_group_size() -> None:
    model = nn.Sequential(nn.Linear(6, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]))

    masks = nvidia_nm_prune_model(model, keep_n=2, group_m=4)

    assert masks == {}
    assert model[0].weight.tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]


class TinyTokenizer:
    eos_token = ""

    def __call__(self, text, *, add_special_tokens=False, truncation=True, max_length=16):
        token_ids = [min(31, max(1, ord(char) % 32)) for char in text][:max_length]
        return {"input_ids": token_ids}


class TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"use_cache": True})()
        self.embed = nn.Embedding(32, 4)
        self.linear = nn.Linear(4, 2, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, **_kwargs):
        hidden = self.embed(input_ids)
        logits = self.linear(hidden)
        return type("Output", (), {"logits": logits, "loss": None})()


def test_wanda_pruning_reaches_rowwise_half_sparsity() -> None:
    model = TinyCausalModel()
    with torch.no_grad():
        model.linear.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [4.0, 3.0, 2.0, 1.0],
                ]
            )
        )

    wanda_prune_model(
        model,
        TinyTokenizer(),
        [PromptResponseRecord(prompt="turn on lamp", response="ok")],
        resolved_family="causal",
        sparsity=0.5,
        device=torch.device("cpu"),
        max_seq_length=16,
        target_max_length=4,
    )

    zeros_per_row = (model.linear.weight == 0).sum(dim=1).tolist()
    assert zeros_per_row == [2, 2]
    report = summarize_linear_sparsity(model, method="wanda")
    assert report.zeros == 4
    assert report.total == 8
    assert report.sparsity == 0.5
