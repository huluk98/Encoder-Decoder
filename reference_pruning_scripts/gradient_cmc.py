import torch
import torch.nn as nn
import json
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

MODEL_PATH = "/nvme1/home/luke/PycharmProjects/iot_t5/sft"   # e.g. "./chatlm-mini"
DATA_PATH = "/nvme1/home/luke/PycharmProjects/t5prune/data/dataset.json"
SAVE_PATH = "/nvme1/home/luke/PycharmProjects/t5prune/pruned_models/gradientnew"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPARSITY = 0.5


class T5Dataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        enc = self.tokenizer(item["prompt"], return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        tgt = self.tokenizer(item["response"], return_tensors="pt", padding="max_length", truncation=True, max_length=128)

        return {
            "input_ids": enc.input_ids.squeeze(),
            "attention_mask": enc.attention_mask.squeeze(),
            "labels": tgt.input_ids.squeeze()
        }


def gradient_prune(model, loader):
    print("🔧 Gradient pruning (Taylor)")

    model.train()

    saliency = {}

    for m in model.modules():
        if isinstance(m, nn.Linear):
            saliency[m] = torch.zeros_like(m.weight.data)

    for batch in loader:

        outputs = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            labels=batch["labels"].to(DEVICE)
        )

        loss = outputs.loss
        loss.backward()

        for m in saliency:
            if m.weight.grad is not None:
                saliency[m] += (m.weight.data * m.weight.grad).abs()

        model.zero_grad()

    for m in saliency:
        score = saliency[m]
        scores = score.view(-1)

        k = int(scores.numel() * (1 - SPARSITY))

        if k <= 0:
            continue

        threshold = torch.topk(scores, k).values.min()

        mask = (score >= threshold)
        m.weight.data *= mask

    return model


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)

    data = json.load(open(DATA_PATH))
    loader = DataLoader(T5Dataset(data, tokenizer), batch_size=4)

    model = gradient_prune(model, loader)

    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print("✅ Gradient done")


if __name__ == "__main__":
    main()