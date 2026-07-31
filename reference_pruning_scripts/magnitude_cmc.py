import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoTokenizer

MODEL_PATH = "models/sft"   # e.g. "./chatlm-mini"
SAVE_PATH = "pruned_models/magnitudenew"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPARSITY = 0.5


def magnitude_prune(model):
    print("🔧 Magnitude pruning (per-layer)")

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):

            W = m.weight.data
            scores = W.abs().view(-1)

            k = int(scores.numel() * (1 - SPARSITY))

            if k <= 0:
                continue

            threshold = torch.topk(scores, k).values.min()

            mask = (W.abs() >= threshold)
            m.weight.data *= mask

    return model


def main():
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = magnitude_prune(model)

    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print("✅ Magnitude done")


if __name__ == "__main__":
    main()