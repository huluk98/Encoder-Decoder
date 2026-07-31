import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoTokenizer

MODEL_PATH = "charent/ChatLM-mini-Chinese"   # e.g. "./chatlm-mini"
SAVE_PATH = "pruned_models/nvidiabase"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def nvidia_prune(model):
    print("🔧 NVIDIA 2:4 pruning")

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):

            W = m.weight.data

            if W.shape[1] % 4 != 0:
                continue

            W_group = W.view(W.shape[0], -1, 4)

            _, idx = torch.topk(torch.abs(W_group), 2, dim=2, largest=False)

            mask = torch.ones_like(W_group)
            mask.scatter_(2, idx, 0)

            m.weight.data = (W_group * mask).view_as(W)

    return model


def main():
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = nvidia_prune(model)

    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print("✅ NVIDIA done")


if __name__ == "__main__":
    main()