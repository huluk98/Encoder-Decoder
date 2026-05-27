import torch
import torch.nn as nn
import json
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

MODEL_PATH = "/nvme1/home/luke/PycharmProjects/iot_t5/sft"   # e.g. "./chatlm-mini"
DATA_PATH = "/nvme1/home/luke/PycharmProjects/t5prune/data/dataset.json"
SAVE_PATH = "/nvme1/home/luke/PycharmProjects/t5prune/pruned_models/wanda"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPARSITY = 0.5
CALIB_BATCHES = 64


# ===== DATA =====
class T5Dataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        enc = self.tokenizer(
            item["prompt"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )

        return {
            "input_ids": enc.input_ids.squeeze(),
            "attention_mask": enc.attention_mask.squeeze()
        }


# ===== ACTIVATIONS =====
def get_activations(model, loader):

    activations = {}
    handles = []

    def hook(module, inp, out):
        x = inp[0].detach()
        x = x.view(-1, x.shape[-1])
        activations.setdefault(module, []).append(x)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(hook))

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= CALIB_BATCHES:
                break

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            # IMPORTANT: use labels (better than shift_right)
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

    for h in handles:
        h.remove()

    for m in activations:
        activations[m] = torch.cat(activations[m], dim=0)

    return activations


# ===== OFFICIAL WANDA =====
def wanda_prune(model, loader):

    print("🔧 OFFICIAL WANDA pruning")

    acts = get_activations(model, loader)

    for m in acts:

        W = m.weight.data  # [out, in]
        X = acts[m]        # [N, in]

        # activation norm per input dimension
        norm = torch.norm(X, dim=0)  # [in]

        # importance
        score = W.abs() * norm.unsqueeze(0)

        # 🔥 PER-ROW PRUNING (THIS IS THE KEY)
        k = int(W.shape[1] * (1 - SPARSITY))

        # top-k per row
        topk = torch.topk(score, k, dim=1).indices

        mask = torch.zeros_like(W)

        for i in range(W.shape[0]):
            mask[i, topk[i]] = 1

        m.weight.data *= mask

    return model


# ===== MAIN =====
def main():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)

    data = json.load(open(DATA_PATH))
    loader = DataLoader(T5Dataset(data, tokenizer), batch_size=4)

    model = wanda_prune(model, loader)

    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print("✅ WANDA done")


if __name__ == "__main__":
    main()