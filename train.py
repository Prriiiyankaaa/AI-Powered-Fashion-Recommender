"""
Phase 2 - Steps 2 & 3: retrain the intent model with the richer schema + SOFT labels.

Replaces the (broken) training cells in bert.ipynb. Run it on a GPU box / Google
Colab - CPU training here is hours. It needs only:

    pip install "transformers>=4.40" torch pandas scikit-learn
    files: fashion_data_v2.csv, fashion_data_v2_heldout.csv, label_maps_v2.json
           (all produced by build_dataset.py)

Output (into ./fashion-bert-v2/):
    model.pt           - state_dict, drop-in for FashionIntentModel in the app
    label_maps.json    - the 6-occasion schema
    tokenizer.json, tokenizer_config.json, vocab.txt

Then Step 7: copy those 4-5 files over ./fashion-bert/ and retest.

Why soft labels: the loss is KL-divergence against a probability *distribution*
per axis. A one-hot row ("party") and a blended row ("party 0.7 / casual 0.3")
are handled by the exact same code path - the blended rows are simply targets
that aren't one-hot, which teaches the model to output graded intent instead of
always spiking to a single label.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizerFast
from sklearn.metrics import f1_score

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
TRAIN_CSV = "fashion_data_v2.csv"
LABEL_MAPS_PATH = "label_maps_v2.json"
BERT_NAME = "bert-base-uncased"
OUT_DIR = "./fashion-bert-v2"

MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 12
PATIENCE = 3            # stop if held-out KL doesn't improve for this many epochs
LR = 1e-5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AXES = ["occasion", "formality", "constraint", "color_tone"]
# model head key -> csv axis name (the app's model calls the 4th head "color")
HEAD_KEYS = {"occasion": "occasion", "formality": "formality",
             "constraint": "constraint", "color": "color_tone"}

torch.manual_seed(SEED)
np.random.seed(SEED)

with open(LABEL_MAPS_PATH) as f:
    LABEL_MAPS = json.load(f)
REVERSE_MAPS = {a: {v: k for k, v in m.items()} for a, m in LABEL_MAPS.items()}


def soft_to_vec(axis, soft_json):
    """'{"party":0.7,"casual":0.3}' -> [p_casual, p_date, ...] in label-index order."""
    d = json.loads(soft_json)
    vec = [0.0] * len(LABEL_MAPS[axis])
    for label, w in d.items():
        vec[LABEL_MAPS[axis][label]] = float(w)
    s = sum(vec)
    return [x / s for x in vec] if s else vec


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class FashionDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        enc = self.tok(row["prompt"], max_length=MAX_LEN, padding="max_length",
                       truncation=True, return_tensors="pt")
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        for axis in AXES:
            item[f"{axis}_target"] = torch.tensor(
                soft_to_vec(axis, row[f"{axis}_soft"]), dtype=torch.float)
        return item


# ------------------------------------------------------------------
# Model  (identical architecture to the app's FashionIntentModel)
# ------------------------------------------------------------------
class FashionIntentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_NAME)
        h = self.bert.config.hidden_size
        self.occasion_head = nn.Linear(h, len(LABEL_MAPS["occasion"]))
        self.formality_head = nn.Linear(h, len(LABEL_MAPS["formality"]))
        self.constraint_head = nn.Linear(h, len(LABEL_MAPS["constraint"]))
        self.color_head = nn.Linear(h, len(LABEL_MAPS["color_tone"]))
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return {
            "occasion": self.occasion_head(cls),
            "formality": self.formality_head(cls),
            "constraint": self.constraint_head(cls),
            "color": self.color_head(cls),
        }


# ------------------------------------------------------------------
# Loss / eval
# ------------------------------------------------------------------
kl = nn.KLDivLoss(reduction="batchmean")


def batch_loss(outputs, batch):
    loss = 0.0
    for head, axis in HEAD_KEYS.items():
        logp = F.log_softmax(outputs[head], dim=1)
        loss = loss + kl(logp, batch[f"{axis}_target"].to(DEVICE))
    return loss


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_kl, n = 0.0, 0
    preds = {a: [] for a in AXES}
    gold = {a: [] for a in AXES}
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        out = model(ids, mask)
        tot_kl += batch_loss(out, batch).item() * len(ids)
        n += len(ids)
        for head, axis in HEAD_KEYS.items():
            preds[axis] += out[head].argmax(1).cpu().tolist()
            gold[axis] += batch[f"{axis}_target"].argmax(1).tolist()
    f1s = {a: f1_score(gold[a], preds[a], average="weighted", zero_division=0) for a in AXES}
    return tot_kl / n, f1s


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
def main():
    df = pd.read_csv(TRAIN_CSV)
    train_df = df[df.split == "train"]
    held_df = df[df.split == "heldout"]
    print(f"train {len(train_df)} | heldout {len(held_df)} | device {DEVICE}")
    print("occasion classes:", LABEL_MAPS["occasion"])

    tok = BertTokenizerFast.from_pretrained(BERT_NAME)
    train_loader = DataLoader(FashionDataset(train_df, tok), batch_size=BATCH_SIZE, shuffle=True)
    held_loader = DataLoader(FashionDataset(held_df, tok), batch_size=BATCH_SIZE)

    model = FashionIntentModel().to(DEVICE)
    opt = AdamW(model.parameters(), lr=LR)

    best_kl, best_state, bad = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            loss = batch_loss(model(ids, mask), batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        val_kl, f1s = evaluate(model, held_loader)
        print(f"epoch {epoch:2d} | train_loss {running/len(train_loader):.4f} "
              f"| heldout_KL {val_kl:.4f} | F1 " +
              " ".join(f"{a[:4]}={f1s[a]:.3f}" for a in AXES))

        if val_kl < best_kl - 1e-4:
            best_kl, best_state, bad = val_kl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stop (no heldout improvement for {PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)

    # ---- save ----
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))
    tok.save_pretrained(OUT_DIR)
    with open(os.path.join(OUT_DIR, "label_maps.json"), "w") as f:
        json.dump(LABEL_MAPS, f)
    print(f"\nsaved -> {OUT_DIR}  (best heldout KL {best_kl:.4f})")

    # ---- Step 4 sanity: does it now output *blends*? ----
    model.eval()
    demo = [
        "beach vacation with my girls, something fun and colourful",
        "it is a birthday, i want something party style but a little casual",
        "job interview tomorrow, need to look professional",
        "office party, mostly work appropriate but slightly festive",
    ]
    print("\n--- inference demo ---")
    for p in demo:
        enc = tok(p, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        print(f'\n"{p}"')
        for head, axis in HEAD_KEYS.items():
            probs = out[head].softmax(1)[0].cpu().tolist()
            top = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:2]
            print(f"  {axis:11s} " + ", ".join(f"{REVERSE_MAPS[axis][i]} {probs[i]:.0%}" for i in top))


if __name__ == "__main__":
    main()
