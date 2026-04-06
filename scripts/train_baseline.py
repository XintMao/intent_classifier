"""
Baseline noise-evaluation experiment.

Trains two distilbert-base-uncased classifiers:
  - Model_A : full labeled dataset
  - Model_B : high-confidence subset (max score >= 0.7)

Outputs accuracy, macro-F1, confusion matrix, and classification report
for both models on their respective test splits.
"""

import argparse
import json
import random
import numpy as np
import torch
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = Path(
    "/leonardo_scratch/large/userexternal/xmao0000/intent-classifier"
    "/data/labeled/labeled_queries.json"
)
SUMMARY_PATH = Path(
    "/leonardo_scratch/large/userexternal/xmao0000/intent-classifier"
    "/data/labeled/summary.json"
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LABEL_MAP = {"recency": 0, "authority": 1, "mechanism": 2, "general": 3}
LABEL_NAMES = ["recency", "authority", "mechanism", "general"]
MODEL_NAME = "distilbert-base-uncased"
SEED = 42
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
MAX_LEN = 128
CONFIDENCE_THRESHOLD = 0.7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: Path):
    with open(path, "r") as f:
        records = json.load(f)
    texts, labels = [], []
    for r in records:
        texts.append(r["query"])
        labels.append(LABEL_MAP[r["primary_intent"]])
    return texts, labels


def filter_high_confidence(path: Path, threshold: float = CONFIDENCE_THRESHOLD):
    with open(path, "r") as f:
        records = json.load(f)
    texts, labels = [], []
    for r in records:
        if max(r["scores"].values()) >= threshold:
            texts.append(r["query"])
            labels.append(LABEL_MAP[r["primary_intent"]])
    return texts, labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------

def build_splits(texts, labels, seed=SEED):
    """Stratified 80/10/10 split."""
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        texts, labels, test_size=0.20, stratify=labels, random_state=seed
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def tokenize(tokenizer, texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors=None,   # return lists so Dataset can index
    )


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def run_experiment(name: str, texts, labels, batch_size: int = BATCH_SIZE):
    print(f"\nTraining {name} on {len(texts)} samples ...")

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = build_splits(texts, labels)

    print(
        f"  train={len(x_train)}  val={len(x_val)}  test={len(x_test)}"
        f"  label dist: {dict(sorted(Counter(y_train).items()))}"
    )

    set_seed(SEED)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    enc_train = tokenize(tokenizer, x_train)
    enc_val   = tokenize(tokenizer, x_val)
    enc_test  = tokenize(tokenizer, x_test)

    ds_train = IntentDataset(enc_train, y_train)
    ds_val   = IntentDataset(enc_val,   y_val)
    ds_test  = IntentDataset(enc_test,  y_test)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, val_preds, val_labels = evaluate(model, val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        print(
            f"  Epoch {epoch}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}"
        )

    _, test_preds, test_labels_arr = evaluate(model, test_loader)

    acc = accuracy_score(test_labels_arr, test_preds)
    macro_f1 = f1_score(test_labels_arr, test_preds, average="macro")
    cm = confusion_matrix(test_labels_arr, test_preds)
    report = classification_report(
        test_labels_arr, test_preds, target_names=LABEL_NAMES
    )

    return acc, macro_f1, cm, report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    batch_size = args.batch_size

    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {batch_size}")

    # --- Optional summary logging ---
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH) as f:
            summary = json.load(f)
        print(f"\n[Summary] total={summary.get('total')}  "
              f"intent_counts={summary.get('primary_intent_counts')}")

    # --- Load datasets ---
    all_texts, all_labels = load_data(DATA_PATH)
    hc_texts,  hc_labels  = filter_high_confidence(DATA_PATH, CONFIDENCE_THRESHOLD)

    size_a = len(all_texts)
    size_b = len(hc_texts)
    retention = size_b / size_a if size_a > 0 else 0.0

    print("\n=== DATASET STATS ===")
    print(f"Full size:       {size_a}")
    print(f"Filtered size:   {size_b}")
    print(f"Retention ratio: {retention:.4f}")

    # --- Train full ---
    acc_a, f1_a, cm_a, rep_a = run_experiment("Model_A (full)", all_texts, all_labels, batch_size)

    # --- Train high-confidence ---
    acc_b, f1_b, cm_b, rep_b = run_experiment("Model_B (high-confidence)", hc_texts, hc_labels, batch_size)

    # --- Final report ---
    print("\n" + "=" * 60)
    print("=== FULL DATASET ===")
    print(f"Accuracy:  {acc_a:.4f}")
    print(f"Macro F1:  {f1_a:.4f}")
    print("Confusion Matrix:")
    print(cm_a)
    print("Classification Report:")
    print(rep_a)

    print("=" * 60)
    print("=== HIGH-CONFIDENCE DATASET ===")
    print(f"Accuracy:  {acc_b:.4f}")
    print(f"Macro F1:  {f1_b:.4f}")
    print("Confusion Matrix:")
    print(cm_b)
    print("Classification Report:")
    print(rep_b)


if __name__ == "__main__":
    main()
