"""
Fine-tune DeBERTa-v3-large as a multi-label intent classifier (teacher model).

Uses BCEWithLogitsLoss + sigmoid outputs since each intent is scored
independently (multi-label, not mutually exclusive).
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
)
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

INTENTS = ["recency", "authority", "mechanism", "general"]
NUM_LABELS = len(INTENTS)


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IntentDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        item = self.records[idx]
        encoding = self.tokenizer(
            item["query"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = torch.tensor(item["labels"], dtype=torch.float32)
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            # token_type_ids not used by DeBERTa-v3 but harmless to omit
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DeBERTaMultiLabel(nn.Module):
    """
    DeBERTa-v3-large backbone with a linear classification head.
    Outputs raw logits (BCEWithLogitsLoss is applied outside during training).
    """

    def __init__(self, model_name: str, num_labels: int, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        # use_safetensors=True avoids torch.load CVE-2025-32434 on torch<2.6
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True, dtype=torch.float32)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.pos_weight = pos_weight  # per-class weight for BCEWithLogitsLoss
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            # Cast labels and pos_weight to match logits dtype (important for bf16/fp16)
            labels = labels.to(dtype=logits.dtype)
            pw = self.pos_weight.to(device=logits.device, dtype=logits.dtype) if self.pos_weight is not None else None
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            loss = criterion(logits, labels)

        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))          # sigmoid
    preds = (probs >= 0.5).astype(int)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    h_loss = hamming_loss(labels, preds)

    # Per-class F1
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    metrics = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "hamming_loss": h_loss,
    }
    for i, intent in enumerate(INTENTS):
        metrics[f"f1_{intent}"] = per_class_f1[i]

    return metrics


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------

def compute_pos_weights(train_records: list[dict]) -> torch.Tensor:
    """
    Compute per-class positive weights for BCEWithLogitsLoss.
    pos_weight[i] = (n_negative[i] / n_positive[i]).
    Clips to [1.0, 10.0] to avoid extreme values.
    """
    n = len(train_records)
    pos_counts = np.array([sum(r["labels"][i] for r in train_records) for i in range(NUM_LABELS)], dtype=float)
    neg_counts = n - pos_counts
    weights = neg_counts / np.maximum(pos_counts, 1.0)
    weights = np.clip(weights, 1.0, 10.0)
    logger.info("Pos weights: %s", dict(zip(INTENTS, weights.round(3).tolist())))
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeBERTa-v3-large multi-label intent classifier")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-name", default="microsoft/deberta-v3-large")
    parser.add_argument("--output-dir", default="models/teacher")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir / "train.log")
    set_seed(args.seed)

    logger.info("Loading splits from %s", args.data_dir)
    data_dir = Path(args.data_dir)
    train_records = json.loads((data_dir / "train.json").read_text())
    val_records   = json.loads((data_dir / "val.json").read_text())
    test_records  = json.loads((data_dir / "test.json").read_text())
    logger.info("Splits: train=%d  val=%d  test=%d", len(train_records), len(val_records), len(test_records))

    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = IntentDataset(train_records, tokenizer, args.max_length)
    val_ds   = IntentDataset(val_records,   tokenizer, args.max_length)
    test_ds  = IntentDataset(test_records,  tokenizer, args.max_length)

    pos_weight = compute_pos_weights(train_records)

    logger.info("Loading model: %s", args.model_name)
    model = DeBERTaMultiLabel(args.model_name, NUM_LABELS, pos_weight=pos_weight)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=str(output_dir / "tensorboard"),
        logging_steps=50,
        save_total_limit=2,
        seed=args.seed,
        report_to=["tensorboard"],
        dataloader_num_workers=4,
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting training")
    trainer.train()

    # ------------------------------------------------------------------
    # Test set evaluation
    # ------------------------------------------------------------------
    logger.info("Evaluating on test set")
    test_output = trainer.predict(test_ds)
    logits = test_output.predictions
    true_labels = np.array([r["labels"] for r in test_records])

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    report = classification_report(
        true_labels,
        preds,
        target_names=INTENTS,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        true_labels,
        preds,
        target_names=INTENTS,
        zero_division=0,
    )
    macro_f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(true_labels, preds, average="micro", zero_division=0)
    h_loss   = hamming_loss(true_labels, preds)

    print("\n=== Test Set Evaluation ===")
    print(report_str)
    print(f"Macro F1     : {macro_f1:.4f}")
    print(f"Micro F1     : {micro_f1:.4f}")
    print(f"Hamming loss : {h_loss:.4f}")

    eval_results = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "hamming_loss": h_loss,
        "per_class": report,
        "threshold_used": 0.5,
    }
    (output_dir / "eval_results.json").write_text(json.dumps(eval_results, indent=2), encoding="utf-8")
    logger.info("Saved eval results to %s", output_dir / "eval_results.json")

    # Save raw predictions for downstream distillation analysis
    test_predictions = [
        {
            "query": test_records[i]["query"],
            "true_labels": true_labels[i].tolist(),
            "pred_labels": preds[i].tolist(),
            "scores": {intent: float(probs[i][j]) for j, intent in enumerate(INTENTS)},
            "primary_intent": test_records[i]["primary_intent"],
        }
        for i in range(len(test_records))
    ]
    (output_dir / "test_predictions.json").write_text(
        json.dumps(test_predictions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved test predictions to %s", output_dir / "test_predictions.json")

    # Save final model and tokenizer
    best_model_path = output_dir / "best_model"
    model.encoder.save_pretrained(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    torch.save(model.classifier.state_dict(), best_model_path / "classifier_head.pt")
    logger.info("Saved best model to %s", best_model_path)


if __name__ == "__main__":
    main()
