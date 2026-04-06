"""
Train DeBERTa-v3-xsmall student via knowledge distillation from DeBERTa-v3-large teacher.

Loss = alpha * soft_loss * T² + (1 - alpha) * hard_loss

Where:
  soft_loss  = BCE(sigmoid(student_logits/T), sigmoid(teacher_logits/T))
  hard_loss  = BCEWithLogitsLoss(student_logits, hard_labels)
  T          = temperature (softens teacher distribution)

Uses BCE (not KL-divergence) because multi-label sigmoid outputs are
independent per-class probabilities — they do not sum to 1.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, hamming_loss
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

logger = logging.getLogger(__name__)

INTENTS = ["recency", "authority", "mechanism", "general"]
NUM_LABELS = len(INTENTS)
TEACHER_MACRO_F1 = 0.9515  # teacher test set result, used for comparison


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

class DistillDataset(Dataset):
    """
    Dataset that supports both soft labels (teacher_logits) and hard labels.
    Test set records don't have teacher_logits — set has_teacher_logits=False.
    """

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int,
        has_teacher_logits: bool = True,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_teacher_logits = has_teacher_logits

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
        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["labels"], dtype=torch.float32),
        }
        if self.has_teacher_logits:
            result["teacher_logits"] = torch.tensor(
                item["teacher_logits"], dtype=torch.float32
            )
        return result


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StudentModel(nn.Module):
    """DeBERTa-v3-xsmall backbone with a linear classification head (4 outputs)."""

    def __init__(self, model_name: str, num_labels: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name, use_safetensors=True, dtype=torch.float32
        )
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """
    Combines soft distillation loss with hard label loss.

    The T² scaling factor on soft_loss ensures gradient magnitudes stay
    comparable to hard_loss regardless of temperature value.
    """
    teacher_soft = torch.sigmoid(teacher_logits / temperature)
    student_soft = torch.sigmoid(student_logits / temperature)
    soft_loss = F.binary_cross_entropy(student_soft, teacher_soft)

    hard_loss = F.binary_cross_entropy_with_logits(student_logits, hard_labels)

    return alpha * soft_loss * (temperature ** 2) + (1 - alpha) * hard_loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    alpha: float,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)

            if "teacher_logits" in batch:
                t_logits = batch["teacher_logits"].to(device)
                loss = distillation_loss(logits, t_logits, labels, temperature, alpha)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits.cpu().float().numpy())
            all_labels.append(labels.cpu().float().numpy())

    all_logits_np = np.concatenate(all_logits, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)

    probs = 1 / (1 + np.exp(-all_logits_np))
    preds = (probs >= 0.5).astype(int)

    return {
        "loss":         total_loss / len(loader),
        "macro_f1":     f1_score(all_labels_np, preds, average="macro",  zero_division=0),
        "micro_f1":     f1_score(all_labels_np, preds, average="micro",  zero_division=0),
        "hamming_loss": hamming_loss(all_labels_np, preds),
        "logits":       all_logits_np,
        "labels":       all_labels_np,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Distill DeBERTa-v3-xsmall student from teacher")
    parser.add_argument("--data-dir",     required=True, help="Soft labels directory")
    parser.add_argument("--test-data",    required=True, help="Path to test.json (original splits)")
    parser.add_argument("--model-name",   default="microsoft/deberta-v3-xsmall")
    parser.add_argument("--output-dir",   default="models/student")
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--learning-rate",type=float, default=5e-5)
    parser.add_argument("--max-length",   type=int,   default=128)
    parser.add_argument("--temperature",  type=float, default=3.0)
    parser.add_argument("--alpha",        type=float, default=0.7)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir / "train.log")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    logger.info("Loading data from %s", data_dir)
    train_records = json.loads((data_dir / "train_with_logits.json").read_text(encoding="utf-8"))
    val_records   = json.loads((data_dir / "val_with_logits.json").read_text(encoding="utf-8"))
    test_records  = json.loads(Path(args.test_data).read_text(encoding="utf-8"))
    logger.info(
        "Splits: train=%d  val=%d  test=%d",
        len(train_records), len(val_records), len(test_records),
    )

    # ------------------------------------------------------------------
    # Tokenizer & datasets
    # ------------------------------------------------------------------
    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = DistillDataset(train_records, tokenizer, args.max_length, has_teacher_logits=True)
    val_ds   = DistillDataset(val_records,   tokenizer, args.max_length, has_teacher_logits=True)
    test_ds  = DistillDataset(test_records,  tokenizer, args.max_length, has_teacher_logits=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,     shuffle=True,  num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,   batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,  batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info("Loading student model: %s", args.model_name)
    model = StudentModel(args.model_name, NUM_LABELS)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Student params: %.1f M", n_params)

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    logger.info(
        "Config: epochs=%d  batch=%d  lr=%s  T=%.1f  alpha=%.2f",
        args.epochs, args.batch_size, args.learning_rate, args.temperature, args.alpha,
    )
    logger.info("Total steps: %d  warmup steps: %d", total_steps, warmup_steps)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_macro_f1 = 0.0
    patience_counter  = 0
    patience          = 5
    best_model_path   = output_dir / "best_model"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            t_logits       = batch["teacher_logits"].to(device)

            optimizer.zero_grad()
            student_logits = model(input_ids, attention_mask)
            loss = distillation_loss(
                student_logits, t_logits, labels, args.temperature, args.alpha
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device, args.temperature, args.alpha)

        logger.info(
            "Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f | "
            "val_macro_f1=%.4f | val_micro_f1=%.4f | val_hamming=%.4f",
            epoch, args.epochs,
            avg_train_loss,
            val_metrics["loss"],
            val_metrics["macro_f1"],
            val_metrics["micro_f1"],
            val_metrics["hamming_loss"],
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            model.encoder.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            torch.save(model.classifier.state_dict(), best_model_path / "classifier_head.pt")
            logger.info("  → New best val macro F1: %.4f — model saved", best_val_macro_f1)
        else:
            patience_counter += 1
            logger.info("  → No improvement (%d/%d)", patience_counter, patience)
            if patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    # ------------------------------------------------------------------
    # Final test evaluation with best model
    # ------------------------------------------------------------------
    logger.info("Loading best model for test evaluation from %s", best_model_path)
    best_model = StudentModel(str(best_model_path), NUM_LABELS)
    best_model.classifier.load_state_dict(
        torch.load(best_model_path / "classifier_head.pt", weights_only=True)
    )
    best_model.to(device)

    # Collect test logits and labels
    best_model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = best_model(input_ids, attention_mask)
            all_logits.append(logits.cpu().float().numpy())
            all_labels.append(batch["labels"].numpy())

    all_logits_np = np.concatenate(all_logits, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)

    probs = 1 / (1 + np.exp(-all_logits_np))
    preds = (probs >= 0.5).astype(int)

    report_str = classification_report(all_labels_np, preds, target_names=INTENTS, zero_division=0)
    report_dict = classification_report(
        all_labels_np, preds, target_names=INTENTS, output_dict=True, zero_division=0
    )
    macro_f1 = f1_score(all_labels_np, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels_np, preds, average="micro", zero_division=0)
    h_loss   = hamming_loss(all_labels_np, preds)

    print("\n=== Test Set Evaluation (Student) ===")
    print(report_str)
    print(f"Macro F1     : {macro_f1:.4f}")
    print(f"Micro F1     : {micro_f1:.4f}")
    print(f"Hamming loss : {h_loss:.4f}")
    print("\n=== Teacher vs Student Comparison ===")
    print(f"Teacher Macro F1 : {TEACHER_MACRO_F1:.4f}")
    print(f"Student Macro F1 : {macro_f1:.4f}")
    print(f"Gap              : {TEACHER_MACRO_F1 - macro_f1:+.4f}")

    eval_results = {
        "macro_f1":           macro_f1,
        "micro_f1":           micro_f1,
        "hamming_loss":       h_loss,
        "best_val_macro_f1":  best_val_macro_f1,
        "teacher_macro_f1":   TEACHER_MACRO_F1,
        "distillation_gap":   round(TEACHER_MACRO_F1 - macro_f1, 4),
        "per_class":          report_dict,
        "threshold_used":     0.5,
        "temperature":        args.temperature,
        "alpha":              args.alpha,
    }
    (output_dir / "eval_results.json").write_text(
        json.dumps(eval_results, indent=2), encoding="utf-8"
    )
    logger.info("Saved eval results → %s", output_dir / "eval_results.json")
    logger.info("Saved best model   → %s", best_model_path)


if __name__ == "__main__":
    main()
