"""
Evaluate a trained intent classifier on a test set.

Loads the saved model checkpoint, runs inference, applies a configurable
threshold to convert sigmoid scores into binary predictions, then reports
macro/micro F1, per-class metrics, and hamming loss.

Works for both the teacher (DeBERTa-v3-large) and the student model
(same interface, different --model-dir).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

INTENTS = ["recency", "authority", "mechanism", "general"]
NUM_LABELS = len(INTENTS)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
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
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class DeBERTaMultiLabel(nn.Module):
    def __init__(self, model_name: str, num_labels: int) -> None:
        super().__init__()
        # use_safetensors=True avoids torch.load CVE-2025-32434 on torch<2.6
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)


def load_model(model_dir: Path, device: torch.device) -> tuple:
    """
    Load encoder + classifier head saved by train_teacher.py.
    Accepts either the parent dir (containing best_model/) or best_model/ directly.
    Returns (model, tokenizer).
    """
    # Support both models/teacher/ and models/teacher/best_model/
    best_model_path = model_dir / "best_model" if (model_dir / "best_model").exists() else model_dir
    logger.info("Loading tokenizer from %s", best_model_path)
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)

    logger.info("Loading model from %s", best_model_path)
    model = DeBERTaMultiLabel(str(best_model_path), NUM_LABELS)

    classifier_head_path = best_model_path / "classifier_head.pt"
    if classifier_head_path.exists():
        model.classifier.load_state_dict(
            torch.load(classifier_head_path, map_location=device, weights_only=True)
        )
        logger.info("Loaded classifier head from %s", classifier_head_path)
    else:
        logger.warning("classifier_head.pt not found; using randomly initialised head")

    model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: nn.Module,
    dataset: IntentDataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return raw logits as (n_samples, num_labels) array."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained intent classifier")
    parser.add_argument("--model-dir", required=True, help="Directory containing best_model/ subdirectory")
    parser.add_argument("--test-data", required=True, help="Path to test.json")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for positive prediction")
    parser.add_argument("--output", required=True, help="Path to save evaluation results JSON")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model_dir = Path(args.model_dir)
    model, tokenizer = load_model(model_dir, device)

    logger.info("Loading test data from %s", args.test_data)
    records = json.loads(Path(args.test_data).read_text())
    logger.info("Loaded %d test records", len(records))

    # Check whether ground-truth labels are present
    has_labels = "labels" in records[0]
    true_labels = np.array([r["labels"] for r in records]) if has_labels else None

    dataset = IntentDataset(records, tokenizer, args.max_length)
    logits = run_inference(model, dataset, args.batch_size, device)

    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= args.threshold).astype(int)

    # Build per-sample output
    predictions = [
        {
            "query": records[i]["query"],
            "scores": {intent: float(probs[i][j]) for j, intent in enumerate(INTENTS)},
            "pred_labels": preds[i].tolist(),
            "true_labels": true_labels[i].tolist() if has_labels else None,
            "primary_intent": records[i].get("primary_intent", ""),
        }
        for i in range(len(records))
    ]

    eval_results: dict = {"threshold": args.threshold, "n_samples": len(records)}

    if has_labels:
        macro_f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
        micro_f1 = f1_score(true_labels, preds, average="micro", zero_division=0)
        h_loss   = hamming_loss(true_labels, preds)
        report_dict = classification_report(
            true_labels, preds, target_names=INTENTS, output_dict=True, zero_division=0
        )
        report_str = classification_report(
            true_labels, preds, target_names=INTENTS, zero_division=0
        )

        print("\n=== Evaluation Results ===")
        print(f"Threshold    : {args.threshold}")
        print(f"Macro F1     : {macro_f1:.4f}")
        print(f"Micro F1     : {micro_f1:.4f}")
        print(f"Hamming loss : {h_loss:.4f}")
        print("\nPer-class report:")
        print(report_str)

        eval_results.update(
            {
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "hamming_loss": h_loss,
                "per_class": report_dict,
            }
        )
    else:
        logger.info("No ground-truth labels found; skipping metric computation")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"eval": eval_results, "predictions": predictions}, indent=2, ensure_ascii=False)
    )
    logger.info("Saved results to %s", output_path)


if __name__ == "__main__":
    main()
