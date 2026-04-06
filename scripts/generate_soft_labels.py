"""
Use the trained teacher model to generate soft labels (raw logits)
for the train and val splits.

Output adds a `teacher_logits` field (list of 4 floats) to each record.
Raw logits are saved — not probabilities — so temperature scaling can be
applied during distillation training.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
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

class QueryDataset(Dataset):
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
# Teacher model loader
# ---------------------------------------------------------------------------

class TeacherModel(nn.Module):
    """Loads the saved teacher encoder + classifier head."""

    def __init__(self, model_dir: str) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_dir, use_safetensors=True, dtype=torch.float32
        )
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, NUM_LABELS)

        head_path = Path(model_dir) / "classifier_head.pt"
        state_dict = torch.load(head_path, weights_only=True)
        self.classifier.load_state_dict(state_dict)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_logits(
    model: TeacherModel,
    records: list[dict],
    tokenizer,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> list[list[float]]:
    dataset = QueryDataset(records, tokenizer, max_length)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    all_logits: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            # Always store as float32 regardless of model dtype
            all_logits.extend(logits.cpu().float().tolist())
            if (i + 1) % 20 == 0:
                logger.info("  Processed %d / %d batches", i + 1, len(loader))

    return all_logits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate soft labels from teacher model")
    parser.add_argument("--model-dir",  required=True, help="Teacher best_model/ directory")
    parser.add_argument("--data-dir",   required=True, help="Splits directory (train.json, val.json)")
    parser.add_argument("--output-dir", required=True, help="Output directory for enriched JSON files")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    setup_logging()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("Loading teacher tokenizer from %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    logger.info("Loading teacher model from %s", args.model_dir)
    model = TeacherModel(args.model_dir)
    model.to(device)
    logger.info(
        "Teacher params: %s M",
        round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
    )

    for split in ("train", "val"):
        src = Path(args.data_dir) / f"{split}.json"
        if not src.exists():
            logger.warning("Skipping %s (file not found)", src)
            continue

        logger.info("Processing split: %s", split)
        records = json.loads(src.read_text(encoding="utf-8"))
        logger.info("  Records: %d", len(records))

        logits_list = generate_logits(
            model, records, tokenizer, args.batch_size, args.max_length, device
        )
        assert len(logits_list) == len(records), "Length mismatch between records and logits"

        enriched = []
        for record, logits in zip(records, logits_list):
            r = dict(record)
            r["teacher_logits"] = logits  # list of 4 floats (raw logits, not sigmoid)
            enriched.append(r)

        out_path = output_dir / f"{split}_with_logits.json"
        out_path.write_text(
            json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("  Saved %d records → %s", len(enriched), out_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
