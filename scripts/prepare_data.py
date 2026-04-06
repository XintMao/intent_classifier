"""
Prepare train/val/test splits from labeled_queries.json.

Converts continuous intent scores to binary multi-label targets using a
configurable threshold, then stratifies the split by primary_intent.
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

INTENTS = ["recency", "authority", "mechanism", "general"]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def scores_to_labels(scores: dict, threshold: float, primary_intent: str) -> list[int]:
    """
    Convert continuous scores to binary labels using the given threshold.
    If all scores fall below the threshold, fall back to primary_intent so
    that every sample has at least one positive label.
    """
    labels = [1 if scores[intent] >= threshold else 0 for intent in INTENTS]
    if sum(labels) == 0:
        fallback_idx = INTENTS.index(primary_intent)
        labels[fallback_idx] = 1
    return labels


def stratified_split(
    data: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Stratified split by primary_intent.
    Splits each stratum proportionally, then merges.
    """
    rng = random.Random(seed)

    # Group indices by primary_intent
    strata: dict[str, list[int]] = {}
    for idx, item in enumerate(data):
        key = item["primary_intent"]
        strata.setdefault(key, []).append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for key, indices in strata.items():
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            # Stratum too small: put everything in train
            logger.warning("Stratum '%s' has only %d samples; all go to train", key, n)
            train_idx.extend(indices)
            continue

        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test : n_test + n_val])
        train_idx.extend(indices[n_test + n_val :])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (
        [data[i] for i in train_idx],
        [data[i] for i in val_idx],
        [data[i] for i in test_idx],
    )


def print_split_stats(name: str, split: list[dict]) -> None:
    n = len(split)
    intent_counts = Counter()
    multi_label = 0
    for item in split:
        active = sum(item["labels"])
        if active > 1:
            multi_label += 1
        for i, intent in enumerate(INTENTS):
            if item["labels"][i] == 1:
                intent_counts[intent] += 1

    print(f"\n  {name}: {n} samples")
    for intent in INTENTS:
        cnt = intent_counts[intent]
        print(f"    {intent:<12}: {cnt:>5} positive  ({100 * cnt / n:.1f}%)")
    print(f"    multi-label  : {multi_label:>5}  ({100 * multi_label / n:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/val/test splits")
    parser.add_argument("--input", required=True, help="Path to labeled_queries.json")
    parser.add_argument("--output-dir", required=True, help="Directory to write split JSON files")
    parser.add_argument("--threshold", type=float, default=0.7, help="Score threshold for positive label")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", args.input)
    raw = json.loads(Path(args.input).read_text())
    logger.info("Loaded %d records", len(raw))

    # Convert scores -> binary labels
    processed = []
    for item in raw:
        labels = scores_to_labels(item["scores"], args.threshold, item["primary_intent"])
        processed.append(
            {
                "query": item["query"],
                "labels": labels,
                "primary_intent": item["primary_intent"],
                "scores": item["scores"],
            }
        )

    # Deduplicate by query text before splitting to prevent leakage.
    # When the same query appears multiple times (generated in different batches),
    # keep the first occurrence (deterministic after sorting by query).
    seen: set[str] = set()
    deduped = []
    for item in sorted(processed, key=lambda x: x["query"]):
        if item["query"] not in seen:
            seen.add(item["query"])
            deduped.append(item)
    n_removed = len(processed) - len(deduped)
    if n_removed:
        logger.info("Removed %d duplicate queries (%d → %d unique)", n_removed, len(processed), len(deduped))
    processed = deduped

    train, val, test = stratified_split(processed, val_ratio=0.1, test_ratio=0.1, seed=args.seed)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{split_name}.json"
        path.write_text(json.dumps(split_data, indent=2, ensure_ascii=False))
        logger.info("Saved %s (%d samples) -> %s", split_name, len(split_data), path)

    # Summary
    print("\n=== Data Split Summary ===")
    print(f"Threshold          : {args.threshold}")
    print(f"Total samples      : {len(processed)}")
    print_split_stats("train", train)
    print_split_stats("val", val)
    print_split_stats("test", test)

    # Label distribution across full dataset
    intent_totals = Counter()
    multi_total = 0
    for item in processed:
        active = sum(item["labels"])
        if active > 1:
            multi_total += 1
        for i, intent in enumerate(INTENTS):
            if item["labels"][i] == 1:
                intent_totals[intent] += 1

    n = len(processed)
    print(f"\n  full dataset positive label rates (threshold={args.threshold}):")
    for intent in INTENTS:
        cnt = intent_totals[intent]
        print(f"    {intent:<12}: {cnt:>5}  ({100 * cnt / n:.1f}%)")
    print(f"    multi-label  : {multi_total:>5}  ({100 * multi_total / n:.1f}%)")


if __name__ == "__main__":
    main()
