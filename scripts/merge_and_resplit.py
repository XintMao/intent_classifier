"""
Merge existing and new labeled data, deduplicate, and re-split into train/val/test.

Uses the same normalize_query function as generate_supplementary.py to ensure
consistent deduplication across the pipeline.
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import normalize_query

logger = logging.getLogger(__name__)

INTENTS = ["recency", "authority", "mechanism", "general"]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def scores_to_labels(scores: dict, threshold: float, primary_intent: str) -> list[int]:
    labels = [1 if scores[intent] >= threshold else 0 for intent in INTENTS]
    if sum(labels) == 0:
        labels[INTENTS.index(primary_intent)] = 1
    return labels


def stratified_split(
    data: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    strata: dict[str, list[int]] = {}
    for idx, item in enumerate(data):
        strata.setdefault(item["primary_intent"], []).append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for key, indices in strata.items():
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, round(n * test_ratio))
        n_val  = max(1, round(n * val_ratio))
        n_train = n - n_val - n_test
        if n_train < 1:
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
    intent_counts: Counter = Counter()
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
    parser = argparse.ArgumentParser(description="Merge and re-split labeled data")
    parser.add_argument("--existing",        required=True, help="Existing labeled_queries.json")
    parser.add_argument("--new",             required=True, help="New labeled_new.json from label_supplementary.py")
    parser.add_argument("--output-labeled",  required=True, help="Output path for merged labeled JSON")
    parser.add_argument("--output-dir",      required=True, help="Directory to write train/val/test splits")
    parser.add_argument("--threshold",       type=float, default=0.7)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    setup_logging()

    # ------------------------------------------------------------------
    # Load and merge
    # ------------------------------------------------------------------
    logger.info("Loading existing: %s", args.existing)
    existing = json.loads(Path(args.existing).read_text(encoding="utf-8"))
    logger.info("Existing records: %d", len(existing))

    logger.info("Loading new: %s", args.new)
    new_data = json.loads(Path(args.new).read_text(encoding="utf-8"))
    logger.info("New records: %d", len(new_data))

    combined = existing + new_data
    logger.info("Combined before dedup: %d", len(combined))

    # ------------------------------------------------------------------
    # Deduplicate using normalize_query (same function as generate_supplementary)
    # ------------------------------------------------------------------
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in combined:
        nq = normalize_query(item["query"])
        if nq not in seen:
            seen.add(nq)
            deduped.append(item)

    n_removed = len(combined) - len(deduped)
    logger.info("Removed %d duplicates — %d unique records remain", n_removed, len(deduped))

    if len(deduped) < 10_000:
        logger.warning(
            "Total records (%d) is below 10,000. Consider generating more supplementary data.",
            len(deduped),
        )

    # ------------------------------------------------------------------
    # Save merged labeled file
    # ------------------------------------------------------------------
    output_labeled = Path(args.output_labeled)
    output_labeled.parent.mkdir(parents=True, exist_ok=True)
    output_labeled.write_text(json.dumps(deduped, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved merged labeled data → %s", output_labeled)

    # ------------------------------------------------------------------
    # Convert scores to hard labels and split
    # ------------------------------------------------------------------
    processed: list[dict] = []
    for item in deduped:
        labels = scores_to_labels(item["scores"], args.threshold, item["primary_intent"])
        processed.append({
            "query":          item["query"],
            "labels":         labels,
            "primary_intent": item["primary_intent"],
            "scores":         item["scores"],
        })

    train, val, test = stratified_split(processed, val_ratio=0.1, test_ratio=0.1, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{split_name}.json"
        path.write_text(json.dumps(split_data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved %s (%d samples) → %s", split_name, len(split_data), path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Merge & Split Summary ===")
    print(f"Existing records        : {len(existing)}")
    print(f"New records             : {len(new_data)}")
    print(f"Combined before dedup   : {len(combined)}")
    print(f"Duplicates removed      : {n_removed}")
    print(f"Unique records          : {len(deduped)}")
    print(f"Threshold               : {args.threshold}")
    print(f"Total after split       : {len(train) + len(val) + len(test)}")
    print_split_stats("train", train)
    print_split_stats("val",   val)
    print_split_stats("test",  test)

    # Verify no leakage
    train_q = set(d["query"] for d in train)
    val_q   = set(d["query"] for d in val)
    test_q  = set(d["query"] for d in test)
    print(f"\n  Leakage check:")
    print(f"    Train ∩ Val:  {len(train_q & val_q)}")
    print(f"    Train ∩ Test: {len(train_q & test_q)}")
    print(f"    Val   ∩ Test: {len(val_q  & test_q)}")


if __name__ == "__main__":
    main()
