"""
Merge existing labeled data with supplementary data, deduplicate, and re-split.

Input:
  data/labeled/labeled_queries.json       (existing, N records)
  data/supplementary/supplementary_queries.json  (new, M records)

Output:
  data/splits/train.json   (80%)
  data/splits/val.json     (10%)
  data/splits/test.json    (10%)

Each output record:
  {"query": "...", "labels": [0,1,0,1], "primary_intent": "...", "scores": {...}}

Labels are generated from scores using threshold=0.7.
If all labels are 0 after thresholding, primary_intent is used as fallback.
"""

import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INTENTS = ["recency", "authority", "mechanism", "general"]
LABEL_THRESHOLD = 0.7
SPLIT_SEED = 42

EXISTING_PATH = Path("data/labeled/labeled_queries.json")
SUPPLEMENT_PATH = Path("data/supplementary/supplementary_queries.json")
OUTPUT_DIR = Path("data/splits")


def normalize_query(q: str) -> str:
    q = q.strip().lower()
    q = re.sub(r'\s+', ' ', q)
    q = re.sub(r'[^\w\s]', '', q)
    return q


def make_labels(scores: dict, primary_intent: str) -> list[int]:
    labels = [1 if scores.get(intent, 0.0) >= LABEL_THRESHOLD else 0 for intent in INTENTS]
    if sum(labels) == 0:
        idx = INTENTS.index(primary_intent)
        labels[idx] = 1
    return labels


def stratified_split(records: list[dict], seed: int) -> tuple[list, list, list]:
    """80/10/10 split stratified by primary_intent."""
    rng = np.random.default_rng(seed)

    by_intent = defaultdict(list)
    for rec in records:
        by_intent[rec["primary_intent"]].append(rec)

    train, val, test = [], [], []
    for intent, items in by_intent.items():
        idx = rng.permutation(len(items)).tolist()
        n = len(items)
        n_val  = max(1, round(n * 0.1))
        n_test = max(1, round(n * 0.1))
        n_train = n - n_val - n_test

        train += [items[i] for i in idx[:n_train]]
        val   += [items[i] for i in idx[n_train:n_train + n_val]]
        test  += [items[i] for i in idx[n_train + n_val:]]

    # Shuffle each split
    for split in (train, val, test):
        order = rng.permutation(len(split)).tolist()
        split[:] = [split[i] for i in order]

    return train, val, test


def verify_no_leakage(train: list, val: list, test: list) -> None:
    train_keys = {normalize_query(r["query"]) for r in train}
    val_keys   = {normalize_query(r["query"]) for r in val}
    test_keys  = {normalize_query(r["query"]) for r in test}

    tv = train_keys & val_keys
    tt = train_keys & test_keys
    vt = val_keys   & test_keys

    if tv or tt or vt:
        logger.error("DATA LEAKAGE DETECTED: train∩val=%d  train∩test=%d  val∩test=%d", len(tv), len(tt), len(vt))
        sys.exit(1)
    logger.info("Data leakage check PASSED: zero overlap across splits")


def main() -> None:
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading existing data: %s", EXISTING_PATH)
    existing = json.loads(EXISTING_PATH.read_text(encoding="utf-8"))
    logger.info("  Existing records: %d", len(existing))

    logger.info("Loading supplementary data: %s", SUPPLEMENT_PATH)
    supplement = json.loads(SUPPLEMENT_PATH.read_text(encoding="utf-8"))
    logger.info("  Supplementary records: %d", len(supplement))

    # ------------------------------------------------------------------
    # Merge and deduplicate (existing first → has priority)
    # ------------------------------------------------------------------
    seen: set[str] = set()
    merged: list[dict] = []

    n_dup_existing = 0
    for rec in existing:
        key = normalize_query(rec["query"])
        if key in seen:
            n_dup_existing += 1
            continue
        seen.add(key)
        merged.append(rec)

    n_dup_supplement = 0
    n_overlap = 0
    for rec in supplement:
        key = normalize_query(rec["query"])
        if key in seen:
            if any(normalize_query(e["query"]) == key for e in existing):
                n_overlap += 1
            else:
                n_dup_supplement += 1
            continue
        seen.add(key)
        merged.append(rec)

    n_discarded = len(existing) + len(supplement) - len(merged)
    logger.info("Deduplication: discarded %d duplicates (%d within-existing, %d within-supplement, %d cross-overlap)",
                n_discarded, n_dup_existing, n_dup_supplement, n_overlap)
    logger.info("Merged unique records: %d", len(merged))

    # ------------------------------------------------------------------
    # Minimum data check
    # ------------------------------------------------------------------
    if len(merged) < 10_000:
        logger.error("Merged total %d < 10,000. Aborting. Please add more data.", len(merged))
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build output records with hard labels
    # ------------------------------------------------------------------
    records: list[dict] = []
    n_fallback = 0
    for rec in merged:
        scores = rec["scores"]
        primary = rec["primary_intent"]
        labels_raw = [1 if scores.get(intent, 0.0) >= LABEL_THRESHOLD else 0 for intent in INTENTS]
        if sum(labels_raw) == 0:
            n_fallback += 1
            labels_raw[INTENTS.index(primary)] = 1
        records.append({
            "query":          rec["query"],
            "labels":         labels_raw,
            "primary_intent": primary,
            "scores":         scores,
        })

    if n_fallback:
        logger.info("Applied primary_intent fallback for %d records with all-zero labels", n_fallback)

    # ------------------------------------------------------------------
    # Stratified split
    # ------------------------------------------------------------------
    train, val, test = stratified_split(records, SPLIT_SEED)
    verify_no_leakage(train, val, test)

    # ------------------------------------------------------------------
    # Save splits
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = OUTPUT_DIR / f"{name}.json"
        path.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved %s.json: %d records → %s", name, len(split), path)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    print("\n========================================")
    print("         MERGE STATISTICS")
    print("========================================")
    print(f"Existing data           : {len(existing):>6,}")
    print(f"Supplementary data      : {len(supplement):>6,}")
    print(f"Duplicates discarded    : {n_discarded:>6,}")
    print(f"  within-existing       : {n_dup_existing:>6,}")
    print(f"  within-supplement     : {n_dup_supplement:>6,}")
    print(f"  cross-overlap         : {n_overlap:>6,}")
    print(f"Merged unique total     : {len(merged):>6,}")
    print()
    print(f"{'Split':<8} {'Count':>7}  {'%':>5}")
    print("-" * 24)
    total = len(records)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:<8} {len(split):>7,}  {len(split)/total*100:>4.1f}%")
    print(f"{'TOTAL':<8} {total:>7,}")

    print("\nPer-intent positive samples (in train):")
    print(f"  {'Intent':<12} {'Count':>6}  {'%':>5}")
    print("  " + "-" * 27)
    for i, intent in enumerate(INTENTS):
        n_pos = sum(1 for r in train if r["labels"][i] == 1)
        print(f"  {intent:<12} {n_pos:>6,}  {n_pos/len(train)*100:>4.1f}%")

    multi_label = sum(1 for r in records if sum(r["labels"]) > 1)
    print(f"\nMulti-label samples (total): {multi_label:,}  ({multi_label/total*100:.1f}%)")

    primary_dist = Counter(r["primary_intent"] for r in records)
    print("\nPrimary intent distribution (total):")
    for intent in INTENTS:
        n = primary_dist[intent]
        print(f"  {intent:<12} {n:>6,}  ({n/total*100:.1f}%)")

    print("\n✓ Splits saved to data/splits/")
    print("✓ Ready to run: sbatch slurm/retrain_full_pipeline.sh")


if __name__ == "__main__":
    main()
