"""
Generate supplementary queries to augment the existing labeled dataset.

Loads existing labeled queries, builds a normalized dedup set, then calls
the Claude API until `--target` NEW (non-duplicate) queries are collected.

Resumable: each batch is saved to data/supplementary/batch_NNNN.json.
On restart, existing batches are loaded and their queries added to the dedup
set before continuing generation.
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import anthropic

# Shared dedup utility
sys.path.insert(0, str(Path(__file__).parent))
from utils import normalize_query

logger = logging.getLogger(__name__)

INTENTS = ["recency", "authority", "mechanism", "general"]

PROMPT_TEMPLATE = """You are generating synthetic search queries for a PubMed scientific literature retrieval system.

Given these real PubMed article topics and example queries, generate {batch_size} NEW and DIVERSE queries that a biomedical researcher might type into a search engine.

INTENT CATEGORIES:
- recency: Seeking the latest, most recent, or emerging research
- authority: Seeking landmark, highly-cited, authoritative, or guideline papers
- mechanism: Seeking molecular, biological, or pathophysiological mechanisms
- general: Broad overview or general information queries

RULES:
1. Each query must be a natural English question a researcher would ask
2. Cover diverse biomedical topics — use the provided topics for INSPIRATION but create original questions
3. About 20% of queries should have MULTIPLE intents (e.g., recency+authority, recency+mechanism)
4. Vary sentence structure: use "What", "How", "Which", "Who", "Are there", "Does", "Can", "Is there evidence" etc.
5. Vary length: some short (5-10 words), some medium (10-20 words), some long (20+ words)
6. Do NOT simply rephrase the example queries — create genuinely new queries on different specific topics
7. Aim for roughly equal distribution across the 4 intents (but natural variation is fine)
8. Make queries specific enough to be realistic PubMed searches (use real gene names, drug names, disease names, techniques etc.)
9. Do NOT generate queries about the following topics: {avoid_topics}. Focus on less common biomedical topics.

REAL PUBMED TOPICS FOR INSPIRATION:
{topics_block}

EXAMPLE QUERIES (for style reference only — generate NEW topics):
{examples_block}

OUTPUT FORMAT: Return ONLY a valid JSON array, no markdown fences, no explanation. Each element must be:
{{"query": "...", "primary_intent": "recency|authority|mechanism|general", "all_intents": ["recency", ...]}}"""


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def load_api_key() -> str:
    key_path = Path("/leonardo_scratch/large/userexternal/xmao0000/intent-classifier/.anthropic_key")
    return key_path.read_text().strip()


def extract_json_array(text: str) -> list:
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON array from response: {text[:300]!r}")


def format_topics_block(topics: list[dict], n: int, rng: random.Random) -> str:
    sample = rng.sample(topics, min(n, len(topics)))
    lines = []
    for t in sample:
        parts = [f"Title: {t['title']}"]
        if t.get("keywords"):
            parts.append(f"Keywords: {', '.join(t['keywords'][:8])}")
        if t.get("mesh_terms"):
            parts.append(f"MeSH: {', '.join(t['mesh_terms'][:5])}")
        if t.get("subjects"):
            parts.append(f"Subjects: {', '.join(t['subjects'][:3])}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def format_examples_block(seeds: list[dict], n: int, rng: random.Random) -> str:
    by_intent: dict[str, list[dict]] = {i: [] for i in INTENTS}
    for s in seeds:
        intent = s.get("intent", "general")
        if intent in by_intent:
            by_intent[intent].append(s)
    chosen = []
    for intent_list in by_intent.values():
        if intent_list:
            chosen.append(rng.choice(intent_list))
    remaining = [s for s in seeds if s not in chosen]
    extra_count = max(0, n - len(chosen))
    if extra_count and remaining:
        chosen.extend(rng.sample(remaining, min(extra_count, len(remaining))))
    return "\n".join(f"[{s.get('intent', 'general')}] {s['query']}" for s in chosen)


def sample_avoid_topics(existing_queries: list[dict], rng: random.Random, n: int = 10) -> str:
    """Extract keyword snippets from existing queries to discourage repetition."""
    sample = rng.sample(existing_queries, min(n, len(existing_queries)))
    # Take first 4-6 words of each query as a topic hint
    snippets = []
    for q in sample:
        text = q.get("query", "")
        words = text.split()[:6]
        snippets.append(" ".join(words))
    return "; ".join(snippets)


def call_api_with_retry(client: anthropic.Anthropic, prompt: str, max_retries: int = 3) -> str:
    delay = 5.0
    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except anthropic.RateLimitError as exc:
            logger.warning("Rate limit (attempt %d/%d): %s — sleeping %.1fs", attempt, max_retries, exc, delay)
            time.sleep(delay)
            delay *= 2
        except anthropic.APIStatusError as exc:
            logger.warning("API error (attempt %d/%d): %s — sleeping %.1fs", attempt, max_retries, exc, delay)
            time.sleep(delay)
            delay *= 2
        except Exception as exc:
            logger.warning("Unexpected error (attempt %d/%d): %s — sleeping %.1fs", attempt, max_retries, exc, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"All {max_retries} API attempts failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate supplementary queries, deduped against existing data")
    parser.add_argument("--existing",   required=True, help="Path to existing labeled_queries.json")
    parser.add_argument("--topics",     required=True, help="Topics JSON from extract_topics.py")
    parser.add_argument("--seeds",      required=True, help="Seed queries JSON")
    parser.add_argument("--output",     required=True, help="Output path for new_queries.json")
    parser.add_argument("--target",     type=int, default=4000, help="Number of NEW queries to collect")
    parser.add_argument("--batch-size", type=int, default=25, help="Queries per API call")
    parser.add_argument("--seed",       type=int, default=123, help="Random seed (different from original to vary generation)")
    args = parser.parse_args()

    output_path = Path(args.output)
    batch_dir   = output_path.parent
    batch_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path("/leonardo_scratch/large/userexternal/xmao0000/intent-classifier")
    setup_logging(project_root / "logs" / "generate_supplementary.log")

    # ------------------------------------------------------------------
    # Build dedup set from existing labeled data
    # ------------------------------------------------------------------
    logger.info("Loading existing data from %s", args.existing)
    existing_records = json.loads(Path(args.existing).read_text(encoding="utf-8"))
    logger.info("Existing records: %d", len(existing_records))

    seen_normalized: set[str] = set()
    for r in existing_records:
        seen_normalized.add(normalize_query(r["query"]))
    logger.info("Unique normalized queries in existing data: %d", len(seen_normalized))

    # ------------------------------------------------------------------
    # Load topics and seeds
    # ------------------------------------------------------------------
    topics = json.loads(Path(args.topics).read_text(encoding="utf-8"))
    seeds  = json.loads(Path(args.seeds).read_text(encoding="utf-8"))
    logger.info("Topics: %d | Seeds: %d", len(topics), len(seeds))

    api_key = load_api_key()
    client  = anthropic.Anthropic(api_key=api_key)
    rng     = random.Random(args.seed)

    # ------------------------------------------------------------------
    # Load already-completed supplementary batches (resumability)
    # ------------------------------------------------------------------
    new_queries: list[dict] = []
    existing_batch_files = sorted(batch_dir.glob("batch_*.json"))
    total_generated = 0
    total_discarded = 0

    for bf in existing_batch_files:
        try:
            batch_data = json.loads(bf.read_text(encoding="utf-8"))
            for item in batch_data:
                nq = normalize_query(item["query"])
                if nq not in seen_normalized:
                    seen_normalized.add(nq)
                    new_queries.append(item)
                else:
                    total_discarded += 1
                total_generated += 1
        except Exception as exc:
            logger.warning("Could not load batch file %s: %s", bf, exc)

    logger.info(
        "Resumed: %d batches loaded, %d new queries so far (%d discarded as duplicates)",
        len(existing_batch_files), len(new_queries), total_discarded,
    )

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------
    batch_idx = len(existing_batch_files)
    intent_counter: Counter = Counter()
    for q in new_queries:
        for intent in q.get("all_intents", [q.get("primary_intent", "general")]):
            intent_counter[intent] += 1

    while len(new_queries) < args.target:
        need = args.target - len(new_queries)
        this_batch = min(args.batch_size, need + 10)  # generate a few extra to absorb dedup losses

        avoid_topics = sample_avoid_topics(existing_records + new_queries, rng, n=10)
        topics_block  = format_topics_block(topics, n=rng.randint(5, 8), rng=rng)
        examples_block = format_examples_block(seeds, n=rng.randint(4, 6), rng=rng)
        prompt = PROMPT_TEMPLATE.format(
            batch_size=this_batch,
            avoid_topics=avoid_topics,
            topics_block=topics_block,
            examples_block=examples_block,
        )

        logger.info(
            "Batch %d — requesting %d queries (need %d more new)…",
            batch_idx + 1, this_batch, need,
        )

        batch_path = batch_dir / f"batch_{batch_idx:04d}.json"

        try:
            response_text = call_api_with_retry(client, prompt)
            raw_batch     = extract_json_array(response_text)
        except Exception as exc:
            logger.error("Batch %d failed: %s — skipping", batch_idx + 1, exc)
            time.sleep(2)
            batch_idx += 1
            continue

        # Validate items
        valid_batch: list[dict] = []
        for item in raw_batch:
            if not isinstance(item, dict) or "query" not in item:
                continue
            item.setdefault("primary_intent", "general")
            if "all_intents" not in item or not isinstance(item["all_intents"], list):
                item["all_intents"] = [item["primary_intent"]]
            item["all_intents"] = [i for i in item["all_intents"] if i in INTENTS] or ["general"]
            if item["primary_intent"] not in INTENTS:
                item["primary_intent"] = item["all_intents"][0]
            valid_batch.append(item)

        # Save raw batch (before dedup) for resumability
        batch_path.write_text(json.dumps(valid_batch, indent=2, ensure_ascii=False), encoding="utf-8")

        # Dedup against seen set
        batch_new = 0
        batch_dup = 0
        for item in valid_batch:
            nq = normalize_query(item["query"])
            total_generated += 1
            if nq in seen_normalized:
                total_discarded += 1
                batch_dup += 1
            else:
                seen_normalized.add(nq)
                new_queries.append(item)
                for intent in item["all_intents"]:
                    intent_counter[intent] += 1
                batch_new += 1

        logger.info(
            "Batch %d done — %d new, %d duplicates discarded (total new: %d / %d)",
            batch_idx + 1, batch_new, batch_dup, len(new_queries), args.target,
        )

        batch_idx += 1
        time.sleep(1.5)

    # Trim to exact target if we overshot
    new_queries = new_queries[:args.target]

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    output_path.write_text(json.dumps(new_queries, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %d new queries → %s", len(new_queries), output_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    dup_rate = 100 * total_discarded / max(total_generated, 1)
    print("\n=== Generation Summary ===")
    print(f"Total API calls made    : {batch_idx}")
    print(f"Total queries generated : {total_generated}")
    print(f"Duplicates discarded    : {total_discarded} ({dup_rate:.1f}%)")
    print(f"New queries retained    : {len(new_queries)}")
    print("Intent distribution (new queries):")
    for intent in INTENTS:
        print(f"  {intent:<12}: {intent_counter[intent]}")
    multi = sum(1 for q in new_queries if len(q.get("all_intents", [])) > 1)
    print(f"Multi-label             : {multi} ({100 * multi / max(len(new_queries), 1):.1f}%)")


if __name__ == "__main__":
    main()
