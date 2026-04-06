"""
Generate synthetic PubMed search queries using the Claude API.

Uses real PubMed topics (from extract_topics.py) as inspiration and
seed queries as few-shot examples. Saves each batch as a separate JSON file
so the job can be resumed if it times out.
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
    """Robustly extract a JSON array from a response that may have prose or fences around it."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Find first '[' and last ']'
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
    # Ensure at least one per intent
    by_intent: dict[str, list[dict]] = {i: [] for i in INTENTS}
    for s in seeds:
        intent = s.get("intent", "general")
        if intent in by_intent:
            by_intent[intent].append(s)

    chosen = []
    for intent_list in by_intent.values():
        if intent_list:
            chosen.append(rng.choice(intent_list))

    # Fill remaining slots from all seeds
    remaining = [s for s in seeds if s not in chosen]
    extra_count = max(0, n - len(chosen))
    if extra_count and remaining:
        chosen.extend(rng.sample(remaining, min(extra_count, len(remaining))))

    lines = []
    for s in chosen:
        lines.append(f"[{s.get('intent', 'general')}] {s['query']}")
    return "\n".join(lines)


def call_api_with_retry(
    client: anthropic.Anthropic,
    prompt: str,
    max_retries: int = 3,
) -> str:
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
    parser = argparse.ArgumentParser(description="Generate synthetic PubMed queries via Claude API")
    parser.add_argument("--topics", required=True, help="Topics JSON from extract_topics.py")
    parser.add_argument("--seeds", required=True, help="Seed queries JSON")
    parser.add_argument("--output-dir", required=True, help="Directory to write batch files")
    parser.add_argument("--total", type=int, default=15000, help="Target total queries to generate")
    parser.add_argument("--batch-size", type=int, default=25, help="Queries per API call")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    project_root = Path("/leonardo_scratch/large/userexternal/xmao0000/intent-classifier")
    setup_logging(project_root / "logs" / "generate_queries.log")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading topics from %s", args.topics)
    topics = json.loads(Path(args.topics).read_text())
    logger.info("Loaded %d topics", len(topics))

    logger.info("Loading seed queries from %s", args.seeds)
    seeds = json.loads(Path(args.seeds).read_text())
    logger.info("Loaded %d seed queries", len(seeds))

    api_key = load_api_key()
    client = anthropic.Anthropic(api_key=api_key)

    rng = random.Random(args.seed)
    n_batches = (args.total + args.batch_size - 1) // args.batch_size

    # Discover already-completed batches for resumability
    existing_batches = {p.stem for p in output_dir.glob("batch_*.json")}
    logger.info("Found %d existing batch files — will skip them", len(existing_batches))

    all_queries: list[dict] = []
    intent_counter: Counter = Counter()
    completed = 0

    for batch_idx in range(n_batches):
        batch_name = f"batch_{batch_idx:04d}"
        batch_path = output_dir / f"{batch_name}.json"

        if batch_name in existing_batches:
            # Load existing batch and count its queries
            try:
                batch_data = json.loads(batch_path.read_text())
                all_queries.extend(batch_data)
                for q in batch_data:
                    for intent in q.get("all_intents", [q.get("primary_intent", "general")]):
                        intent_counter[intent] += 1
                completed += 1
                continue
            except Exception as exc:
                logger.warning("Could not load existing batch %s: %s — regenerating", batch_name, exc)

        # Determine how many queries to generate in this batch
        remaining_queries = args.total - len(all_queries)
        if remaining_queries <= 0:
            logger.info("Reached target of %d queries — stopping", args.total)
            break
        this_batch_size = min(args.batch_size, remaining_queries)

        topics_block = format_topics_block(topics, n=rng.randint(5, 8), rng=rng)
        examples_block = format_examples_block(seeds, n=rng.randint(4, 6), rng=rng)
        prompt = PROMPT_TEMPLATE.format(
            batch_size=this_batch_size,
            topics_block=topics_block,
            examples_block=examples_block,
        )

        logger.info("Batch %d/%d — requesting %d queries…", batch_idx + 1, n_batches, this_batch_size)

        try:
            response_text = call_api_with_retry(client, prompt)
            batch_queries = extract_json_array(response_text)
        except Exception as exc:
            logger.error("Batch %d failed: %s — skipping", batch_idx, exc)
            time.sleep(2)
            continue

        # Validate and normalise
        valid: list[dict] = []
        for item in batch_queries:
            if not isinstance(item, dict) or "query" not in item:
                continue
            item.setdefault("primary_intent", "general")
            if "all_intents" not in item or not isinstance(item["all_intents"], list):
                item["all_intents"] = [item["primary_intent"]]
            # Clamp to known intents
            item["all_intents"] = [i for i in item["all_intents"] if i in INTENTS] or ["general"]
            if item["primary_intent"] not in INTENTS:
                item["primary_intent"] = item["all_intents"][0]
            valid.append(item)

        batch_path.write_text(json.dumps(valid, indent=2, ensure_ascii=False))
        all_queries.extend(valid)
        for q in valid:
            for intent in q["all_intents"]:
                intent_counter[intent] += 1

        completed += 1
        logger.info(
            "Batch %d done — got %d valid queries (total so far: %d)",
            batch_idx + 1, len(valid), len(all_queries),
        )

        # Rate limiting
        time.sleep(1.5)

    # Merge all batches into a single file
    merged_path = output_dir / "all_generated.json"
    merged_path.write_text(json.dumps(all_queries, indent=2, ensure_ascii=False))
    logger.info("Saved %d queries to %s", len(all_queries), merged_path)

    # Summary
    print("\n=== Generation Summary ===")
    print(f"Total queries generated : {len(all_queries)}")
    print(f"Batches completed       : {completed}")
    print("Intent distribution:")
    for intent in INTENTS:
        print(f"  {intent:<12}: {intent_counter[intent]}")
    multi_label = sum(1 for q in all_queries if len(q.get("all_intents", [])) > 1)
    print(f"Multi-label queries     : {multi_label} ({100 * multi_label / max(len(all_queries), 1):.1f}%)")


if __name__ == "__main__":
    main()
