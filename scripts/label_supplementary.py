"""
Label supplementary queries with per-intent confidence scores using the Claude API.

Same logic and prompt as label_queries.py, but reads from a different input file
and writes to a separate output/cache directory so it doesn't conflict with the
original labeling run.

Output format matches data/labeled/labeled_queries.json exactly:
  [{"query": "...", "scores": {...}, "primary_intent": "..."}, ...]
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

import anthropic

logger = logging.getLogger(__name__)

INTENTS = ["recency", "authority", "mechanism", "general"]

LABELING_PROMPT_TEMPLATE = """You are an expert annotator for a PubMed literature search intent classifier.

For each query below, rate the presence of each intent on a scale from 0.0 to 1.0:

INTENT DEFINITIONS:
- recency (0.0-1.0): How much does this query seek recent/latest/newest/emerging research?
- authority (0.0-1.0): How much does this query seek landmark/authoritative/highly-cited/guideline work?
- mechanism (0.0-1.0): How much does this query seek molecular/biological/pathophysiological mechanisms or pathways?
- general (0.0-1.0): How much is this a broad overview or general information query?

SCORING GUIDE:
- 0.9-1.0: Very strong, unambiguous signal for this intent
- 0.7-0.8: Clear signal present
- 0.4-0.6: Ambiguous or weak signal
- 0.1-0.3: Minimal signal
- 0.0: No signal at all

IMPORTANT: A query CAN score high on MULTIPLE intents. Score each intent independently.
Example: "What are the latest landmark studies on X?" -> recency: 0.9, authority: 0.9, mechanism: 0.1, general: 0.2

QUERIES TO LABEL:
{queries_block}

OUTPUT FORMAT: Return ONLY a valid JSON array, no markdown fences, no explanation. Each element must be:
{{"query": "...", "recency": 0.0, "authority": 0.0, "mechanism": 0.0, "general": 0.0}}"""


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
    end   = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON array from response: {text[:300]!r}")


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
            logger.warning("Rate limit (attempt %d/%d): %s -- sleeping %.1fs", attempt, max_retries, exc, delay)
            time.sleep(delay)
            delay *= 2
        except anthropic.APIStatusError as exc:
            logger.warning("API error (attempt %d/%d): %s -- sleeping %.1fs", attempt, max_retries, exc, delay)
            time.sleep(delay)
            delay *= 2
        except Exception as exc:
            logger.warning("Unexpected error (attempt %d/%d): %s -- sleeping %.1fs", attempt, max_retries, exc, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"All {max_retries} API attempts failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Label supplementary queries via Claude API")
    parser.add_argument("--input",      required=True, help="new_queries.json from generate_supplementary.py")
    parser.add_argument("--output",     required=True, help="Output path for labeled_new.json")
    parser.add_argument("--batch-size", type=int, default=20, help="Queries per API call")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    project_root = Path("/leonardo_scratch/large/userexternal/xmao0000/intent-classifier")
    setup_logging(project_root / "logs" / "label_supplementary.log")

    # Per-batch cache in same directory as output, separate from original label_cache
    cache_dir = output_path.parent / "label_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading queries from %s", args.input)
    all_queries = json.loads(Path(args.input).read_text(encoding="utf-8"))
    logger.info("Loaded %d queries to label", len(all_queries))

    api_key = load_api_key()
    client  = anthropic.Anthropic(api_key=api_key)

    n_batches  = (len(all_queries) + args.batch_size - 1) // args.batch_size
    all_results: list[dict] = []

    for batch_idx in range(n_batches):
        batch_name = f"label_batch_{batch_idx:04d}"
        cache_path = cache_dir / f"{batch_name}.json"

        start = batch_idx * args.batch_size
        end   = min(start + args.batch_size, len(all_queries))
        batch_queries = all_queries[start:end]

        # Resumability
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                all_results.extend(cached)
                logger.info(
                    "Batch %d/%d -- loaded from cache (%d items)", batch_idx + 1, n_batches, len(cached)
                )
                continue
            except Exception as exc:
                logger.warning("Could not load cache for batch %d: %s -- reprocessing", batch_idx, exc)

        queries_block = "\n".join(f"{i + 1}. {q['query']}" for i, q in enumerate(batch_queries))
        prompt = LABELING_PROMPT_TEMPLATE.format(queries_block=queries_block)

        logger.info("Labeling batch %d/%d (%d queries)…", batch_idx + 1, n_batches, len(batch_queries))

        try:
            response_text = call_api_with_retry(client, prompt)
            raw_labels    = extract_json_array(response_text)
        except Exception as exc:
            logger.error("Batch %d failed: %s -- skipping", batch_idx, exc)
            time.sleep(2)
            continue

        batch_results: list[dict] = []
        for idx, item in enumerate(raw_labels):
            if idx >= len(batch_queries):
                break
            if not isinstance(item, dict):
                continue
            query_text = batch_queries[idx]["query"]
            scores = {}
            for intent in INTENTS:
                try:
                    scores[intent] = float(item.get(intent, 0.0))
                except (TypeError, ValueError):
                    scores[intent] = 0.0
            primary = max(INTENTS, key=lambda i: scores[i])
            batch_results.append({
                "query":          query_text,
                "scores":         scores,
                "primary_intent": primary,
            })

        cache_path.write_text(json.dumps(batch_results, indent=2, ensure_ascii=False), encoding="utf-8")
        all_results.extend(batch_results)

        logger.info(
            "Batch %d done -- %d items (total so far: %d / %d)",
            batch_idx + 1, len(batch_results), len(all_results), len(all_queries),
        )
        time.sleep(1.5)

    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %d labeled queries → %s", len(all_results), output_path)

    # Summary
    primary_counts: Counter = Counter(r["primary_intent"] for r in all_results)
    print("\n=== Labeling Summary ===")
    print(f"Total queries labeled : {len(all_results)}")
    print("Primary intent counts:")
    for intent in INTENTS:
        print(f"  {intent:<12}: {primary_counts.get(intent, 0)}")

    intent_scores: dict[str, list[float]] = {i: [] for i in INTENTS}
    for r in all_results:
        for intent in INTENTS:
            intent_scores[intent].append(r["scores"][intent])
    print("Per-intent score stats (mean ± std, median):")
    for intent in INTENTS:
        vals = intent_scores[intent]
        if vals:
            print(
                f"  {intent:<12}: {mean(vals):.3f} ± {stdev(vals) if len(vals) > 1 else 0:.3f}"
                f"  (median {median(vals):.3f})"
            )


if __name__ == "__main__":
    main()
