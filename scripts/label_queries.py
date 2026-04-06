"""
Label generated queries with per-intent confidence scores using the Claude API.

No threshold filtering is applied here. All queries are kept with their raw
scores so that threshold selection, multi-label splitting, and OOD detection
can be done offline during the training phase.

Outputs:
  - labeled_queries.json  -- all queries with raw intent scores and primary intent
  - summary.json          -- score statistics and distribution per intent
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
    """Robustly extract a JSON array from a response that may have prose or fences around it."""
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


# Score bucket boundaries for distribution analysis.
# Buckets: [0.0, 0.1), [0.1, 0.2), ..., [0.8, 0.9), [0.9, 1.0]
BUCKET_EDGES = [round(i * 0.1, 1) for i in range(11)]  # 0.0 to 1.0


def score_bucket(score: float) -> int:
    """Return the bucket index (0-9) for a given score."""
    if score >= 1.0:
        return 9
    return min(int(score * 10), 9)


def bucket_label(idx: int) -> str:
    lo = round(idx * 0.1, 1)
    hi = round(lo + 0.1, 1)
    if idx == 9:
        return f"[{lo:.1f}, 1.0]"
    return f"[{lo:.1f}, {hi:.1f})"


def compute_summary(all_results: list[dict]) -> dict:
    """
    Compute summary statistics over all labeled queries.

    - primary_intent_counts: how many queries have each intent as their primary
    - score_statistics: per-intent mean, std, min, max, median over all queries
    - score_distribution: per-intent query count per 0.1-wide score bucket,
      useful for choosing a threshold offline
    """
    primary_counts: Counter = Counter()
    intent_scores: dict[str, list[float]] = {i: [] for i in INTENTS}
    # Distribution: intent -> list of 10 bucket counts
    distribution: dict[str, list[int]] = {i: [0] * 10 for i in INTENTS}

    for item in all_results:
        primary_counts[item["primary_intent"]] += 1
        for intent in INTENTS:
            score = item["scores"][intent]
            intent_scores[intent].append(score)
            distribution[intent][score_bucket(score)] += 1

    score_statistics = {}
    for intent in INTENTS:
        vals = intent_scores[intent]
        if vals:
            score_statistics[intent] = {
                "mean":   round(mean(vals), 4),
                "std":    round(stdev(vals) if len(vals) > 1 else 0.0, 4),
                "min":    round(min(vals), 4),
                "max":    round(max(vals), 4),
                "median": round(median(vals), 4),
            }

    # Convert distribution lists to labeled dicts for readability
    score_distribution = {
        intent: {bucket_label(idx): count for idx, count in enumerate(counts)}
        for intent, counts in distribution.items()
    }

    return {
        "total": len(all_results),
        "primary_intent_counts": dict(primary_counts),
        "score_statistics": score_statistics,
        "score_distribution": score_distribution,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Label queries with confidence scores via Claude API")
    parser.add_argument("--input", required=True, help="all_generated.json from generate_queries.py")
    parser.add_argument("--output", required=True, help="Output path for labeled queries JSON")
    parser.add_argument("--batch-size", type=int, default=20, help="Queries per API call")
    args = parser.parse_args()

    project_root = Path("/leonardo_scratch/large/userexternal/xmao0000/intent-classifier")
    setup_logging(project_root / "logs" / "label_queries.log")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-batch cache directory for resumability
    cache_dir = output_path.parent / "label_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading queries from %s", args.input)
    all_queries = json.loads(Path(args.input).read_text())
    logger.info("Loaded %d queries", len(all_queries))

    api_key = load_api_key()
    client = anthropic.Anthropic(api_key=api_key)

    n_batches = (len(all_queries) + args.batch_size - 1) // args.batch_size
    all_results: list[dict] = []

    for batch_idx in range(n_batches):
        batch_name = f"label_batch_{batch_idx:04d}"
        cache_path = cache_dir / f"{batch_name}.json"

        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(all_queries))
        batch_queries = all_queries[start:end]

        # Resumability: load from cache if this batch already completed
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                all_results.extend(cached)
                logger.info("Batch %d/%d -- loaded from cache (%d items)", batch_idx + 1, n_batches, len(cached))
                continue
            except Exception as exc:
                logger.warning("Could not load cache for batch %d: %s -- reprocessing", batch_idx, exc)

        queries_block = "\n".join(
            f"{i + 1}. {q['query']}" for i, q in enumerate(batch_queries)
        )
        prompt = LABELING_PROMPT_TEMPLATE.format(queries_block=queries_block)

        logger.info("Labeling batch %d/%d (%d queries)...", batch_idx + 1, n_batches, len(batch_queries))

        try:
            response_text = call_api_with_retry(client, prompt)
            raw_labels = extract_json_array(response_text)
        except Exception as exc:
            logger.error("Batch %d failed: %s -- skipping", batch_idx, exc)
            time.sleep(2)
            continue

        # Match API response back to input queries by position
        batch_results: list[dict] = []
        for idx, item in enumerate(raw_labels):
            if idx >= len(batch_queries):
                break
            if not isinstance(item, dict):
                continue

            query_text = batch_queries[idx]["query"]
            scores = {}
            for intent in INTENTS:
                val = item.get(intent, 0.0)
                try:
                    scores[intent] = float(val)
                except (TypeError, ValueError):
                    scores[intent] = 0.0

            primary = max(INTENTS, key=lambda i: scores[i])

            batch_results.append({
                "query": query_text,
                "scores": scores,
                "primary_intent": primary,
            })

        # Write cache before appending so a crash here is recoverable
        cache_path.write_text(json.dumps(batch_results, indent=2, ensure_ascii=False))
        all_results.extend(batch_results)

        logger.info(
            "Batch %d done -- %d items (total so far: %d)",
            batch_idx + 1, len(batch_results), len(all_results),
        )

        time.sleep(1.5)

    # Write main output
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    logger.info("Saved %d labeled queries to %s", len(all_results), output_path)

    # Write summary
    summary = compute_summary(all_results)
    summary_path = output_path.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Saved summary to %s", summary_path)

    # Print summary to stdout
    print("\n=== Labeling Summary ===")
    print(f"Total queries labeled   : {summary['total']}")
    print("Primary intent counts:")
    for intent in INTENTS:
        print(f"  {intent:<12}: {summary['primary_intent_counts'].get(intent, 0)}")
    print("Per-intent score stats (mean +/- std, median):")
    for intent in INTENTS:
        s = summary["score_statistics"].get(intent, {})
        print(
            f"  {intent:<12}: {s.get('mean', 0):.3f} +/- {s.get('std', 0):.3f}"
            f"  (median {s.get('median', 0):.3f},"
            f"  min {s.get('min', 0):.3f}, max {s.get('max', 0):.3f})"
        )
    print("Score distribution (queries per 0.1 bucket):")
    for intent in INTENTS:
        buckets = summary["score_distribution"].get(intent, {})
        row = "  ".join(f"{v:>5}" for v in buckets.values())
        print(f"  {intent:<12}: {row}")
    labels_row = "  ".join(f"{k:>5}" for k in list(summary["score_distribution"][INTENTS[0]].keys()))
    print(f"  {'buckets':<12}: {labels_row}")


if __name__ == "__main__":
    main()
