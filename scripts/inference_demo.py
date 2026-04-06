"""
Inference demo for the ONNX intent classifier.

Usage:
  # Demo with built-in queries:
  python scripts/inference_demo.py

  # Single query:
  python scripts/inference_demo.py --query "What is the latest research on EGFR mutations?"

  # Batch from file (JSON array of query strings, or one per line):
  python scripts/inference_demo.py --queries-file my_queries.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

INTENTS = ["recency", "authority", "mechanism", "general"]

DEMO_QUERIES = [
    "What is the latest research on periodontal disease and cardiovascular risk?",
    "Which landmark papers established the link between chronic inflammation and atherosclerosis?",
    "How does SIRT3 protect against diabetic cardiomyopathy at the mitochondrial level?",
    "What dietary interventions are recommended for cardiovascular disease prevention?",
    "What are the latest landmark studies on EGFR mutations in lung cancer?",
]


def load_model(model_dir: str):
    """Load tokenizer and ONNX InferenceSession from model_dir."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    model_path = Path(model_dir)
    quant_path  = model_path / "model_quantized.onnx"
    plain_path  = model_path / "model.onnx"

    onnx_file = quant_path if quant_path.exists() else plain_path
    if not onnx_file.exists():
        print(f"ERROR: No ONNX model found in {model_dir}", file=sys.stderr)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 1   # deterministic single-threaded latency
    session = ort.InferenceSession(
        str(onnx_file),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    # Load max_length from model_info if available
    max_length = 128
    info_path = model_path / "model_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        max_length = info.get("max_length", 128)

    return tokenizer, session, max_length, str(onnx_file)


def predict(tokenizer, session, query: str, max_length: int) -> tuple[dict, float]:
    """
    Run inference on a single query.
    Returns (scores_dict, latency_ms).
    scores_dict maps intent -> sigmoid probability.
    """
    enc = tokenizer(
        query,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    ort_inputs = {
        "input_ids":      enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }

    t0 = time.perf_counter()
    logits = session.run(["logits"], ort_inputs)[0][0]   # shape (4,)
    latency_ms = (time.perf_counter() - t0) * 1000

    probs = 1.0 / (1.0 + np.exp(-logits))   # sigmoid
    scores = {intent: float(probs[i]) for i, intent in enumerate(INTENTS)}
    return scores, latency_ms


def format_result(query: str, scores: dict, threshold: float, latency_ms: float) -> str:
    lines = [f'Query: "{query}"', "", "Intent scores:"]
    for intent in INTENTS:
        lines.append(f"  {intent:<12}: {scores[intent]:.2f}")

    primary = max(INTENTS, key=lambda i: scores[i])
    active  = [i for i in INTENTS if scores[i] >= threshold]

    lines.append("")
    lines.append(f"Primary intent: {primary}")
    lines.append(
        f"Active intents (threshold={threshold}): "
        + (", ".join(active) if active else "none")
    )

    if not active:
        lines.append("WARNING: Possible OOD query — no intent exceeds threshold")

    lines.append(f"Inference time: {latency_ms:.1f}ms")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Intent classifier inference demo")
    parser.add_argument("--model-dir",     default="models/onnx",
                        help="Directory with ONNX model and tokenizer")
    parser.add_argument("--query",         default=None,
                        help="Single query string")
    parser.add_argument("--queries-file",  default=None,
                        help="JSON file with list of query strings")
    parser.add_argument("--threshold",     type=float, default=0.5,
                        help="Multi-label threshold (default: 0.5)")
    args = parser.parse_args()

    tokenizer, session, max_length, onnx_file = load_model(args.model_dir)
    print(f"Loaded model: {onnx_file}\n")

    # Collect queries
    if args.query:
        queries = [args.query]
    elif args.queries_file:
        raw = json.loads(Path(args.queries_file).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            queries = [str(q) for q in raw]
        else:
            print("ERROR: --queries-file must contain a JSON array of strings", file=sys.stderr)
            sys.exit(1)
    else:
        print("No query provided — running built-in demo queries.\n")
        queries = DEMO_QUERIES

    # Warm-up run
    predict(tokenizer, session, "warm up", max_length)

    for query in queries:
        scores, latency_ms = predict(tokenizer, session, query, max_length)
        print(format_result(query, scores, args.threshold, latency_ms))
        print("-" * 60)


if __name__ == "__main__":
    main()
