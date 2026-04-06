"""
Accurate single-threaded CPU latency benchmark for the ONNX intent classifier.

Uses intra_op_num_threads=1 and inter_op_num_threads=1 to avoid thread
scheduling noise from HPC CPU affinity restrictions.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

QUERIES = {
    "short  (~5 words)":  "How to treat hypertension?",
    "medium (~15 words)": "What are the latest findings on gut microbiota and cardiovascular disease?",
    "long   (~25 words)": (
        "What are the most recent randomized controlled trials evaluating the efficacy "
        "of immunotherapy combinations for advanced non-small cell lung cancer?"
    ),
}

WARMUP_RUNS = 50
BENCH_RUNS  = 500


def load_session(model_dir: str):
    import onnxruntime as ort
    model_path = Path(model_dir)
    quant = model_path / "model_quantized.onnx"
    plain = model_path / "model.onnx"
    onnx_file = quant if quant.exists() else plain
    if not onnx_file.exists():
        print(f"ERROR: No ONNX model found in {model_dir}", file=sys.stderr)
        sys.exit(1)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_file), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    return session, str(onnx_file)


def bench_query(session, tokenizer, query: str, max_length: int) -> dict:
    enc = tokenizer(
        query,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    ort_inputs = {
        "input_ids":      enc["input_ids"].astype("int64"),
        "attention_mask": enc["attention_mask"].astype("int64"),
    }

    # Warmup
    for _ in range(WARMUP_RUNS):
        session.run(["logits"], ort_inputs)

    # Benchmark
    times = []
    for _ in range(BENCH_RUNS):
        t0 = time.perf_counter()
        session.run(["logits"], ort_inputs)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "mean":   float(np.mean(times)),
        "median": float(np.median(times)),
        "p95":    float(np.percentile(times, 95)),
        "p99":    float(np.percentile(times, 99)),
        "min":    float(np.min(times)),
        "max":    float(np.max(times)),
        "tokens": enc["input_ids"].shape[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-threaded CPU latency benchmark")
    parser.add_argument("--model-dir", default="models/onnx")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    model_dir = args.model_dir
    session, onnx_file = load_session(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    max_length = 128
    info_path = Path(model_dir) / "model_info.json"
    if info_path.exists():
        max_length = json.loads(info_path.read_text())["max_length"]

    print(f"Model : {onnx_file}")
    print(f"Config: intra_op_threads=1, inter_op_threads=1, warmup={WARMUP_RUNS}, runs={BENCH_RUNS}")
    print()

    results = {}
    for label, query in QUERIES.items():
        print(f"Benchmarking [{label}]")
        print(f"  Query: \"{query[:60]}{'…' if len(query) > 60 else ''}\"")
        stats = bench_query(session, tokenizer, query, max_length)
        results[label] = stats
        print(f"  Mean={stats['mean']:.2f}ms  Median={stats['median']:.2f}ms  "
              f"P95={stats['p95']:.2f}ms  P99={stats['p99']:.2f}ms  "
              f"Min={stats['min']:.2f}ms  Max={stats['max']:.2f}ms")
        print()

    # Summary table
    print("=" * 72)
    print(f"{'Query length':<22} {'Mean':>7} {'Median':>8} {'P95':>7} {'P99':>7} {'Min':>7} {'Max':>7}")
    print("-" * 72)
    for label, stats in results.items():
        print(f"{label:<22} {stats['mean']:>6.2f}ms {stats['median']:>7.2f}ms "
              f"{stats['p95']:>6.2f}ms {stats['p99']:>6.2f}ms "
              f"{stats['min']:>6.2f}ms {stats['max']:>6.2f}ms")
    print("=" * 72)


if __name__ == "__main__":
    main()
