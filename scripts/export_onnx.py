"""
Export DeBERTa-v3-xsmall student to ONNX and apply INT8 dynamic quantization.

Steps:
  A. Export ONNX (tries torch.onnx.export first, falls back to optimum)
  B. INT8 dynamic quantization via onnxruntime
  C. Copy tokenizer files
  D. Validate: run test set, compare Macro F1 vs PyTorch baseline
  E. Measure single-query latency (CPU)
  F. Save model_info.json metadata
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

INTENTS = ["recency", "authority", "mechanism", "general"]
NUM_LABELS = len(INTENTS)


# ---------------------------------------------------------------------------
# Model (mirrors distill_student.py)
# ---------------------------------------------------------------------------

class StudentModel(nn.Module):
    def __init__(self, model_dir: str) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_dir, use_safetensors=True, dtype=torch.float32
        )
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, NUM_LABELS)
        head_path = Path(model_dir) / "classifier_head.pt"
        state = torch.load(head_path, weights_only=True, map_location="cpu")
        self.classifier.load_state_dict(state)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# ONNX wrapper (single input dict — easier for torch.onnx.export)
# ---------------------------------------------------------------------------

class ONNXWrapper(nn.Module):
    """Wraps StudentModel to accept positional tensors for torch.onnx.export."""

    def __init__(self, model: StudentModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, attention_mask)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_with_torch(
    model: StudentModel,
    tokenizer,
    output_path: Path,
    max_length: int,
) -> bool:
    """Try torch.onnx.export. Returns True on success."""
    wrapper = ONNXWrapper(model)
    wrapper.eval()

    sample = tokenizer(
        "What is the latest research on periodontal disease?",
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    try:
        logger.info("Attempting torch.onnx.export (opset 14)…")
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (input_ids, attention_mask),
                str(output_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids":      {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits":         {0: "batch_size"},
                },
                opset_version=14,
                do_constant_folding=True,
            )
        logger.info("torch.onnx.export succeeded → %s", output_path)
        return True
    except Exception as exc:
        logger.warning("torch.onnx.export failed: %s", exc)
        return False


def export_with_optimum(model_dir: str, output_dir: Path) -> bool:
    """Fallback: use HuggingFace optimum to export. Returns True on success."""
    try:
        logger.info("Falling back to optimum ORTModelForFeatureExtraction…")
        # optimum export for custom models requires going through the transformers
        # SequenceClassification interface; since we have a custom head, we do a
        # manual export via optimum's lower-level API.
        from optimum.exporters.onnx import main_export
        main_export(
            model_name_or_path=model_dir,
            output=output_dir,
            task="feature-extraction",
            opset=13,
            no_post_process=True,
        )
        # Rename to expected path if needed
        exported = output_dir / "model.onnx"
        if not exported.exists():
            candidates = list(output_dir.glob("*.onnx"))
            if candidates:
                candidates[0].rename(exported)
        logger.info("optimum export succeeded → %s", exported)
        return True
    except Exception as exc:
        logger.error("optimum export also failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_int8(input_path: Path, output_path: Path) -> None:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    logger.info("Applying INT8 dynamic quantization…")
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
    logger.info("Quantization done → %s", output_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_onnx_inference(session, tokenizer, queries: list[str], max_length: int) -> np.ndarray:
    """Run ONNX session on a list of queries. Returns raw logits (N, 4)."""
    all_logits = []
    for query in queries:
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
        logits = session.run(["logits"], ort_inputs)[0]   # (1, 4)
        all_logits.append(logits[0])
    return np.array(all_logits)


def validate(session, tokenizer, test_data: list[dict], max_length: int, threshold: float = 0.5) -> float:
    queries = [d["query"] for d in test_data]
    true_labels = np.array([d["labels"] for d in test_data])

    logits = run_onnx_inference(session, tokenizer, queries, max_length)
    probs  = 1.0 / (1.0 + np.exp(-logits))   # sigmoid
    preds  = (probs >= threshold).astype(int)

    # Fallback: if no label predicted for a row, pick argmax
    for i in range(len(preds)):
        if preds[i].sum() == 0:
            preds[i, np.argmax(probs[i])] = 1

    macro_f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
    return float(macro_f1)


def measure_latency(session, tokenizer, max_length: int, n_runs: int = 100) -> tuple[float, float]:
    query = "What is the latest research on periodontal disease and cardiovascular risk?"
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

    # Warm-up
    for _ in range(5):
        session.run(["logits"], ort_inputs)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(["logits"], ort_inputs)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = float(np.mean(times))
    p95_ms = float(np.percentile(times, 95))
    return avg_ms, p95_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export student model to ONNX + INT8 quantization")
    parser.add_argument("--model-dir",  default="models/student/best_model")
    parser.add_argument("--output-dir", default="models/onnx")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--test-data",  default="data/splits/test.json")
    parser.add_argument("--threshold",  type=float, default=0.5)
    args = parser.parse_args()

    setup_logging()

    output_dir  = Path(args.output_dir)
    model_dir   = Path(args.model_dir)
    onnx_path   = output_dir / "model.onnx"
    quant_path  = output_dir / "model_quantized.onnx"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step A: Load PyTorch model and export ONNX
    # ------------------------------------------------------------------
    logger.info("Loading PyTorch student from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = StudentModel(str(model_dir))
    model.eval()

    success = export_with_torch(model, tokenizer, onnx_path, args.max_length)
    if not success:
        success = export_with_optimum(str(model_dir), output_dir)
    if not success:
        logger.error("All ONNX export methods failed. Aborting.")
        sys.exit(1)

    size_fp32_mb = onnx_path.stat().st_size / 1024 / 1024
    logger.info("ONNX model size: %.1f MB", size_fp32_mb)

    # ------------------------------------------------------------------
    # Step B: INT8 dynamic quantization
    # ------------------------------------------------------------------
    quantize_int8(onnx_path, quant_path)
    size_int8_mb = quant_path.stat().st_size / 1024 / 1024
    logger.info("Quantized model size: %.1f MB (compression %.1fx)", size_int8_mb, size_fp32_mb / size_int8_mb)

    # ------------------------------------------------------------------
    # Step C: Save tokenizer
    # ------------------------------------------------------------------
    logger.info("Saving tokenizer to %s", output_dir)
    tokenizer.save_pretrained(str(output_dir))

    # ------------------------------------------------------------------
    # Step D: Validate on test set
    # ------------------------------------------------------------------
    import onnxruntime as ort

    logger.info("Loading quantized model for validation…")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(quant_path), sess_options=sess_opts,
                                   providers=["CPUExecutionProvider"])

    logger.info("Loading test data from %s", args.test_data)
    test_data = json.loads(Path(args.test_data).read_text(encoding="utf-8"))
    logger.info("Test samples: %d", len(test_data))

    logger.info("Running ONNX inference on test set…")
    onnx_f1 = validate(session, tokenizer, test_data, args.max_length, args.threshold)
    logger.info("ONNX Macro F1: %.4f", onnx_f1)

    pytorch_f1 = 0.9494  # from eval_results.json
    gap = pytorch_f1 - onnx_f1
    logger.info("PyTorch Macro F1 : %.4f", pytorch_f1)
    logger.info("ONNX Macro F1    : %.4f", onnx_f1)
    logger.info("Quantization gap : %.4f (%.2f%%)", gap, gap * 100)
    if gap > 0.01:
        logger.warning(
            "F1 gap %.4f exceeds 1%% — consider static INT8 quantization with calibration data.",
            gap,
        )

    # ------------------------------------------------------------------
    # Step E: Latency benchmark
    # ------------------------------------------------------------------
    logger.info("Benchmarking single-query latency (100 runs)…")
    avg_ms, p95_ms = measure_latency(session, tokenizer, args.max_length)
    logger.info("Avg latency: %.2f ms | p95: %.2f ms", avg_ms, p95_ms)

    # ------------------------------------------------------------------
    # Step F: Save metadata
    # ------------------------------------------------------------------
    model_info = {
        "model_name":            "deberta-v3-xsmall-intent-classifier",
        "base_model":            "microsoft/deberta-v3-xsmall",
        "num_labels":            NUM_LABELS,
        "label_names":           INTENTS,
        "max_length":            args.max_length,
        "quantization":          "INT8 dynamic",
        "onnx_opset":            14,
        "model_size_mb":         round(size_fp32_mb, 2),
        "quantized_size_mb":     round(size_int8_mb, 2),
        "compression_ratio":     round(size_fp32_mb / size_int8_mb, 2),
        "avg_latency_ms":        round(avg_ms, 2),
        "p95_latency_ms":        round(p95_ms, 2),
        "test_macro_f1_pytorch": pytorch_f1,
        "test_macro_f1_onnx":    round(onnx_f1, 4),
        "quantization_gap":      round(gap, 4),
        "threshold":             args.threshold,
    }
    info_path = output_dir / "model_info.json"
    info_path.write_text(json.dumps(model_info, indent=2), encoding="utf-8")
    logger.info("Saved model_info.json → %s", info_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== ONNX Export Summary ===")
    print(f"ONNX model size      : {size_fp32_mb:.1f} MB")
    print(f"Quantized size       : {size_int8_mb:.1f} MB  ({size_fp32_mb / size_int8_mb:.1f}x compression)")
    print(f"PyTorch Macro F1     : {pytorch_f1:.4f}")
    print(f"ONNX Macro F1        : {onnx_f1:.4f}")
    print(f"Quantization gap     : {gap:+.4f}  ({gap * 100:+.2f}%)")
    print(f"Avg latency (CPU)    : {avg_ms:.2f} ms")
    print(f"P95 latency (CPU)    : {p95_ms:.2f} ms")
    print(f"\nOutput directory     : {output_dir.resolve()}")
    print(f"  model.onnx         : {size_fp32_mb:.1f} MB  (FP32)")
    print(f"  model_quantized.onnx: {size_int8_mb:.1f} MB  (INT8 — use this for deployment)")
    print(f"  model_info.json    : metadata")


if __name__ == "__main__":
    main()
