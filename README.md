# Intent Classifier for PubMed Retrieval

Multi-label intent classifier for biomedical search queries. Given a free-text query, the model assigns independent confidence scores for four intents: `recency`, `authority`, `mechanism`, `general`.

**Deployment model:** DeBERTa-v3-xsmall (INT8 quantized ONNX) — 83 MB, ~39 ms CPU latency, macro-F1 = 0.945.

## Model Performance

| Metric | Value |
|---|---|
| Test macro-F1 (PyTorch) | 0.9494 |
| Test macro-F1 (ONNX INT8) | 0.9451 |
| Quantization F1 gap | 0.0043 |
| Mean CPU latency (single-thread) | ~39 ms |
| P95 CPU latency | ~40 ms |
| Quantized model size | 83 MB |

Latency measured on a single CPU core (intra/inter_op_threads=1), 500 runs after 50 warmup iterations, query lengths 5–25 words.

## Quick Start

```python
from scripts.inference_demo import load_model, predict

session, tokenizer = load_model("models/onnx/model_quantized.onnx")
scores = predict(session, tokenizer, "What are the latest RCTs on SGLT2 inhibitors?")
# {'recency': 1.0, 'authority': 0.0, 'mechanism': 0.98, 'general': 0.0}
```

Or from the command line:

```bash
python scripts/inference_demo.py --model models/onnx/model_quantized.onnx \
    --query "What are the latest RCTs on SGLT2 inhibitors?"
```

## Directory Structure

```
intent-classifier/
├── data/
│   ├── queries_seed.json                        # 30 seed queries (supervisor-provided)
│   ├── topics/                                  # extracted PubMed metadata [not committed]
│   ├── generated/                               # Claude-generated queries, per-batch [not committed]
│   ├── labeled/
│   │   ├── labeled_queries.json                 # 10 120 annotated queries (main dataset)
│   │   └── summary.json                         # intent distribution statistics
│   ├── supplementary/
│   │   └── supplementary_queries.json           # extra queries added in data augmentation
│   ├── soft_labels/                             # teacher logits for distillation [not committed]
│   └── splits/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── models/
│   └── onnx/
│       ├── model_quantized.onnx                 # deployed model (INT8, 83 MB)
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── model_info.json                      # performance & metadata summary
├── scripts/                                     # all pipeline scripts (see below)
├── slurm/                                       # SLURM job scripts for Leonardo HPC
├── .anthropic_key                               # API key — never committed
└── requirements.txt
```

## Full Training Pipeline

The pipeline runs in seven stages. All stages are implemented as resumable SLURM jobs.

### Stage 0 — Environment setup

```bash
sbatch slurm/setup_train_env.sh     # creates train_venv/ and installs requirements.txt
```

### Stage 1 — Extract PubMed topics (~1–2 h, no API)

```bash
sbatch slurm/extract_topics.sh
```

Scans 5 000 random PMC XML files from the Leonardo corpus and writes `data/topics/pubmed_topics.json`.

### Stage 2 — Generate queries (~4–6 h, Claude API)

```bash
sbatch slurm/generate_queries.sh
```

Makes ~600 API calls to generate 15 000 diverse queries in batches of 25. Each batch is saved to `data/generated/batch_NNNN.json` — resubmitting a timed-out job skips completed batches automatically.

### Stage 3 — Label queries (~6–8 h, Claude API)

```bash
sbatch slurm/label_queries.sh
```

Assigns per-intent confidence scores (0.0–1.0) to every query using Claude. Produces `data/labeled/labeled_queries.json` (10 120 records) and `data/labeled/summary.json`.

### Stage 4 — Prepare splits + train teacher (~2–4 h, GPU)

```bash
sbatch slurm/prepare_data.sh       # stratified train/val/test split
sbatch slurm/train_teacher.sh      # fine-tune DeBERTa-v3-large (teacher)
```

`prepare_data.py` thresholds continuous scores into binary labels (default threshold 0.5) and stratifies by primary intent. `train_teacher.py` fine-tunes `microsoft/deberta-v3-large` with BCEWithLogitsLoss.

### Stage 5 — Data augmentation (optional)

```bash
sbatch slurm/supplement_data.sh    # generate + label supplementary queries
```

Runs `generate_supplementary.py` → `label_supplementary.py` → `merge_supplement.py` → `merge_and_resplit.py` to expand the dataset without duplicates.

### Stage 6 — Knowledge distillation (~1–2 h, GPU)

```bash
sbatch slurm/generate_soft_labels.sh   # teacher logits → data/soft_labels/
sbatch slurm/distill_student.sh        # train DeBERTa-v3-xsmall student
```

Loss = α · KL(soft) · T² + (1 − α) · BCE(hard). Student is `microsoft/deberta-v3-xsmall`.

### Stage 7 — Export, quantize & evaluate

```bash
sbatch slurm/export_onnx.sh            # ONNX export + INT8 dynamic quantization
sbatch slurm/final_evaluation.sh       # latency benchmark + ambiguous query demo
```

`export_onnx.py` exports to ONNX (opset 14) and applies INT8 dynamic quantization via `onnxruntime`, achieving 3.26× size compression with only 0.43% F1 drop.

To retrain the full pipeline from scratch:

```bash
sbatch slurm/retrain_full_pipeline.sh
```

## Dataset Statistics

| Split | Records |
|---|---|
| Train | ~8 090 |
| Val | ~1 010 |
| Test | ~1 020 |
| **Total** | **10 120** |

Primary intent distribution (full dataset): recency 28%, mechanism 29%, authority 23%, general 19%.

## Resumability

All multi-hour API jobs (`generate_queries.py`, `label_queries.py`, `generate_supplementary.py`, `label_supplementary.py`) cache completed batches to disk. Resubmitting a timed-out job picks up from where it left off automatically.

## Repository Contents

Files tracked in this repository (large intermediate data and PyTorch checkpoints are excluded via `.gitignore`):

| Path | Description |
|---|---|
| `data/labeled/labeled_queries.json` | 10 120 annotated training queries |
| `data/labeled/summary.json` | Intent distribution statistics |
| `data/splits/{train,val,test}.json` | Train/val/test splits |
| `data/queries_seed.json` | 30 seed queries |
| `data/supplementary/supplementary_queries.json` | Augmentation queries |
| `models/onnx/model_quantized.onnx` | Deployed INT8 model (83 MB) |
| `models/onnx/tokenizer*.json` | Tokenizer files |
| `models/onnx/model_info.json` | Performance & metadata |
| `scripts/` | All pipeline scripts |
| `slurm/` | SLURM job scripts |
