# Intent Classifier — Step 1: Data Generation & Labeling

Multi-label intent classifier for PubMed search queries. Intents: `recency`, `authority`, `mechanism`, `general`.

## Directory Structure

```
intent-classifier/
├── data/
│   ├── queries_seed.json          # 30 supervisor-provided seed queries
│   ├── topics/pubmed_topics.json  # extracted PubMed metadata (step 1 output)
│   ├── generated/                 # per-batch generated queries + all_generated.json (step 2 output)
│   └── labeled/                   # labeled_queries.json, ood_queries.json, summary.json (step 3 output)
├── scripts/
│   ├── extract_topics.py          # step 1: parse PMC XML files
│   ├── generate_queries.py        # step 2: Claude API query generation
│   └── label_queries.py           # step 3: Claude API confidence scoring
├── slurm/
│   ├── extract_topics.sh
│   ├── generate_queries.sh
│   └── label_queries.sh
├── logs/                          # Python log files
├── .anthropic_key                 # Anthropic API key (not committed)
└── requirements.txt
```

## Running the Pipeline

### Setup (handled automatically by step 1's SLURM script)

The first SLURM job creates a Python venv and installs dependencies from `requirements.txt`.
Subsequent jobs simply activate the existing venv.

```bash
# Run from the project root
cd /leonardo_scratch/large/userexternal/xmao0000/intent-classifier
```

### Step 1 — Extract PubMed topics (no API, ~1–2 h)

```bash
sbatch slurm/extract_topics.sh
```

Scans 5000 random PMC XML files from `/leonardo_scratch/large/userexternal/mvesterb/all/` and
writes `data/topics/pubmed_topics.json`.

### Step 2 — Generate queries (Claude API, ~4–6 h)

```bash
sbatch slurm/generate_queries.sh
```

Depends on step 1. Makes ~600 API calls to generate 15 000 queries in batches of 25.
Each batch is saved as `data/generated/batch_NNNN.json` — if the job times out, resubmit
and it will skip already-completed batches. Merges everything into `data/generated/all_generated.json`.

### Step 3 — Label queries (Claude API, ~6–8 h)

```bash
sbatch slurm/label_queries.sh
```

Depends on step 2. Labels each query with per-intent confidence scores (0.0–1.0).
Queries where all scores < 0.7 are saved to `data/labeled/ood_queries.json`.
Final outputs:
- `data/labeled/labeled_queries.json` — main training dataset
- `data/labeled/ood_queries.json` — OOD queries for analysis
- `data/labeled/summary.json` — dataset statistics

## Resumability

Both `generate_queries.py` and `label_queries.py` cache completed batches to disk.
Resubmitting a timed-out job picks up from where it left off automatically.

## Final Outputs (for Step 2: Model Training)

| File | Description |
|---|---|
| `data/labeled/labeled_queries.json` | Multi-label annotated training dataset |
| `data/labeled/ood_queries.json` | OOD queries for analysis / new category discovery |
| `data/labeled/summary.json` | Intent distribution and score statistics |
