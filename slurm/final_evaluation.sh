#!/bin/bash
#SBATCH --job-name=final_eval
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_025
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/final_eval_%j.out
#SBATCH --error=slurm/logs/final_eval_%j.err

module load python/3.11.7

export HF_HOME=/leonardo_scratch/large/userexternal/xmao0000/hf_cache
export TRANSFORMERS_CACHE=/leonardo_scratch/large/userexternal/xmao0000/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /leonardo_scratch/large/userexternal/xmao0000/intent-classifier
source train_venv/bin/activate

echo "=========================================="
echo "Part 1: Latency Benchmark"
echo "=========================================="
python scripts/benchmark_latency.py --model-dir models/onnx

echo ""
echo "=========================================="
echo "Part 2: Ambiguous Query Tests"
echo "=========================================="

python scripts/inference_demo.py --model-dir models/onnx --threshold 0.5 --query "What do we know about inflammation and heart disease?"
echo "------------------------------------------------------------"
python scripts/inference_demo.py --model-dir models/onnx --threshold 0.5 --query "Tell me about BRCA1"
echo "------------------------------------------------------------"
python scripts/inference_demo.py --model-dir models/onnx --threshold 0.5 --query "What are the latest authoritative guidelines on the molecular pathways of diabetes?"
echo "------------------------------------------------------------"
python scripts/inference_demo.py --model-dir models/onnx --threshold 0.5 --query "How to treat hypertension"
echo "------------------------------------------------------------"
python scripts/inference_demo.py --model-dir models/onnx --threshold 0.5 --query "What is known about covid vaccines and myocarditis risk in young adults?"
echo "------------------------------------------------------------"
python scripts/inference_demo.py --model-dir models/onnx --threshold 0.5 --query "Are there any recent landmark reviews on the signaling mechanisms of PD-1 checkpoint inhibitors?"
echo "------------------------------------------------------------"
python scripts/inference_demo.py --model-dir models/onnx --threshold 0.5 --query "What causes Alzheimer disease?"
echo "------------------------------------------------------------"
python scripts/inference_demo.py --model-dir models/onnx --threshold 0.5 --query "Which new studies have explored how SGLT2 inhibitors protect the heart at the cellular level?"
echo "------------------------------------------------------------"

echo ""
echo "=========================================="
echo "FINAL EVALUATION COMPLETE"
echo "=========================================="
