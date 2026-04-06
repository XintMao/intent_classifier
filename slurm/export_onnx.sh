#!/bin/bash
#SBATCH --job-name=export_onnx
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_025
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/export_onnx_%j.out
#SBATCH --error=slurm/logs/export_onnx_%j.err

# No GPU needed — ONNX export and quantization run on CPU.

module load python/3.11.7

export HF_HOME=/leonardo_scratch/large/userexternal/xmao0000/hf_cache
export TRANSFORMERS_CACHE=/leonardo_scratch/large/userexternal/xmao0000/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /leonardo_scratch/large/userexternal/xmao0000/intent-classifier
source train_venv/bin/activate

mkdir -p models/onnx slurm/logs

echo "=== Checking onnxruntime / onnx / optimum ==="
python -c "import onnx, onnxruntime, optimum; print('onnx', onnx.__version__, '| onnxruntime', onnxruntime.__version__, '| optimum', optimum.__version__)"

echo ""
echo "=== Step A-F: Export, quantize, validate, benchmark ==="
python scripts/export_onnx.py \
    --model-dir  models/student/best_model \
    --output-dir models/onnx \
    --max-length 128 \
    --test-data  data/splits/test.json \
    --threshold  0.5

echo ""
echo "=== Running inference demo ==="
python scripts/inference_demo.py \
    --model-dir  models/onnx \
    --threshold  0.5
