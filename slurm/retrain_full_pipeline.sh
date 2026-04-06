#!/bin/bash
#SBATCH --job-name=retrain_full
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_025
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/retrain_full_%j.out
#SBATCH --error=slurm/logs/retrain_full_%j.err

module load python/3.11.7
module load cuda/12.3

export HF_HOME=/leonardo_scratch/large/userexternal/xmao0000/hf_cache
export TRANSFORMERS_CACHE=/leonardo_scratch/large/userexternal/xmao0000/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /leonardo_scratch/large/userexternal/xmao0000/intent-classifier
source train_venv/bin/activate

mkdir -p models/teacher models/student models/onnx data/soft_labels slurm/logs

echo "=========================================="
echo "Step 1/5: Train teacher (DeBERTa-v3-large)"
echo "=========================================="
python scripts/train_teacher.py \
    --data-dir data/splits \
    --model-name microsoft/deberta-v3-large \
    --output-dir models/teacher \
    --epochs 10 \
    --batch-size 8 \
    --learning-rate 2e-5 \
    --max-length 128 \
    --warmup-ratio 0.1 \
    --weight-decay 0.01 \
    --seed 42

if [ $? -ne 0 ]; then
    echo "ERROR: Teacher training failed. Aborting pipeline."
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2/5: Generate soft labels"
echo "=========================================="
python scripts/generate_soft_labels.py \
    --model-dir models/teacher/best_model \
    --data-dir  data/splits \
    --output-dir data/soft_labels \
    --batch-size 32 \
    --max-length 128

if [ $? -ne 0 ]; then
    echo "ERROR: Soft label generation failed. Aborting pipeline."
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 3/5: Distill student (DeBERTa-v3-xsmall)"
echo "=========================================="
python scripts/distill_student.py \
    --data-dir      data/soft_labels \
    --test-data     data/splits/test.json \
    --model-name    microsoft/deberta-v3-xsmall \
    --output-dir    models/student \
    --epochs        20 \
    --batch-size    32 \
    --learning-rate 5e-5 \
    --max-length    128 \
    --temperature   3.0 \
    --alpha         0.7 \
    --warmup-ratio  0.1 \
    --weight-decay  0.01 \
    --seed          42

if [ $? -ne 0 ]; then
    echo "ERROR: Student distillation failed. Aborting pipeline."
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 4/5: Export ONNX + INT8 quantization"
echo "=========================================="
python scripts/export_onnx.py \
    --model-dir  models/student/best_model \
    --output-dir models/onnx \
    --max-length 128 \
    --test-data  data/splits/test.json

if [ $? -ne 0 ]; then
    echo "ERROR: ONNX export failed. Aborting pipeline."
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 5/5: Inference demo"
echo "=========================================="
python scripts/inference_demo.py \
    --model-dir models/onnx \
    --threshold 0.5

echo ""
echo "=========================================="
echo "FULL PIPELINE COMPLETE"
echo "=========================================="
echo ""
echo "Final model: models/onnx/model_quantized.onnx"
cat models/onnx/model_info.json 2>/dev/null || true
