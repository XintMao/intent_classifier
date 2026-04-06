#!/bin/bash
#SBATCH --job-name=train_teacher
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_025
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/train_teacher_%j.out
#SBATCH --error=slurm/logs/train_teacher_%j.err

module load python/3.11.7
module load cuda/12.3

export HF_HOME=/leonardo_scratch/large/userexternal/xmao0000/hf_cache
export TRANSFORMERS_CACHE=/leonardo_scratch/large/userexternal/xmao0000/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /leonardo_scratch/large/userexternal/xmao0000/intent-classifier
source train_venv/bin/activate

mkdir -p models/teacher slurm/logs

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
