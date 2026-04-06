#!/bin/bash
#SBATCH --job-name=soft_labels
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_025
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/soft_labels_%j.out
#SBATCH --error=slurm/logs/soft_labels_%j.err

module load python/3.11.7
module load cuda/12.3

export HF_HOME=/leonardo_scratch/large/userexternal/xmao0000/hf_cache
export TRANSFORMERS_CACHE=/leonardo_scratch/large/userexternal/xmao0000/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /leonardo_scratch/large/userexternal/xmao0000/intent-classifier
source train_venv/bin/activate

mkdir -p data/soft_labels slurm/logs

python scripts/generate_soft_labels.py \
    --model-dir models/teacher/best_model \
    --data-dir  data/splits \
    --output-dir data/soft_labels \
    --batch-size 32 \
    --max-length 128
