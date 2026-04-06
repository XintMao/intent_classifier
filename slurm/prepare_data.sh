#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_025
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/prepare_data_%j.out
#SBATCH --error=slurm/logs/prepare_data_%j.err

module load python/3.11.7

export HF_HOME=/leonardo_scratch/large/userexternal/xmao0000/hf_cache
export TRANSFORMERS_CACHE=/leonardo_scratch/large/userexternal/xmao0000/hf_cache

cd /leonardo_scratch/large/userexternal/xmao0000/intent-classifier
source train_venv/bin/activate

mkdir -p data/splits slurm/logs

python scripts/prepare_data.py \
    --input data/labeled/labeled_queries.json \
    --output-dir data/splits \
    --threshold 0.7 \
    --seed 42
