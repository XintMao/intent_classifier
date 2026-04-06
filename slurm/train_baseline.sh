#!/bin/bash
#SBATCH --job-name=train_baseline
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_025
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/train_baseline_%j.out
#SBATCH --error=slurm/logs/train_baseline_%j.err

module load python/3.11.7
module load cuda/12.3

export HF_HOME=/leonardo_scratch/large/userexternal/xmao0000/hf_cache
export TRANSFORMERS_CACHE=/leonardo_scratch/large/userexternal/xmao0000/hf_cache

cd /leonardo_scratch/large/userexternal/xmao0000/intent-classifier

mkdir -p slurm/logs

source train_venv/bin/activate

python scripts/train_baseline.py --batch-size 8
