#!/bin/bash
#SBATCH --job-name=label_queries
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_025
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/label_queries_%j.out
#SBATCH --error=slurm/logs/label_queries_%j.err

module load python/3.11.7

cd /leonardo_scratch/large/userexternal/xmao0000/intent-classifier
source venv/bin/activate

mkdir -p data/labeled logs

python scripts/label_queries.py \
    --input data/generated/all_generated.json \
    --output data/labeled/labeled_queries.json \
    --batch-size 20
