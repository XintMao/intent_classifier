#!/bin/bash
#SBATCH --job-name=gen_queries
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_025
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/gen_queries_%j.out
#SBATCH --error=slurm/logs/gen_queries_%j.err

module load python/3.11.7

cd /leonardo_scratch/large/userexternal/xmao0000/intent-classifier
source venv/bin/activate

mkdir -p data/generated logs

python scripts/generate_queries.py \
    --topics data/topics/pubmed_topics.json \
    --seeds data/queries_seed.json \
    --output-dir data/generated \
    --total 15000 \
    --batch-size 25
