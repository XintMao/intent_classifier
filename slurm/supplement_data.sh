#!/bin/bash
#SBATCH --job-name=supplement
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_025
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/supplement_%j.out
#SBATCH --error=slurm/logs/supplement_%j.err

module load python/3.11.7

cd /leonardo_scratch/large/userexternal/xmao0000/intent-classifier
source venv/bin/activate

mkdir -p data/supplementary slurm/logs

echo "=== Step A: Generating supplementary queries ==="
python scripts/generate_supplementary.py \
    --existing  data/labeled/labeled_queries.json \
    --topics    data/topics/pubmed_topics.json \
    --seeds     data/queries_seed.json \
    --output    data/supplementary/new_queries.json \
    --target    4000 \
    --batch-size 25

echo "=== Step B: Labeling new queries ==="
python scripts/label_supplementary.py \
    --input      data/supplementary/new_queries.json \
    --output     data/supplementary/labeled_new.json \
    --batch-size 20

echo "=== Step C: Merging and re-splitting ==="
python scripts/merge_and_resplit.py \
    --existing       data/labeled/labeled_queries.json \
    --new            data/supplementary/labeled_new.json \
    --output-labeled data/labeled/labeled_queries_merged.json \
    --output-dir     data/splits \
    --threshold      0.7 \
    --seed           42
