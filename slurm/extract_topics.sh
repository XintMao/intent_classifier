#!/bin/bash
#SBATCH --job-name=extract_topics
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_025
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/extract_topics_%j.out
#SBATCH --error=slurm/logs/extract_topics_%j.err

module load python/3.11.7

cd /leonardo_scratch/large/userexternal/xmao0000/intent-classifier

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Create log directory
mkdir -p slurm/logs logs

python scripts/extract_topics.py \
    --input-dir /leonardo_scratch/large/userexternal/mvesterb/all \
    --output data/topics/pubmed_topics.json \
    --sample-size 5000 \
    --seed 42
