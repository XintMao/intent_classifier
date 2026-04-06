#!/bin/bash
#SBATCH --job-name=setup_train_venv
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_025
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/setup_train_venv_%j.out
#SBATCH --error=slurm/logs/setup_train_venv_%j.err

module load python/3.11.7
module load cuda/12.3

export HF_HOME=/leonardo_scratch/large/userexternal/xmao0000/hf_cache
export TRANSFORMERS_CACHE=/leonardo_scratch/large/userexternal/xmao0000/hf_cache

cd /leonardo_scratch/large/userexternal/xmao0000/intent-classifier

mkdir -p slurm/logs

python -m venv train_venv
source train_venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate scikit-learn tensorboard tqdm

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
python -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"

# Pre-download distilbert weights so compute nodes don't need internet
python -c "
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
print('Downloading distilbert-base-uncased ...')
DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)
print('Done.')
"
