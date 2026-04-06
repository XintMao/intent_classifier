#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_025
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/setup_env_%j.out
#SBATCH --error=slurm/logs/setup_env_%j.err

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
pip install transformers datasets accelerate scikit-learn tensorboard tqdm sentencepiece tiktoken safetensors

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
python -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"

# Pre-download DeBERTa-v3-large weights to the scratch cache so compute
# nodes do not need outbound internet access during training.
python -c "
from transformers import AutoTokenizer, AutoModel
print('Downloading microsoft/deberta-v3-large tokenizer and model weights...')
AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
AutoModel.from_pretrained('microsoft/deberta-v3-large')
print('Download complete.')
"
