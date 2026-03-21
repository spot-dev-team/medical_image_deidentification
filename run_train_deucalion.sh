#!/bin/bash
#SBATCH --job-name=mede_train
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --account=f202500001hpcvlabepicureg
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# 1. Caminhos
PROJ_DIR="/projects/F202500001HPCVLABEPICURE/andresousa615"
CONDA_ENV="${PROJ_DIR}/env_mede"

cd ${PROJ_DIR}
mkdir -p logs

# 2. Ativar o Ambiente Conda
# Primeiro carregamos o módulo para ter o comando 'conda' disponível
module load Anaconda3 || module load miniconda3
source activate ${CONDA_ENV}

echo "Ambiente ativado: ${CONDA_ENV}"
echo "A verificar GPU e Tesseract..."
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available())"
tesseract --version

# 3. Executar o treino
echo "A iniciar o treino no Deucalion..."
srun python train_seg.py --config cluster_train.yaml --e 100 --gpu 0
