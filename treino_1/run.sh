#!/bin/bash
#SBATCH --job-name=MedNeXt_Paper_t1
#SBATCH --account=haslab
#SBATCH --partition=rtx4060         # Partição correta para a RTX 4060 Ti
#SBATCH --exclude=aurora04          # Vital: Evitar o nó com erro de CUDA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2           # Conforme restrição de memória/IO do Spot
#SBATCH --mem=16G                   # RAM do sistema para carregar volumes 3D
#SBATCH --time=48:00:00             # MedNeXt 3D é pesado, 48h é mais seguro
#SBATCH --output=logs/mednext_%j.log
#SBATCH --error=logs/mednext_%j.err

# 1. Carregar Módulos Identificados
module purge
module load CUDA/12.3.0


# VITAL: Tem de ter o P maiúsculo e estar ANTES de ativar o venv
module load Python/3.10.8

# 2. Ativar Ambiente Virtual (venv)# Nota: Ajusta o caminho se o teu venv estiver noutro local
source /home/andresousa615/rempe/mede_code/mednext_venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. Verificação de GPU
echo "A iniciar job no nó: $(hostname)"
python -c "import torch; print(f'GPU detetada: {torch.cuda.is_available()} - {torch.cuda.get_device_name(0)}')"


# 4. Criar pasta de logs se não existir
mkdir -p logs

# 5. Executar o Treino
# O script assume que estás na pasta mede_code
python train_seg.py --config cluster_train.yaml --e 100 --gpu 0
