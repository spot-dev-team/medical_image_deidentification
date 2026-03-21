#!/bin/bash
#SBATCH --job-name=MedNeXt_Paper_t1_deucalion
#SBATCH --account=f202500001hpcvlabepicureg
#SBATCH --partition=normal-a100-40         
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32           
#SBATCH --mem=128G                   # RAM do sistema para carregar volumes 3D
#SBATCH --time=12:00:00             # MedNeXt 3D é pesado, 48h é mais seguro
#SBATCH --output=logs/mednext_t1_deucalion_%j.log
#SBATCH --error=logs/mednext_t1_deucalion_%j.err

# 1. Carregar Módulos (Geralmente o ambiente conda já traz o necessário)
module purge

# 2. Ativar o Ambiente (Caminho absoluto para o ambiente no /projects)
# Primeiro garantimos que o comando 'activate' está disponível no script
eval "$(conda shell.bash hook)"
source activate /projects/F202500001HPCVLABEPICURE/andresousa615/env_mede

# 3. PYTHONPATH e Diretório de Trabalho
PROJ_DIR="/projects/F202500001HPCVLABEPICURE/andresousa615/rempe"
cd $PROJ_DIR
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"

# 4. Verificação de GPU
echo "A iniciar job no nó: $(hostname)"
python -c "import torch; print(f'GPU detetada: {torch.cuda.is_available()} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NENHUMA\"}')"


# 5. Executar o Treino
# Garante que criaste o ficheiro configs/cluster_train.yaml
python train_seg.py --config configs/cluster_train.yaml --e 100 --gpu 0
