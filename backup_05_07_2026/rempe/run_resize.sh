#!/bin/bash
#SBATCH --job-name=resize_dataset
#SBATCH --account=f202500001hpcvlabepicureg
#SBATCH --partition=normal-a100-40  # Podes mudar para uma partição só de CPU se o Deucalion tiver, para não gastares quota de GPU
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1            # Não precisamos de GPU para redimensionar offline
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=32          # 32 núcleos é excelente para processar imagens em paralelo rapidamente
#SBATCH --mem=128G                  # RAM suficiente para carregar ficheiros gigantes de 16MB+ sem crashar
#SBATCH --time=04:00:00             # 4 horas de limite máximo (deve demorar muito menos)
#SBATCH --output=logs/resize_%j.log
#SBATCH --error=logs/resize_%j.err

echo "=========================================================="
echo " INICIANDO PRÉ-PROCESSAMENTO  (Job ID: $SLURM_JOB_ID)"
echo "=========================================================="

# ==============================================================================
# 1. AMBIENTE E MÓDULOS (Igual ao teu master_pipeline)
# ==============================================================================
module purge
module load Python/3.10.8
module load Miniconda3/23.5.2-0

eval "$(conda shell.bash hook)"
source activate /projects/F202500001HPCVLABEPICURE/andresousa615/env_mede

PROJ_DIR="/projects/F202500001HPCVLABEPICURE/andresousa615/rempe"
cd $PROJ_DIR
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"

# ==============================================================================
# 2. DEFINIR CAMINHOS DOS DADOS
# ==============================================================================
BASE_DATA_DIR="/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/processed_datasets/processed_datasets"
TREINO_DIR="${BASE_DATA_DIR}/treino"

# ==============================================================================
# 3. EXECUTAR REDIMENSIONAMENTO
# ==============================================================================
echo "----------------------------------------------------------"
echo "[FASE 1] A redimensionar os dados de TREINO e VALIDACAO para 128x128x128..."
srun python preprocessing/resize_datasets.py --dir $TREINO_DIR

echo "----------------------------------------------------------"
echo "✅ PROCESSAMENTO CONCLUÍDO COM SUCESSO!"
echo "Os exames estão otimizados e prontos para o DataLoader."
echo "=========================================================="