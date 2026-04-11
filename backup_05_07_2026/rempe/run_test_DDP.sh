#!/bin/bash
#SBATCH --job-name=test_ddp_rempe
#SBATCH --account=f202500001hpcvlabepicureg
#SBATCH --partition=normal-a100-40 
#SBATCH --nodes=2 
#SBATCH --gpus-per-node=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G 
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_ddp_%j.log
#SBATCH --error=logs/test_ddp_%j.err

# 1. Preparar o ambiente
module purge
module load Python/3.10.8
module load Miniconda3/23.5.2-0
module load CUDA/12.4.0

eval "$(conda shell.bash hook)"
source activate /projects/F202500001HPCVLABEPICURE/andresousa615/env_mede

# 2. PYTHONPATH e Diretório de Trabalho
PROJ_DIR="/projects/F202500001HPCVLABEPICURE/andresousa615/rempe"
cd $PROJ_DIR
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"

# 3. Obter o NOME do Head Node para o DDP
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}

echo "--- A iniciar Avaliação Física Distribuída (DDP) ---"
echo "Master Node Name: $head_node"

# 4. Variáveis de Rede para o PyTorch
export GLOO_SOCKET_IFNAME=^lo,docker0
export NCCL_SOCKET_IFNAME=^lo,docker0
export LOGLEVEL=INFO
export OMP_NUM_THREADS=4 

start_time=$(date +%s)

# ==============================================================================
# 5. CONFIGURAÇÃO DE CAMINHOS (IMPORTANTE: Atualiza estas variáveis)
# ==============================================================================
# Coloca aqui o nome base do modelo e o ID do job de treino que queres testar
MODEL_NAME_BASE="mednext_0.0005_mednext_aurora_16gb_vram"
TRAIN_JOB_ID="100" # Substitui pelo ID real do teu treino com sucesso

BASE_RESULTS="${PROJ_DIR}/results/train_${MODEL_NAME_BASE}_${TRAIN_JOB_ID}"
#BASE_RESULTS="${PROJ_DIR}/results/train_mednext_0.0005_mednext_aurora_16gb_vram"
WEIGHTS="${BASE_RESULTS}/best_${MODEL_NAME_BASE}.pt"
INFERENCE_DIR="${BASE_RESULTS}/inference_test_masks"
CSV_METRICS="${BASE_RESULTS}/per_exam_metrics.csv"
TXT_METRICS="${BASE_RESULTS}/real_test_metrics.txt"
# ==============================================================================

# 6. Executar inferência com torchrun
srun torchrun \
    --nnodes=2 \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node:29500 \
    test_DDP.py \
    --weights "$WEIGHTS" \
    --out_dir "$INFERENCE_DIR" \
    --metrics_csv "$CSV_METRICS" \
    --metrics_txt "$TXT_METRICS"

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "----------------------------------"
echo "Inferência concluída em $duration segundos"
echo "----------------------------------"