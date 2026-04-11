#!/bin/bash
#SBATCH --job-name=apenas_defacing_score
#SBATCH --account=f202500001hpcvlabepicureg
#SBATCH --partition=normal-a100-40 
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=128
#SBATCH --mem=400G 
#SBATCH --time=02:00:00  
#SBATCH --output=logs/apenas_defacing_%j.log
#SBATCH --error=logs/apenas_defacing_%j.err

# ==============================================================================
# 0. AMBIENTE E MÓDULOS
# ==============================================================================
module purge
module load Python/3.10.8
module load Miniconda3/23.5.2-0
module load CUDA/12.4.0
module load Xvfb/21.1.8-GCCcore-12.3.0
module load Mesa/23.1.4-GCCcore-12.3.0

eval "$(conda shell.bash hook)"
source activate /projects/F202500001HPCVLABEPICURE/andresousa615/env_mede

PROJ_DIR="/projects/F202500001HPCVLABEPICURE/andresousa615/rempe"
cd $PROJ_DIR
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"

# ==============================================================================
# Rede DDP
# ==============================================================================
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}

# 1. GLOO DE VOLTA (Obrigatório para o Rendezvous do c10d)
export GLOO_SOCKET_IFNAME=^lo,docker0
export NCCL_SOCKET_IFNAME=^lo,docker0
export LOGLEVEL=INFO
export OMP_NUM_THREADS=32 

# 2. GERAR PORTA ALEATÓRIA (Entre 20000 e 30000 para evitar colisões)
RDZV_PORT=$(( 20000 + RANDOM % 10000 ))
# ==============================================================================
# VARIÁVEIS 
# ==============================================================================
MODEL_BASE="mednext_0.0005_mednext_aurora_16gb_vram"
RUN_DIR="${PROJ_DIR}/results/train_${MODEL_BASE}_1120800"

# Variáveis exclusivas da Fase 5
DIR_PARES="${RUN_DIR}/pares_2d"
DEFACING_REPORT="${RUN_DIR}/defacing_report.txt"

echo "=========================================================="
echo " A EXECUTAR APENAS A AUDITORIA BIOMÉTRICA (FASE 5)"
echo " Alvo: $RUN_DIR"
echo "=========================================================="
pipeline_start=$(date +%s)

# ==============================================================================
# FASE 5: CÁLCULO DO DEFACING SCORE (DLIB DDP)
# ==============================================================================
echo "[FASE 5] A Iniciar Auditoria Biométrica (Defacing Score)..."
start_time=$(date +%s)

srun torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node:$RDZV_PORT \
    after_training/calculate_defacing_score_ddp.py \
    --dir_pares "$DIR_PARES" \
    --output_report "$DEFACING_REPORT"

    
end_time=$(date +%s)
echo "[FASE 5 CONCLUÍDA] Duração: $((end_time - start_time)) segundos."
echo "----------------------------------------------------------"

pipeline_end=$(date +%s)
echo "TEMPO TOTAL: $((pipeline_end - pipeline_start)) segundos."
echo "=========================================================="