#!/bin/bash
#SBATCH --job-name=rempe_ddp_2gpus
#SBATCH --account=f202500001hpcvlabepicureg
#SBATCH --partition=normal-a100-40 
#SBATCH --nodes=2 
#SBATCH --gpus-per-node=2 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G 
#SBATCH --time=24:00:00
#SBATCH --output=logs/rempe_ddp_%j.log
#SBATCH --error=logs/rempe_ddp_%j.err
# REMOVIDO: --exclusive e --mem=0 para o SLURM te poder "encaixar" em nós parcialmente usados

# 1. Preparar o ambiente
module purge
module load Python/3.10.8
module load Miniconda3/23.5.2-0
module load CUDA/12.4.0

eval "$(conda shell.bash hook)"
source activate /projects/F202500001HPCVLABEPICURE/andresousa615/env_mede


# 2. Obter o NOME do Head Node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}

echo "--- A iniciar treino de teste do Rempe (2 Nós, 1 GPU cada) ---"
echo "Master Node Name: $head_node"

# 3. Forçar o PyTorch a ignorar redes de Loopback
export GLOO_SOCKET_IFNAME=^lo,docker0
export NCCL_SOCKET_IFNAME=^lo,docker0
export LOGLEVEL=INFO

# Limitar as threads ao número de CPUs alocados (32)
export OMP_NUM_THREADS=32 

start_time=$(date +%s)

# 4. Executar com torchrun
srun torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node:29500 \
    train_seg_DDP.py --epochs 100 --config cluster_train.yaml

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "----------------------------------"
echo "Treino concluído em $duration segundos"
echo "----------------------------------"