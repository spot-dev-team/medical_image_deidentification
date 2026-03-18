#!/bin/bash
#SBATCH --job-name=mede_train
#SBATCH --partition=normal-a100-40     # Partição de GPU (A100 40GB)
#SBATCH --nodes=1                      # 1 Nó
#SBATCH --gpus=1                       # 1 GPU
#SBATCH --ntasks=1                     # 1 Tarefa
#SBATCH --cpus-per-task=32             # OBRIGATÓRIO: 32 CPUs por cada 1 GPU no Deucalion
#SBATCH --time=24:00:00                # Tempo limite (HH:MM:SS)
#SBATCH --account=<SEU_ACCOUNT_G>      # SUBSTITUA pelo seu account (terminado em 'g')
#SBATCH --output=logs/train_%j.out     # Log de saída (%j é o ID do job)
#SBATCH --error=logs/train_%j.err      # Log de erro

# 1. Preparar o ambiente
# Nota: Garanta que o seu código e ambiente estão em /projects/$PROJECT/
# Se usar um virtualenv ou conda, ative-o aqui:
# source /projects/<seu_projeto>/venv/bin/activate

# 2. Criar pasta de logs se não existir
mkdir -p logs

# 3. Executar o treino
# O srun garante que o job é gerido corretamente pelo Slurm
echo "A iniciar o treino no Deucalion..."
srun python train_seg.py --config cluster_train.yaml --e 100 --gpu 0
