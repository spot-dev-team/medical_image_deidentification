#!/bin/bash
#SBATCH --job-name=mede_train
#SBATCH --partition=normal-a100-40     # Partição standard de GPU (A100 40GB)
#SBATCH --nodes=1                      # Usar 1 nó
#SBATCH --ntasks=1                     # 1 tarefa (o teu script python)
#SBATCH --gpus=1                       # Alocar 1 GPU
#SBATCH --cpus-per-task=32             # OBRIGATÓRIO NO DEUCALION: 32 CPUs para 1 GPU
#SBATCH --time=24:00:00                # Tempo limite (ajusta se necessário, máx 48h)
#SBATCH --account=<O_TEU_PROJETO>g     # SUBSTITUI: o teu ID de projeto (geralmente termina em 'g')
#SBATCH --mem=64G                      # Memória RAM do sistema (ajusta se precisares de mais)
#SBATCH --output=logs/train_%j.out     # Ficheiro de log (ID do job no nome)
#SBATCH --error=logs/train_%j.err      # Ficheiro de erros

# 1. Criar pasta de logs se não existir
mkdir -p logs

# 2. Carregar o ambiente (escolhe UM dos métodos abaixo e descomenta)

## SE USARES CONDA:
# source /projects/<o_teu_projeto>/.conda/etc/profile.d/conda.sh
# conda activate <o_teu_env>

## SE USARES VIRTUALENV:
# source /projects/<o_teu_projeto>/venv/bin/activate

# 3. Executar o Treino
# Nota: No Deucalion, deves correr o código a partir de /projects/ para evitar limites de I/O
echo "A iniciar treino no nó: $SLURM_NODELIST"

srun python train_seg.py --config cluster_train.yaml --e 100 --gpu 0
