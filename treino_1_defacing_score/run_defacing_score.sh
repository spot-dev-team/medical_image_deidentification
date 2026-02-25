#!/bin/bash

# ================= CONFIGURAÇÃO SLURM =================
#SBATCH --job-name=defacing_score_t1      # Nome da tarefa
#SBATCH --output=logs/defacing_%j_t1.log     # Ficheiro de saída (%j insere o ID do job)
#SBATCH --error=logs/defacing_%j_t1.err      # Ficheiro de erros
#SBATCH --account=haslab
#SBATCH --partition=rtx4060                   # Partição das GPUs
#SBATCH --ntasks=1                        # Uma única tarefa
#SBATCH --cpus-per-task=4                 # CPUs para auxiliar no carregamento das imagens
#SBATCH --mem=16G                         # Memória RAM necessária
#SBATCH --time=02:00:00                   # Tempo máximo (2 horas é suficiente para 21 exames)
# ======================================================



# 1. Limpar e carregar módulos necessários (conforme a configuração do Aurora)
module purge
module load Python/3.10.8
module load CUDA/12.3.0
module load CMake/3.24.3  # Adicionado para suporte ao dlib
module load GCC/12.2.0    # Adicionado para suporte ao dlib

# 3. Ativar o teu ambiente virtual onde o dlib e face_recognition estão instalados
# Ajusta o caminho se o ambiente estiver noutro diretório
source /home/andresousa615/rempe/mede_code/mednext_venv/bin/activate

# 4. O TRUQUE PARA O CLUSTER LER OS PACOTES DO TEU UTILIZADOR
export PYTHONPATH="/home/andresousa615/.local/lib/python3.10/site-packages:$PYTHONPATH"


# 5. Executar o script de avaliação
python -u /home/andresousa615/rempe/mede_code/after_training/calculate_defacing_score_v2.py

