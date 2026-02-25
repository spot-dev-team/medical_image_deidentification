#!/bin/bash

# ================= CONFIGURAÇÃO SLURM =================
#SBATCH --job-name=teste_cruzado_t1       # Nome da tarefa
#SBATCH --output=logs/cruzado_%j_t1.log   # Ficheiro de saída em tempo real
#SBATCH --error=logs/cruzado_%j_t1.err    # Ficheiro de erros
#SBATCH --account=haslab
#SBATCH --partition=rtx4060               # Partição das GPUs
#SBATCH --ntasks=1                        # Uma única tarefa
#SBATCH --cpus-per-task=4                 # CPUs para auxiliar no carregamento
#SBATCH --mem=16G                         # Memória RAM necessária
#SBATCH --time=02:00:00                   # Tempo máximo
#SBATCH --exclude=aurora04                # Exclusão do nó aurora04
# ======================================================

# 1. Carregar os módulos do sistema
module purge
module load Python/3.10.8
module load CUDA/12.3.0
module load CMake/3.24.3
module load GCC/12.2.0

# 2. Ativar o ambiente virtual da Spot
source /home/andresousa615/rempe/mede_code/mednext_venv/bin/activate

# 3. Injetar a diretoria local no PYTHONPATH para aceder à biblioteca dlib/face_recognition
export PYTHONPATH="/home/andresousa615/.local/lib/python3.10/site-packages:$PYTHONPATH"

# 4. Executar o script em modo unbuffered (-u) para visualizar os prints imediatamente
python -u /home/andresousa615/rempe/mede_code/after_training/teste_aleatorio_defacing_score.py