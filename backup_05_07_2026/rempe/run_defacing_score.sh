#!/bin/bash

# ================= CONFIGURAÇÃO SLURM =================
#SBATCH --job-name=defacing_score_t1      # Nome da tarefa
#SBATCH --output=logs/defacing_score_%j_t1.log     # Ficheiro de saída (%j insere o ID do job)
#SBATCH --error=logs/defacing_score_%j_t1.err      # Ficheiro de erros
#SBATCH --account=f202500001hpcvlabepicureg
#SBATCH --partition=normal-a100-40         
#SBATCH --ntasks=1                        # Uma única tarefa
#SBATCH --cpus-per-task=32                 # CPUs para auxiliar no carregamento das imagens
#SBATCH --mem=64G                         # Memória RAM necessária
#SBATCH --time=02:00:00                   # Tempo máximo (2 horas é suficiente para 21 exames)
#SBATCH --gpus=1
# ======================================================



# 1. Limpar e carregar módulos necessários (conforme a configuração do Aurora)
module purge
module load Python/3.10.8
module load Miniconda3/23.5.2-0
module load CMake/3.24.3
module load GCC/12.2.0

# Ativar o ambiente virtual
eval "$(conda shell.bash hook)"
source activate /projects/F202500001HPCVLABEPICURE/andresousa615/env_mede


# 3. PYTHONPATH e Diretório de Trabalho
PROJ_DIR="/projects/F202500001HPCVLABEPICURE/andresousa615/rempe"
cd $PROJ_DIR
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"


# 5. Executar o script de avaliação
python -u /projects/F202500001HPCVLABEPICURE/andresousa615/rempe/after_training/calculate_defacing_score_v2.py --e 100 --gpu 0
echo "Cálculo de Defacing Score concluído."

