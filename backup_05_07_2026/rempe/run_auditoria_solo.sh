#!/bin/bash

# ================= CONFIGURAÇÃO SLURM =================
#SBATCH --job-name=auditoria_visual_t1    # Nome da tarefa
#SBATCH --output=logs/auditoria_%j_t1.log # Ficheiro de saída (%j insere o ID do job)
#SBATCH --error=logs/auditoria_%j_t1.err  # Ficheiro de erros
#SBATCH --account=f202500001hpcvlabepicurex     
#SBATCH --partition=normal-x86            # Partição padrão x86
#SBATCH --ntasks=1                        # Uma única tarefa
#SBATCH --cpus-per-task=4                 # CPUs para auxiliar no carregamento das imagens
#SBATCH --mem=16G                         # Memória RAM necessária
#SBATCH --time=02:00:00                   # Tempo máximo
# ======================================================

# 1. Carregar os módulos do sistema necessários para a compilação C++ / CUDA
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


# 4. Executar o script em modo unbuffered (-u) para escrita em tempo real no ficheiro .log
srun python /projects/F202500001HPCVLABEPICURE/andresousa615/rempe/after_training/auditoria_visual_solo.py
