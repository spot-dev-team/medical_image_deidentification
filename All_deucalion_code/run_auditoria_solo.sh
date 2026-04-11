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
module load CMake/3.24.3
module load GCC/12.2.0


# 2. Ativar o Ambiente Virtual
source /home/andresousa615/rempe/mede_code/mede_venv_deucalion/bin/activate


# 3. PYTHONPATH
export PYTHONPATH="/projects/<project_id>/<user>/medical_image_deidentification:$PYTHONPATH"
 
# 4. Executar o script em modo unbuffered (-u) para escrita em tempo real no ficheiro .log
python -u /home/andresousa615/rempe/mede_code/after_training/auditoria_visual_solo.py