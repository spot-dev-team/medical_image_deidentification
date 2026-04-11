#!/bin/bash
#SBATCH --job-name=render_mede
#SBATCH --output=logs/render_%j.log
#SBATCH --error=logs/render_%j.err
#SBATCH --account=f202500001hpcvlabepicurex        
#SBATCH --partition=normal-x86         # Partição padrão x86
#SBATCH --nodes=1                      # Para renderização 1 nó costuma chegar
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16             # Mais cores aceleram o processamento NIfTI
#SBATCH --mem=32G                      # Deucalion tem muita RAM por nó
#SBATCH --time=02:00:00


# 1. Limpar e carregar módulos (adaptado ao Deucalion)
module purge
module load Xvfb/21.1.8-GCCcore-12.3.0 # O módulo que mencionaste!
module load Mesa/23.1.4-GCCcore-12.3.0 # Necessário para o VTK renderizar via CPU
module load Python/3.10.8  # Usa a mesma versão que usaste para criar o venv


# 2. Ativar o Ambiente Virtual (no /projects/)
source /home/andresousa615/rempe/mede_code/mede_venv_deucalion/bin/activate

# 3. PYTHONPATH (ajustado para os novos caminhos)
#Substitui <path_do_codigo> pelo caminho onde tens a pasta 'mede' no Deucalion
export PYTHONPATH="/projects/<project_id>/<user>/medical_image_deidentification:$PYTHONPATH"
 
# 4. Execução com xvfb-run (Essencial para PyVista em clusters headless)
# O Deucalion não tem ecrã físico, o xvfb cria um ecrã virtual na memória.
#xvfb-run -a python -u /home/andresousa615/rempe/mede_code/after_training/process_all_exames_generate_2D_image.py

xvfb-run -a -s "-screen 0 1600x1200x24 +extension GLX +render" python -u /home/andresousa615/rempe/mede_code/after_training/process_all_exames_generate_2D_image.py