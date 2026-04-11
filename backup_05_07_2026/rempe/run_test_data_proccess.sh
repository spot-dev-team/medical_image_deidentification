#!/bin/bash
#SBATCH --job-name=test_metrics_deucalion_t1
#SBATCH --output=logs/test_metrics_deucalion_t1_%j.log
#SBATCH --error=logs/test_metrics_deucalion_t1_%j.err
#SBATCH --account=f202500001hpcvlabepicurex
#SBATCH --partition=normal-x86   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                 # CPUs para auxiliar no carregamento das imagens
#SBATCH --mem=16G                         # Memória RAM necessária
#SBATCH --time=04:00:00                   # Tempo máximo       

# 1. Carregar módulos do Aurora (O Deucalion requer o módulo Miniconda)
module purge
module load Miniconda3/23.5.2-0
module load CMake/3.24.3
module load GCC/12.2.0

# 2. Ativar Ambiente Virtual da Spot de forma rigorosa
eval "$(conda shell.bash hook)"
conda activate /projects/F202500001HPCVLABEPICURE/andresousa615/env_mede

# Verificar no log se o Python certo foi carregado
echo "Python em uso:"
which python


# 3. Forçar o PYTHONPATH para carregar os ficheiros da arquitetura MedNeXt locais
# 3. PYTHONPATH e Diretório de Trabalho
PROJ_DIR="/projects/F202500001HPCVLABEPICURE/andresousa615/rempe"
cd $PROJ_DIR
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"


# 4. Executar inferência (modo unbuffered para ver os prints no .log)
python -u test_data_process.py