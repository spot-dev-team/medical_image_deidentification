#!/bin/bash
#SBATCH --job-name=real_test_metrics_t1
#SBATCH --output=logs/real_test_metrics_t1_%j.log
#SBATCH --error=logs/real_test_metrics_t1_%j.err
#SBATCH --account=haslab
#SBATCH --partition=rtx4060
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --exclude=aurora04
#SBATCH --cpus-per-task=2          


# 1. Carregar módulos do Aurora
module purge
module load Python/3.10.8
module load CUDA/12.3.0

# 2. Ativar Ambiente Virtual da Spot
source /home/andresousa615/rempe/mede_code/mednext_venv/bin/activate

# 3. Forçar o PYTHONPATH para carregar os ficheiros da arquitetura MedNeXt locais
export PYTHONPATH="/home/andresousa615/rempe/mede_code:$PYTHONPATH"

# 4. Executar inferência (modo unbuffered para ver os prints no .log)
python -u test_data_process.py