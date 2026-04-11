#!/bin/bash
#SBATCH --job-name=inference_deucalion_t1
#SBATCH --output=logs/inference_deucalion_t1_%j.log
#SBATCH --error=logs/inference_deucalion_t1_%j.err
#SBATCH --account=f202500001hpcvlabepicureg
#SBATCH --partition=normal-a100-40         
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32           
#SBATCH --mem=64G 
#SBATCH --time=3:00:00       


echo "A iniciar tarefa de inferência na arquitetura Aurora..."
echo "Nó alocado: $SLURM_JOB_NODELIST"

# Carregar o ambiente correto (idêntico ao treino)
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

# 4. Verificação de GPU
echo "A iniciar job no nó: $(hostname)"
python -c "import torch; print(f'GPU detetada: {torch.cuda.is_available()} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NENHUMA\"}')"


# Executar o script
python run_inference.py --e 100 --gpu 0

echo "Inferência concluída."