#!/bin/bash
#SBATCH --job-name=inference_deucalion_t1
#SBATCH --account=f202500001hpcvlabepicureg
#SBATCH --partition=normal-a100-40         
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32           
#SBATCH --mem=100G 
#SBATCH --time=03:00:00       
#SBATCH --output=logs/inference_counitng_preprocessing_%j.out
#SBATCH --error=logs/inference_counitng_preprocessing_%j.err

echo "=========================================================="
echo " A iniciar tarefa de inferência na arquitetura Aurora..."
echo " Nó alocado: $SLURM_JOB_NODELIST"
echo "=========================================================="

# ==============================================================================
# 1. AMBIENTE E MÓDULOS
# ==============================================================================
module purge
module load Python/3.10.8
module load Miniconda3/23.5.2-0
module load CUDA/12.4.0  # Adicionado para garantir compatibilidade com o PyTorch
module load CMake/3.24.3
module load GCC/12.2.0

eval "$(conda shell.bash hook)"
source activate /projects/F202500001HPCVLABEPICURE/andresousa615/env_mede

# ==============================================================================
# 2. OTIMIZAÇÃO DE CPU / CONTROLO DE THREADS (Single GPU)
# ==============================================================================
# Como tens 32 CPUs exclusivos para 1 processo, vamos libertar 16 threads 
# para a matemática de CPU. Isto acelera drasticamente as leituras do Nibabel
# e os redimensionamentos espaciais do TorchIO/SimpleITK!
export OMP_NUM_THREADS=16 
export MKL_NUM_THREADS=16 
export OPENBLAS_NUM_THREADS=16 
export VECLIB_MAXIMUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=16 

# ==============================================================================
# 3. PYTHONPATH E DIRETÓRIO DE TRABALHO
# ==============================================================================
PROJ_DIR="/projects/F202500001HPCVLABEPICURE/andresousa615/rempe"
cd $PROJ_DIR
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"

# ==============================================================================
# 4. EXECUÇÃO
# ==============================================================================
echo "A iniciar job no nó: $(hostname)"
python -c "import torch; print(f'GPU detetada: {torch.cuda.is_available()} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NENHUMA\"}')"

start_time=$(date +%s)

# Boa prática: Usar o srun para garantir o binding correto dos recursos alocados
srun python run_inference.py --gpu 0

end_time=$(date +%s)
echo "=========================================================="
echo "Inferência concluída em $((end_time - start_time)) segundos."
echo "=========================================================="