#!/bin/bash
#SBATCH --job-name=MedNeXt_inference_t3
#SBATCH --account=haslab
#SBATCH --output=logs/inference_t3_%j.log
#SBATCH --partition=rtx4060
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --exclude=aurora04         
#SBATCH --cpus-per-task=2          




echo "A iniciar tarefa de inferência na arquitetura Aurora..."
echo "Nó alocado: $SLURM_JOB_NODELIST"

# Carregar o ambiente correto (idêntico ao treino)
module purge
module load Python/3.10.8
module load CUDA/12.3.0

# Ativar o ambiente virtual
source /home/andresousa615/rempe/mede_code/mednext_venv/bin/activate

# Otimização de memória PyTorch para inferência fluida
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Executar o script
python run_inference.py

echo "Inferência concluída."