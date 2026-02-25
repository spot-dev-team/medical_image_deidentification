#!/bin/bash
#SBATCH --job-name=test_t3
#SBATCH --output=logs/test_t3_%j.log
#SBATCH --partition=rtx4060
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00     
#SBATCH --mem=16G
#SBATCH --account=haslab
#SBATCH --exclude=aurora04  




module purge
module load Python/3.10.8
module load CUDA/12.3.0
source /home/andresousa615/rempe/mede_code/mednext_venv/bin/activate

python train_test_data.py