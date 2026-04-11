#!/bin/bash
#SBATCH --job-name=plot_treino
#SBATCH --account=f202500001hpcvlabepicureg
#SBATCH --partition=normal-a100-40 
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G 
#SBATCH --time=00:15:00  
#SBATCH --output=logs/plot_%j.log
#SBATCH --error=logs/plot_%j.err



TRAIN_CSV_METRICS="/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/results/train_mednext_0.0005_mednext_aurora_16gb_vram_1084609/training_metrics.csv"
TRAIN_PLOT_IMG="/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/results/train_mednext_0.0005_mednext_aurora_16gb_vram_1084609/training_plot.png"

# ==============================================================================
# 2. AMBIENTE E MÓDULOS
# ==============================================================================
module purge
module load Python/3.10.8
module load Miniconda3/23.5.2-0

eval "$(conda shell.bash hook)"
source activate /projects/F202500001HPCVLABEPICURE/andresousa615/env_mede

PROJ_DIR="/projects/F202500001HPCVLABEPICURE/andresousa615/rempe"
cd $PROJ_DIR
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"

# ==============================================================================
# 3. EXECUÇÃO
# ==============================================================================
echo "=========================================================="
echo " A GERAR GRÁFICOS DE TREINO"
echo " CSV Input: $TRAIN_CSV_METRICS"
echo " Imagem Output: $TRAIN_PLOT_IMG"
echo "=========================================================="
start_time=$(date +%s)

python /projects/F202500001HPCVLABEPICURE/andresousa615/rempe/postprocessing/plot_results.py \
    --csv_file "$TRAIN_CSV_METRICS" \
    --output_img "$TRAIN_PLOT_IMG"

end_time=$(date +%s)
echo "[CONCLUÍDO] Duração: $((end_time - start_time)) segundos."
echo "=========================================================="