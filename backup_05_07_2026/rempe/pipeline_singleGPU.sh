#!/bin/bash
#SBATCH --job-name=pipeline_1GPU_baseline
#SBATCH --account=f202500001hpcvlabepicureg
#SBATCH --partition=normal-a100-40 
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G 
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_1GPU_baseline_%j.log
#SBATCH --error=logs/pipeline_1GPU_baseline_%j.err

# ==============================================================================
# 0. AMBIENTE E MÓDULOS
# ==============================================================================
module purge
module load Python/3.10.8
module load Miniconda3/23.5.2-0
module load CUDA/12.4.0
module load Xvfb/21.1.8-GCCcore-12.3.0 # Para PyVista Headless
module load Mesa/23.1.4-GCCcore-12.3.0

eval "$(conda shell.bash hook)"
source activate /projects/F202500001HPCVLABEPICURE/andresousa615/env_mede

PROJ_DIR="/projects/F202500001HPCVLABEPICURE/andresousa615/rempe"
cd $PROJ_DIR
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"


# ==============================================================================
# VARIÁVEIS DO PIPELINE
# ==============================================================================
EPOCHS=200
CONFIG_YAML="cluster_train.yaml"
MODEL_BASE="mednext_0.0005_mednext_aurora_16gb_vram" 

# Diretório base gerado por este job específico
RUN_DIR="${PROJ_DIR}/results/train_${MODEL_BASE}_${SLURM_JOB_ID}"

WEIGHTS="${RUN_DIR}/best_${MODEL_BASE}.pt"
TRAIN_CSV_METRICS="${RUN_DIR}/training_metrics.csv"
TRAIN_PLOT_IMG="${RUN_DIR}/training_plot.png"

INFERENCE_DIR="${RUN_DIR}/inference_test_masks"
TEST_CSV_METRICS="${RUN_DIR}/test_metrics.csv"
TEST_TXT_METRICS="${RUN_DIR}/test_metrics.txt"

DIR_PARES="${RUN_DIR}/pares_2d"
DEFACING_REPORT="${RUN_DIR}/defacing_report.txt"

DIR_ORIGINAL="/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/processed_datasets/processed_datasets/teste"

echo "=========================================================="
echo " INICIANDO PIPELINE FIM-A-FIM SINGLE-GPU (Job ID: $SLURM_JOB_ID)"
echo " Head Node: $(hostname)"
echo " Working Dir: $RUN_DIR"
echo "=========================================================="
pipeline_start=$(date +%s)

# ==============================================================================
# FASE 1: TREINO (SINGLE GPU)
# ==============================================================================
echo "[FASE 1] A Iniciar Treino (MedNeXt)..."
start_time=$(date +%s)

srun python train_seg_singleGPU.py --epochs $EPOCHS --config configs/$CONFIG_YAML

end_time=$(date +%s)
echo "[FASE 1 CONCLUÍDA] Duração: $((end_time - start_time)) segundos."
echo "----------------------------------------------------------"

# ==============================================================================
# FASE 2: PLOT DOS RESULTADOS DO TREINO
# ==============================================================================
echo "[FASE 2] A Gerar Gráficos de Treino..."
python /projects/F202500001HPCVLABEPICURE/andresousa615/rempe/postprocessing/plot_results.py --csv_file "$TRAIN_CSV_METRICS" --output_img "$TRAIN_PLOT_IMG"
echo "[FASE 2 CONCLUÍDA]"
echo "----------------------------------------------------------"

# ==============================================================================
# FASE 3: AVALIAÇÃO FÍSICA (TESTE)
# ==============================================================================
echo "[FASE 3] A Iniciar Inferência e Métricas de Teste..."
start_time=$(date +%s)
mkdir -p "$INFERENCE_DIR"

srun python test_singleGPU.py \
    --weights "$WEIGHTS" \
    --out_dir "$INFERENCE_DIR" \
    --metrics_csv "$TEST_CSV_METRICS" \
    --metrics_txt "$TEST_TXT_METRICS"

end_time=$(date +%s)
echo "[FASE 3 CONCLUÍDA] Duração: $((end_time - start_time)) segundos."
echo "----------------------------------------------------------"

# Proteção para as Fases Finais (Evitar que o CPU atrase a geração do dlib)
export OMP_NUM_THREADS=2

# ==============================================================================
# FASE 4: RENDERIZAÇÃO 2D (PYVISTA + XVFB)
# ==============================================================================
echo "[FASE 4] A Renderizar Imagens 2D (Ray-Casting Paralelo)..."
start_time=$(date +%s)

# Uso do xvfb-run diretamente com o python (sem torchrun)
srun xvfb-run -a -s "-screen 0 1600x1200x24 +extension GLX +render" \
    python /projects/F202500001HPCVLABEPICURE/andresousa615/rempe/after_training/process_all_exams_singleGPU.py \
    --dir_defaced "$INFERENCE_DIR" \
    --dir_original "$DIR_ORIGINAL" \
    --dir_pares "$DIR_PARES"

end_time=$(date +%s)
echo "[FASE 4 CONCLUÍDA] Duração: $((end_time - start_time)) segundos."
echo "----------------------------------------------------------"

# ==============================================================================
# FASE 5: CÁLCULO DO DEFACING SCORE (DLIB)
# ==============================================================================
echo "[FASE 5] A Iniciar Cálculo do Defacing Score..."
start_time=$(date +%s)

srun python /projects/F202500001HPCVLABEPICURE/andresousa615/rempe/after_training/calculate_defacing_score_singleGPU.py \
    --dir_pares "$DIR_PARES" \
    --output_report "$DEFACING_REPORT"

end_time=$(date +%s)
echo "[FASE 5 CONCLUÍDA] Duração: $((end_time - start_time)) segundos."
echo "----------------------------------------------------------"

# ==============================================================================
# RESUMO FINAL
# ==============================================================================
pipeline_end=$(date +%s)
pipeline_duration=$((pipeline_end - pipeline_start))
hours=$((pipeline_duration / 3600))
mins=$(((pipeline_duration % 3600) / 60))

echo "=========================================================="
echo " PIPELINE FINALIZADO COM SUCESSO!"
echo " Tempo Total: ${hours}h ${mins}m"
echo " Todos os resultados guardados em:"
echo " $RUN_DIR"
echo "=========================================================="