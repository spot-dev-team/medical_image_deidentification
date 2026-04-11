#!/bin/bash
#SBATCH --job-name=resume_unet_pipeline
#SBATCH --account=f202500001hpcvlabepicureg
#SBATCH --partition=normal-a100-40 
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=128
#SBATCH --mem=400G 
#SBATCH --time=02:00:00  
#SBATCH --output=logs/resume_pipeline_unet_%j.log
#SBATCH --error=logs/resume_pipeline_unet_%j.err

# ==============================================================================
# 0. AMBIENTE E MÓDULOS
# ==============================================================================
module purge
module load Python/3.10.8
module load Miniconda3/23.5.2-0
module load CUDA/12.4.0
module load Xvfb/21.1.8-GCCcore-12.3.0
module load Mesa/23.1.4-GCCcore-12.3.0

eval "$(conda shell.bash hook)"
source activate /projects/F202500001HPCVLABEPICURE/andresousa615/env_mede

PROJ_DIR="/projects/F202500001HPCVLABEPICURE/andresousa615/rempe"
cd $PROJ_DIR
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"

# Rede DDP
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
export NCCL_SOCKET_IFNAME=^lo,docker0
export LOGLEVEL=INFO
export OMP_NUM_THREADS=32 

# ==============================================================================
# VARIÁVEIS (HARDCODED PARA A TUA ÚLTIMA EXECUÇÃO DE SUCESSO)
# ==============================================================================
# Apontamos diretamente para a pasta do ID 1077344
MODEL_BASE="unet3d_0.0005_unet3d_aurora_16gb_vram"
#"mednext_0.0005_mednext_aurora_16gb_vram"
#unet3d_0.0005_unet3d_aurora_16gb_vram

RUN_DIR="${PROJ_DIR}/results/train_${MODEL_BASE}_1094177"

# Variáveis da Fase 3
WEIGHTS="${RUN_DIR}/best_${MODEL_BASE}.pt" 
TEST_CSV_METRICS="${RUN_DIR}/per_exam_metrics.csv"
TEST_TXT_METRICS="${RUN_DIR}/real_test_metrics.txt"

# Variáveis da Fase 4 e 5
INFERENCE_DIR="${RUN_DIR}/inference_test_masks"
DIR_PARES="${RUN_DIR}/pares_2d"
DEFACING_REPORT="${RUN_DIR}/defacing_report.txt"
DIR_ORIGINAL="/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/processed_datasets/processed_datasets/teste"

TRAIN_CSV_METRICS="${RUN_DIR}/training_metrics.csv"
TRAIN_PLOT_IMG="${RUN_DIR}/training_plot.png"

echo "=========================================================="
echo " A RETOMAR PIPELINE A PARTIR DA FASE 2"
echo " Alvo: $RUN_DIR"
echo "=========================================================="
pipeline_start=$(date +%s)

# ==============================================================================
# FASE 2: GERAÇÃO DE GRÁFICOS DE TREINO
# ==============================================================================
#echo "[FASE 2] A gerar gráficos de treino..."
#start_time=$(date +%s)

# Aqui não usamos DDP, um único nó chega para o Pandas/Matplotlib
#python /projects/F202500001HPCVLABEPICURE/andresousa615/rempe/postprocessing/plot_results.py --csv_file "$TRAIN_CSV_METRICS" --output_img "$TRAIN_PLOT_IMG"

#end_time=$(date +%s)
#echo "[FASE 2 CONCLUÍDA] Duração: $((end_time - start_time)) segundos."
#echo "----------------------------------------------------------"



# ==============================================================================
# FASE 3: AVALIAÇÃO FÍSICA E GERAÇÃO DE MÁSCARAS/ANON (TESTE DDP)
# ==============================================================================
echo "[FASE 3] A Iniciar Inferência e Métricas de Teste..."
start_time=$(date +%s)

mkdir -p "$INFERENCE_DIR"

srun torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node:29500 \
    test_DDP.py \
    --weights "$WEIGHTS" \
    --out_dir "$INFERENCE_DIR" \
    --metrics_csv "$TEST_CSV_METRICS" \
    --metrics_txt "$TEST_TXT_METRICS"

end_time=$(date +%s)
echo "[FASE 3 CONCLUÍDA] Duração: $((end_time - start_time)) segundos."
echo "----------------------------------------------------------"


# ==============================================================================
# FASE 4: RENDERIZAÇÃO 2D (PYVISTA DDP + XVFB)
# ==============================================================================
echo "[FASE 4] A Renderizar Imagens 2D (Ray-Casting Paralelo)..."
start_time=$(date +%s)

xvfb-run -a -s "-screen 0 1600x1200x24 +extension GLX +render" \
    srun torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node:29500 \
    after_training/process_all_exams_ddp.py \
    --dir_defaced "$INFERENCE_DIR" \
    --dir_original "$DIR_ORIGINAL" \
    --dir_pares "$DIR_PARES"

end_time=$(date +%s)
echo "[FASE 4 CONCLUÍDA] Duração: $((end_time - start_time)) segundos."
echo "----------------------------------------------------------"


# ==============================================================================
# FASE 5: CÁLCULO DO DEFACING SCORE (DLIB DDP)
# ==============================================================================
echo "[FASE 5] A Iniciar Auditoria Biométrica (Defacing Score)..."
start_time=$(date +%s)

srun torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node:29500 \
    after_training/calculate_defacing_score_ddp.py \
    --dir_pares "$DIR_PARES" \
    --output_report "$DEFACING_REPORT"

end_time=$(date +%s)
echo "[FASE 5 CONCLUÍDA] Duração: $((end_time - start_time)) segundos."
echo "----------------------------------------------------------"

pipeline_end=$(date +%s)
echo "TEMPO TOTAL DESTA RETOMA: $((pipeline_end - pipeline_start)) segundos."