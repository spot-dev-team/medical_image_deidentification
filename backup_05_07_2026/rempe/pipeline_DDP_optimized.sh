#!/bin/bash
#SBATCH --job-name=rempe_pipeline_4gpus_MEDNEXT_Profiling
#SBATCH --account=f202500001hpcvlabepicureg
#SBATCH --partition=normal-a100-40 
#SBATCH --nodes=2 
#SBATCH --gpus-per-node=4 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G 
#SBATCH --time=12:00:00
#SBATCH --output=logs/pipeline_mednext_8GPUS_%j.log
#SBATCH --error=logs/pipeline_mednext_8GPUS_%j.err

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

# Rede DDP
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
#export GLOO_SOCKET_IFNAME=^lo,docker0
export NCCL_SOCKET_IFNAME=^lo,docker0
export LOGLEVEL=INFO
#export OMP_NUM_THREADS=32 
# ==============================================================================
# OTIMIZAÇÃO DE CPU / CONTROLO DE THREADS (Max 50 threads ativas para 64 cores p/node, assumindo que um core apenas lida com 1 thread)
# ==============================================================================
export OMP_NUM_THREADS=2 #Limita as threads do OpenMP (Open Multi-Processing). É o standard da indústria para paralelismo em C/C++. A maior parte das funções de array do NumPy e operações de tensores do PyTorch feitas no CPU respeitam este limite.
export MKL_NUM_THREADS=2 #Limita as threads do Intel MKL (Math Kernel Library). É uma biblioteca ultra-otimizada para álgebra linear (matrizes). Muitas vezes o PyTorch e o NumPy vêm compilados com o MKL por ser extremamente rápido.
export OPENBLAS_NUM_THREADS=2 #Limita as threads do OpenBLAS. É a alternativa open-source ao Intel MKL. Dependendo de como o teu ambiente Conda foi instalado, o NumPy pode estar a usar o OpenBLAS para multiplicar as matrizes 3D das imagens.
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
#O VecLib é mais relevante em sistemas da Apple, mas o NumExpr é um avaliador numérico rápido muito usado no ecossistema de Data Science do Python (Pandas, SciPy).

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2 # Limita as threads do SimpleITK (Insight Toolkit).
#A biblioteca TorchIO que usas para ler os ficheiros .nii.gz e para aplicar as deformações médicas (como a RandomElasticDeformation) não usa as bibliotecas normais, usa o SimpleITK


# ==============================================================================
# VARIÁVEIS DO PIPELINE
# ==============================================================================
# Define aqui o que o Treino vai usar. Os restantes passos vão ler estas pastas!
EPOCHS=200
CONFIG_YAML="cluster_train.yaml"

# O nome do ficheiro gerado no __init__ do TrainNetwork
MODEL_BASE="mednext_0.0005_mednext_aurora_16gb_vram" 
#mednext_0.0005_mednext_aurora_16gb_vram
#unet3d_0.0005_unet3d_aurora_16gb_vram

# Diretório base gerado por este job específico
# (O teu train_seg_DDP.py cria "train_{model_name}_{SLURM_JOB_ID}")
RUN_DIR="${PROJ_DIR}/results/train_${MODEL_BASE}_${SLURM_JOB_ID}"

# Caminhos Dinâmicos
WEIGHTS="${RUN_DIR}/best_${MODEL_BASE}.pt"
TRAIN_CSV_METRICS="${RUN_DIR}/training_metrics.csv"
TRAIN_PLOT_IMG="${RUN_DIR}/training_plot.png"

INFERENCE_DIR="${RUN_DIR}/inference_test_masks"
TEST_CSV_METRICS="${RUN_DIR}/test_metrics.csv"
TEST_TXT_METRICS="${RUN_DIR}/test_metrics.txt"

DIR_PARES="${RUN_DIR}/pares_2d"
DEFACING_REPORT="${RUN_DIR}/defacing_report.txt"

# Diretório onde tens os ficheiros originais (Niftis puros)
DIR_ORIGINAL="/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/processed_datasets/processed_datasets/teste"

EARLY_STOP_FLAG=""
#se False: ""
# se True: "--earlystop"

# Caminho da Snapshot para Retoma
#SNAPSHOT_ID="1084604"

echo "=========================================================="
echo " INICIANDO PIPELINE FIM-A-FIM (Job ID: $SLURM_JOB_ID)"
echo " Head Node: $head_node"
echo " Working Dir: $RUN_DIR"
echo "2 NÓS - 2 GPUS/NÓ"
echo "=========================================================="
pipeline_start=$(date +%s)

start_time=$(date +%s)
# Adicionada a flag --resume com o caminho da snapshot
srun torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node:29500 \
    train_seg_DDP_optimized.py --epochs $EPOCHS --config $CONFIG_YAML $EARLY_STOP_FLAG

end_time=$(date +%s)
echo "[FASE 1 CONCLUÍDA] Duração: $((end_time - start_time)) segundos."
echo "----------------------------------------------------------"


# ==============================================================================
# FASE 2: PLOT DOS RESULTADOS DO TREINO
# ==============================================================================
echo "[FASE 2] A Gerar Gráficos de Treino..."

# Aqui não usamos DDP, um único nó chega para o Pandas/Matplotlib
python /projects/F202500001HPCVLABEPICURE/andresousa615/rempe/postprocessing/plot_results.py --csv_file "$TRAIN_CSV_METRICS" --output_img "$TRAIN_PLOT_IMG"

echo "[FASE 2 CONCLUÍDA]"
echo "----------------------------------------------------------"


# ==============================================================================
# FASE 3: AVALIAÇÃO FÍSICA (TESTE DDP)
# ==============================================================================
echo "[FASE 3] A Iniciar Inferência e Métricas de Teste..."
start_time=$(date +%s)

# Criar a pasta de inferência explicitamente
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

# Como o PyVista precisa do Xvfb, encapsulamos o srun dentro do xvfb-run
# Passamos --nproc_per_node=4 (em vez de 2) para maximizar o uso de CPUs, já que PyVista é CPU-Bound
xvfb-run -a -s "-screen 0 1600x1200x24 +extension GLX +render" \
    srun torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node:29500 \
    /projects/F202500001HPCVLABEPICURE/andresousa615/rempe/after_training/process_all_exams_ddp.py \
    --dir_defaced "$INFERENCE_DIR" \
    --dir_original "$DIR_ORIGINAL" \
    --dir_pares "$DIR_PARES"

end_time=$(date +%s)
echo "[FASE 4 CONCLUÍDA] Duração: $((end_time - start_time)) segundos."
echo "----------------------------------------------------------"


# ==============================================================================
# FASE 5: CÁLCULO DO DEFACING SCORE (DLIB DDP)
# ==============================================================================
echo "[FASE 5] A Iniciar Cálculo do Defacing Score..."
start_time=$(date +%s)

srun torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node:29500 \
    /projects/F202500001HPCVLABEPICURE/andresousa615/rempe/after_training/calculate_defacing_score_ddp.py \
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