#!/bin/bash
#SBATCH --job-name=build-mede
#SBATCH --partition=dev-x86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --account=f202500001hpcvlabepicure

# 1. Configurar pastas (Cache no Projects para poupar a Home)
PROJ_DIR="/projects/F202500001HPCVLABEPICURE/andresousa615"
export SINGULARITY_CACHEDIR="${PROJ_DIR}/cache"
# Deixamos o TMPDIR na Home temporariamente (como pediste) para ver se resolve o erro de mapeamento
# Mas atenção: se der erro de "No space left on device", teremos de voltar ao Projects.
export SINGULARITY_TMPDIR="${HOME}/tmp_build"
mkdir -p $SINGULARITY_CACHEDIR $SINGULARITY_TMPDIR

IMAGE_PATH="${PROJ_DIR}/mede_v1.sif"
cd ${PROJ_DIR}

echo "A tentar build SEM fakeroot..."
# Build sem --fakeroot porque o novo .def não usa apt-get
singularity build "${IMAGE_PATH}" Singularity.def

if [ $? -eq 0 ]; then
    echo "Sucesso!"
    rm -rf $SINGULARITY_TMPDIR
else
    echo "O build falhou novamente."
    echo "Dica: Executa 'cat /etc/subuid | grep $USER' no terminal."
    echo "Se não aparecer nada, tens de pedir ao suporte do Deucalion para ativar o fakeroot para ti."
    exit 1
fi
