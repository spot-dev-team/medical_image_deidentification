# Manual de Contexto: Infraestrutura HPC da Spot (Deucalion e Lab Cluster)

## PARTE I: Supercomputador Deucalion

### 1. Visão Geral da Infraestrutura
O Deucalion é um supercomputador (HPC) localizado em Portugal, gerido pelo MACC. A submissão, agendamento e alocação de recursos computacionais (CPUs, GPUs, RAM) são controlados exclusivamente pelo **SLURM Workload Manager**.
* **Acesso:** Feito via nós de login (ex: `login.deucalion.macc.fccn.pt`).
* **Regra de Ouro:** Nunca se executa código pesado diretamente no nó de login. Todo o treino de modelos ou processamento de dados tem de ser submetido para os nós de computação via SLURM.

### 2. Estrutura de Diretórios
A infraestrutura tem diferentes sistemas de ficheiros. É crítico usar o correto:
* `$HOME` (ex: `/home/andresousa615`): Quota muito pequena. Apenas para scripts básicos e ficheiros de configuração.
* `$SCRATCH` ou `/projects/` (ex: `/projects/F202500001HPCVLABEPICURE/andresousa615/`): Armazenamento de alta capacidade e alta performance (geralmente Lustre). **Todos os datasets (ADNI, IXI), ambientes Conda, contentores Apptainer e ficheiros `.h5` / `.pt` devem residir e ser executados a partir daqui.**

### 3. Gestão de Software (Módulos)
O Deucalion usa o sistema `module` para carregar software otimizado para a arquitetura:
* `module avail`: Lista o software disponível.
* `module load Miniconda3/23.5.2-0`: Carrega o Conda do sistema.
* `module load CUDA/12.2.2`: Carrega drivers e bibliotecas C++ para as GPUs.
* `module purge`: Limpa todos os módulos carregados (recomendado no início de cada script para garantir um ambiente limpo).

### 4. O Agendador SLURM
O SLURM gere filas de espera baseadas em partições e recursos solicitados.

#### 4.1. Comandos Essenciais de Terminal
* `sbatch <script.sh>`: Submete um job para a fila de espera. Devolve um JobID.
* `squeue -u <username>`: Mostra o estado dos jobs do utilizador (PD = Pendente, R = A correr).
* `scancel <JobID>`: Cancela/Mata um job específico.
* `sacct -j <JobID> --format=JobID,JobName,State,Elapsed,Timelimit`: Consulta o histórico e o tempo real de execução de um job terminado.

#### 4.2. Estrutura Padrão de um Job Script (.sh)
O ficheiro Bash submetido pelo `sbatch` contém diretivas `#SBATCH` no cabeçalho, seguidas do código de execução. 

**Exemplo prático para Treino de Deep Learning em GPU:**

```bash
#!/bin/bash
#SBATCH --job-name=Spot_Model_Training
#SBATCH --account=f202500001hpcvlabepicureg  # Projeto/Conta de faturação
#SBATCH --partition=normal-a100-40         # Partição com GPUs A100 (40GB)
#SBATCH --nodes=1                          # Número de nós físicos solicitados
#SBATCH --ntasks=1                         # Número de processos principais
#SBATCH --gpus=1                           # Número de GPUs solicitadas
#SBATCH --cpus-per-task=32                 # No Deucalion, alocar ~32 CPUs por cada GPU A100
#SBATCH --mem=128G                         # Memória RAM do sistema (útil para volumes 3D)
#SBATCH --time=18:00:00                    # Walltime máximo (HH:MM:SS)
#SBATCH --output=logs/treino_%j.log        # Ficheiro de standard output (%j é substituído pelo JobID)
#SBATCH --error=logs/treino_%j.err         # Ficheiro de standard error

# 1. Preparar o ambiente base
module purge
module load Miniconda3/23.5.2-0
module load CUDA/12.2.2

# 2. Ativar o ambiente Python (Caminho absoluto obrigatório)
eval "$(conda shell.bash hook)"
conda activate /projects/F202500001HPCVLABEPICURE/andresousa615/env_mede

# 3. Mudar para a diretoria de trabalho correta
PROJ_DIR="/projects/F202500001HPCVLABEPICURE/andresousa615/rempe"
cd $PROJ_DIR

# 4. Executar o código em modo Unbuffered (-u) para ter logs em tempo real
python -u train_seg.py