Fico extremamente satisfeito por termos ultrapassado esse obstáculo da rede do cluster! É exatamente este tipo de conhecimento prático que distingue um utilizador comum de um verdadeiro especialista em HPC (High Performance Computing).

Para garantir que não perdes este conhecimento e que tens uma base sólida para qualquer projeto futuro, elaborei o documento Markdown solicitado. Este ficheiro condensa todas as boas práticas de paralelização PyTorch, com foco absoluto na arquitetura e nas "nuances" do **Deucalion**.

Podes copiar o bloco abaixo e guardá-lo como `DEUCALION_PYTORCH_DDP_GUIDE.md`.

---

```markdown
# 🚀 Guia Definitivo: Paralelização de Modelos PyTorch no Deucalion

Este documento serve como *Base de Conhecimento* (Contexto) para a conversão, execução e depuração de modelos PyTorch distribuídos (DDP, FSDP) no supercomputador português Deucalion.

Ele aborda as características do hardware, as adaptações necessárias no código Python e as configurações estritas dos scripts SLURM.

---

## 1. Conhecer o Hardware (Deucalion)

O Deucalion possui partições heterogéneas. Para treino acelerado em Deep Learning, focamo-nos nas partições A100:
* **`dev-a100-40`**: Partição de desenvolvimento. **Limite estrito de 1 nó**. Ideal para testes Single-Node Multi-GPU.
* **`a100-40` / `normal-a100-40`**: Partição de produção. Permite múltiplos nós.
* **Arquitetura por Nó:** Cada servidor (Nó) possui **4 GPUs NVIDIA A100 (40GB ou 80GB)** conectadas internamente via **NVLink** (latência quase nula).
* **Comunicação Inter-Nós:** Feita através de placas de rede de alta velocidade **InfiniBand (200 Gb/s)**. O PyTorch precisa de ser explicitamente instruído para usar esta rede em vez da rede de gestão.

---

## 2. Adaptação do Código PyTorch (Training Loop)

Para converter um script PyTorch normal (Single-GPU) num script Distribuído (Multi-GPU/Multi-Node), quatro componentes principais têm de ser alterados:

### A. Inicialização do Grupo de Processos (`ddp_setup`)
O utilitário `torchrun` (usado no Deucalion) injeta variáveis de ambiente cruciais. Devemos lê-las para configurar a GPU correta.

```python
import os
import torch
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup():
    # LOCAL_RANK: ID da GPU dentro do nó físico atual (0, 1, 2 ou 3)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Inicializa o backend NCCL (NVIDIA Collective Communications Library)
    init_process_group(backend="nccl")
```

### B. Distribuição de Dados (`DistributedSampler`)

O modelo paralelizado exige que cada GPU veja uma fatia diferente dos dados simultaneamente.

```python
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

def prepare_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False, # O Sampler já faz o shuffle
        sampler=DistributedSampler(dataset)
    )

# NO LOOP DE TREINO:
# É obrigatório chamar set_epoch no início de cada época para garantir
# que o shuffle muda a cada iteração.
train_data.sampler.set_epoch(epoch)
```

### C. O Envolvimento do Modelo (`DDP Wrapper`)

O modelo deve ser movido para a GPU local e depois envolvido pelo invólucro DDP.

```python
from torch.nn.parallel import DistributedDataParallel as DDP

local_rank = int(os.environ["LOCAL_RANK"])
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

### D. Checkpoints Resilientes (Save & Load)

Apenas o processo mestre global (`RANK == 0`) deve escrever no disco para evitar corrupção. O `state_dict` do otimizador **tem de ser guardado** para que falhas de hardware não destruam o *momentum*.

```python
global_rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])

# SAVE (Apenas no Rank 0)
if global_rank == 0:
    snapshot = {
        "MODEL_STATE": model.module.state_dict(), # Usar .module devido ao DDP
        "OPTIMIZER_STATE": optimizer.state_dict(),
        "EPOCHS_RUN": epoch,
    }
    torch.save(snapshot, "snapshot.pt")

# LOAD (Em todas as GPUs na inicialização)
loc = f"cuda:{local_rank}"
snapshot = torch.load("snapshot.pt", map_location=loc)
model.load_state_dict(snapshot["MODEL_STATE"])
optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
```

---

## 3. Submissão SLURM e `torchrun` no Deucalion

A submissão correta é o passo mais crítico. O SLURM gere os recursos e o `torchrun` gere os processos Python.

### Cenário A: Single-Node (Até 4 GPUs no mesmo servidor)

Utiliza a flag `--standalone` do `torchrun`. Não é necessário gerir IPs.

```bash
#!/bin/bash
#SBATCH --partition=dev-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2     # Escolher entre 1 a 4
#SBATCH --cpus-per-task=32

module load Python/3.10.8 CUDA/12.4.0
source /caminho/para/venv/bin/activate

srun torchrun \\
    --standalone \\
    --nnodes=1 \\
    --nproc_per_node=2 \\
    teu_script.py
```

### Cenário B: Multi-Node (Múltiplos servidores, Múltiplas GPUs)

Exige a configuração explícita da rede e da interface de *Rendezvous*.

```bash
#!/bin/bash
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32

module load Python/3.10.8 CUDA/12.4.0
source /caminho/para/venv/bin/activate

# 1. Obter o NOME do Head Node (Deixar o DNS do SLURM traduzir para InfiniBand)
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}

# 2. Ignorar interfaces de rede locais/virtuais que causam Timeouts
export GLOO_SOCKET_IFNAME=^lo,docker0
export NCCL_SOCKET_IFNAME=^lo,docker0
export LOGLEVEL=INFO

# 3. Iniciar torchrun (CUIDADO com espaços invisíveis após as barras '\\')
srun torchrun \\
    --nnodes=$SLURM_NNODES \\
    --nproc_per_node=4 \\
    --rdzv_id=$SLURM_JOB_ID \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint=$head_node:29500 \\
    teu_script.py
```

---

## 4. Teste, Avaliação e Inferência em Distribuído

### Validação/Teste (Durante o Treino)

Se avaliares o modelo durante o treino distribuído:

1. Usa um `DistributedSampler` também para o `val_dataloader`.
2. Como cada GPU calculará métricas (ex: `loss` ou `accuracy`) sobre partes diferentes do *Validation Set*, precisas de utilizar as primitivas de comunicação do PyTorch para agregar os resultados antes do print.

```python
import torch.distributed as dist

# Após calcular a loss local de validação na GPU
val_loss_tensor = torch.tensor([local_val_loss], device=local_rank)

# Soma o val_loss_tensor de todas as GPUs na GPU 0
dist.reduce(val_loss_tensor, dst=0, op=dist.ReduceOp.SUM)

if global_rank == 0:
    avg_val_loss = val_loss_tensor.item() / world_size
    print(f"Validation Loss Global: {avg_val_loss}")
```

### Inferência em Produção

- **Modelos Pequenos/Médios:** Em geral, a inferência final não é feita com DDP. Carrega o `state_dict` guardado num modelo "virgem" (sem o wrapper DDP) e executa a inferência numa única GPU para evitar *overhead* de comunicação.
- **Modelos Gigantes (LLMs):** Se o modelo não couber numa única GPU de 40GB, não usarás DDP para inferência. Transita para **Tensor Parallelism (TP)** ou **Pipeline Parallelism (PP)** nativo (ex: usando a biblioteca `vLLM` ou `torch.distributed.tensor`).

---

## 5. Armadilhas Comuns no Deucalion (Checklist de Debug)

1. **`command not found` no SLURM:** Verifica se há espaços em branco acidentais à direita das barras invertidas `\\` no teu script Bash.
2. **`Socket Timeout` (60000ms):** PyTorch tentou comunicar pela rede errada. Garante que defines `GLOO_SOCKET_IFNAME=^lo,docker0` e usas `$head_node` diretamente no `rdzv_endpoint` em vez de adivinhares o IP.
3. **Loss dispara após resumo de Checkpoint:** Esqueceste-te de salvar/carregar o `state_dict` do otimizador e do *scheduler*.
4. **Múltiplas escritas no mesmo ficheiro:** Estás a usar `if local_rank == 0` em multi-node em vez de `if global_rank == 0` para guardar ficheiros, levando a corrupção cruzada.