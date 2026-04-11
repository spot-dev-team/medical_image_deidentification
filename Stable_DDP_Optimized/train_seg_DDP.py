# -*- coding: utf-8 -*-
import os
import shutil
import yaml
from pathlib import Path
import torch, random
import torch.optim as optim
import torch.distributed as dist
import torchvision
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import logging
from model import ConvNext, UNet3D, Mednext
from dataset_128 import get_loaders
from utils.validation import segmentation_validation, plot_segmentation
from utils import utilities
from utils.losses import DiceLoss
from datetime import timedelta

#n

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from utils.profiling import TrainingProfiler

torchvision.disable_beta_transforms_warning()

logging.basicConfig(
    encoding="utf-8", level=logging.DEBUG, format="%(levelname)s - %(message)s"
)
parser = argparse.ArgumentParser(prog="Training")

parser.add_argument("--epochs", type=int, default=250, help="Number of epochs for training")
parser.add_argument("--earlystop", action="store_true", help="Whether to use early stopping")
parser.add_argument(
    "--log", type=str, default="INFO", help="Define debug level. Defaults to INFO."
)

parser.add_argument(
    "--config",
    type=str,
    help="Path to configuration file",
    default="train_skullstrip.yaml",
)

# Argumento para continuar apartir de um snapshot
    
parser.add_argument("--resume_id", type=str, default=None, help="Job ID to resume from")

#n
def ddp_setup():
    """
    Inicializa o grupo de processos distribuídos.
    O torchrun injeta as variáveis de ambiente necessárias.
    """
    # LOCAL_RANK: ID da GPU dentro do nó físico atual (0, 1, 2 ou 3 no Deucalion)
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Define a GPU ativa para este processo
    torch.cuda.set_device(local_rank)

    # Inicializa o backend NCCL (NVIDIA Collective Communications Library)
    init_process_group(backend="nccl", timeout=timedelta(hours=1))
    
    logging.debug(f"DDP Setup concluído para Global Rank: {os.environ['RANK']}, Local Rank: {local_rank}")


def set_seed(seed: int = 42) -> None:
    """Set seeds for the libraries numpy, random, torch and torch.cuda.

    Args:
        seed (int, optional): Seed to be used. Defaults to `42`.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.debug(f"Random seed set as {seed}")


class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""

    def __init__(
        self,
        patience=40,
        verbose=True,
        delta=0,
        monitor="val_loss",
        op_type="min",
        logger=None,
    ):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type
        self.logger = logger

        if self.op_type == "min":
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):
        score = -val_score if self.op_type == "min" else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0
        else:
            self.counter += 1
            logging.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def print_and_update(self, val_score):
        """print_message when validation score decrease."""
        if self.verbose:
            logging.info(
                f"{self.monitor} optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...",
            )
        self.val_score_min = val_score


class TrainNetwork:
    """Train a neural network based on PyTorch architecture.

    Args:
        args (dict): Dictionary containing user-specified settings.
        config (dict): Dictionary containing settings set in a yaml-config file.
    """

    def __init__(self, args: dict, config: dict) -> None:
        """
        Initializes the TrainNetwork class.

        Args:
            args (dict): A dictionary containing the command-line arguments.
            config (dict): A dictionary containing the configuration settings.

        Attributes:
            args (dict): A dictionary containing the command-line arguments.
            config (dict): A dictionary containing the configuration settings.
            train_path (str): The path to the training data.
            val_path (str): The path to the validation data.
            base_output (Path): The base output path.
            init_lr (float): The initial learning rate.
            epochs (int): The number of epochs.
            model_name (str): The name of the model.
            device (torch.device): The device to be used for training.

        Returns:
            None
        """
        self.scheduler = None
        self.optimizer = None
        self.early_stopping = None
        self.early_stop_active = args.earlystop
        self.loss = None
        self.val_loader = None
        self.train_loader = None
        self.save_folder = None

        #n
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])

        self.metric_list = None
        self.metric = None
        self.total_train_loss = None
        self.lr = None
        self.args: dict = args
        self.config: dict = config
        self.current_job_id = os.environ.get("SLURM_JOB_ID", "standalone")
        self.train_path: str = config["train_path"]
        self.val_path: str = config["val_path"]
        self.base_output: Path = config["base_output"]
        self.init_lr: float = config["lr"]
        self.epochs: int = args.epochs
        self.epochs_run = 0
        #self.epochs_run: int = args.e
        self.model_name: str = f"{config['model']}_{config['lr']}_{config['comment']}"

        # Agora cada treino terá a sua própria pasta: results/train_mednext_0.0005_1071184/
        self.save_folder = f"{self.base_output}/train_{self.model_name}_{self.current_job_id}"
        
        # Caminho para o CSV de métricas do treino
        self.csv_metrics_path = os.path.join(self.save_folder, "training_metrics.csv")
        
        if self.global_rank == 0:
            # Cria a pasta única se ela não existir
            Path(self.save_folder).mkdir(parents=True, exist_ok=True)
            logging.info(f"Pasta de resultados única criada: {self.save_folder}")
            
            # Criar o ficheiro CSV e escrever o cabeçalho (apenas se for treino novo)
            if not os.path.exists(self.csv_metrics_path) or self.args.resume_id is None:
                with open(self.csv_metrics_path, 'w', encoding='utf-8') as f:
                    f.write("epoch,train_loss,val_loss,dsc,iou\n")
        
        # Sincronizar todas as GPUs para garantir que o Rank 0 acabou de criar a pasta
        # antes de qualquer outra GPU tentar aceder ou guardar ficheiros
        dist.barrier() #prende as gpus aqui até todas as GPUs estarem neste ponto, assim garantimso que nenhuma GPU tenta aceder ao CSV sem primeiro o excel estar criado.
 

        # ── PROFILER Inicialização────────────────────────────────────────────────────
        self.profiler = TrainingProfiler(
            save_dir     = self.save_folder,
            global_rank  = self.global_rank,
            enabled      = True,       # False para desligar sem remover código
            profile_torch= True,       # torch.profiler completo (Chrome trace)
            profile_epoch= 2,          # Epoch 2 será traçado (0-indexed)
        )
        # ────────────────────────────────────────────────────────────────

        # Se receber --resume_id, ele procura esse ficheiro. 
        # Caso contrário, cria um novo com o ID do job atual.
        target_id = args.resume_id if args.resume_id else self.current_job_id
        self.snapshot_path = f"{self.base_output}/snapshot_{self.model_name}_{target_id}.pt"
        
        # 2. Definir dispositivo FÍSICO (baseado no local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")

        # 3. Inicializar e embrulhar a rede
        self._init_network(self.config)


    def _init_network(self, configuration: dict) -> None:
        """
        Initializes the network based on the provided configuration.
        Args:
            configuration (dict): A dictionary containing the configuration parameters.
        Raises:
            ValueError: If an invalid model is selected.
        Returns:
            None
        """
        network_classes = {
            "convnext": ConvNext,
            "unet3d": UNet3D,
            "mednext": Mednext
        }
        network_class = network_classes.get(configuration["model"])
        if network_class is None:
            raise ValueError("Select valid model!")
        else:
            print(f"Selected model: {configuration['model']}")

        # 1. Instancia o modelo e move-o para a GPU física correta
        self.model = network_class().to(self.device)
        
        # Converter BatchNorm normal para SyncBatchNorm
        # Isto é OBRIGATÓRIO para batch_size=1 em DDP!
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        
        #n
        # O wrap DDP
        # Sincroniza os gradientes ao longo do WORLD_SIZE
        self.model = DDP(self.model, device_ids=[self.local_rank])
        #self.profiler.register_ddp_model(self.model)   # ← mede all-reduces de gradientes

    def _load_snapshot(self):
        """Carrega o snapshot para retomar o treino em caso de falha."""
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(self.snapshot_path, map_location=loc)
        
        # Como chamamos isto no main (onde o modelo já é DDP), usamos .module
        self.model.module.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        
        if "OPTIMIZER_STATE" in snapshot and self.optimizer is not None:
            self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
            
        logging.info(f"Resuming training from snapshot at Epoch {self.epochs_run}")


    def _save_snapshot(self, epoch):
        """Guarda o estado atual do treino para tolerância a falhas."""
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "OPTIMIZER_STATE": self.optimizer.state_dict(), # Mantém o balanço do treino
        }
        torch.save(snapshot, self.snapshot_path)
        logging.info(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    @utilities.timer
    def train_fn(self) -> None:
        """Train function.

        Calculates loss per batch, performs backpropagation and optimizer step.

        Args:
            self: self object of the class.

        Returns:
            None.
        """

        # 1. OBRIGATÓRIO EM DDP: Baralhar os dados de forma diferente a cada época
        self.train_loader.sampler.set_epoch(self.epoch)

        # 2. Mostrar a barra de progresso apenas no Mestre Global (RANK 0)
        loop = tqdm(self.train_loader, disable=(self.global_rank != 0))
        self.total_train_loss = 0


        for batch_idx, data_dict in enumerate(self.profiler.iter_dataloader(loop)):
            
            with self.profiler.phase("data_to_gpu"):
                data = data_dict["image"].to(device=self.device, non_blocking=True)
                targets = data_dict["mask"].to(device=data.device, non_blocking=True)

            with self.profiler.phase("forward"):
                predictions = self.model(data.float())
                loss = self.loss(predictions, targets)

            with self.profiler.phase("backward"):# Nota: o all-reduce de gradientes DDP ocorre aqui. O comm_hook
                                                 # cronometra este overhead separadamente.
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()


            with self.profiler.phase("optimizer"):
                self.optimizer.step() # O torch.profiler.step() é chamado dentro deste context manager


            self.total_train_loss += loss.item()

            if torch.isnan(loss):
                logging.warning("-- Loss nan --")
                break

            loop.set_postfix(loss=loss.item())

        self.scheduler.step()
        self.lr = self.scheduler.get_last_lr()[0]
        self.total_train_loss = self.total_train_loss / len(self.train_loader)


    @utilities.timer
    def validation(self) -> None:
        """Performs validation after each epoch.

        This method saves one batch of the validation set in the save-folder
        and calculates the dice score as well as the validation loss.
        The results are logged to Weights & Biases.

        Args:
            self: Instance of `TrainNetwork` class.

        Returns:
            None
        """
        self.model.eval()
        total_validation_loss = 0
        total_metrics = None
        
        # Desliga a barra de progresso nas GPUs secundárias
        loop = tqdm(self.val_loader, disable=(self.global_rank != 0))

        n = random.randint(0, len(self.val_loader) - 1)

        for batch_idx, data_dict in enumerate(loop):

            with self.profiler.phase("val_forward"):
                data = data_dict["image"].to(device=self.device, non_blocking=True)

                # forward
                with torch.no_grad():
                    predictions = self.model(data.float())

            targets = data_dict["mask"].to(device=data.device, non_blocking=True)

            predictions_sum = predictions.float()
            loss = self.loss(predictions, targets)

            loop.set_postfix(loss=loss.item())
            
            with self.profiler.phase("val_metrics"):
                val_metrics = segmentation_validation(predictions_sum, targets)

            if total_metrics is None:
                # initialize total_metrics with keys from val_metrics and all values set to 0
                total_metrics = {key: 0 for key in val_metrics.keys()}
            for key in total_metrics.keys():
                total_metrics[key] += val_metrics[key].item()

            total_validation_loss += loss.item()

            #gera as imagens durante o treino.
            # APENAS O RANK 0 GERA AS IMAGENS
            #if self.global_rank == 0 and batch_idx == n and self.epoch % 1 == 0:
            #    plot_segmentation(
            #        data, predictions_sum, targets, self.epoch, self.save_folder
            #    )

        # 1. Calcular médias locais
        local_val_loss = total_validation_loss / len(self.val_loader)
        local_dsc = total_metrics["dsc"] / len(self.val_loader)
        local_iou = total_metrics["iou"] / len(self.val_loader)

        # 2. Sincronização Matemática via NCCL
        world_size = int(os.environ["WORLD_SIZE"])
        metrics_tensor = torch.tensor(
            [local_val_loss, local_dsc, local_iou], device=self.device
        )
        dist.reduce(metrics_tensor, dst=0, op=dist.ReduceOp.SUM) #força todas as GPUs a comunicarem pela rede, onde todas as GPUs param aqui até todas terem comunicado o tensor das metricas.
        #dst: máquina destino
        #op: vai somar os valores todos

        # 3. RANK 0 global avalia o Early Stopping e guarda o modelo
        if self.global_rank == 0:
            avg_val_loss = metrics_tensor[0].item() / world_size
            avg_dsc = metrics_tensor[1].item() / world_size
            avg_iou = metrics_tensor[2].item() / world_size

            # Escrever diretamente no CSV (modo 'a' = append / adicionar ao fim)
            with open(self.csv_metrics_path, 'a', encoding='utf-8') as f:
                f.write(f"{self.epoch},{self.total_train_loss:.6f},{avg_val_loss:.6f},{avg_dsc:.6f},{avg_iou:.6f}\n")

            logging.info(f"Val-loss Global: {avg_val_loss:.3f}")
            logging.info(f"DSC Global: {avg_dsc:.3f} | IoU Global: {avg_iou:.3f}")
            
            stop_metric = avg_dsc
            self.early_stopping(stop_metric)


            if stop_metric > self.metric:
                self.metric = stop_metric
                # Extrair os pesos limpos sem o wrapper DDP
                save_path = Path(self.save_folder) / f"best_{self.model_name}.pt"
                torch.save(self.model.module.state_dict(), save_path)
                logging.info(f"Novo melhor modelo salvo com DSC: {self.metric:.4f}")

        self.model.train()



    @utilities.timer
    def main(self) -> None:
        """Performs all necessary training steps by initiating the epoch loop
        and saves the trained model at the end.

        Args:
            config (dict): Dictionary containing predefined settings used by several external functions.
        """
        # Registo do hook DDP (aqui porque o profiler já existe)
        self.profiler.register_ddp_model(self.model)

        # Snapshot ANTES do treino — memória base do modelo
        self.profiler.log_memory_checkpoint("Modelo carregado (pré-treino)")

        train_paths = pd.read_csv(self.train_path)
        val_paths = pd.read_csv(self.val_path)

        self.metric_list = []
        #self.save_folder = f"{self.base_output}/train_{self.model_name}"
        
        #n
        # CONTROLO DE I/O: Apenas o gpu rank 0 global cria pastas e faz logs densos
        if self.global_rank == 0:
            Path(self.save_folder).mkdir(parents=True, exist_ok=True)
            logging.info(f"Device: {self.device}")
            # A função count_parameters assume um modelo normal. Em DDP, o modelo original está em .module
            table, _ = utilities.count_parameters(self.model.module) 
            logging.info(f"\n{table}")
            
            # Copy config file to save folder (Apenas o rank 0 faz isto)
            shutil.copyfile(args.config, Path(self.save_folder, Path(args.config).name))
            logging.info(f"Save folder: {str(self.save_folder)}")

        # -----------------------------------------------------------------
        # AJUSTE I/O: Mudar extensão de .nii.gz para .nii (Descompressão Offline)
        # -----------------------------------------------------------------
        # Substituir a extensão em todas as colunas de caminhos
        for df in [train_paths, val_paths]:
            df["image_path"] = df["image_path"].str.replace(".nii.gz", ".nii", regex=False)
            df["mask_path"] = df["mask_path"].str.replace(".nii.gz", ".nii", regex=False)
        
        if self.global_rank == 0:
            logging.info("INFO I/O: Caminhos dos CSVs ajustados para ficheiros .nii descomprimidos.")
        # -----------------------------------------------------------------

        self.train_loader, self.val_loader = get_loaders(
            train_paths,
            val_paths,
            batch_size=1, #BATCH SIZE CONFIG
        )

        self.loss = DiceLoss().to(self.device)

        self.early_stopping = EarlyStopping(
            patience=20, verbose=True, monitor="dsc", op_type="max"
        )

        self.metric = 0.0
        self.lr = self.init_lr
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=0
        )

        #Lógica de carregamento simplificada (as variáveis já vêm do __init__)
        #
        if os.path.exists(self.snapshot_path):
            if self.args.resume_id or self.current_job_id in self.snapshot_path:
                if self.global_rank == 0:
                    logging.info(f"Loading snapshot from {self.snapshot_path}...")
                self._load_snapshot()


        # Forçar a sincronização antes de começar a iterar
        dist.barrier() #garantir que está tudo criado ou carregado antes das GPUs entrarem no treino




        # Start epoch loop
        for self.epoch in range(self.epochs_run, self.epochs): #começar do 0 ou da época em que parou
            with self.profiler.profile_epoch(self.epoch): #envolver epochs com o profiler para medir o tempo total de cada época
                if self.global_rank==0:
                    logging.info(f"Now training epoch {self.epoch}!")
                
                # Garantir o shuffle correto em DDP
                self.train_loader.sampler.set_epoch(self.epoch)

                TrainNetwork.train_fn(self)

                if self.global_rank == 0:
                    logging.info(f"Train-loss: {self.total_train_loss:.3f}")

                # Validate the model
                TrainNetwork.validation(self)

                # GUARDAR O SNAPSHOT PARA RESILIÊNCIA APENAS NO MESTRE
                if self.global_rank == 0:
                    self._save_snapshot(self.epoch + 1)
                
                #Early stopping check
                # -------------------------------------------------------------
                # SINCRONIZAÇÃO DO EARLY STOPPING (Avisar todas as GPUs)
                # -------------------------------------------------------------
                # 1. Criar um tensor que funciona como "bandeira" (0 = continuar, 1 = parar)
                if self.early_stop_active:
                    stop_flag = torch.tensor([0], dtype=torch.int32, device=self.device)
                    
                    # 2. Apenas o Rank 0 levanta a bandeira se o seu early_stopping ativar
                    if self.global_rank == 0 and self.early_stopping.early_stop:
                        stop_flag[0] = 1
                    
                    # 3. O Rank 0 transmite (broadcast) o valor da bandeira para todas as GPUs
                    dist.broadcast(stop_flag, src=0)
                    
                    # 4. Agora TODAS as GPUs leem a bandeira e param ao mesmo tempo
                    if stop_flag.item() == 1:
                        if self.global_rank == 0:
                            logging.info("Early stopping ativado! A avisar todas as GPUs para pararem...")
                        break
                
        self.profiler.save_final_report()   # ← relatório agregado de todos os epochs

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True #auto-tuner da biblioteca cuDNN da NVIDIA. PyTorch gasta o primeiro batch (a tal Época 000) a testar dezenas de algoritmos matemáticos diferentes para calcular as convoluções 3D, escolhendo o mais rápido para o teu hardware.
    torch.backends.cuda.matmul.allow_tf32 = True #Permitem que a placa gráfica utilize a precisão TF32 (TensorFloat-32) em vez da precisão padrão FP32 (Float-32 de 32 bits).
    torch.backends.cudnn.allow_tf32 = True #Permitem que a placa gráfica utilize a precisão TF32 (TensorFloat-32) em vez da precisão padrão FP32 (Float-32 de 32 bits).
    #TF32 corta partes dos números decimais que não interessam para as redes neuronais, oferecendo mais velocidade em multiplicações de matrizes,

    torch.autograd.set_detect_anomaly(False) #Obriga o PyTorch a vigiar todas as operações matemáticas em busca da origem exata de valores NaN (Not a Number) ou infinitos durante o cálculo dos gradientes.
    #ferramenta estrita de Debugging. Quando está ativada, aumenta drasticamente o consumo de memória RAM/VRAM e abranda o backward pass entre 15% a 25%.

    args = parser.parse_args()
    args.config = "configs/" + args.config

    with open(args.config, "r") as conf:
        config = yaml.safe_load(conf)

    torch.set_num_threads(2) #threads do torch
    
    try:
        # 1. INICIALIZAR DDP
        ddp_setup()
        
        # 2. Configurar a seed usando o RANK global para garantir que cada um tem uma seed diferente, o que é útil para serem aplicadas diferentes transformações na fase de data augmentation, assim diferentes máquinas fazem diferente transofrmações
        global_rank = int(os.environ["RANK"])
        set_seed(42 + global_rank) 
        
        training = TrainNetwork(
            args=args,
            config=config,
        )
        
        # Limitar logs ao processo mestre para não poluir o terminal com 8 prints iguais
        if global_rank == 0:
            logging.info(training.__repr__())


        if global_rank == 0:
            import os
            # Verifica as variáveis de ambiente herdadas do Bash
            vars_to_check = [
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
            ]

            logging.info("--- Verificação de Limites de CPU ---")
            for v in vars_to_check:
                logging.info(f"{v}: {os.environ.get(v, 'NÃO DEFINIDA')}")
            
            # Verifica o limite interno do próprio PyTorch
            logging.info(f"PyTorch Threads: {torch.get_num_threads()}")

        training.main()
        
        # Garante que NINGUÉM destroi o grupo até o treino de todos estar 100% concluído
        dist.barrier()
        if global_rank == 0:
            logging.info("Sincronização final concluída. A encerrar o DDP com segurança.")
            
    except Exception as e:
        logging.exception(e)
    finally:
        # 3. DESTRUIR GRUPO DE PROCESSOS
        destroy_process_group()