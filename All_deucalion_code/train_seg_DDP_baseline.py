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
from dataset_DDP_baseline import get_loaders
from utils.validation import segmentation_validation, plot_segmentation
from utils import utilities
from utils.losses import DiceLoss
from datetime import timedelta

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

def ddp_setup():
    """
    Inicializa o grupo de processos distribuídos.
    O torchrun injeta as variáveis de ambiente necessárias.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl", timeout=timedelta(hours=1))
    logging.debug(f"DDP Setup concluído para Global Rank: {os.environ['RANK']}, Local Rank: {local_rank}")


def set_seed(seed: int = 42) -> None:
    """Set seeds for the libraries numpy, random, torch and torch.cuda."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Na versão não otimizada, isto estava ativo e travava o benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
        if self.verbose:
            logging.info(
                f"{self.monitor} optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...",
            )
        self.val_score_min = val_score


class TrainNetwork:
    def __init__(self, args: dict, config: dict) -> None:
        self.scheduler = None
        self.optimizer = None
        self.early_stopping = None
        self.early_stop_active = args.earlystop
        self.loss = None
        self.val_loader = None
        self.train_loader = None
        self.save_folder = None

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
        self.model_name: str = f"{config['model']}_{config['lr']}_{config['comment']}"

        self.save_folder = f"{self.base_output}/train_{self.model_name}_{self.current_job_id}"
        self.csv_metrics_path = os.path.join(self.save_folder, "training_metrics.csv")
        
        if self.global_rank == 0:
            Path(self.save_folder).mkdir(parents=True, exist_ok=True)
            logging.info(f"Pasta de resultados única criada: {self.save_folder}")
            
            if not os.path.exists(self.csv_metrics_path) or self.args.resume_id is None:
                with open(self.csv_metrics_path, 'w', encoding='utf-8') as f:
                    f.write("epoch,train_loss,val_loss,dsc,iou\n")
        
        dist.barrier()

        # Profiler base ativo
        self.profiler = TrainingProfiler(
            save_dir     = self.save_folder,
            global_rank  = self.global_rank,
            enabled      = True,
            profile_torch= True,
            profile_epoch= 2,
        )

        target_id = args.resume_id if args.resume_id else self.current_job_id
        self.snapshot_path = f"{self.base_output}/snapshot_{self.model_name}_{target_id}.pt"
        
        self.device = torch.device(f"cuda:{self.local_rank}")
        self._init_network(self.config)

    def _init_network(self, configuration: dict) -> None:
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

        self.model = network_class().to(self.device)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(self.snapshot_path, map_location=loc)
        self.model.module.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        
        if "OPTIMIZER_STATE" in snapshot and self.optimizer is not None:
            self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
            
        logging.info(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
        }
        torch.save(snapshot, self.snapshot_path)
        logging.info(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    @utilities.timer
    def train_fn(self) -> None:
        self.train_loader.sampler.set_epoch(self.epoch)

        loop = tqdm(self.train_loader, disable=(self.global_rank != 0))
        self.total_train_loss = 0

        for batch_idx, data_dict in enumerate(self.profiler.iter_dataloader(loop)):
            
            with self.profiler.phase("data_to_gpu"):
                data = data_dict["image"].to(device=self.device, non_blocking=True)
                targets = data_dict["mask"].to(device=data.device, non_blocking=True)

            with self.profiler.phase("forward"):
                predictions = self.model(data.float())
                loss = self.loss(predictions, targets)

            with self.profiler.phase("backward"):
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

            with self.profiler.phase("optimizer"):
                self.optimizer.step()

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
        self.model.eval()
        total_validation_loss = 0
        total_metrics = None
        
        loop = tqdm(self.val_loader, disable=(self.global_rank != 0))
        n = random.randint(0, len(self.val_loader) - 1)

        for batch_idx, data_dict in enumerate(loop):

            with self.profiler.phase("val_forward"):
                data = data_dict["image"].to(device=self.device, non_blocking=True)
                with torch.no_grad():
                    predictions = self.model(data.float())

            targets = data_dict["mask"].to(device=data.device, non_blocking=True)
            predictions_sum = predictions.float()
            loss = self.loss(predictions, targets)
            loop.set_postfix(loss=loss.item())
            
            with self.profiler.phase("val_metrics"):
                val_metrics = segmentation_validation(predictions_sum, targets)

            if total_metrics is None:
                total_metrics = {key: 0 for key in val_metrics.keys()}
            for key in total_metrics.keys():
                total_metrics[key] += val_metrics[key].item()

            total_validation_loss += loss.item()

        local_val_loss = total_validation_loss / len(self.val_loader)
        local_dsc = total_metrics["dsc"] / len(self.val_loader)
        local_iou = total_metrics["iou"] / len(self.val_loader)

        world_size = int(os.environ["WORLD_SIZE"])
        metrics_tensor = torch.tensor(
            [local_val_loss, local_dsc, local_iou], device=self.device
        )
        dist.reduce(metrics_tensor, dst=0, op=dist.ReduceOp.SUM) 

        if self.global_rank == 0:
            avg_val_loss = metrics_tensor[0].item() / world_size
            avg_dsc = metrics_tensor[1].item() / world_size
            avg_iou = metrics_tensor[2].item() / world_size

            with open(self.csv_metrics_path, 'a', encoding='utf-8') as f:
                f.write(f"{self.epoch},{self.total_train_loss:.6f},{avg_val_loss:.6f},{avg_dsc:.6f},{avg_iou:.6f}\n")

            logging.info(f"Val-loss Global: {avg_val_loss:.3f}")
            logging.info(f"DSC Global: {avg_dsc:.3f} | IoU Global: {avg_iou:.3f}")
            
            stop_metric = avg_dsc
            self.early_stopping(stop_metric)

            if stop_metric > self.metric:
                self.metric = stop_metric
                save_path = Path(self.save_folder) / f"best_{self.model_name}.pt"
                torch.save(self.model.module.state_dict(), save_path)
                logging.info(f"Novo melhor modelo salvo com DSC: {self.metric:.4f}")

        self.model.train()

    @utilities.timer
    def main(self) -> None:
        self.profiler.register_ddp_model(self.model)
        self.profiler.log_memory_checkpoint("Modelo carregado (pré-treino)")

        train_paths = pd.read_csv(self.train_path)
        val_paths = pd.read_csv(self.val_path)
        self.metric_list = []
        
        if self.global_rank == 0:
            Path(self.save_folder).mkdir(parents=True, exist_ok=True)
            logging.info(f"Device: {self.device}")
            table, _ = utilities.count_parameters(self.model.module) 
            logging.info(f"\n{table}")
            shutil.copyfile(args.config, Path(self.save_folder, Path(args.config).name))
            logging.info(f"Save folder: {str(self.save_folder)}")

        # =================================================================
        # INTERCETAR E REDIRECIONAR OS CAMINHOS
        # =================================================================
        for df in [train_paths, val_paths]:
            # 1. Redirecionar para a pasta com as dimensões originais (gigantes)
            df["image_path"] = df["image_path"].str.replace("/treino/", "/treino_original_dimensions/", regex=False)
            df["mask_path"] = df["mask_path"].str.replace("/treino/", "/treino_original_dimensions/", regex=False)
            
            # 2. Apenas retirar a compressão (manter o nome original do ficheiro!)
            df["image_path"] = df["image_path"].str.replace(".nii.gz", ".nii", regex=False)
            df["mask_path"] = df["mask_path"].str.replace(".nii.gz", ".nii", regex=False)
        # =================================================================

        self.train_loader, self.val_loader = get_loaders(
            train_paths,
            val_paths,
            batch_size=1,
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

        if os.path.exists(self.snapshot_path):
            if self.args.resume_id or self.current_job_id in self.snapshot_path:
                if self.global_rank == 0:
                    logging.info(f"Loading snapshot from {self.snapshot_path}...")
                self._load_snapshot()

        dist.barrier()

        for self.epoch in range(self.epochs_run, self.epochs):
            with self.profiler.profile_epoch(self.epoch):
                if self.global_rank==0:
                    logging.info(f"Now training epoch {self.epoch}!")
                
                self.train_loader.sampler.set_epoch(self.epoch)
                TrainNetwork.train_fn(self)

                if self.global_rank == 0:
                    logging.info(f"Train-loss: {self.total_train_loss:.3f}")

                TrainNetwork.validation(self)

                if self.global_rank == 0:
                    self._save_snapshot(self.epoch + 1)
                
                if self.early_stop_active:
                    stop_flag = torch.tensor([0], dtype=torch.int32, device=self.device)
                    if self.global_rank == 0 and self.early_stopping.early_stop:
                        stop_flag[0] = 1
                    
                    dist.broadcast(stop_flag, src=0)
                    if stop_flag.item() == 1:
                        if self.global_rank == 0:
                            logging.info("Early stopping ativado! A avisar todas as GPUs para pararem...")
                        break
                
        self.profiler.save_final_report()

if __name__ == "__main__":
    # Removemos o cudnn.benchmark e o allow_tf32 que libertaram a gráfica
    
    args = parser.parse_args()
    args.config = "configs/" + args.config

    with open(args.config, "r") as conf:
        config = yaml.safe_load(conf)

    # Removemos o torch.set_num_threads(2) aqui também
    
    try:
        ddp_setup()
        
        # Voltamos à seed fixa para todas as GPUs (o que criava redundância de augmentation)
        set_seed(42) 
        
        training = TrainNetwork(
            args=args,
            config=config,
        )
        
        # O bloco de debug de variáveis de ambiente foi removido daqui
        
        global_rank = int(os.environ["RANK"])
        if global_rank == 0:
            logging.info(training.__repr__())

        training.main()
        
        dist.barrier()
        if global_rank == 0:
            logging.info("Sincronização final concluída. A encerrar o DDP com segurança.")
            
    except Exception as e:
        logging.exception(e)
    finally:
        destroy_process_group()
        