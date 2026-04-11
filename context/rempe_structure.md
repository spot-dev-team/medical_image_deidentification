Project Context: mede (Medical De-Identification) - DDP Training Refactor

  1. Project Overview
  The mede project is a medical imaging de-identification tool. The current focus is on parallelizing the training
  process of segmentation models (skull-stripping and defacing) using PyTorch Distributed Data Parallel (DDP). The
  models process 3D medical volumes (MRI/CT).

  2. Core Training Components

  train_seg.py (Main Entry Point)
   * Purpose: Orchestrates the training lifecycle.
   * Class TrainNetwork:
       * __init__: Initializes config, model, and device. Currently uses a single GPU (cuda:args.gpu).
       * _init_network: Instantiates the selected model architecture.
       * train_fn: The per-epoch training loop. Handles forward pass, loss calculation (DiceLoss), backpropagation, and
         optimizer steps.
       * validation: The per-epoch validation loop. Calculates DSC/IoU metrics and saves the best model.
       * main: Sets up DataLoaders, Optimizer (AdamW), and Scheduler (CosineAnnealingLR).
   * Current State: Procedural and single-process. Uses argparse for GPU selection.

  dataset.py (Data Loading & Preprocessing)
   * Class SegmentationDataset:
       * Loads 3D NIfTI images and masks using nibabel.
       * Resampling: All volumes are resampled to "RAS" orientation.
       * Augmentation: Uses torchio (tio) for 3D transforms (Affine, Elastic, BiasField, Noise, etc.).
       * Normalization: Min-max normalization per volume.
       * Upsampling: All volumes are upsampled/downsampled to a fixed size [128, 128, 128].
   * Function get_loaders: Creates standard PyTorch DataLoader instances.
       * Issue for DDP: Needs to incorporate torch.utils.data.distributed.DistributedSampler.

  model.py (Architectures)
   * Supported Models:
       * ConvNext: 2D-based architecture with a loop to process 3D slices.
       * UNet3D: Standard 3D UNet implementation.
       * Mednext: Wrapper for the specialized MedNeXt architecture.
   * Requirement for DDP: Models must be wrapped in torch.nn.parallel.DistributedDataParallel.

  utils/validation.py & utils/losses.py
   * Metrics: Uses torchmetrics.functional for Dice (DSC) and Jaccard (IoU).
       * Critical: Functional metrics in DDP do not automatically aggregate results across GPUs.
   * Loss: DiceLoss calculates loss on flattened tensors.
   * Visualization: plot_segmentation saves PNG slices. In DDP, this should only happen on the master process (Rank 0).

  3. Current Data Flow
   1. CSV files (train.csv, val.csv) provide paths to .nii.gz files.
   2. DataLoader fetches batches (currently batch_size=1 in main).
   3. SegmentationDataset performs 3D resampling and resizing on-the-fly.
   4. Model outputs logits -> DiceLoss -> Optimizer update.
   5. Validation runs every epoch on the full validation set.

  4. DDP Implementation Requirements
  To parallelize this, the following changes are expected:
   1. Process Initialization: Setup dist.init_process_group (using nccl backend).
   2. Multiprocessing: Wrap the execution in torch.multiprocessing.spawn or adapt it for torchrun.
   3. Data Partitioning: Implement DistributedSampler in get_loaders to ensure each GPU sees a different subset of data.
   4. Model Wrapping: Wrap models with DDP and handle device assignment based on local_rank.
   5. Metric Syncing: Aggregate validation metrics (DSC/IoU) and training loss across all processes using
      dist.all_reduce.
   6. Logging/Saving: Ensure only the Rank 0 process handles logging.info, plt.imsave, and torch.save.
   7. Reproducibility: Update set_seed to account for process rank.

  5. File Structure Reference
   - train_seg.py: Training logic.
   - dataset.py: 3D Data loading and torchio transforms.
   - model.py: Model definitions.
   - utils/validation.py: Metric calculation.
   - utils/losses.py: Loss functions.
   - configs/*.yaml: Training hyperparameters.

  ---

  Instruções para o Gemini Pro:
  "Com base no ficheiro de contexto acima, ajuda-me a refatorar o projeto mede para suportar treino distribuído com
  PyTorch DDP. Quero manter a compatibilidade com as transformações do torchio e garantir que as métricas de validação
  são sincronizadas corretamente entre as GPUs."




  ---
