import os
import time
import torch
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
import logging

# Importar do teu código adaptado/original
from model import Mednext
from dataset_optimized import InferenceDataset, resample, dcm2nifti, nifti2dcm

logging.basicConfig(level=logging.INFO)

def _deidentify_header(nii_image, keys=["descrip", "intent_name"]):
    """Lógica exata do dicom_skullstrip_defacing.py"""
    for key in keys:
        if key in nii_image.header:
            nii_image.header[key] = b""
    return nii_image

def run_original_exact(model_weights_path, csv_test_path, output_path, processes=16, gpu=0):
    
    # =========================================================================
    # 1. CONFIGURAÇÕES EXATAS DO deidentify.py
    # =========================================================================
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(True)
    torch.set_num_threads(processes)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    stem = "deface"
    
    # =========================================================================
    # 2. INICIALIZAÇÃO DA CLASSE Inference
    # =========================================================================
    model = Mednext().to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
    model.eval()

    # =========================================================================
    # 3. ADAPTAÇÃO PARA O TEU CSV (Como autorizado)
    # =========================================================================
    df = pd.read_csv(csv_test_path)
    df["image_path"] = df["image_path"].str.replace(".nii.gz", ".nii", regex=False)
    valid_paths = {"image_path": df["image_path"].tolist()}
    
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # O get_inference_loader original instancia o InferenceDataset e o DataLoader assim:
    inference_dataset = InferenceDataset(valid_paths)
    test_loader = DataLoader(
        inference_dataset, 
        batch_size=1, 
        num_workers=4, 
        pin_memory=True, 
        prefetch_factor=4
    )

    times = []

    # =========================================================================
    # 4. LOOP EXATO DO dicom_skullstrip_defacing.py (Sem cronómetros alterados)
    # =========================================================================
    for data in tqdm(test_loader, desc="Inference (Batch)"):
        start_time = time.time()

        image = data["image"].to(device)

        with torch.no_grad():
            pred = model(image.float())

        pred = pred.detach()

        for idx in range(image.shape[0]):
            file_path = data["file_name"][idx]
            file_lower = file_path.lower()
            is_nifti = file_lower.endswith('.nii') or file_lower.endswith('.nii.gz')
            is_npy = file_lower.endswith('.npy')

            if is_nifti:
                nifti_img = nib.load(file_path)
                nifti_img = resample(nifti_img)
                input_volume = nifti_img.get_fdata()
                affine = nifti_img.affine
                header = nifti_img.header
            elif is_npy:
                input_volume = np.load(file_path)
                if input_volume.ndim == 4 and input_volume.shape[0] == 1:
                    input_volume = input_volume[0]
                if input_volume.ndim != 3:
                    raise ValueError(
                        f"Unsupported numpy array shape {input_volume.shape} in {file_path}. Expected 3D volume."
                    )
                affine = np.eye(4)
                header = None
            else:
                nifti_img = dcm2nifti(file_path, transpose=False)
                input_volume = nifti_img.get_fdata()
                affine = nifti_img.affine
                header = nifti_img.header

            size = np.transpose(input_volume, (2, 0, 1)).shape

            if pred.ndim == 5:
                pred_numpy = torch.nn.Upsample(size=size)(
                    pred[idx].unsqueeze(dim=0)).cpu().numpy().squeeze().squeeze()
            else:
                pred_numpy = (
                    v2.functional.resize(
                        pred[idx], size[-2:], antialias=True, interpolation=InterpolationMode.BICUBIC
                    )
                    .numpy()
                )
            if pred_numpy.ndim == 2:
                pred_numpy = pred_numpy[None, ...]
                
            pred_volume = input_volume * (np.transpose(pred_numpy > 0.5, (1, 2, 0)))

            if is_npy:
                out_path = Path(output_path) / f"{Path(Path(file_path).stem).stem}_{stem}.npy"
                np.save(out_path, pred_volume)
            else:
                img = nib.Nifti1Image(pred_volume, affine=affine, header=header)
                img = _deidentify_header(img)

                if is_nifti:
                    nib.save(
                        img,
                        f"{output_path}/{Path(Path(file_path).stem).stem}_{stem}.nii.gz",
                    )
                else:
                    nifti2dcm(img, file_path, f"{output_path}")

            end_time = time.time()

            times.append(round((end_time - start_time), 3))

            logging.info(f"Time for volume: {round((end_time - start_time), 3)} seconds.")


if __name__ == "__main__":
    TEST_CSV = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/data/test_noise_cleanse_v2.csv"
    OUTPUT = "/projects/F202500001HPCVLABEPICURE/andresousa615/original_code_rempe/medical_image_deidentification-main/mede"
    PESOS = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/results/train_mednext_0.0005_mednext_aurora_16gb_vram_1120800/best_mednext_0.0005_mednext_aurora_16gb_vram.pt"
    
    # Executa a inferência exata. Usa 16 processos (adequado ao teu Bash atual) e a GPU 0
    run_original_exact(PESOS, TEST_CSV, OUTPUT, processes=16, gpu=0)