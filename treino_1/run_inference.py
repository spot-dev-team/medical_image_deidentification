import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import os
import time
import pandas as pd
from pathlib import Path

# Importamos apenas a arquitetura. Ignoramos o dataset.py para evitar o erro do DICOM
from model import ConvNext, UNet3D, Mednext

# --- 1. Nova Estrutura de Dados: Dataset Nativo para NIfTI ---
class SpotNiftiDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        # Lemos os caminhos diretamente do CSV
        df = pd.read_csv(csv_path)
        self.paths = df['image_path'].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        file_path = self.paths[idx]
        
        # Carregar NIfTI cru
        orig_data = nib.load(file_path).get_fdata()
        
        # Converter para Tensor e adicionar canal: [1, Z, Y, X]
        tensor_data = torch.tensor(orig_data, dtype=torch.float32).unsqueeze(0)
        
        # Normalização de Intensidade (Z-Score idêntico ao treino)
        mean, std = tensor_data.mean(), tensor_data.std()
        if std > 0:
            tensor_data = (tensor_data - mean) / std
            
        # Redimensionar para o tamanho do MedNeXt: [1, 128, 128, 128]
        tensor_resized = F.interpolate(
            tensor_data.unsqueeze(0), 
            size=(128, 128, 128), 
            mode='trilinear', 
            align_corners=False
        ).squeeze(0)

        return {
            "image": tensor_resized,
            "file_path": file_path,
            # Guardamos a estrutura do formato original [Z, Y, X] para reverter a máscara depois
            "orig_shape": torch.tensor(orig_data.shape) 
        }

def run_inference(model_weights_path, csv_test_path, output_dir):
    print("A inicializar o modelo...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 2. Carregar o modelo completo guardado
    model = torch.load(model_weights_path, map_location=device)
    model.to(device)
    model.eval() 

    # 3. Inicializar o nosso novo DataLoader que liga as imagens ao modelo
    print(f"A carregar NIfTIs a partir de: {csv_test_path}")
    dataset = SpotNiftiDataset(csv_path=csv_test_path)
    inference_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    total_seg_time = 0.0
    total_anon_time = 0.0
    total_pipeline_time = 0.0
    num_volumes = 0

    with torch.no_grad(): 
        for batch in inference_loader:
            images = batch["image"].float().to(device)
            file_path = batch["file_path"][0] 
            
            # Recuperar as dimensões originais (ex: 256x256x170) para esta imagem específica
            orig_shape = tuple(batch["orig_shape"][0].tolist())
            
            print(f"\nA processar ficheiro: {os.path.basename(file_path)}")
            
            # ---> t0: INÍCIO DA SEGMENTAÇÃO (GPU) <---
            t0 = time.perf_counter()
            
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            binary_mask = (probs > 0.5).float() 
            
            # Esticar a máscara 128^3 para o tamanho e geometria originais do exame
            restored_mask = F.interpolate(binary_mask, size=orig_shape, mode='nearest')
            
            mask_array = restored_mask.squeeze(0).squeeze(0).cpu().numpy()
            
            # ---> t1: FIM DA SEGMENTAÇÃO / INÍCIO DA ANONIMIZAÇÃO <---
            t1 = time.perf_counter()
            
            orig_nifti = nib.load(file_path)
            orig_data = orig_nifti.get_fdata()
            real_affine = orig_nifti.affine 
            
            anonymized_data = np.copy(orig_data)
            anonymized_data[mask_array == 1] = 0
            
            # --- CORREÇÃO DO NOME DO FICHEIRO PARA EVITAR OVERWRITE ---
            patient_id = os.path.basename(os.path.dirname(file_path))
            base_name = os.path.basename(file_path)
            novo_nome = f"{patient_id}_{base_name.replace('.nii.gz', '_anon.nii.gz')}"
            anon_output_path = os.path.join(output_dir, novo_nome)
            
            nib.save(nib.Nifti1Image(anonymized_data.astype(orig_data.dtype), real_affine), anon_output_path)
            
            # ---> t2: FIM DA ANONIMIZAÇÃO E GRAVAÇÃO <---
            t2 = time.perf_counter()
            
            seg_time = t1 - t0
            anon_time = t2 - t1
            total_vol_time = t2 - t0
            
            total_seg_time += seg_time
            total_anon_time += anon_time
            total_pipeline_time += total_vol_time
            num_volumes += 1
            
            print(f"  -> Tempo Segmentação (GPU): {seg_time:.4f} s")
            print(f"  -> Tempo Anonimização e I/O: {anon_time:.4f} s")
            print(f"  -> Tempo Total do Volume:   {total_vol_time:.4f} s")

    # 4. Agregação e Relatório
    if num_volumes > 0:
        avg_seg = total_seg_time / num_volumes
        avg_anon = total_anon_time / num_volumes
        avg_total = total_pipeline_time / num_volumes
        output_file = os.path.join(output_dir, "inference_performance_report.txt")
        
        with open(output_file, "w") as f:
            f.write("="*55 + "\n")
            f.write("MÉTRICAS DE PERFORMANCE DA INFERÊNCIA\n\n")
            f.write("="*55 + "\n")
            f.write(f"Total de exames processados:   {num_volumes}\n")
            f.write(f"Média de Segmentação (GPU):    {avg_seg:.4f} segundos/exame\n")
            f.write(f"Média de Anonimização (CPU):   {avg_anon:.4f} segundos/exame\n")
            f.write("-" * 55 + "\n")
            f.write(f"Tempo MÉDIO TOTAL por exame:   {avg_total:.4f} segundos\n")
            f.write(f"Capacidade de processamento:   {(1.0/avg_total):.2f} exames/segundo\n")
            f.write("="*55 + "\n")
            
        print("\n" + "="*55)
        print("MÉTRICAS DE PERFORMANCE DA INFERÊNCIA (Spot)")
        print("="*55)
        print(f"Total de exames processados:   {num_volumes}")
        print(f"Média de Segmentação (GPU):    {avg_seg:.4f} segundos/exame")
        print(f"Média de Anonimização (CPU):   {avg_anon:.4f} segundos/exame")
        print("-" * 55)
        print(f"Tempo MÉDIO TOTAL por exame:   {avg_total:.4f} segundos")
        print("="*55)

if __name__ == "__main__":
    PESOS = "/home/andresousa615/rempe/mede_code/results/train_mednext_0.0005_mednext_aurora_16gb_vram/mednext_0.0005_mednext_aurora_16gb_vram"
    TEST_CSV = "/home/andresousa615/rempe/mede_code/mede/data/test.csv"
    OUTPUT = "/home/andresousa615/rempe/mede_code/resultados_inferencia/"
    
    torch.backends.cudnn.enabled = False
    run_inference(PESOS, TEST_CSV, OUTPUT)