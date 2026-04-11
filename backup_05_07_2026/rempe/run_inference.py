import torch
import os
import time
import warnings
import pandas as pd
import torchio as tio
from pathlib import Path
from torch.utils.data import DataLoader

from model import ConvNext, UNet3D, Mednext

# Esconde os avisos chatos
warnings.filterwarnings("ignore")

def custom_collate(batch):
    # Retorna o objeto Subject diretamente, obrigatório porque os NIfTIs têm tamanhos originais diferentes
    return batch[0]

def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def run_inference(model_weights_path, csv_test_path, output_dir):
    print("A inicializar o modelo MedNeXt (Otimização Clínica - Final)...")
    
    # =====================================================================
    # PASSO 4: FLAGS CUDNN CORRETAS
    # =====================================================================
    torch.backends.cudnn.enabled = True      
    torch.backends.cudnn.benchmark = False   
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = Mednext()
    state_dict = torch.load(model_weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"A carregar exames a partir de: {csv_test_path}")
    df = pd.read_csv(csv_test_path)
    df["image_path"] = df["image_path"].str.replace(".nii.gz", ".nii", regex=False)
    file_paths = df['image_path'].tolist()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # PASSO 3 & 5: DATALOADER ASSÍNCRONO + NORMALIZAÇÃO MIN-MAX
    # =====================================================================
    subjects = []
    for file_path in file_paths:
        patient_id = os.path.basename(os.path.dirname(file_path))
        subjects.append(tio.Subject(
            image=tio.ScalarImage(file_path),
            patient_id=patient_id,
            file_path=file_path
        ))
        
    preprocessing_transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resize((128, 128, 128)),
        tio.Lambda(min_max_norm)
    ])
    
    dataset = tio.SubjectsDataset(subjects, transform=preprocessing_transform)
    
    loader = DataLoader(
        dataset, 
        batch_size=1, 
        num_workers=4, 
        prefetch_factor=2, 
        collate_fn=custom_collate
    )

    total_preprocessing_time, total_seg_time, total_anon_time, total_pipeline_time, total_save_time, total_pipeline_no_save_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    num_volumes = len(file_paths)

    # PASSO 2: INFERÊNCIA FP16 COM AUTOCAST
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16): 
        for subject in loader:
            patient_id = subject['patient_id']
            file_path = subject['file_path']
            print(f"\nA processar paciente: {patient_id}")
            
            t_preprocessing = time.perf_counter()
            t0 = time.perf_counter()

            input_tensor = subject.image.data.unsqueeze(0).float().to(device)
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            
            binary_mask = (probs > 0.5).float().cpu().squeeze(0) 
            
            # 8. Operação de Anonimização
            # Lemmos a imagem original 100% limpa primeiro
            raw_img = tio.ScalarImage(file_path)

            # =====================================================================
            # SOLUÇÃO DEFINITIVA DO RESAMPLE FÍSICO (Bypass ao Bug do DataLoader)
            # =====================================================================
            # 1. Colocamos a máscara num objecto à parte (ela ainda está em 128x128x128)
            mask_subject = tio.Subject(prediction=tio.LabelMap(
                tensor=binary_mask, affine=subject.image.affine
            ))
            
            # 2. Em vez de tentarmos invocar o "histórico" (que o DataLoader apagou),
            # Mandamos o TorchIO amoldar a máscara às propriedades físicas rigorosas
            # da raw_img que acabámos de ler do disco! Fica perfeitamente mapeada a 160.
            resample_to_orig = tio.Resample(raw_img)
            mask_orig = resample_to_orig(mask_subject)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            orig_data = raw_img.data[0].numpy()
            
            # A máscara vai agora garantir os 160 (ou qual seja o tamanho real do paciente)
            mask_array = mask_orig.prediction.data[0].numpy() > 0.5
            
            # Defacing real sem quebras
            orig_data[mask_array] = 0
            
            base_name = os.path.basename(file_path)
            ext = ".nii.gz" if base_name.endswith(".nii.gz") else ".nii"
            novo_nome_anon = f"{patient_id}_{base_name.replace(ext, f'_anon{ext}')}"
            anon_output_path = os.path.join(output_dir, novo_nome_anon)
            
            anonymized_image = tio.ScalarImage(tensor=torch.tensor(orig_data).unsqueeze(0), affine=raw_img.affine)
            
            t2 = time.perf_counter()

            anonymized_image.save(anon_output_path)
            
            t3 = time.perf_counter()
            
            preprocessing_time = t0 - t_preprocessing
            seg_time = t1 - t0
            anon_time = t2 - t1
            save_time = t3 - t2
            total_no_save_time = t2 - t0
            total_vol_time = t3 - t0
            
            total_preprocessing_time += preprocessing_time
            total_seg_time += seg_time
            total_anon_time += anon_time
            total_save_time += save_time
            total_pipeline_no_save_time += total_no_save_time
            total_pipeline_time += total_vol_time
            
            print(f"  -> Tempo Segmentação + Mapeamento Geométrico: {seg_time:.4f} s")
            print(f"  -> Tempo Anonimização (CPU): {anon_time:.4f} s")
            print(f"  -> Tempo Gravação (Disco): {save_time:.4f} s")

    if num_volumes > 0:
        print("\n=== SCRIPT DE INFERÊNCIA CONCLUÍDO ===")
        print(f"Exames processados. Verifica a pasta {output_dir}")
        print(f"Tempo total de pré-processamento (no loop principal): {total_preprocessing_time:.4f} s")
        print(f"Tempo total de segmentação (GPU+Resample): {total_seg_time:.4f} s")
        print(f"Tempo total de anonimização: {total_anon_time:.4f} s")
        print(f"Tempo total de gravação: {total_save_time:.4f} s")
        print(f"Tempo total do pipeline sem gravação: {total_pipeline_no_save_time:.4f} s")
        print(f"Tempo total do pipeline: {total_pipeline_time:.4f} s")

        avg_preprocessing_time = total_preprocessing_time / num_volumes
        avg_seg_time = total_seg_time / num_volumes
        avg_anon_time = total_anon_time / num_volumes
        avg_save_time = total_save_time / num_volumes
        avg_total_pipeline_no_save_time = total_pipeline_no_save_time / num_volumes
        avg_total_time = total_pipeline_time / num_volumes

        print(f"\nTempo médio de segmentação por exame: {avg_seg_time:.4f} s")
        print(f"Tempo médio de anonimização por exame: {avg_anon_time:.4f} s")
        print(f"Tempo médio de gravação por exame: {avg_save_time:.4f} s")
        print(f"Tempo médio total sem gravação por exame: {avg_total_pipeline_no_save_time:.4f} s")
        print(f"Tempo médio total por exame: {avg_total_time:.4f} s")
    else:
        print("Nenhum exame encontrado para processar.")

if __name__ == "__main__":
    TEST_CSV = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/data/test_noise_cleanse_v2.csv"
    OUTPUT = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/results/train_mednext_0.0005_mednext_aurora_16gb_vram_1120800/resultados_inferencia/"
    PESOS = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/results/train_mednext_0.0005_mednext_aurora_16gb_vram_1120800/best_mednext_0.0005_mednext_aurora_16gb_vram.pt"
    
    run_inference(PESOS, TEST_CSV, OUTPUT)