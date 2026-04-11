import torch
import os
import time
import pandas as pd
import torchio as tio
from pathlib import Path

from model import ConvNext, UNet3D, Mednext

def run_inference(model_weights_path, csv_test_path, output_dir):
    print("A inicializar o modelo MedNeXt...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Instanciar a arquitetura vazia do modelo
    model = Mednext()
    
    # 2. Carregar o dicionário de pesos (adicionamos weights_only=True para apagar o aviso)
    state_dict = torch.load(model_weights_path, map_location=device, weights_only=True)
    
    # 3. Injetar os pesos na arquitetura
    model.load_state_dict(state_dict)
    
    # 4. Mover para GPU e colocar em modo de avaliação
    model.to(device)
    model.eval()

    print(f"A carregar exames a partir de: {csv_test_path}")
    df = pd.read_csv(csv_test_path)
    
    # Ajuste por precaução: forçar caminhos do CSV a .nii caso ainda digam .nii.gz
    df["image_path"] = df["image_path"].str.replace(".nii.gz", ".nii", regex=False)
    file_paths = df['image_path'].tolist()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Instanciar transformações isoladas
    canonical_transform = tio.ToCanonical()
    resize_transform = tio.Resize((128, 128, 128))
    intensity_transform = tio.ZNormalization()

    total_preprocessing_time, total_seg_time, total_anon_time, total_pipeline_time, total_save_time, total_pipeline_no_save_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    num_volumes = len(file_paths)

    with torch.no_grad(): 
        for file_path in file_paths:
            patient_id = os.path.basename(os.path.dirname(file_path))
            print(f"\nA processar paciente: {patient_id}")
            
            t_preprocessing = time.perf_counter()
            
            # 1. Carregar com TorchIO
            subject = tio.Subject(image=tio.ScalarImage(file_path))
            
            # =====================================================================
            # 2. GARANTIR ORIENTAÇÃO PADRÃO (RAS) SEMPRE
            # =====================================================================
            subject_spatial = canonical_transform(subject)
            
            # =====================================================================
            # 3. VERIFICAÇÃO DE DIMENSÕES (na imagem já orientada)
            # =====================================================================
            is_already_128 = (subject_spatial.spatial_shape == (128, 128, 128))
            
            if is_already_128:
                print("  -> Exame já está a 128x128x128. A ignorar Resize.")
            else:
                print(f"  -> Exame está a {subject_spatial.spatial_shape}. A aplicar Resize.")
                subject_spatial = resize_transform(subject_spatial)
            # =====================================================================
            
            # 4. Aplicar a normalização de intensidade (Não entra no histórico espacial)
            subject_ready = intensity_transform(subject_spatial)
            

            t0 = time.perf_counter()

            # 5. Enviar para a GPU e prever
            input_tensor = subject_ready.image.data.unsqueeze(0).float().to(device)
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            
            # Binarizar na CPU
            binary_mask = (probs > 0.5).float().cpu().squeeze(0) 
            
            # 6. O SEGREDO DO ALINHAMENTO ABSOLUTO
            pred_label = tio.LabelMap(tensor=binary_mask, affine=subject_spatial.image.affine)
            subject_spatial.add_image(pred_label, 'prediction')
            
            # 7. Reverter TODAS as transformações espaciais (Resize se aplicado, e Canonical sempre)
            inverse_transform = subject_spatial.get_composed_history().inverse()
            subject_orig_space = inverse_transform(subject_spatial)
            
            t1 = time.perf_counter()
            
            # 8. Operação de Anonimização
            orig_data = subject_orig_space.image.data[0].numpy()
            mask_array = subject_orig_space.prediction.data[0].numpy() > 0.5
            
            orig_data[mask_array] = 0
            
            # Construir nomes e caminhos de forma segura (funciona com .nii e .nii.gz)
            base_name = os.path.basename(file_path)
            ext = ".nii.gz" if base_name.endswith(".nii.gz") else ".nii"
            
            novo_nome_anon = f"{patient_id}_{base_name.replace(ext, f'_anon{ext}')}"
            novo_nome_mask = f"{patient_id}_{base_name.replace(ext, f'_mask{ext}')}"
            
            anon_output_path = os.path.join(output_dir, novo_nome_anon)
            mask_output_path = os.path.join(output_dir, novo_nome_mask)
            
            # 9. Preparar Ficheiros com o TorchIO
            anonymized_image = tio.ScalarImage(tensor=torch.tensor(orig_data).unsqueeze(0), affine=subject_orig_space.image.affine)
            
            t2 = time.perf_counter()

            # Guardar Ficheiros
            anonymized_image.save(anon_output_path)
            subject_orig_space.prediction.save(mask_output_path)
            
            t3 = time.perf_counter()
            
            # Tempos
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
            
            print(f"  -> Tempo Segmentação (GPU): {seg_time:.4f} s")
            print(f"  -> Tempo Anonimização: {anon_time:.4f} s")

    if num_volumes > 0:
        print("\n=== SCRIPT DE INFERÊNCIA CONCLUÍDO ===")
        print(f"Exames processados. Verifica a pasta {output_dir}")
        print(f"Tempo total de pré-processamento: {preprocessing_time:.4f} s")
        print(f"Tempo total de segmentação: {total_seg_time:.4f} s")
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

        print(f"Tempo médio de segmentação por exame: {avg_seg_time:.4f} s")
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
    
    torch.backends.cudnn.enabled = False
    run_inference(PESOS, TEST_CSV, OUTPUT)