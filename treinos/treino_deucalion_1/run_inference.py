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
    
    # Carregar o modelo
    model = torch.load(model_weights_path, map_location=device)
    model.to(device)
    model.eval() 

    print(f"A carregar exames a partir de: {csv_test_path}")
    df = pd.read_csv(csv_test_path)
    
    file_paths = df['image_path'].tolist()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---> CORREÇÃO: Separar Espacial de Intensidade <---
    # O TorchIO consegue inverter perfeitamente transformações espaciais, 
    # mas o ZNormalization não tem inverso.
    spatial_transform = tio.Compose([
        tio.ToCanonical(),           
        tio.Resize((128, 128, 128))  
    ])
    intensity_transform = tio.ZNormalization()

    total_seg_time, total_anon_time, total_pipeline_time, total_save_time, total_pipeline_no_save_time = 0.0, 0.0, 0.0, 0.0, 0.0
    num_volumes = len(file_paths)

    with torch.no_grad(): 
        for file_path in file_paths:
            patient_id = os.path.basename(os.path.dirname(file_path))
            print(f"\nA processar paciente: {patient_id}")
            
            #Operação de Segmentação    ---
            t0 = time.perf_counter()
        

            # 1. Carregar com TorchIO
            subject = tio.Subject(image=tio.ScalarImage(file_path))
            
            # 2. Aplicar APENAS transformações espaciais (Ficam registadas no histórico para inversão)
            subject_spatial = spatial_transform(subject)
            
            # 3. Aplicar a normalização de intensidade (Não fica no histórico da máscara)
            subject_ready = intensity_transform(subject_spatial)
            
            # 4. Enviar para a GPU e prever
            input_tensor = subject_ready.image.data.unsqueeze(0).float().to(device)
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            
            # Binarizar na CPU
            binary_mask = (probs > 0.5).float().cpu().squeeze(0) 
            
            # 5. O SEGREDO DO ALINHAMENTO ABSOLUTO
            # Anexamos a máscara 128x128 ao objeto que tem o histórico de Resize/Rotação
            pred_label = tio.LabelMap(tensor=binary_mask, affine=subject_spatial.image.affine)
            subject_spatial.add_image(pred_label, 'prediction')
            
            # Comandamos o TorchIO para desdobrar a máscara para o tamanho original e posição exata
            inverse_transform = subject_spatial.get_composed_history().inverse()
            subject_orig_space = inverse_transform(subject_spatial)
            #fim operação de segmentação

            #Operação de Anonimização
            t1 = time.perf_counter()
            
            # 6. Operação de Anonimização (Matrizes perfeitamente encaixadas)
            orig_data = subject_orig_space.image.data[0].numpy()
            mask_array = subject_orig_space.prediction.data[0].numpy() > 0.5
            
            orig_data[mask_array] = 0
            
            # Construir nomes e caminhos
            base_name = os.path.basename(file_path)
            novo_nome_anon = f"{patient_id}_{base_name.replace('.nii.gz', '_anon.nii.gz')}"
            novo_nome_mask = f"{patient_id}_{base_name.replace('.nii.gz', '_mask.nii.gz')}"
            
            anon_output_path = os.path.join(output_dir, novo_nome_anon)
            mask_output_path = os.path.join(output_dir, novo_nome_mask)
            
            # 7. Guardar Ficheiros com o TorchIO
            # Imagem Anonimizada
            anonymized_image = tio.ScalarImage(tensor=torch.tensor(orig_data).unsqueeze(0), affine=subject_orig_space.image.affine)
            
            t2 = time.perf_counter()

            anonymized_image.save(anon_output_path)
            
            # ---> REQUISITO: Guardar a máscara separada para o 3D Slicer <---
            subject_orig_space.prediction.save(mask_output_path)
            t3 = time.perf_counter()
            
            
            seg_time = t1 - t0
            anon_time = t2 - t1
            save_time = t3 - t2
            total_no_save_time = t2 - t0
            total_vol_time = t3 - t0

            total_seg_time += seg_time
            total_anon_time += anon_time
            total_save_time += save_time
            total_pipeline_no_save_time += total_no_save_time
            total_pipeline_time += total_vol_time
            
            print(f"  -> Tempo Segmentação (GPU): {seg_time:.4f} s")
            print(f"  -> Tempo Anonimização: {anon_time:.4f} s")

    if num_volumes > 0:
        print("\n=== SCRIPT DE Inferência CONCLUÍDO ===")
        print(f"Exames processados. Verifica a pasta {output_dir}")
        print(f"Tempo total de segmentação: {total_seg_time:.4f} s")
        print(f"Tempo total de anonimização: {total_anon_time:.4f} s")
        print(f"Tempo total de gravação: {total_save_time:.4f} s")
        print(f"Tempo total do pipeline sem gravação: {total_pipeline_no_save_time:.4f} s")
        print(f"Tempo total do pipeline: {total_pipeline_time:.4f} s")

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
    #PESOS = "/home/andresousa615/rempe/mede_code/results/train_mednext_0.0005_mednext_aurora_16gb_vram/mednext_0.0005_mednext_aurora_16gb_vram"
    TEST_CSV = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/data/test.csv"
    OUTPUT = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/resultados_inferencia/"
    PESOS = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/results/train_mednext_0.0005_mednext_aurora_16gb_vram/mednext_0.0005_mednext_aurora_16gb_vram"
    torch.backends.cudnn.enabled = False
    run_inference(PESOS, TEST_CSV, OUTPUT)