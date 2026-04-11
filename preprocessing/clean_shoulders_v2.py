import os
import nibabel as nib
import numpy as np
from scipy.ndimage import median_filter

# ================= CONFIGURAÇÃO =================
BASE_DIR = r"E:\Tese\Datasets\Rempe\processed_datasets\processed_datasets\teste_noise_cleanse\_REJECTED"
MIN_JUMP_VOXELS = 8     # Sensibilidade do degrau
PERSISTENCE_CHECK = 5   # Quantas fatias o recuo tem de se manter para ser validado
# ================================================

def remove_shoulders_top_down():
    print("A iniciar corte de ombros com Varredura Inversa (Top-Down)...")
    count = 0

    for exam_name in os.listdir(BASE_DIR):
        path_exam = os.path.join(BASE_DIR, exam_name)
        if not os.path.isdir(path_exam):
            continue

        # Definição de inputs (prioridade ao clean_1)
        path_clean_1 = os.path.join(path_exam, "mask_GT_clean_1.nii.gz")
        path_base = os.path.join(path_exam, "mask_GT.nii.gz")
        path_out = os.path.join(path_exam, "mask_GT_clean_3.nii.gz")
        
        inp = path_clean_1 if os.path.exists(path_clean_1) else path_base
        if not os.path.exists(inp): continue

        try:
            nifti = nib.load(inp)
            data = nifti.get_fdata().astype(np.uint8)
            
            # 1. Perfil Sagital (Igual ao anterior)
            # Colapsar eixo X
            sagittal = np.any(data > 0, axis=0) 
            
            # Extrair o contorno posterior (Menor Y para cada Z)
            z_dim = data.shape[2]
            y_profile = np.full(z_dim, np.nan)
            
            for z in range(z_dim):
                ys = np.where(sagittal[:, z])[0]
                if len(ys) > 0:
                    y_profile[z] = np.min(ys)
            
            # 2. Varredura Inversa: Iterar do topo (Z max) para a base (Z 0)
            # Procuramos o momento em que a máscara "recua" (Y aumenta) drasticamente
            # Nota: Em RAS, Y menor = Costas, Y maior = Frente.
            # O pescoço tem Y menor (mais atrás). O queixo tem Y maior (mais à frente).
            
            cut_found = False
            z_cut_final = 0
            y_limit_final = 0
            
            # Começamos a iterar de cima para baixo
            # Valid_z_indices garante que só olhamos onde há máscara
            valid_z = np.where(~np.isnan(y_profile))[0]
            if len(valid_z) < 10: continue
            
            # Inverter a lista para percorrer de cima para baixo
            top_down_z = valid_z[::-1] 
            
            for i in range(len(top_down_z) - 1):
                z_curr = top_down_z[i]
                z_next = top_down_z[i+1] # O próximo pixel ABAIXO
                
                y_curr = y_profile[z_curr] # Face/Queixo
                y_next = y_profile[z_next] # Potencial Pescoço
                
                # Se y_next for significativamente MENOR que y_curr, 
                # significa que a máscara foi para trás (entrou no pescoço)
                jump = y_curr - y_next 
                
                if jump > MIN_JUMP_VOXELS:
                    # 3. Validação de Persistência
                    # Verificar se nas próximas N fatias abaixo, o Y se mantém recuado
                    is_stable = True
                    for k in range(1, PERSISTENCE_CHECK + 1):
                        if i + 1 + k >= len(top_down_z): break
                        z_check = top_down_z[i + 1 + k]
                        if y_profile[z_check] > (y_curr - MIN_JUMP_VOXELS/2):
                            is_stable = False # Falso alarme, voltou para a frente
                            break
                    
                    if is_stable:
                        z_cut_final = z_next # O corte é feito na entrada do pescoço
                        y_limit_final = y_curr # Cortamos tudo o que estiver atrás do queixo
                        cut_found = True
                        break # Encontrámos o queixo, paramos a procura!
            
            # 4. Aplicação do Corte
            data_new = np.copy(data)
            if cut_found:
                # Cortar tudo o que está ABAIXO do queixo (Z < z_cut) 
                # E que está ATRÁS do queixo (Y < y_limit)
                # Nota: Em RAS, "atrás" significa Y menor.
                
                # Criar grelhas de coordenadas
                _, YY, ZZ = np.meshgrid(
                    np.arange(data.shape[0]),
                    np.arange(data.shape[1]),
                    np.arange(data.shape[2]),
                    indexing='ij'
                )
                
                mask_ombros = (ZZ < z_cut_final) & (YY < y_limit_final)
                data_new[mask_ombros] = 0
                
                print(f"[{count+1}] {exam_name}: Queixo detetado em Z={z_cut_final}. Corte aplicado.")
            else:
                print(f"[{count+1}] {exam_name}: Perfil contínuo. Nenhum queixo óbvio detetado.")
                
            nib.save(nib.Nifti1Image(data_new, nifti.affine, nifti.header), path_out)
            count += 1

        except Exception as e:
            print(f"Erro em {exam_name}: {e}")

if __name__ == "__main__":
    remove_shoulders_top_down()