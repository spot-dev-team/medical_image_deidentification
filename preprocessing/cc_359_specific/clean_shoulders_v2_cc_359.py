import os
import nibabel as nib
import numpy as np
from scipy.ndimage import median_filter

# ================= CONFIGURAÇÃO =================
BASE_DIR = r"E:\Tese\Datasets\Rempe\CC_359\cc_359_GT_Masks\ge"
MIN_JUMP_VOXELS = 8     # Sensibilidade do degrau
PERSISTENCE_CHECK = 5   # Quantas fatias o recuo tem de se manter para ser validado
# ================================================

def remove_shoulders_top_down():
    print("A iniciar corte de ombros com Varredura Inversa (Top-Down)...")
    count = 0

    for file_name in os.listdir(BASE_DIR):
        # Processar apenas ficheiros NIfTI e ignorar os que já foram cortados
        if not file_name.endswith('_closed.nii.gz') or '_shoulderless' in file_name:
            continue

        inp = os.path.join(BASE_DIR, file_name)
        
        # Gerar o novo nome de output dinamicamente
        out_name = file_name.replace("_closed.nii.gz", "_shoulderless.nii.gz")
        path_out = os.path.join(BASE_DIR, out_name)

        try:
            nifti = nib.load(inp)
            data = nifti.get_fdata().astype(np.uint8)
            
            # 1. Perfil Sagital
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
            cut_found = False
            z_cut_final = 0
            y_limit_final = 0
            
            valid_z = np.where(~np.isnan(y_profile))[0]
            if len(valid_z) < 10: continue
            
            # Inverter a lista para percorrer de cima para baixo
            top_down_z = valid_z[::-1] 
            
            for i in range(len(top_down_z) - 1):
                z_curr = top_down_z[i]
                z_next = top_down_z[i+1] # O próximo pixel ABAIXO
                
                y_curr = y_profile[z_curr] # Face/Queixo
                y_next = y_profile[z_next] # Potencial Pescoço
                
                jump = y_curr - y_next 
                
                if jump > MIN_JUMP_VOXELS:
                    # 3. Validação de Persistência
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
                # Criar grelhas de coordenadas
                _, YY, ZZ = np.meshgrid(
                    np.arange(data.shape[0]),
                    np.arange(data.shape[1]),
                    np.arange(data.shape[2]),
                    indexing='ij'
                )
                
                mask_ombros = (ZZ < z_cut_final) & (YY < y_limit_final)
                data_new[mask_ombros] = 0
                
                print(f"[{count+1}] {file_name}: Queixo detetado em Z={z_cut_final}. Corte aplicado.")
            else:
                print(f"[{count+1}] {file_name}: Perfil contínuo. Nenhum queixo óbvio detetado.")
                
            nib.save(nib.Nifti1Image(data_new, nifti.affine, nifti.header), path_out)
            count += 1

        except Exception as e:
            print(f"Erro em {file_name}: {e}")

if __name__ == "__main__":
    remove_shoulders_top_down()