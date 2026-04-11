import os
import nibabel as nib
import numpy as np

# ================= CONFIGURAÇÃO =================
BASE_DIR = r"E:\Tese\Datasets\Rempe\CC_359\cc_359_GT_Masks\ge\_REJECTED"

# Critério 1: Salto Abrupto (O teu método original para pescoços direitos)
MIN_JUMP_VOXELS = 8     
PERSISTENCE_CHECK = 5   

# Critério 2: Rampa Diagonal (Novo método para pescoços inclinados)
RAMP_DROP_TOTAL = 10    # Quanto o contorno tem de recuar no total da rampa
RAMP_WINDOW = 8         # Quantas fatias usamos para medir a rampa
# ================================================

def remove_shoulders_robust():
    print("A iniciar corte de ombros robusto (Saltos e Rampas)...")
    count = 0

    for file_name in os.listdir(BASE_DIR):
        if not file_name.endswith('shoulderless.nii.gz'):
            continue

        inp = os.path.join(BASE_DIR, file_name)
        out_name = file_name.replace("shoulderless.nii.gz", "_shoulderless_robust.nii.gz")
        path_out = os.path.join(BASE_DIR, out_name)

        try:
            nifti = nib.load(inp)
            data = nifti.get_fdata().astype(np.uint8)
            
            # 1. Perfil Sagital (Igual ao original)
            sagittal = np.any(data > 0, axis=0) 
            z_dim = data.shape[2]
            y_profile = np.full(z_dim, np.nan)
            
            for z in range(z_dim):
                ys = np.where(sagittal[:, z])[0]
                if len(ys) > 0:
                    y_profile[z] = np.min(ys)
            
            valid_z = np.where(~np.isnan(y_profile))[0]
            if len(valid_z) < 10: continue
            
            top_down_z = valid_z[::-1] 
            
            cut_found = False
            z_cut_final = 0
            y_limit_final = 0
            
            # Limite de segurança: nunca cortar acima de 30% do topo da máscara
            z_safety_limit = top_down_z[int(len(top_down_z) * 0.3)]

            # 2. Varredura Híbrida
            for i in range(len(top_down_z) - RAMP_WINDOW):
                z_curr = top_down_z[i]
                
                # Prevenção: Não cortar demasiado alto na cabeça
                if z_curr > z_safety_limit:
                    continue

                # --- MÉTODO A: O Salto Abrupto (O teu original) ---
                z_next = top_down_z[i+1] 
                y_curr = y_profile[z_curr] 
                y_next = y_profile[z_next] 
                
                jump = y_curr - y_next 
                
                if jump > MIN_JUMP_VOXELS:
                    is_stable = True
                    for k in range(1, PERSISTENCE_CHECK + 1):
                        if i + 1 + k >= len(top_down_z): break
                        z_check = top_down_z[i + 1 + k]
                        if y_profile[z_check] > (y_curr - MIN_JUMP_VOXELS/2):
                            is_stable = False
                            break
                    if is_stable:
                        z_cut_final = z_next 
                        y_limit_final = y_curr 
                        cut_found = True
                        method = "Salto"
                        break
                
                # --- MÉTODO B: A Rampa Diagonal (Novo) ---
                # Olha 'RAMP_WINDOW' fatias para a frente. Se o contorno recuou 
                # consistentemente 'RAMP_DROP_TOTAL' voxels, encontrámos o pescoço!
                z_future = top_down_z[i + RAMP_WINDOW - 1]
                y_future = y_profile[z_future]
                
                ramp_drop = y_curr - y_future
                
                if ramp_drop > RAMP_DROP_TOTAL:
                    # Verificar se é uma descida consistente (sem voltar para a frente)
                    is_consistent = True
                    for k in range(1, RAMP_WINDOW):
                        z_mid = top_down_z[i + k]
                        # Se algum ponto a meio do caminho estiver mais à frente que o início
                        if y_profile[z_mid] > y_curr + 2: 
                            is_consistent = False
                            break
                            
                    if is_consistent:
                        z_cut_final = z_curr # O corte começa no início da descida
                        y_limit_final = y_curr
                        cut_found = True
                        method = "Rampa"
                        break

            # 3. Aplicação do Corte (Seguro)
            data_new = np.copy(data)
            if cut_found:
                _, YY, ZZ = np.meshgrid(
                    np.arange(data.shape[0]),
                    np.arange(data.shape[1]),
                    np.arange(data.shape[2]),
                    indexing='ij'
                )
                
                # Só apaga o que está ABAIXO do corte e ATRÁS do limite do queixo
                mask_ombros = (ZZ <= z_cut_final) & (YY <= y_limit_final)
                data_new[mask_ombros] = 0
                
                print(f"[{count+1}] {file_name}: Pescoço ({method}) em Z={z_cut_final}. Corte aplicado.")
            else:
                print(f"[{count+1}] {file_name}: Perfil contínuo. Nenhum pescoço detetado.")
                
            nib.save(nib.Nifti1Image(data_new, nifti.affine, nifti.header), path_out)
            count += 1

        except Exception as e:
            print(f"Erro em {file_name}: {e}")

if __name__ == "__main__":
    remove_shoulders_robust()