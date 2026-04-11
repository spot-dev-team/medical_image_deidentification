import os
import numpy as np
import nibabel as nib

# ================= CONFIGURAÇÃO =================
PROCESSED_DIR = r"E:\Tese\Datasets\Rempe\CC_359\cc_359_GT_Masks\ge"

# O sufixo da máscara que queres usar como entrada
TARGET_SUFFIX = "_closed.nii.gz" 
# ================================================

def apply_scanline_fencing(mask_data):
    """
    Preenche cavidades internas usando interseção de varreduras.
    Evita a criação de "tendas" em zonas salientes como o nariz.
    """
    # 1. Varredura Esquerda-Direita (Eixo 0)
    # Deteta o espaço trancado entre o tecido mais à esquerda e mais à direita
    cx = np.cumsum(mask_data, axis=0)
    cx_rev = np.cumsum(mask_data[::-1, :, :], axis=0)[::-1, :, :]
    fill_x = (cx > 0) & (cx_rev > 0)

    # 2. Varredura Frente-Trás (Eixo 1)
    # Deteta o espaço trancado entre o tecido mais à frente e mais atrás
    cy = np.cumsum(mask_data, axis=1)
    cy_rev = np.cumsum(mask_data[:, ::-1, :], axis=1)[:, ::-1, :]
    fill_y = (cy > 0) & (cy_rev > 0)

    # 3. A Interseção (O truque para evitar o triângulo do nariz)
    # Só preenche se o voxel estiver "fechado" em ambas as direções!
    fenced_mask = fill_x & fill_y
    
    # Opcional: Se quiseres ser ainda mais estrito, podes adicionar o eixo Z (Cima-Baixo)
    # cz = np.cumsum(mask_data, axis=2)
    # cz_rev = np.cumsum(mask_data[:, :, ::-1], axis=2)[:, :, ::-1]
    # fill_z = (cz > 0) & (cz_rev > 0)
    # fenced_mask = fenced_mask & fill_z

    return fenced_mask.astype(np.uint8)

def process_fast_fencing():
    if not os.path.exists(PROCESSED_DIR):
        print(f"❌ Pasta não encontrada: {PROCESSED_DIR}")
        return

    mask_files = [f.name for f in os.scandir(PROCESSED_DIR) if f.is_file() and f.name.endswith(TARGET_SUFFIX)]
    
    print(f"⚡ A aplicar Fast Scanline Fencing em {len(mask_files)} máscaras...")

    count = 0

    for i, mask_filename in enumerate(mask_files):
        inp = os.path.join(PROCESSED_DIR, mask_filename)
        
        # Cria um nome para a nova máscara vedada
        out_name = mask_filename.replace(TARGET_SUFFIX, "_closed.nii.gz")
        out_path = os.path.join(PROCESSED_DIR, out_name)

        try:
            nifti_img = nib.load(inp)
            mask_data = nifti_img.get_fdata().astype(np.uint8)
            
            # Aplicar o novo algoritmo rápido
            fenced_data = apply_scanline_fencing(mask_data)
            
            # Verificar alterações
            diff = np.sum(fenced_data) - np.sum(mask_data)
            
            new_img = nib.Nifti1Image(fenced_data, nifti_img.affine, nifti_img.header)
            nib.save(new_img, out_path)
            
            print(f"[{i+1}] ✨ {mask_filename}: Concluído instantaneamente (Preenchidos {int(diff)} voxels ocos).")
            count += 1

        except Exception as e:
            print(f"❌ Erro em {mask_filename}: {e}")

    print(f"\n🏁 Terminado! {count} máscaras seladas à velocidade da luz.")

if __name__ == "__main__":
    process_fast_fencing()