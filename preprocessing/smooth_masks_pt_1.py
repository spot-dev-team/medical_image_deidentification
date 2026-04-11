import os
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_closing

# ================= CONFIGURAÇÃO =================
# A pasta onde tens as máscaras atuais
MASK_DIR = r"E:\Tese\Datasets\Rempe\CC_359\cc_359_GT_Masks\ge"

# O sufixo dos ficheiros que queres processar (ajusta se necessário)
TARGET_SUFFIX = "_mask.nii.gz"

# Quantas "passagens de espátula" queres dar. 
# 1 ou 2 costuma ser o ideal. Valores muito altos distorcem a face original.
CLOSING_ITERATIONS = 2 
# ================================================

def apply_morphological_closing(mask_data, iterations=1):
    """ 
    Suaviza a superfície exterior e fecha fendas finas 
    sem alterar o volume global da segmentação.
    """
    closed_mask = binary_closing(mask_data, iterations=iterations).astype(np.uint8)
    return closed_mask

def process_closing():
    if not os.path.exists(MASK_DIR):
        print(f"❌ Pasta não encontrada: {MASK_DIR}")
        return

    # Procura apenas os ficheiros alvo
    mask_files = [f.name for f in os.scandir(MASK_DIR) if f.is_file() and f.name.endswith(TARGET_SUFFIX)]
    
    print(f"🧱 A aplicar Fechamento Morfológico ({CLOSING_ITERATIONS} iterações) em {len(mask_files)} máscaras...")

    count = 0

    for i, mask_filename in enumerate(mask_files):
        mask_path = os.path.join(MASK_DIR, mask_filename)
        
        # Cria um nome para a nova máscara alisada
        out_name = mask_filename.replace(TARGET_SUFFIX, "_closed.nii.gz")
        out_path = os.path.join(MASK_DIR, out_name)

        try:
            # Carregar a imagem
            nifti_img = nib.load(mask_path)
            mask_data = nifti_img.get_fdata().astype(np.uint8)
            
            # Aplicar o Fechamento
            smoothed_data = apply_morphological_closing(mask_data, iterations=CLOSING_ITERATIONS)
            
            # Verificar se houve alterações reais
            diff = np.sum(smoothed_data) - np.sum(mask_data)
            
            # Guardar o resultado
            new_img = nib.Nifti1Image(smoothed_data, nifti_img.affine, nifti_img.header)
            nib.save(new_img, out_path)
            
            print(f"[{i+1}] ✨ {mask_filename}: Suavizado (Adicionados {int(diff)} voxels para tapar fendas).")
            count += 1

        except Exception as e:
            print(f"❌ Erro em {mask_filename}: {e}")

    print(f"\n🏁 Terminado! {count} máscaras suavizadas com sucesso.")

if __name__ == "__main__":
    process_closing()