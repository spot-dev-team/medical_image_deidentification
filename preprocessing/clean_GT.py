import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label, binary_opening

# ================= CONFIGURAÇÃO =================
# Pasta onde estão as máscaras que queres limpar (ex: philips, siemens ou ge)
MASK_DIR = r"E:\Tese\Datasets\Rempe\CC_359\cc_359_GT_Masks\ge"

# Pasta onde estão os exames originais (necessário para o filtro de intensidade)
ORIGINAL_DIR = r"E:\Tese\Datasets\Rempe\Original\Original\ge"

# 1. INTENSIDADE: Filtra o que é escuro demais (ar)
NOISE_PERCENTILE = 40

# 2. MORFOLOGIA: Tamanho da "faca" que corta as ligações finas
OPENING_ITERATIONS = 1 
# ================================================

def keep_largest_component(mask_data):
    """ Mantém apenas a maior ilha conectada. """
    labeled_mask, num_features = label(mask_data)
    if num_features == 0: return mask_data
    counts = np.bincount(labeled_mask.ravel())
    counts[0] = 0
    largest_label = counts.argmax()
    clean_mask = np.zeros_like(mask_data)
    clean_mask[labeled_mask == largest_label] = 1
    return clean_mask

def filter_by_intensity(mask_data, original_data, percentile=15):
    """ Remove zonas de baixa intensidade (fundo). """
    values_in_mask = original_data[mask_data > 0]
    if values_in_mask.size == 0: return mask_data

    non_zero_values = values_in_mask[values_in_mask > 1e-3]
    if non_zero_values.size == 0:
        threshold = np.max(values_in_mask) + 1 
    else:
        threshold = np.percentile(non_zero_values, percentile)
    
    filtered_mask = np.zeros_like(mask_data, dtype=np.uint8)
    filtered_mask[(mask_data > 0) & (original_data > threshold)] = 1
    return filtered_mask

def apply_morphological_opening(mask_data, iterations=1):
    """
    Aplica erosão seguida de dilatação.
    Objetivo: Quebrar conexões finas (ruído) mantendo o volume principal (cara).
    """
    opened_mask = binary_opening(mask_data, iterations=iterations).astype(np.uint8)
    return opened_mask

def process_masks():
    if not os.path.exists(MASK_DIR):
        print(f"❌ Pasta de máscaras não encontrada: {MASK_DIR}")
        return
    if not os.path.exists(ORIGINAL_DIR):
        print(f"❌ Pasta de originais não encontrada: {ORIGINAL_DIR}")
        return

    # MUDANÇA: Procurar ficheiros diretamente
    mask_files = [f.name for f in os.scandir(MASK_DIR) if f.is_file() and f.name.endswith('.nii.gz')]
    
    print(f"🧹 A refinar máscaras (V3 - Morfologia) para {len(mask_files)} ficheiros...")

    cleaned_count = 0

    for i, mask_filename in enumerate(mask_files):
        mask_path = os.path.join(MASK_DIR, mask_filename)
        
        # Mapear o nome da máscara para o nome do ficheiro original
        # Ex: "CC0001_philips_15_55_M_mask.nii.gz" -> "CC0001_philips_15_55_M.nii.gz"
        orig_filename = mask_filename.replace("_mask.nii.gz", ".nii.gz")
        orig_path = os.path.join(ORIGINAL_DIR, orig_filename)

        if not os.path.exists(orig_path):
            print(f"⚠️ [{i+1}] Original não encontrado para a máscara {mask_filename}. A saltar.")
            continue

        try:
            mask_img = nib.load(mask_path)
            orig_img = nib.load(orig_path)
            
            mask_data = mask_img.get_fdata().astype(np.uint8)
            orig_data = orig_img.get_fdata()
            
            original_vol = np.sum(mask_data)

            # --- PIPELINE DE LIMPEZA V3 ---
            
            # 1. Filtro de Intensidade (Remove o "preto/ar")
            step1_data = filter_by_intensity(mask_data, orig_data, percentile=NOISE_PERCENTILE)
            
            # 2. Abertura Morfológica (Quebra as pontes finas das "cortinas")
            step2_data = apply_morphological_opening(step1_data, iterations=OPENING_ITERATIONS)
            
            # 3. Manter Maior Componente (Elimina as cortinas agora isoladas)
            final_clean_data = keep_largest_component(step2_data)
            
            clean_vol = np.sum(final_clean_data)
            diff = original_vol - clean_vol
            
            if diff > 0:
                new_img = nib.Nifti1Image(final_clean_data, mask_img.affine, mask_img.header)
                nib.save(new_img, mask_path)
                print(f"[{i+1}] ✨ {mask_filename}: Refinado (Removidos {int(diff)} voxels)")
                cleaned_count += 1
            else:
                print(f"[{i+1}] ✅ {mask_filename}: Estável.")

        except Exception as e:
            print(f"❌ {mask_filename}: Erro - {e}")

    print(f"\n🏁 Terminado! {cleaned_count} máscaras refinadas.")

if __name__ == "__main__":
    process_masks()