import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label, binary_opening

# ================= CONFIGURAÇÃO =================
PROCESSED_DIR = r"E:\Tese\Datasets\Rempe\processed_datasets\treino"

# 1. INTENSIDADE: Filtra o que é escuro demais (ar)
# Podes manter 25 se funcionou bem, ou baixar para 20 se a morfologia for agressiva
NOISE_PERCENTILE = 20 

# 2. MORFOLOGIA: Tamanho da "faca" que corta as ligações finas
# Aumenta para 2 ou 3 se as bordas continuarem lá.
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
    # binary_opening usa uma estrutura padrão (cruz) se não especificarmos outra
    # iterations define o quão agressivo é o corte
    opened_mask = binary_opening(mask_data, iterations=iterations).astype(np.uint8)
    return opened_mask

def process_masks():
    if not os.path.exists(PROCESSED_DIR):
        print(f"❌ Pasta não encontrada: {PROCESSED_DIR}")
        return

    exam_folders = [f.path for f in os.scandir(PROCESSED_DIR) if f.is_dir()]
    print(f"🧹 A refinar máscaras (V3 - Morfologia) para {len(exam_folders)} exames...")

    cleaned_count = 0

    for i, exam_path in enumerate(exam_folders):
        exam_name = os.path.basename(exam_path)
        mask_path = os.path.join(exam_path, "mask_GT.nii.gz")
        
        possible_orig = ["image.nii.gz", "raw.nii.gz"]
        orig_path = None
        for f in possible_orig:
            if os.path.exists(os.path.join(exam_path, f)):
                orig_path = os.path.join(exam_path, f)
                break

        if not os.path.exists(mask_path) or orig_path is None:
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
                print(f"[{i+1}] ✨ {exam_name}: Refinado (Removidos {int(diff)} voxels)")
                cleaned_count += 1
            else:
                print(f"[{i+1}] ✅ {exam_name}: Estável.")

        except Exception as e:
            print(f"❌ {exam_name}: Erro - {e}")

    print(f"\n🏁 Terminado! {cleaned_count} máscaras refinadas.")

if __name__ == "__main__":
    process_masks()