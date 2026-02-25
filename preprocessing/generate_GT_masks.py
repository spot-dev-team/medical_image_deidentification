import os
import numpy as np
import nibabel as nib

# ================= CONFIGURAÇÃO =================
# Pasta onde estão os exames
PROCESSED_DIR = r"E:\Tese\Datasets\Rempe\dataset_test_complete"
# ================================================

def create_mask_strict(original_path, defaced_path, output_path):
    try:
        # Carregar imagens
        img_orig = nib.load(original_path)
        img_def = nib.load(defaced_path)
        
        # Carregar dados como float para evitar erros de tipos (int vs float)
        data_orig = img_orig.get_fdata()
        data_def = img_def.get_fdata()
        
        # Verificar dimensões
        if data_orig.shape != data_def.shape:
            print(f"   ❌ Erro: Dimensões incompatíveis ({data_orig.shape} vs {data_def.shape}).")
            return False

        # --- NOVA LÓGICA (A Regra dos Zeros) ---
        
        # 1. Onde é que a imagem Defaced é ZERO (ou muito próximo de zero)?
        # Usamos < 1e-4 para apanhar zeros mesmo que sejam 0.0000001
        is_zero_in_defaced = (np.abs(data_def) < 1e-4)

        # 2. Onde é que a imagem Original NÃO ERA ZERO?
        # Isto evita criar máscara no ar/fundo da imagem
        has_tissue_in_original = (np.abs(data_orig) > 1e-4)

        # 3. A Máscara é a interseção: Tinha tecido no original E desapareceu no defaced
        mask = np.logical_and(is_zero_in_defaced, has_tissue_in_original).astype(np.uint8)

        # --- FIM DA NOVA LÓGICA ---

        # Validação Rápida
        vol_mask = np.sum(mask)
        if vol_mask == 0:
            print(f"   ⚠️  Aviso: Máscara vazia (O Defaced não tinha zeros onde devia?).")
            # Forçar gravação mesmo assim para debug
        
        # Guardar
        mask_nifti = nib.Nifti1Image(mask, img_orig.affine, img_orig.header)
        nib.save(mask_nifti, output_path)
        return True

    except Exception as e:
        print(f"   ❌ Erro de processamento: {e}")
        return False

def generate_masks():
    if not os.path.exists(PROCESSED_DIR):
        print(f"❌ Pasta não encontrada: {PROCESSED_DIR}")
        return

    exam_folders = [f.path for f in os.scandir(PROCESSED_DIR) if f.is_dir()]
    print(f"🚀 A gerar máscaras (Lógica: Zeros) para {len(exam_folders)} exames...")

    count = 0
    for i, exam_path in enumerate(exam_folders):
        exam_name = os.path.basename(exam_path)
        
        # Identificar ficheiros
        possible_files = ["image.nii.gz", "raw.nii.gz"]
        input_file = None
        for f in possible_files:
            if os.path.exists(os.path.join(exam_path, f)):
                input_file = os.path.join(exam_path, f)
                break
        
        if not input_file:
            continue 

        defaced_file = input_file.replace(".nii.gz", "_defaced.nii.gz")


        mask_output = os.path.join(exam_path, "mask_GT.nii.gz")

        if os.path.exists(mask_output):
            print(f"⚠️  {exam_name}: Máscara já existe. A saltar.")
            os.remove(mask_output)  # Forçar re-criação para corrigir más anteriores
            print(f"   🧹 Máscara antiga removida para regeneração.")

        if os.path.exists(defaced_file):
            # Vou forçar a re-escrita mesmo que já exista, para corrigir as más anteriores
            print(f"[{i+1}] A corrigir máscara: {exam_name}")
            success = create_mask_strict(input_file, defaced_file, mask_output)
            if success:
                count += 1
        else:
            print(f"❌ {exam_name}: Faltava o ficheiro defaced.")

    print(f"\n🏁 Terminado! {count} máscaras geradas.")

if __name__ == "__main__":
    generate_masks()