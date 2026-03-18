import os
import subprocess
import glob
from pathlib import Path
import shutil
import numpy as np
import nibabel as nib  # Necessário para ler as imagens e calcular a máscara

# ================= CONFIGURAÇÃO =================
# Pasta raiz onde estão as pastas dos exames
ROOT_DIR = r"E:\Tese\Datasets\Rempe\filtered_datasets"

# Nome da distribuição WSL
WSL_DISTRO = "Ubuntu-22.04" 

# Caminhos Linux (Baseados na tua configuração atual)
FSL_DIR_LINUX = "/home/andresousa615/fsl"
PYDEFACE_BIN = "/home/andresousa615/.local/bin/pydeface"
# ================================================

def to_wsl_path(win_path):
    """Converte um caminho Windows para o formato WSL."""
    path = Path(win_path).resolve()
    drive = path.drive.lower().replace(':', '')
    rel_path = path.relative_to(path.anchor).as_posix()
    return f"/mnt/{drive}/{rel_path}"

def create_mask_from_difference(original_path, defaced_path, output_mask_path):
    """
    Gera a máscara subtraindo a imagem defaced da original.
    Máscara = |Original - Defaced| > 0
    """
    try:
        # Carregar as imagens NIfTI
        img_orig = nib.load(original_path)
        img_def = nib.load(defaced_path)
        
        data_orig = img_orig.get_fdata()
        data_def = img_def.get_fdata()
        
        # Calcular a diferença absoluta
        diff = np.abs(data_orig - data_def)
        
        # Criar máscara binária (onde a diferença for maior que um valor ínfimo)
        mask_data = np.zeros_like(data_orig, dtype=np.uint8)
        mask_data[diff > 1e-4] = 1 
        
        # Guardar a nova máscara usando o header da imagem original
        new_img = nib.Nifti1Image(mask_data, img_orig.affine, img_orig.header)
        nib.save(new_img, output_mask_path)
        return True
    except Exception as e:
        print(f"      ❌ Erro ao calcular máscara via subtração: {e}")
        return False

def process_exams():
    if not os.path.exists(ROOT_DIR):
        print(f"❌ Erro: A pasta {ROOT_DIR} não existe.")
        return

    exam_folders = [f.path for f in os.scandir(ROOT_DIR) if f.is_dir()]
    print(f"📂 Encontrados {len(exam_folders)} exames para processar em: {ROOT_DIR}\n")

    for i, exam_path in enumerate(exam_folders):
        exam_name = os.path.basename(exam_path)
        print(f"[{i+1}/{len(exam_folders)}] Processando: {exam_name} ...")

        # 1. Identificar ficheiro de entrada
        possible_files = ["image.nii.gz", "raw.nii.gz"]
        input_file = None
        for f in possible_files:
            full_path = os.path.join(exam_path, f)
            if os.path.exists(full_path):
                input_file = full_path
                break
        
        if not input_file:
            print(f"   ⚠️  Saltado: Sem imagem válida.")
            continue

        # Caminhos finais
        final_mask_name = f"pydeface_mask_{exam_name}.nii.gz"
        final_mask_path = os.path.join(exam_path, final_mask_name)
        
        # O ficheiro que o PyDeface gera sempre (Defaced Image)
        defaced_image_path = input_file.replace(".nii.gz", "_defaced.nii.gz")

        # Se a máscara final já existe, saltamos
        if os.path.exists(final_mask_path):
            print(f"   ✅ Máscara final já existe. A saltar.")
            continue

        # 2. Executar PyDeface (apenas se a imagem defaced ainda não existir)
        if not os.path.exists(defaced_image_path):
            wsl_input = to_wsl_path(input_file)
            
            cmd = (
                f"export FSLDIR={FSL_DIR_LINUX}; "
                f". {FSL_DIR_LINUX}/etc/fslconf/fsl.sh; "
                f"export PATH=\"{FSL_DIR_LINUX}/bin:$PATH\"; " 
                f"{PYDEFACE_BIN} '{wsl_input}' --force"
            )
            
            try:
                # capture_output=False permite ver o output do FSL a acontecer
                subprocess.run(["wsl", "-d", WSL_DISTRO, "bash", "-c", cmd], check=True, capture_output=False)
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Erro ao executar PyDeface no WSL: {e}")
                continue
        else:
            print(f"   ℹ️  Imagem defaced já existe. A calcular apenas a máscara...")

        # 3. Gerar a Máscara via Python (Subtração)
        if os.path.exists(defaced_image_path):
            success = create_mask_from_difference(input_file, defaced_image_path, final_mask_path)
            
            if success:
                print(f"   ✨ Sucesso! Máscara criada: {final_mask_name}")
                # Opcional: Apagar a imagem defaced para poupar espaço, já que só queres a máscara
                # os.remove(defaced_image_path)
        else:
            print(f"   ❌ Erro Crítico: O PyDeface terminou mas não gerou {os.path.basename(defaced_image_path)}")

if __name__ == "__main__":
    process_exams()