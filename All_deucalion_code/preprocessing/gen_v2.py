import os
import subprocess
import uuid
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= CONFIGURAÇÃO =================
ROOT_DIR = r"E:\Tese\Datasets\Rempe\filtered_datasets"
WSL_DISTRO = "Ubuntu-22.04" 
FSL_DIR_LINUX = "/home/andresousa615/fsl"
PYDEFACE_BIN = "/home/andresousa615/.local/bin/pydeface"
MAX_WORKERS = 6  # Mantém os 6 workers para ser rápido
# ================================================

def to_wsl_path(win_path):
    path = Path(win_path).resolve()
    drive = path.drive.lower().replace(':', '')
    rel_path = path.relative_to(path.anchor).as_posix()
    return f"/mnt/{drive}/{rel_path}"

def process_single_exam(exam_path):
    exam_name = os.path.basename(exam_path)

    # 1. Identificar input
    possible_files = ["image.nii.gz", "raw.nii.gz"]
    input_file = None
    for f in possible_files:
        full_path = os.path.join(exam_path, f)
        if os.path.exists(full_path):
            input_file = full_path
            break
    
    if not input_file:
        return f"⚠️  {exam_name}: Saltado (Sem imagem original)."

    # O objetivo agora é APENAS gerar este ficheiro
    defaced_image_path = input_file.replace(".nii.gz", "_defaced.nii.gz")

    if os.path.exists(defaced_image_path):
        return f"✅ {exam_name}: Já existe."

    # 2. Preparar Comando WSL (Copy-Process-Copy)
    wsl_input = to_wsl_path(input_file)
    wsl_output_defaced = to_wsl_path(defaced_image_path)
    
    # ID único para não haver conflitos na pasta /tmp entre os 6 workers
    unique_id = str(uuid.uuid4())[:8]
    linux_tmp_in = f"/tmp/in_{unique_id}.nii.gz"
    # O Pydeface adiciona _defaced automaticamente ao output, temos de prever isso
    linux_tmp_out_created = f"/tmp/in_{unique_id}_defaced.nii.gz"

    cmd = (
        f"export FSLDIR={FSL_DIR_LINUX}; "
        f". {FSL_DIR_LINUX}/etc/fslconf/fsl.sh; "
        f"export PATH=\"{FSL_DIR_LINUX}/bin:$PATH\"; " 
        f"cp '{wsl_input}' {linux_tmp_in}; "            # Copia para Linux
        f"{PYDEFACE_BIN} {linux_tmp_in} --force; "     # Processa
        f"cp {linux_tmp_out_created} '{wsl_output_defaced}'; " # Devolve ao Windows
        f"rm {linux_tmp_in} {linux_tmp_out_created}"  # Limpa
    )
    
    try:
        # capture_output=True para o terminal não ficar uma confusão
        subprocess.run(["wsl", "-d", WSL_DISTRO, "bash", "-c", cmd], check=True, capture_output=True)
        
        # Verificar se o ficheiro chegou mesmo ao Windows
        if os.path.exists(defaced_image_path):
            return f"✨ {exam_name}: Defaced gerado com sucesso."
        else:
            return f"❌ {exam_name}: O comando correu mas o ficheiro não apareceu."

    except subprocess.CalledProcessError:
        return f"❌ {exam_name}: Erro interno no PyDeface/WSL."

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"❌ Erro: A pasta {ROOT_DIR} não existe.")
        return

    exam_folders = [f.path for f in os.scandir(ROOT_DIR) if f.is_dir()]
    total_exams = len(exam_folders)
    
    print(f"🚀 A gerar APENAS imagens defaced.")
    print(f"📂 Exames: {total_exams} | Workers: {MAX_WORKERS}")
    print("-" * 40)
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_exam, folder): folder for folder in exam_folders}
        
        finished_count = 0
        for future in as_completed(futures):
            finished_count += 1
            print(f"[{finished_count}/{total_exams}] {future.result()}")

if __name__ == "__main__":
    main()