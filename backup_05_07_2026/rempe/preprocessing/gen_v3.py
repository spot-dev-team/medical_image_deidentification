#tem um timeout para nunca encravar em nenhum exame, e continua a processar os restantes. O timeout é de 20 minutos, o que é mais do que suficiente para a maioria dos exames, mesmo os mais complexos. Se um exame demorar mais do que isso, ele é cancelado e o script avança para o próximo, garantindo que o processamento geral não seja interrompido por um único caso problemático.


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

# REDUZI PARA 4 PARA GARANTIR ESTABILIDADE DURANTE A NOITE
MAX_WORKERS = 4  
# ================================================

def to_wsl_path(win_path):
    path = Path(win_path).resolve()
    drive = path.drive.lower().replace(':', '')
    rel_path = path.relative_to(path.anchor).as_posix()
    return f"/mnt/{drive}/{rel_path}"

def process_single_exam(exam_path):
    exam_name = os.path.basename(exam_path)

    possible_files = ["image.nii.gz", "raw.nii.gz"]
    input_file = None
    for f in possible_files:
        full_path = os.path.join(exam_path, f)
        if os.path.exists(full_path):
            input_file = full_path
            break
    
    if not input_file:
        return f"⚠️  {exam_name}: Saltado (Sem imagem original)."

    defaced_image_path = input_file.replace(".nii.gz", "_defaced.nii.gz")

    # VERIFICAÇÃO DE PROGRESSO: Se já existe, salta imediatamente
    if os.path.exists(defaced_image_path):
        return f"✅ {exam_name}: Já existe (Recuperado)."

    # Preparar Comando WSL
    wsl_input = to_wsl_path(input_file)
    wsl_output_defaced = to_wsl_path(defaced_image_path)
    
    unique_id = str(uuid.uuid4())[:8]
    linux_tmp_in = f"/tmp/in_{unique_id}.nii.gz"
    linux_tmp_out_created = f"/tmp/in_{unique_id}_defaced.nii.gz"

    cmd = (
        f"export FSLDIR={FSL_DIR_LINUX}; "
        f". {FSL_DIR_LINUX}/etc/fslconf/fsl.sh; "
        f"export PATH=\"{FSL_DIR_LINUX}/bin:$PATH\"; " 
        f"cp '{wsl_input}' {linux_tmp_in}; "            
        f"{PYDEFACE_BIN} {linux_tmp_in} --force; "     
        f"cp {linux_tmp_out_created} '{wsl_output_defaced}'; " 
        f"rm {linux_tmp_in} {linux_tmp_out_created}"  
    )
    
    try:
        # NOVIDADE: timeout=1200 segundos (20 minutos)
        # Se demorar mais que isto, o Python mata o processo e avança.
        subprocess.run(
            ["wsl", "-d", WSL_DISTRO, "bash", "-c", cmd], 
            check=True, 
            capture_output=True,
            timeout=1200 
        )
        
        if os.path.exists(defaced_image_path):
            return f"✨ {exam_name}: Defaced gerado com sucesso."
        else:
            return f"❌ {exam_name}: Ficheiro não gerado (Erro silencioso)."

    except subprocess.TimeoutExpired:
        # Se passar dos 20 minutos, cai aqui
        return f"⏰ {exam_name}: TIMEOUT - Cancelado por demorar mais de 20min."
        
    except subprocess.CalledProcessError:
        return f"❌ {exam_name}: Erro interno no PyDeface."

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"❌ Erro: A pasta {ROOT_DIR} não existe.")
        return

    exam_folders = [f.path for f in os.scandir(ROOT_DIR) if f.is_dir()]
    total_exams = len(exam_folders)
    
    print(f"🚀 A retomar processamento (Com Timeout de 20min).")
    print(f"📂 Exames Totais: {total_exams} | Workers: {MAX_WORKERS}")
    print("-" * 40)
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_exam, folder): folder for folder in exam_folders}
        
        finished_count = 0
        for future in as_completed(futures):
            finished_count += 1
            print(f"[{finished_count}/{total_exams}] {future.result()}")

if __name__ == "__main__":
    main()