import os
import shutil
from pathlib import Path


# Isto separa os exames que já foram defaced dos que ainda não foram, movendo os processados para uma pasta "processed_datasets".


# ================= CONFIGURAÇÃO =================

SOURCE_DIR = r"E:\Tese\Datasets\Rempe\filtered_datasets"
DEST_DIR = r"E:\Tese\Datasets\Rempe\processed_datasets"

# ================================================

def move_processed_exams():
    # Criar pasta de destino se não existir
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"📁 Pasta criada: {DEST_DIR}")

    # Listar todas as pastas de exames
    if not os.path.exists(SOURCE_DIR):
        print("❌ A pasta de origem não existe.")
        return

    exam_folders = [f.path for f in os.scandir(SOURCE_DIR) if f.is_dir()]
    print(f"🔍 A analisar {len(exam_folders)} pastas em procura de ficheiros defaced...")

    moved_count = 0

    for exam_path in exam_folders:
        exam_name = os.path.basename(exam_path)
        
        # 1. Verificar qual o nome do ficheiro original para deduzir o defaced
        possible_files = ["image.nii.gz", "raw.nii.gz"]
        found_defaced = False
        
        for f in possible_files:
            # Constrói o nome esperado: image.nii.gz -> image_defaced.nii.gz
            defaced_name = f.replace(".nii.gz", "_defaced.nii.gz")
            defaced_path = os.path.join(exam_path, defaced_name)
            
            if os.path.exists(defaced_path):
                found_defaced = True
                break
        
        # 2. Se encontrou o defaced, move a pasta inteira
        if found_defaced:
            dest_path = os.path.join(DEST_DIR, exam_name)
            
            # Verificar se já existe no destino para não dar erro
            if os.path.exists(dest_path):
                print(f"⚠️  {exam_name}: Já existe no destino. A saltar.")
            else:
                try:
                    shutil.move(exam_path, dest_path)
                    print(f"✅ {exam_name}: Movido para processados.")
                    moved_count += 1
                except Exception as e:
                    print(f"❌ {exam_name}: Erro ao mover - {e}")

    print(f"\n🏁 Concluído. Total movidos: {moved_count}")

if __name__ == "__main__":
    move_processed_exams()