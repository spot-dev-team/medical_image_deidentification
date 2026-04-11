import os
import pandas as pd
from sklearn.model_selection import train_test_split
from glob import glob

# ================= CONFIGURAÇÃO =================
# Pastas onde tens os teus exames organizados
# (Uso 'r' antes das strings para o Windows não se confundir com as barras)
TRAIN_VAL_SOURCE = r"/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/processed_datasets/processed_datasets/treino"
TEST_SOURCE = r"/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/processed_datasets/processed_datasets/teste"


# Onde vamos guardar os CSVs para o cluster ler
OUTPUT_CSV_DIR = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/mede/data"
# ================================================

def get_image_mask_pairs(folder_path):
    """
    Percorre uma pasta e devolve uma lista de dicionários com os caminhos
    absolutos de cada par (imagem, máscara).
    """
    if not os.path.exists(folder_path):
        print(f"⚠️ Aviso: A pasta {folder_path} não existe.")
        return []

    # Listar subpastas (cada subpasta é um exame)
    exam_folders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    dataset = []

    print(f"🔍 A analisar {folder_path}...")

    for exam in exam_folders:
        # 1. Procurar Imagem Original (image.nii.gz ou raw.nii.gz)
        img_path = None
        if os.path.exists(os.path.join(exam, "image.nii.gz")):
            img_path = os.path.join(exam, "image.nii.gz")
        elif os.path.exists(os.path.join(exam, "raw.nii.gz")):
            img_path = os.path.join(exam, "raw.nii.gz")
        
        # 2. Procurar Máscara Ground Truth (que gerámos antes)
        mask_path = os.path.join(exam, "mask_GT.nii.gz")

        # 3. Validar se ambos existem
        if img_path and os.path.exists(mask_path):
            dataset.append({
                "image_path": os.path.abspath(img_path),
                "mask_path": os.path.abspath(mask_path)
            })
    
    return dataset

def main():
    if not os.path.exists(OUTPUT_CSV_DIR):
        os.makedirs(OUTPUT_CSV_DIR)

    # --- PASSO 1: Preparar Treino e Validação ---
    # Lemos tudo o que está na pasta "treino"
    all_train_data = get_image_mask_pairs(TRAIN_VAL_SOURCE)
    
    if len(all_train_data) == 0:
        print("❌ Erro: Não encontrei exames na pasta de treino.")
        return

    # Dividimos automaticamente: 80% Treino, 20% Validação
    # O random_state=42 garante que a divisão é sempre igual (reprodutível)
    train_data, val_data = train_test_split(all_train_data, test_size=0.2, random_state=42)

    # --- PASSO 2: Preparar Teste ---
    # Lemos tudo o que está na pasta "teste" (não dividimos nada aqui)
    test_data = get_image_mask_pairs(TEST_SOURCE)

    # --- PASSO 3: Salvar CSVs ---
    # Converter para DataFrames do Pandas
    df_train = pd.DataFrame(train_data)
    df_val = pd.DataFrame(val_data)
    df_test = pd.DataFrame(test_data)

    # Guardar ficheiros
    train_csv = os.path.join(OUTPUT_CSV_DIR, "train.csv")
    val_csv = os.path.join(OUTPUT_CSV_DIR, "val.csv")
    test_csv = os.path.join(OUTPUT_CSV_DIR, "test.csv")

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    
    # Só guardamos o test.csv se houver dados, para não dar erro
    if not df_test.empty:
        df_test.to_csv(test_csv, index=False)

    print("-" * 30)
    print(f"✅ CSVs gerados com sucesso em '{OUTPUT_CSV_DIR}/':")
    print(f"   📂 Treino:    {len(df_train)} exames -> {train_csv}")
    print(f"   📂 Validação: {len(df_val)} exames -> {val_csv}")
    print(f"   📂 Teste:     {len(df_test)} exames -> {test_csv}")
    print("-" * 30)

if __name__ == "__main__":
    main()
