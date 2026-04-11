# resize_datasets_offline.py
import os
import torchio as tio
from glob import glob
from tqdm import tqdm
import argparse

def process_folder(base_dir):
    print(f"A procurar ficheiros em: {base_dir}")
    
    # Procura todos os ficheiros .nii recursivamente em todas as subpastas
    nii_files = glob(os.path.join(base_dir, '**', '*.nii'), recursive=True)
    
    if not nii_files:
        print("Nenhum ficheiro .nii encontrado! Verifica se a pasta está correta.")
        return

    print(f"Foram encontrados {len(nii_files)} ficheiros para redimensionar.")
    
    # 1. ToCanonical: Garante a orientação padrão RAS para todos os eixos
    canonical_transform = tio.ToCanonical()
    
    # 2. Resize: Força a grelha para 128x128x128
    # (O TorchIO aplica a interpolação matematicamente correta dependendo do tipo de imagem)
    resize_transform = tio.Resize((128, 128, 128))

    for filepath in tqdm(nii_files, desc="A redimensionar exames"):
        try:
            filename = os.path.basename(filepath).lower()
            
            # Detetar se é uma máscara com base no nome do ficheiro
            # Apanha o "mask_GT_clean_1.nii", "mask_GT_clean_original.nii" e "mask_GT.nii"
            if "mask" in filename or "gt" in filename:
                img = tio.LabelMap(filepath)
            # Apanha o "image.nii"
            else:
                img = tio.ScalarImage(filepath)
                
            subject = tio.Subject(img=img)
            
            # Aplicar as transformações
            subject = canonical_transform(subject)
            subject = resize_transform(subject)
            
            # Gravar o ficheiro com o novo tamanho (escreve por cima do antigo)
            # Como tens a pasta "treino_original_dimensions", isto é seguro!
            subject['img'].save(filepath)
            
        except Exception as e:
            print(f"\n[ERRO] Falha ao processar o ficheiro {filepath}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Caminho para a pasta 'treino'")
    args = parser.parse_args()
    
    process_folder(args.dir)
    print("Redimensionamento offline concluído com sucesso!")

#python resize_datasets.py --dir /projects/F202500001HPCVLABEPICURE/andresousa615/rempe/processed_datasets/processed_datasets/treino