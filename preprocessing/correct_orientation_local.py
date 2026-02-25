import os
import shutil
import nibabel as nib

# Entrada e Saída (Caminhos separados evitam o [Errno 13])
DATASET_DIR = r"E:\Tese\Datasets\Rempe\dataset_test_complete"
OUTPUT_DIR = r"E:\Tese\Datasets\Rempe\dataset_test_complete_orientation_corrected"

def standardize_exam_orientation(input_base, output_base):
    print(f"A criar dataset Canónico em: {output_base}")
    count_modified = 0
    count_ok = 0
    
    for root_dir, _, files in os.walk(input_base):
        img_path = None
        mask_path = os.path.join(root_dir, "mask_GT.nii.gz")
        
        # Encontrar a imagem original
        for f in ["image.nii.gz", "raw.nii.gz"]:
            if f in files:
                img_path = os.path.join(root_dir, f)
                break
                
        if img_path and os.path.exists(mask_path):
            # 1. Replicar a estrutura de pastas no diretório de output
            rel_path = os.path.relpath(root_dir, input_base)
            target_dir = os.path.join(output_base, rel_path)
            os.makedirs(target_dir, exist_ok=True)
            
            target_img_path = os.path.join(target_dir, os.path.basename(img_path))
            target_mask_path = os.path.join(target_dir, "mask_GT.nii.gz")

            # 2. Carregar matrizes originais (Apenas Leitura - Sem bloqueio do Windows)
            img_nifti = nib.load(img_path)
            mask_nifti = nib.load(mask_path)
            
            current_orientation = nib.aff2axcodes(img_nifti.affine)
            
            if current_orientation == ('R', 'A', 'S'):
                # Se já estiver correto, copiamos os ficheiros diretamente para poupar processamento
                shutil.copy2(img_path, target_img_path)
                shutil.copy2(mask_path, target_mask_path)
                count_ok += 1
                continue 
            
            print(f"A transpor matrizes [{current_orientation} -> RAS]: {os.path.basename(root_dir)}")
            
            # 3. Transformação Matemática para o formato RAS
            canonical_img = nib.as_closest_canonical(img_nifti)
            canonical_mask = nib.as_closest_canonical(mask_nifti)
            
            # 4. Gravar num caminho completamente novo e seguro
            nib.save(canonical_img, target_img_path)
            nib.save(canonical_mask, target_mask_path)
            
            count_modified += 1
                
    print("\n" + "="*50)
    print("NOVO DATASET GERADO COM SUCESSO")
    print("="*50)
    print(f"Exames copiados (já em RAS): {count_ok}")
    print(f"Exames matematicamente corrigidos: {count_modified}")
    print("="*50)

if __name__ == "__main__":
    # Garante que não há processos Python pendentes e corre o script
    standardize_exam_orientation(DATASET_DIR, OUTPUT_DIR)