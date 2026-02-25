import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

# ================= CONFIGURAÇÃO =================
BASE_DIR = r"E:\Tese\Datasets\Rempe"
DIR_ORIGINAL = os.path.join(BASE_DIR, "processed_datasets\\treino")
DIR_CANONICAL = os.path.join(BASE_DIR, "processed_datasets_canonical\\treino")
# ================================================

# MARGEM DE SEGURANÇA: Número de voxels a expandir a partir da fronteira do cérebro.
# 2 iterações costumam representar cerca de 2 a 3 mm de segurança física extra.
SAFETY_MARGIN_VOXELS = 3


def remove_brain_intersection():
    print("A iniciar a limpeza de interseção das máscaras da Spot...")
    count_processed = 0
    count_errors = 0

    # Iterar pelos exames na pasta canónica
    for exam_name in os.listdir(DIR_CANONICAL):
        count_processed += 1
        print(f"\nProcessando exame: {exam_name}, ({count_processed}/{len(os.listdir(DIR_CANONICAL))})")
        path_canonical_exam = os.path.join(DIR_CANONICAL, exam_name)
        
        # Ignorar ficheiros soltos, iterar apenas sobre pastas
        if not os.path.isdir(path_canonical_exam):
            continue

        # Definir caminhos de entrada e saída
        path_gt_face = os.path.join(path_canonical_exam, "mask_GT.nii.gz")
        path_brain_original = os.path.join(DIR_ORIGINAL, exam_name, "mask.nii.gz")
        path_output = os.path.join(path_canonical_exam, "mask_GT_clean_1.nii.gz")

        if os.path.exists(path_output):
            print(f"[Aviso] Ficheiro de saída já existe para {exam_name}, vou apagar.")
            os.remove(path_output)

        # Verificar se ambos os ficheiros existem antes de tentar ler
        if not os.path.exists(path_gt_face) or not os.path.exists(path_brain_original):
            print(f"[Aviso] Ficheiros em falta para o exame: {exam_name}")
            continue

        try:
            # 1. Carregar a máscara da face (Já convertida para RAS)
            nifti_gt_face = nib.load(path_gt_face)
            data_gt_face = nifti_gt_face.get_fdata().astype(np.uint8)

            # 2. Carregar a máscara do cérebro e forçar o alinhamento para RAS
            # Isto garante que o voxel [X, Y, Z] de uma matriz corresponde ao mesmo espaço físico na outra
            nifti_brain_raw = nib.load(path_brain_original)
            nifti_brain_ras = nib.as_closest_canonical(nifti_brain_raw)
            data_brain = nifti_brain_ras.get_fdata().astype(np.uint8)

            # Segurança: Validar que as dimensões bateram certo após a transposição
            if data_gt_face.shape != data_brain.shape:
                print(f"[Erro] Dimensões incompatíveis em {exam_name}: Face {data_gt_face.shape} vs Cérebro {data_brain.shape}")
                count_errors += 1
                continue

            data_brain_expanded = binary_dilation(data_brain, iterations=SAFETY_MARGIN_VOXELS).astype(np.uint8)

            # 3. Operação Matricial: Subtrair a interseção
            # Copiamos a máscara da face e colocamos a '0' todos os píxeis que também estão ativos na máscara do cérebro
            data_clean_gt = np.copy(data_gt_face)
            data_clean_gt[(data_gt_face > 0) & (data_brain_expanded > 0)] = 0

            # 4. Guardar o novo NIfTI utilizando a geometria afim da face (RAS)
            nifti_clean = nib.Nifti1Image(data_clean_gt, nifti_gt_face.affine, nifti_gt_face.header)
            nib.save(nifti_clean, path_output)
            
            # Cálculo de volume afetado para feedback no terminal
            voxels_removidos = np.sum(data_gt_face) - np.sum(data_clean_gt)
            print(f"[{count_processed + 1}] {exam_name}: Limpeza concluída (Removidos {voxels_removidos} voxels de cérebro).")
           
        except Exception as e:
            print(f"[Erro] Falha ao processar {exam_name}: {e}")
            

    print("-" * 55)
    print(f"Processo concluído: {count_processed} máscaras refinadas.")
    if count_errors > 0:
        print(f"Ocorreram erros em {count_errors} exames.")

if __name__ == "__main__":
    remove_brain_intersection()