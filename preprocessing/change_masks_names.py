# este código serviu para passar o nome dos ficheiros de máscara de "mask_GT_clean_3.nii.gz" para "mask_GT.nii.gz" e o "mask_GT.nii.gz" para "mask_GT_original.nii.gz", para manter a consistência com o que é carregado no QA Reviewer. Não tem outra função.

import os

# ================= CONFIGURAÇÃO =================
BASE_DIR = r"E:\Tese\Datasets\Rempe"
SUB_DIRS = ["dataset_test_complete_orientation_corrected"]
# ================================================

def organizar_mascaras_treino():
    print("A iniciar a organização de ficheiros para o novo treino...")
    
    count_renamed_3 = 0
    count_renamed_1 = 0
    count_skipped = 0

    for sub in SUB_DIRS:
        target_dir = os.path.join(BASE_DIR, sub)
        
        if not os.path.exists(target_dir):
            print(f"[Aviso] Pasta não encontrada: {target_dir}")
            continue
            
        print(f"\nA processar a pasta: {sub.upper()}")
        
        for exam_name in os.listdir(target_dir):
            exam_path = os.path.join(target_dir, exam_name)
            
            if not os.path.isdir(exam_path):
                continue
                
            path_gt = os.path.join(exam_path, "mask_GT.nii.gz")
            path_gt_original = os.path.join(exam_path, "mask_GT_original.nii.gz")
            path_clean_1 = os.path.join(exam_path, "mask_GT_clean_1.nii.gz")
            path_clean_3 = os.path.join(exam_path, "mask_GT_clean_3.nii.gz")
            
            # Lógica 1: Prioridade Máxima (clean_3)
            if os.path.exists(path_clean_3):
                # Salvaguardar o original se ainda não foi feito
                if os.path.exists(path_gt) and not os.path.exists(path_gt_original):
                    os.rename(path_gt, path_gt_original)
                elif os.path.exists(path_gt) and os.path.exists(path_gt_original):
                    os.remove(path_gt) # Limpa o caminho para o novo ficheiro
                    
                os.rename(path_clean_3, path_gt)
                count_renamed_3 += 1
                
            # Lógica 2: Prioridade Secundária (clean_1)
            elif os.path.exists(path_clean_1):
                if os.path.exists(path_gt) and not os.path.exists(path_gt_original):
                    os.rename(path_gt, path_gt_original)
                elif os.path.exists(path_gt) and os.path.exists(path_gt_original):
                    os.remove(path_gt)
                    
                os.rename(path_clean_1, path_gt)
                count_renamed_1 += 1
                
            # Lógica 3: Passar à frente
            else:
                count_skipped += 1

    # Relatório Final
    print("\n" + "=" * 55)
    print("RESUMO DA OPERAÇÃO DE ORGANIZAÇÃO")
    print("=" * 55)
    print(f"Exames atualizados com clean_3: {count_renamed_3}")
    print(f"Exames atualizados com clean_1: {count_renamed_1}")
    print(f"Exames ignorados (sem máscaras limpas): {count_skipped}")
    print("=" * 55)

if __name__ == "__main__":
    organizar_mascaras_treino()