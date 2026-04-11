"""
Este código é uma alternativa para calcular o dicescore sobre os dados 
de teste para garantir que não há falhas (Corrige alinhamento e intensidade).
""""""
Código de Avaliação Hard Dice (Com Exportação de Métricas por Exame)
"""

"""
================================================================================
AVALIAÇÃO CLÍNICA / MUNDO REAL (FICHEIRO OFICIAL DE TESTE)
================================================================================
Objetivo: Simular o caso de uso real da Spot num ambiente clínico. 
Este script carrega os ficheiros NIfTI originais, executa a inferência, 
e projeta as máscaras geradas de volta para o tamanho físico original do paciente.

Características:
- Guarda as predições físicas em disco (.nii.gz) para visualização médica.
- Calcula métricas 'Hard Dice' exatas baseadas na anatomia real.
- Gera um relatório CSV detalhado (exame a exame) para auditar falhas e sucessos.

-> É ESTE O FICHEIRO A USAR PARA RESULTADOS FINAIS, PAPERS E RELATÓRIOS.
================================================================================
"""


import os
import torch
import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch.nn.functional as F

# ================= CONFIGURAÇÃO =================
TEST_CSV = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/data/test.csv"
BEST_WEIGHTS = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/results/train_mednext_0.0005_mednext_aurora_16gb_vram/mednext_0.0005_mednext_aurora_16gb_vram"
INFERENCE_OUT_DIR = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/results/inference_test_masks"
METRICS_TXT = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/logs/real_test_metrics.txt"
# NOVO: Ficheiro CSV para análise individual
INDIVIDUAL_METRICS_CSV = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/logs/per_exam_metrics.csv"
# ================================================

def resample_to_ras(nifti_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Reorienta a imagem NIfTI para RAS (Direita, Anterior, Superior).
    """
    orig_orientation = nib.orientations.io_orientation(nifti_img.affine)
    target_orientation = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(orig_orientation, target_orientation)
    return nifti_img.as_reoriented(transform)

def calculate_hard_metrics(gt_data, pred_data):
    """
    Calcula o Hard Dice e IoU em matrizes booleanas (0 e 1).
    """
    gt_bin = (gt_data > 0).astype(bool)
    pred_bin = (pred_data > 0).astype(bool)
    
    intersection = np.logical_and(gt_bin, pred_bin).sum()
    gt_sum = gt_bin.sum()
    pred_sum = pred_bin.sum()
    
    dice = (2.0 * intersection) / (gt_sum + pred_sum + 1e-8)
    iou = intersection / (gt_sum + pred_sum - intersection + 1e-8)
    return dice, iou

def run_physical_evaluation():
    print("A iniciar Inferência Física e Avaliação Hard Dice para a Spot...")
    
    os.makedirs(INFERENCE_OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Carregar Modelo
    model = torch.load(BEST_WEIGHTS, map_location=device)
    model.to(device)
    model.eval()
    torch.backends.cudnn.enabled = False

    df_test = pd.read_csv(TEST_CSV)
    
    total_dsc = 0.0
    total_iou = 0.0
    valid_exams = 0
    
    # NOVO: Lista para guardar os resultados individuais
    exam_results = []
    
    print(f"Total de exames a processar: {len(df_test)}")

    with torch.no_grad():
        for index, row in tqdm(df_test.iterrows(), total=len(df_test)):
            img_path = row['image_path']
            mask_path = row['mask_path']
            exam_id = os.path.basename(os.path.dirname(img_path))
            
            # 2. Carregar ficheiros físicos e garantir formato RAS
            img_nifti_raw = nib.load(img_path)
            gt_nifti_raw = nib.load(mask_path)
            
            img_nifti_ras = resample_to_ras(img_nifti_raw)
            gt_nifti_ras = resample_to_ras(gt_nifti_raw)
            
            orig_data = img_nifti_ras.get_fdata().copy()
            gt_data = gt_nifti_ras.get_fdata().copy()
            
            # 3. Transposição e Tensor [Batch, Channel, X, Y, Z]
            img_tensor = torch.tensor(orig_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # 4. REDIMENSIONAMENTO ESTRUTURAL
            target_inference_size = (128, 128, 128) 
            img_tensor_resized = F.interpolate(img_tensor, size=target_inference_size, mode='trilinear', align_corners=False)
            
            # 5. NORMALIZAÇÃO MIN-MAX
            tensor_min = img_tensor_resized.min()
            tensor_max = img_tensor_resized.max()
            if tensor_max > 0:
                img_tensor_resized = (img_tensor_resized - tensor_min) / (tensor_max - tensor_min)
            
            # 6. INFERÊNCIA
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits_resized = model(img_tensor_resized)
            
            # 7. RESTAURAR DIMENSÕES
            original_spatial_shape = orig_data.shape
            logits_original_size = F.interpolate(logits_resized.float(), size=original_spatial_shape, mode='trilinear', align_corners=False)
            
            probs = torch.sigmoid(logits_original_size)
            pred_tensor = (probs > 0.5).float()
            
            pred_data = pred_tensor.squeeze().cpu().numpy()
            
            # 8. EXPORTAÇÃO PARALELA (GT vs PRED)
            pred_out_path = os.path.join(INFERENCE_OUT_DIR, f"{exam_id}_pred_mask.nii.gz")
            pred_nifti = nib.Nifti1Image(pred_data.astype(np.uint8), img_nifti_ras.affine, img_nifti_ras.header)
            nib.save(pred_nifti, pred_out_path)
            
            gt_out_path = os.path.join(INFERENCE_OUT_DIR, f"{exam_id}_gt_mask.nii.gz")
            gt_nifti_export = nib.Nifti1Image(gt_data.astype(np.uint8), img_nifti_ras.affine, img_nifti_ras.header)
            nib.save(gt_nifti_export, gt_out_path)
            
            # 9. Cálculo de Métricas Matemáticas Diretas
            dsc, iou = calculate_hard_metrics(gt_data, pred_data)
            
            total_dsc += dsc
            total_iou += iou
            valid_exams += 1
            
            # NOVO: Guardar o registo individual
            exam_results.append({
                "Exam_ID": exam_id,
                "Dice_Score": round(dsc, 4),
                "IoU": round(iou, 4)
            })

            del img_tensor, img_tensor_resized, logits_resized, logits_original_size, probs, pred_tensor
            torch.cuda.empty_cache()

    avg_dsc = total_dsc / valid_exams
    avg_iou = total_iou / valid_exams
    
    # NOVO: Converter os resultados individuais para um DataFrame e guardar em CSV
    df_results = pd.DataFrame(exam_results)
    df_results.to_csv(INDIVIDUAL_METRICS_CSV, index=False)
    
    with open(METRICS_TXT, 'w') as f:
        f.write("RESULTADOS FINAIS - HARD DICE (VOLUMES FÍSICOS NIFTI)\n")
        f.write("="*55 + "\n")
        f.write(f"Average DSC (Dice Score): {avg_dsc:.4f}\n")
        f.write(f"Average IoU (Jaccard):    {avg_iou:.4f}\n")
        f.write("="*55 + "\n")
        
    print("\n" + "="*55)
    print("RESULTADOS FINAIS - HARD DICE (VOLUMES FÍSICOS NIFTI)")
    print("="*55)
    print(f"Average DSC (Dice Score): {avg_dsc:.4f}")
    print(f"Average IoU (Jaccard):    {avg_iou:.4f}")
    print(f"\n✅ Métricas individuais exportadas para: {INDIVIDUAL_METRICS_CSV}")
    print("="*55)

if __name__ == "__main__":
    run_physical_evaluation()