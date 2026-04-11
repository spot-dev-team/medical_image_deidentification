import os
import argparse
import torch
import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import ConvNext, UNet3D, Mednext

# FASE 1: Importar exatamente o que usaste no script offline
import torchio as tio
import torchmetrics.functional as f_metrics

import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

# ================= CONFIGURAÇÃO =================
TEST_CSV = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/data/test_noise_cleanse_v2.csv"
# ================================================

def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl")
    return local_rank, global_rank, world_size


# O DataLoader agora apenas entrega os caminhos. 
# O TorchIO vai fazer a magia física e segura dentro do loop de inferência para evitar erros de batching de diferentes tamanhos.
class PhysicalEvalDataset(Dataset):
    def __init__(self, df_chunk):
        self.df = df_chunk.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        exam_id = os.path.basename(os.path.dirname(row['image_path']))
        return row['image_path'], row['mask_path'], exam_id


# ================= FUNÇÃO PRINCIPAL DE INFERÊNCIA =================
def run_physical_evaluation(weights_path, out_dir, metrics_csv, metrics_txt):
    local_rank, global_rank, world_size = ddp_setup()
    device = torch.device(f"cuda:{local_rank}")
    
    if global_rank == 0:
        print("A iniciar Inferência Física Distribuída (Geometria Alinhada com Treino)...")
        os.makedirs(out_dir, exist_ok=True)
    dist.barrier()

    model = Mednext() 
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    torch.backends.cudnn.enabled = False

    df_test = pd.read_csv(TEST_CSV)
    df_test["image_path"] = df_test["image_path"].str.replace(".nii.gz", ".nii", regex=False)
    if "mask_path" in df_test.columns:
        df_test["mask_path"] = df_test["mask_path"].str.replace(".nii.gz", ".nii", regex=False)

    chunks = np.array_split(df_test, world_size)
    my_chunk = chunks[global_rank]
    
    if global_rank == 0:
        print(f"Total de exames a processar: {len(df_test)}")

    dataset = PhysicalEvalDataset(my_chunk)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    exam_results = []
    loop = tqdm(dataloader, disable=(global_rank != 0), desc="Progresso Global (Rank 0)")

    with torch.no_grad():
        for batch in loop:
            img_path = batch[0][0]
            mask_path = batch[1][0]
            exam_id = batch[2][0]
            
            # =========================================================================
            # PASSO 1: REPLICAR O TEU CÓDIGO OFFLINE EXATAMENTE (TorchIO Pipeline)
            # =========================================================================
            subject = tio.Subject(
                image=tio.ScalarImage(img_path),
                mask=tio.LabelMap(mask_path)
            )
            
            # 1. ToCanonical (Igual ao offline)
            subject = tio.ToCanonical()(subject)
            
            # Guardamos os dados originais já alinhados em RAS para o Defacing
            orig_data_ras = subject['image'].data.squeeze(0).numpy().copy()
            orig_gt_ras = subject['mask'].data.squeeze(0).numpy().copy()
            
            # Guardamos a geometria física exata para conseguirmos reconstruir depois!
            orig_shape_ras = subject['image'].shape[1:] 
            orig_affine_ras = subject['image'].affine.copy()
            
            # 2. Resize para 128³ (Igual ao offline - Mantém a proporção real)
            subject_128 = tio.Resize((128, 128, 128))(subject)
            img_128 = subject_128['image'].data.squeeze(0)
            gt_128 = subject_128['mask'].data.squeeze(0)
            affine_128 = subject_128['image'].affine.copy()
            
            # 3. Normalização
            tensor_min = img_128.min()
            tensor_max = img_128.max()
            if tensor_max > 0:
                img_128 = (img_128 - tensor_min) / (tensor_max - tensor_min)
            
            # Enviar para GPU
            img_tensor = img_128.unsqueeze(0).unsqueeze(0).to(device)
            gt_tensor = gt_128.unsqueeze(0).unsqueeze(0).to(device)
            
            # =========================================================================
            # PASSO 2: INFERÊNCIA E MÉTRICAS JUSTAS (A 128³)
            # =========================================================================
            logits = model(img_tensor.float())
            predictions = torch.sigmoid(logits)
            
            # O Dice é calculado a 128³ com Soft Dice (espelhado do validation.py)
            dsc = f_metrics.dice(predictions.float(), gt_tensor.int(), ignore_index=0)
            iou = f_metrics.jaccard_index(predictions.float(), gt_tensor.int(), num_classes=2, ignore_index=0)
            
            exam_results.append({
                "Exam_ID": exam_id,
                "Dice_Score": round(dsc.item(), 4),
                "IoU": round(iou.item(), 4)
            })
            
            # =========================================================================
            # PASSO 3: RESTAURAR O TAMANHO FÍSICO COM TORCHIO (Fim das distorções)
            # =========================================================================
            probs_cpu = predictions.cpu().squeeze(0) # Tira o batch
            
            # Colocamos a previsão do modelo "dentro" do ecossistema TorchIO a 128³
            pred_subject = tio.Subject(
                pred=tio.ScalarImage(tensor=probs_cpu, affine=affine_128)
            )
            
            # O TorchIO vai ler as propriedades geométricas originais que guardámos
            # e vai reverter a imagem perfeitamente para o tamanho e rácio original!
            pred_subject_restored = tio.Resize(orig_shape_ras)(pred_subject)
            
            # Máscara final a tamanho gigante (Binarizada a 0.5)
            pred_mask_orig = (pred_subject_restored['pred'].data.squeeze(0).numpy() > 0.5).astype(np.uint8)
            
            # MATEMÁTICA DO DEFACING (Na resolução gigante)
            defaced_data = np.where(pred_mask_orig > 0, 0, orig_data_ras)
            
            # EXPORTAÇÕES NIFTI (Com o Affine original puro)
            pred_out_path = os.path.join(out_dir, f"{exam_id}_pred_mask.nii")
            nib.save(nib.Nifti1Image(pred_mask_orig, orig_affine_ras), pred_out_path)
            
            gt_out_path = os.path.join(out_dir, f"{exam_id}_gt_mask.nii")
            nib.save(nib.Nifti1Image(orig_gt_ras.astype(np.uint8), orig_affine_ras), gt_out_path)
            
            anon_out_path = os.path.join(out_dir, f"{exam_id}_anon.nii")
            nib.save(nib.Nifti1Image(defaced_data.astype(orig_data_ras.dtype), orig_affine_ras), anon_out_path)

    # 4. RECOLHA DE DADOS NO MESTRE GLOBAL
    gathered_results = [None for _ in range(world_size)]
    dist.gather_object(exam_results, gathered_results if global_rank == 0 else None, dst=0)

    # 5. CÁLCULO FINAL E RELATÓRIO
    if global_rank == 0:
        all_exam_results = [item for sublist in gathered_results for item in sublist]
        df_results = pd.DataFrame(all_exam_results)
        df_results.to_csv(metrics_csv, index=False)
        
        avg_dsc = df_results["Dice_Score"].mean()
        avg_iou = df_results["IoU"].mean()
        
        with open(metrics_txt, 'w') as f:
            f.write("RESULTADOS FINAIS - SOFT DICE (Geometria TorchIO Alinhada)\n")
            f.write("="*55 + "\n")
            f.write(f"Total Exames Avaliados: {len(df_results)}\n")
            f.write(f"Average DSC (Dice Score): {avg_dsc:.4f}\n")
            f.write(f"Average IoU (Jaccard):    {avg_iou:.4f}\n")
            f.write("="*55 + "\n")
            
        print("\n" + "="*55)
        print("RESULTADOS FINAIS - AVALIAÇÃO FÍSICA ALINHADA")
        print("="*55)
        print(f"Total Exames Avaliados: {len(df_results)}")
        print(f"Average DSC (Dice Score): {avg_dsc:.4f}")
        print(f"Average IoU (Jaccard):    {avg_iou:.4f}")
        print(f"\n✅ Relatório de auditoria guardado em: {metrics_csv}")
        print("="*55)

    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Caminho para o modelo treinado")
    parser.add_argument("--out_dir", type=str, required=True, help="Pasta de saída das máscaras NIfTI")
    parser.add_argument("--metrics_csv", type=str, required=True, help="Ficheiro CSV final")
    parser.add_argument("--metrics_txt", type=str, required=True, help="Ficheiro TXT final")
    args = parser.parse_args()

    run_physical_evaluation(args.weights, args.out_dir, args.metrics_csv, args.metrics_txt)