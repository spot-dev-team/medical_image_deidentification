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
# Importações DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

# ================= CONFIGURAÇÃO =================
TEST_CSV = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/data/test.csv"
# ================================================

def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl")
    return local_rank, global_rank, world_size

def resample_to_ras(nifti_img: nib.Nifti1Image) -> nib.Nifti1Image:
    orig_orientation = nib.orientations.io_orientation(nifti_img.affine)
    target_orientation = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(orig_orientation, target_orientation)
    return nifti_img.as_reoriented(transform)

def calculate_hard_metrics(gt_data, pred_data):
    gt_bin = (gt_data > 0).astype(bool)
    pred_bin = (pred_data > 0).astype(bool)
    intersection = np.logical_and(gt_bin, pred_bin).sum()
    gt_sum = gt_bin.sum()
    pred_sum = pred_bin.sum()
    dice = (2.0 * intersection) / (gt_sum + pred_sum + 1e-8)
    iou = intersection / (gt_sum + pred_sum - intersection + 1e-8)
    return dice, iou

# ================= CLASSE DATASET PERSONALIZADA =================
class PhysicalEvalDataset(Dataset):
    def __init__(self, df_chunk):
        """Recebe apenas a "fatia" do DataFrame que pertence a esta GPU"""
        self.df = df_chunk.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx] #pega na linha do csv
        img_path = row['image_path'] #saca o path da imagem
        mask_path = row['mask_path']
        exam_id = os.path.basename(os.path.dirname(img_path))
        
        # 1. Carregar ficheiros físicos e garantir formato RAS
        img_nifti_ras = resample_to_ras(nib.load(img_path))
        gt_nifti_ras = resample_to_ras(nib.load(mask_path))
        
        orig_data = img_nifti_ras.get_fdata(dtype=np.float32)
        gt_data = gt_nifti_ras.get_fdata(dtype=np.float32)
        orig_shape = torch.tensor(orig_data.shape) # Guardar shape original
        
        # 2. Transposição e Tensor [Channel, X, Y, Z]
        img_tensor = torch.tensor(orig_data).unsqueeze(0).unsqueeze(0) # Adiciona batch temporário
        
        # 3. REDIMENSIONAMENTO ESTRUTURAL (Feito pelo CPU Worker)
        img_tensor_resized = F.interpolate(img_tensor, size=(128, 128, 128), mode='trilinear', align_corners=False)
        img_tensor_resized = img_tensor_resized.squeeze(0) # Remove o batch [1, 128, 128, 128]
        
        # 4. NORMALIZAÇÃO MIN-MAX
        tensor_min = img_tensor_resized.min()
        tensor_max = img_tensor_resized.max()
        if tensor_max > 0:
            img_tensor_resized = (img_tensor_resized - tensor_min) / (tensor_max - tensor_min)
            
        gt_tensor = torch.tensor(gt_data, dtype=torch.uint8)
        
        # Devolvemos tudo o que a GPU precisa para inferir e guardar
        return img_tensor_resized, gt_tensor, img_path, exam_id, orig_shape

# ================= FUNÇÃO PRINCIPAL DE INFERÊNCIA =================
def run_physical_evaluation(weights_path, out_dir, metrics_csv, metrics_txt):
    local_rank, global_rank, world_size = ddp_setup()
    device = torch.device(f"cuda:{local_rank}")
    
    if global_rank == 0:
        print("A iniciar Inferência Física Distribuída (Híbrida com DataLoader)...")
        os.makedirs(out_dir, exist_ok=True)
    dist.barrier() # Sincroniza todas as GPUs

    # 1. Carregar Modelo - alterar se for outro modelo
    model = Mednext() 
    #UNet3D() #Mednext() 
    
    # Carregar o dicionário de pesos do disco
    state_dict = torch.load(weights_path, map_location=device)
    
    # Injetar os pesos na arquitetura
    model.load_state_dict(state_dict)
    
    # Passar para a GPU e meter em modo de avaliação
    model.to(device)
    model.eval()


    torch.backends.cudnn.enabled = False

    # 2. Divisão Manual do Dataset (Sem perda de dados)
    df_test = pd.read_csv(TEST_CSV)
    chunks = np.array_split(df_test, world_size)
    my_chunk = chunks[global_rank]
    
    if global_rank == 0:
        print(f"Total de exames a processar: {len(df_test)}")

    # 3. Inicializar o DataLoader Local (I/O Rápido)
    dataset = PhysicalEvalDataset(my_chunk)
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, # 4 workers por GPU é o "sweet spot" para inferência
        pin_memory=True
    )

    exam_results = []
    loop = tqdm(dataloader, disable=(global_rank != 0), desc="Progresso Global (Rank 0)")

    with torch.no_grad():
        for batch in loop:
            # Desempacotar os dados fornecidos pelo DataLoader
            img_tensor_resized, gt_tensor, img_path_tuple, exam_id_tuple, orig_shape = batch
            
            # Como batch_size=1, extraímos os valores das tuplas/tensores
            img_path = img_path_tuple[0]
            exam_id = exam_id_tuple[0]
            spatial_shape = tuple(orig_shape[0].tolist())
            
            gt_data = gt_tensor.squeeze().numpy()
            img_tensor_resized = img_tensor_resized.to(device)
            
            # INFERÊNCIA
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits_resized = model(img_tensor_resized)
            
            # RESTAURAR DIMENSÕES
            logits_original_size = F.interpolate(logits_resized.float(), size=spatial_shape, mode='trilinear', align_corners=False)
            probs = torch.sigmoid(logits_original_size)
            pred_tensor = (probs > 0.5).float()
            pred_data = pred_tensor.squeeze().cpu().numpy()
            
            # LER METADADOS E DADOS ORIGINAIS COMPLETOS PARA O DEFACING
            header_nifti = resample_to_ras(nib.load(img_path))
            orig_data_full = header_nifti.get_fdata(dtype=np.float32)
            
            # MATEMÁTICA DO DEFACING (A tua lógica médica)
            # Onde a máscara previu face (> 0.5), apagamos (colocamos a 0)
            defaced_data = np.where(pred_data > 0.5, 0, orig_data_full)
            
            # EXPORTAÇÃO 1: A Máscara (para auditoria)
            pred_out_path = os.path.join(out_dir, f"{exam_id}_pred_mask.nii.gz")
            nib.save(nib.Nifti1Image(pred_data.astype(np.uint8), header_nifti.affine, header_nifti.header), pred_out_path)
            
            # EXPORTAÇÃO 2: A Máscara Ground Truth
            gt_out_path = os.path.join(out_dir, f"{exam_id}_gt_mask.nii.gz")
            nib.save(nib.Nifti1Image(gt_data.astype(np.uint8), header_nifti.affine, header_nifti.header), gt_out_path)
            
            # EXPORTAÇÃO 3: O Exame Anonimizado (Para a Fase 4)
            anon_out_path = os.path.join(out_dir, f"{exam_id}_anon.nii.gz")
            nib.save(nib.Nifti1Image(defaced_data.astype(orig_data_full.dtype), header_nifti.affine, header_nifti.header), anon_out_path)

            
            # MÉTRICAS
            dsc, iou = calculate_hard_metrics(gt_data, pred_data)
            exam_results.append({
                "Exam_ID": exam_id,
                "Dice_Score": round(dsc, 4),
                "IoU": round(iou, 4)
            })

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
            f.write("RESULTADOS FINAIS - HARD DICE (DDP + DataLoader)\n")
            f.write("="*55 + "\n")
            f.write(f"Total Exames Avaliados: {len(df_results)}\n")
            f.write(f"Average DSC (Dice Score): {avg_dsc:.4f}\n")
            f.write(f"Average IoU (Jaccard):    {avg_iou:.4f}\n")
            f.write("="*55 + "\n")
            
        print("\n" + "="*55)
        print("RESULTADOS FINAIS - AVALIAÇÃO FÍSICA")
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