import torch
import pandas as pd
from tqdm import tqdm

from model import ConvNext, UNet3D, Mednext
from dataset import get_loaders 
from utils.validation import segmentation_validation

def run_test_evaluation(csv_path, weights_path, output_path):
    print(f"A iniciar avaliação de teste para a Spot.")
    print(f"Ficheiro de dados: {csv_path}")
    print(f"Modelo carregado: {weights_path}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Preparar Modelo
    model = torch.load(weights_path, map_location=device)
    model.to(device)
    model.eval() 
    torch.backends.cudnn.enabled = False

    # 2. Estrutura de Dados: DataLoader de Teste
    # LER O CSV PRIMEIRO: Transforma o ficheiro físico num DataFrame
    df_test = pd.read_csv(csv_path)

    # Passar o DataFrame para a função usando os argumentos corretos (no plural)
    # O num_workers foi removido porque já está fixo no teu dataset.py
    _, test_loader = get_loaders(
        train_paths=df_test, 
        val_paths=df_test,
        batch_size=1
    )

    total_dsc = 0.0
    total_iou = 0.0
    num_batches = len(test_loader)

    print(f"Total de exames NIfTI a processar: {num_batches}")

    # 3. Ciclo de Inferência e Avaliação
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="A testar volumes 3D"):
            images = batch["image"].float().to(device)
            targets = batch["mask"].float().to(device)
            
            # Forward pass (Tensores de entrada -> Logits brutos)
            logits = model(images)
            
            # Passar os logits diretamente! A função segmentation_validation 
            # já contém a instrução `torch.sigmoid(predictions)`
            scores = segmentation_validation(logits, targets)
            
            total_dsc += scores["dsc"].item()
            total_iou += scores["iou"].item()

    # 4. Agregação de Resultados Finais
    avg_dsc = total_dsc / num_batches
    avg_iou = total_iou / num_batches
    with open(output_path, 'w') as f:
        f.write("RESULTADOS FINAIS DE TESTE (CONJUNTO CEGO)\n")
        f.write("="*50 + "\n")
        f.write(f"Average DSC (Dice Score): {avg_dsc:.4f}\n")
        f.write(f"Average IoU (Jaccard):    {avg_iou:.4f}\n")
        f.write("="*50 + "\n")
        f.close()
        
    print("\n" + "="*50)
    print("RESULTADOS FINAIS DE TESTE (CONJUNTO CEGO)")
    print("="*50)
    print(f"Average DSC (Dice Score): {avg_dsc:.4f}")
    print(f"Average IoU (Jaccard):    {avg_iou:.4f}")
    print("="*50)
    
if __name__ == "__main__":
    TEST_CSV = "/home/andresousa615/rempe/mede_code/mede/data/test.csv"
    BEST_WEIGHTS = "/home/andresousa615/rempe/mede_code/results/train_mednext_0.0005_mednext_aurora_16gb_vram/mednext_0.0005_mednext_aurora_16gb_vram"
    OUTPUT = "/home/andresousa615/rempe/mede_code/logs/test_evaluation.txt"
    
    run_test_evaluation(TEST_CSV, BEST_WEIGHTS,OUTPUT)

    