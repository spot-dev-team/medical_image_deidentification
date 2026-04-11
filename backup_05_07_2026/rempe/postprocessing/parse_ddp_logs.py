import re
import pandas as pd

def parse_slurm_log(log_path, output_csv):
    # Expressões regulares atualizadas para refletir as variáveis Globais do DDP
    re_epoch = re.compile(r"Now training epoch (\d+)")
    re_train = re.compile(r"Train-loss:\s+([0-9.]+)")
    re_val = re.compile(r"Val-loss Global:\s+([0-9.]+)")
    re_metrics = re.compile(r"DSC Global:\s+([0-9.]+)\s+\|\s+IoU Global:\s+([0-9.]+)")

    # Estrutura de dados para armazenar as épocas
    parsed_data = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "dsc": [],
        "iou": []
    }

    current_epoch = None
    current_train_loss = None
    current_val_loss = None

    print(f"A processar o ficheiro: {log_path} ...")
    
    # Usamos errors='replace' para que caracteres corrompidos 
    # sejam substituídos por um sinal de interrogação em vez de dar erro.
    with open(log_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            # 1. Capturar o número exato da Época
            match_epoch = re_epoch.search(line)
            if match_epoch:
                current_epoch = int(match_epoch.group(1))
                
            # 2. Capturar Train Loss
            match_train = re_train.search(line)
            if match_train:
                current_train_loss = float(match_train.group(1))
            
            # 3. Capturar Val Loss (Global)
            match_val = re_val.search(line)
            if match_val:
                current_val_loss = float(match_val.group(1))
                
            # 4. Capturar Métricas (Global)
            match_metrics = re_metrics.search(line)
            if match_metrics:
                dsc = float(match_metrics.group(1))
                iou = float(match_metrics.group(2))
                
                # Quando apanhamos o DSC/IoU, o ciclo de log dessa época está fechado. Guardamos!
                if current_epoch is not None:
                    parsed_data["epoch"].append(current_epoch)
                    parsed_data["train_loss"].append(current_train_loss)
                    parsed_data["val_loss"].append(current_val_loss)
                    parsed_data["dsc"].append(dsc)
                    parsed_data["iou"].append(iou)

    # Converter o dicionário num DataFrame estruturado e guardar em CSV
    df = pd.DataFrame(parsed_data)
    df.to_csv(output_csv, index=False)
    print(f"Sucesso! Dados extraídos para: {output_csv}")
    print("\nResumo das primeiras linhas:")
    print(df.head())

if __name__ == "__main__":
    # Ajusta os nomes dos ficheiros conforme a tua pasta
    LOG_FILE = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/logs/rempe_ddp_1071218.err"
    CSV_OUTPUT = "/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/logs/training_metrics_ddp_1071218.csv"
    
    parse_slurm_log(LOG_FILE, CSV_OUTPUT)