import re
import pandas as pd
import argparse

def parse_slurm_log(log_path, output_csv):
    # Expressões regulares para capturar os números exatos
    re_train = re.compile(r"Train-loss:\s+([0-9.]+)")
    re_val = re.compile(r"Val-loss:\s+([0-9.]+)")
    re_metrics = re.compile(r"DSC:\s+([0-9.]+)\s+\|\s+IoU:\s+([0-9.]+)")

    # Estrutura de dados para armazenar as épocas
    parsed_data = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "dsc": [],
        "iou": []
    }

    current_train_loss = None
    current_val_loss = None
    epoch_counter = 0

    print(f"A processar o ficheiro: {log_path} ...")
    
    with open(log_path, 'r') as file:
        for line in file:
            # Capturar Train Loss
            match_train = re_train.search(line)
            if match_train:
                current_train_loss = float(match_train.group(1))
            
            # Capturar Val Loss
            match_val = re_val.search(line)
            if match_val:
                current_val_loss = float(match_val.group(1))
                
            # Capturar Métricas (Isto acontece no final do ciclo de validação de cada época)
            match_metrics = re_metrics.search(line)
            if match_metrics:
                dsc = float(match_metrics.group(1))
                iou = float(match_metrics.group(2))
                
                # Quando apanhamos o DSC/IoU, temos a época completa. Guardamos nas listas.
                parsed_data["epoch"].append(epoch_counter)
                parsed_data["train_loss"].append(current_train_loss)
                parsed_data["val_loss"].append(current_val_loss)
                parsed_data["dsc"].append(dsc)
                parsed_data["iou"].append(iou)
                
                epoch_counter += 1

    # Converter o dicionário num DataFrame estruturado e guardar em CSV
    df = pd.DataFrame(parsed_data)
    df.to_csv(output_csv, index=False)
    print(f"Sucesso! Dados extraídos para: {output_csv}")
    print(df.head()) # Imprime as primeiras linhas para confirmar

if __name__ == "__main__":
    # Ajusta os nomes dos ficheiros conforme a tua pasta
    LOG_FILE = "../logs/mednext_173580.err" 
    CSV_OUTPUT = "../logs/training_metrics.csv"
    
    parse_slurm_log(LOG_FILE, CSV_OUTPUT)