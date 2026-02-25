import pandas as pd
import matplotlib.pyplot as plt

def plot_training_results(csv_path, output_img="../logs/training_plot.png"):
    # Carregar a matriz de dados
    df = pd.read_csv(csv_path)
    
    # Criar uma figura com 2 subgráficos (Loss em cima, DSC/IoU em baixo)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # 1. Gráfico de Loss
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='red', linewidth=2)
    ax1.set_ylabel('Loss (Dice Loss)')
    ax1.set_title('Progressão da Loss por Época')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # 2. Gráfico de Métricas (DSC e IoU)
    ax2.plot(df['epoch'], df['dsc'], label='DSC (Dice Score)', color='green', linewidth=2)
    ax2.plot(df['epoch'], df['iou'], label='IoU (Jaccard Index)', color='orange', linewidth=2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score (0.0 a 1.0)')
    ax2.set_title('Métricas de Validação')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"Gráfico guardado com sucesso em: {output_img}")

if __name__ == "__main__":
    CSV_FILE = "../logs/training_metrics.csv"
    plot_training_results(CSV_FILE)