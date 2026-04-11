import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_results(csv_path, output_img):
    if not os.path.exists(csv_path):
        print(f"[Erro] O ficheiro CSV não foi encontrado: {csv_path}")
        return

    print(f"A gerar gráficos a partir de: {csv_path}")
    
    # Carregar a matriz de dados principal
    df = pd.read_csv(csv_path)
    
    # Se o treino parou muito cedo (ex: 1 época), não vale a pena fazer gráficos complexos
    if len(df) < 2:
        print("[Aviso] Dados insuficientes para gerar gráficos de progressão (menos de 2 épocas).")
        return

    # --- TRATAMENTO DE DADOS (EMA) ---
    # Parâmetro de suavização: 0.2 significa que o novo valor pesa 20% e o histórico 80%
    SMOOTH_ALPHA = 0.2
    
    # Influenciamos o DataFrame adicionando novas Series (colunas) calculadas 
    # a partir das colunas originais, aplicando o método de média móvel exponencial
    df['train_loss_ema'] = df['train_loss'].ewm(alpha=SMOOTH_ALPHA, adjust=False).mean()
    df['val_loss_ema'] = df['val_loss'].ewm(alpha=SMOOTH_ALPHA, adjust=False).mean()
    df['dsc_ema'] = df['dsc'].ewm(alpha=SMOOTH_ALPHA, adjust=False).mean()
    df['iou_ema'] = df['iou'].ewm(alpha=SMOOTH_ALPHA, adjust=False).mean()
    # ---------------------------------

    # Criar uma figura com 2 subgráficos
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # 1. Gráfico de Loss
    # A) Plotar os dados vetoriais em bruto com baixa opacidade (alpha=0.25)
    ax1.plot(df['epoch'], df['train_loss'], alpha=0.25, color='blue', label='Train Loss (Raw)')
    ax1.plot(df['epoch'], df['val_loss'], alpha=0.25, color='red', label='Val Loss (Raw)')
    # B) Plotar os dados vetoriais suavizados por cima com espessura maior
    ax1.plot(df['epoch'], df['train_loss_ema'], linewidth=2.5, color='blue', label='Train Loss (EMA)')
    ax1.plot(df['epoch'], df['val_loss_ema'], linewidth=2.5, color='red', label='Val Loss (EMA)')
    
    ax1.set_ylabel('Loss (Dice Loss)')
    ax1.set_title('Progressão da Loss por Época')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # 2. Gráfico de Métricas (DSC e IoU)
    # A) Plotar os dados vetoriais em bruto (transparentes)
    ax2.plot(df['epoch'], df['dsc'], alpha=0.25, color='green', label='DSC (Raw)')
    ax2.plot(df['epoch'], df['iou'], alpha=0.25, color='orange', label='IoU (Raw)')
    # B) Plotar os dados vetoriais suavizados (sólidos)
    ax2.plot(df['epoch'], df['dsc_ema'], linewidth=2.5, color='green', label='DSC (EMA)')
    ax2.plot(df['epoch'], df['iou_ema'], linewidth=2.5, color='orange', label='IoU (EMA)')
    
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score (0.0 a 1.0)')
    ax2.set_title('Métricas de Validação')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"✅ Gráfico guardado com sucesso em: {output_img}")

if __name__ == "__main__":
    # ✅ NOVO: Configuração do argparse para o Pipeline Bash
    parser = argparse.ArgumentParser(description="Gerador de gráficos de treino")
    parser.add_argument("--csv_file", type=str, required=True, help="Caminho para o ficheiro CSV de métricas de treino")
    parser.add_argument("--output_img", type=str, required=True, help="Caminho e nome do ficheiro PNG de saída")
    
    args = parser.parse_args()
    
    plot_training_results(args.csv_file, args.output_img)