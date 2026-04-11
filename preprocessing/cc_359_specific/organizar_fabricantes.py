import os
import shutil

# ================= CONFIGURAÇÃO =================
# A pasta onde tens as 359 máscaras todas misturadas
PASTA_ORIGEM = r"E:\Tese\Datasets\Rempe\CC_359\cc_359_GT_Masks\cc_359_masks"

# A pasta "mãe" onde vamos criar as 3 subpastas
PASTA_BASE = r"E:\Tese\Datasets\Rempe\CC_359\cc_359_GT_Masks"
# ================================================

def organizar_por_maquina():
    if not os.path.exists(PASTA_ORIGEM):
        print(f"❌ Erro: A pasta {PASTA_ORIGEM} não existe.")
        return

    # Definir os nomes das máquinas que queremos procurar
    maquinas = ["philips", "siemens", "ge"]
    
    # Dicionário para guardar o caminho das novas pastas e contar quantos ficheiros movemos
    pastas_destino = {}
    contadores = {m: 0 for m in maquinas}
    contadores["desconhecido"] = 0

    # Criar as 3 subpastas (se não existirem)
    for maquina in maquinas:
        caminho = os.path.join(PASTA_BASE, maquina)
        os.makedirs(caminho, exist_ok=True)
        pastas_destino[maquina] = caminho

    print("🚀 A iniciar a separação das máscaras por fabricante...\n")

    # Varrer todos os ficheiros na pasta original
    for nome_ficheiro in os.listdir(PASTA_ORIGEM):
        caminho_completo = os.path.join(PASTA_ORIGEM, nome_ficheiro)
        
        # Ignorar se for uma pasta
        if not os.path.isfile(caminho_completo):
            continue

        nome_minusculas = nome_ficheiro.lower()
        movido = False

        # Verificar a qual fabricante pertence
        for maquina in maquinas:
            if maquina in nome_minusculas:
                destino = os.path.join(pastas_destino[maquina], nome_ficheiro)
                shutil.move(caminho_completo, destino)
                contadores[maquina] += 1
                movido = True
                break # Já encontrámos a máquina, não precisa de procurar mais
        
        if not movido:
            print(f"⚠️ Aviso: O ficheiro '{nome_ficheiro}' não tem a marca no nome.")
            contadores["desconhecido"] += 1

    # Resumo Final
    print("-" * 40)
    print("✅ Separação concluída! Resumo:")
    for maquina in maquinas:
        print(f" 📂 {maquina.capitalize()}: {contadores[maquina]} ficheiros")
    
    if contadores["desconhecido"] > 0:
        print(f" ❓ Desconhecidos/Não movidos: {contadores['desconhecido']} ficheiros")
    print("-" * 40)

if __name__ == "__main__":
    organizar_por_maquina()