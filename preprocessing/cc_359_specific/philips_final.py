import os
import shutil
import re

# ================= CONFIGURAÇÃO =================
PASTA_ORIGEM = r"E:\Tese\Datasets\Rempe\CC_359\cc_359_GT_Masks\ge"
PASTA_DESTINO = r"E:\Tese\Datasets\Rempe\CC_359\cc_359_GT_Masks\ge_final"
# ================================================

def copiar_e_renomear():
    # 1. Verificar se a pasta de origem existe
    if not os.path.exists(PASTA_ORIGEM):
        print(f"❌ Erro: Pasta de origem não encontrada: {PASTA_ORIGEM}")
        return

    # 2. Criar a pasta de destino (se não existir)
    if not os.path.exists(PASTA_DESTINO):
        os.makedirs(PASTA_DESTINO)
        print(f"📁 Pasta de destino criada: {PASTA_DESTINO}")
    else:
        print(f"📁 Pasta de destino já existe: {PASTA_DESTINO}")

    # 3. Listar apenas os ficheiros que terminam com .nii.gz
    ficheiros = [f for f in os.listdir(PASTA_ORIGEM) if f.endswith(".nii.gz")]
    
    if len(ficheiros) == 0:
        print(f"⚠️ Não foram encontrados ficheiros .nii.gz na origem.")
        return

    print(f"🚀 A iniciar cópia e renomeação de {len(ficheiros)} exames...\n")

    # 4. Criar o padrão Regex
    # ^ : Começa pelo início da string
    # (CC\d{4}_ge_\d+_\d+_[FM]) : Grupo que captura a base do ID até ao sexo (F ou M)
    padrao = re.compile(r"^(CC\d{4}_ge_\d+_\d+_[FM])")

    count = 0
    for nome_antigo in ficheiros:
        caminho_antigo = os.path.join(PASTA_ORIGEM, nome_antigo)
        
        # Procurar o padrão no nome do ficheiro atual
        match = padrao.match(nome_antigo)
        
        if match:
            # Extrair a base limpa (ex: "CC0301_ge_3_53_F")
            base_nome = match.group(1) 
            
            # Construir o nome final desejado
            nome_novo = f"{base_nome}_mask.nii.gz"
            caminho_novo = os.path.join(PASTA_DESTINO, nome_novo)

            try:
                # 5. Copiar o ficheiro preservando metadados
                shutil.copy2(caminho_antigo, caminho_novo)
                print(f"✅ {nome_antigo}  -->  {nome_novo}")
                count += 1
            except Exception as e:
                print(f"❌ Erro ao copiar {nome_antigo}: {e}")
        else:
            print(f"⚠️ Ficheiro ignorado (não segue o padrão CCXXXX_ge...): {nome_antigo}")

    print(f"\n🏁 Terminado! {count} ficheiros prontos na pasta 'ge_final'.")

if __name__ == "__main__":
    copiar_e_renomear()