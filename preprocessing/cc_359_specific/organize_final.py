import os
import shutil

# ================= CONFIGURAÇÃO =================
# Pasta final onde estão as máscaras agora (e onde vão ficar as subpastas)
PASTA_FINAL = r"E:\Tese\Datasets\Rempe\CC_359\cc_359_final"

# Pasta raiz onde estão as subpastas com os originais (ge, siemens, philips)
PASTA_ORIGINAIS = r"E:\Tese\Datasets\Rempe\Original\Original"
MARCAS = ["philips", "ge", "siemens"]
# ================================================

def organizar_dataset_final():
    if not os.path.exists(PASTA_FINAL):
        print(f"❌ Pasta de máscaras não encontrada: {PASTA_FINAL}")
        return

    # Listar todas as máscaras soltas na pasta
    mascaras = [f for f in os.listdir(PASTA_FINAL) if f.endswith("_mask.nii.gz")]
    print(f"🚀 Encontradas {len(mascaras)} máscaras prontas para emparelhar...\n")

    count = 0
    for mascara_nome in mascaras:
        # Extrair o ID do exame (tira o "_mask.nii.gz")
        id_exame = mascara_nome.replace("_mask.nii.gz", "")
        
        # O nome que o original deve ter
        original_nome = f"{id_exame}.nii.gz"
        
        # Procurar ativamente o original nas três pastas possíveis
        caminho_original = None
        for marca in MARCAS:
            tentativa = os.path.join(PASTA_ORIGINAIS, marca, original_nome)
            if os.path.exists(tentativa):
                caminho_original = tentativa
                break
                
        if caminho_original is None:
            print(f"⚠️ Aviso: Original não encontrado para a máscara {mascara_nome}. A saltar.")
            continue
            
        # Criar a nova subpasta para este exame específico
        pasta_exame = os.path.join(PASTA_FINAL, id_exame)
        if not os.path.exists(pasta_exame):
            os.makedirs(pasta_exame)
            
        # Definir os caminhos de destino dentro da nova subpasta
        destino_mascara = os.path.join(pasta_exame, mascara_nome)
        destino_original = os.path.join(pasta_exame, original_nome)
        
        try:
            # 1. MOVER a máscara para dentro da subpasta (para limpar a raiz)
            caminho_mascara_atual = os.path.join(PASTA_FINAL, mascara_nome)
            shutil.move(caminho_mascara_atual, destino_mascara)
            
            # 2. COPIAR o original para não o apagarmos da source
            shutil.copy2(caminho_original, destino_original)
            
            print(f"✅ {id_exame} -> Emparelhado com sucesso!")
            count += 1
            
        except Exception as e:
            print(f"❌ Erro ao organizar o exame {id_exame}: {e}")

    print(f"\n🏁 Organização concluída! {count} exames emparelhados nas suas subpastas.")

if __name__ == "__main__":
    organizar_dataset_final()