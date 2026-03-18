import os
import face_recognition

# ================= CONFIGURAÇÃO =================
DIR_PARES = r"/home/andresousa615/rempe/mede_code/after_training/pares_exames_non_def_and_def"
TOLERANCE = 0.6 
# ================================================

def extrair_id_paciente(nome_pasta):
    """
    Manipula a string do nome da pasta para devolver um identificador único de paciente.
    """
    if nome_pasta.startswith("infant_"):
        # Os infant são únicos, o ID é o próprio nome da pasta
        return nome_pasta
    elif "__" in nome_pasta:
        # Pega em "002_S_0413__2006..." e devolve apenas "002_S_0413"
        return nome_pasta.split("__")[0]
    else:
        # Fallback de segurança
        return nome_pasta

def calculate_defacing_score_crossed_strict():
    print("A iniciar o Teste de Falsos Positivos Cruzados (Identidades Estritas) para a Spot...\n")
    
    # 1. Carregar e ordenar a lista de pastas de exames
    exam_ids = []
    for item in os.listdir(DIR_PARES):
        if os.path.isdir(os.path.join(DIR_PARES, item)):
            exam_ids.append(item)
            
    exam_ids.sort() # Ordenar garante consistência
    num_exames = len(exam_ids)
    
    if num_exames < 2:
        print("Erro: Precisas de pelo menos 2 pastas de exames para fazer o cruzamento.")
        return

    total_exames_testados = 0
    originais_com_face_detetada = 0
    defaced_sucesso = 0
    defaced_falha = 0

    exames_falha_controlo = []
    exames_falha_defacing = []

    # 2. Iterar sobre a lista de exames
    for i in range(num_exames):
        exam_id_defaced = exam_ids[i]
        id_paciente_defaced = extrair_id_paciente(exam_id_defaced)
        
        # 3. Lógica para encontrar o próximo exame de um paciente DIFERENTE
        j = (i + 1) % num_exames
        id_paciente_original = extrair_id_paciente(exam_ids[j])
        
        # Continua a avançar no índice até o ID do paciente ser diferente
        while id_paciente_original == id_paciente_defaced:
            j = (j + 1) % num_exames
            id_paciente_original = extrair_id_paciente(exam_ids[j])
            
            # Failsafe: se der a volta completa e todos os exames forem da mesma pessoa
            if j == i:
                break
                
        if j == i:
            print(f"Aviso: Não foi possível encontrar um cruzamento válido para {exam_id_defaced} (só existe 1 paciente).")
            continue

        exam_id_original = exam_ids[j]
        total_exames_testados += 1
        
        path_img_defaced = os.path.join(DIR_PARES, exam_id_defaced, f"{exam_id_defaced}_defaced_image.png")
        path_img_original = os.path.join(DIR_PARES, exam_id_original, f"{exam_id_original}_original_image.png")
        
        if not os.path.exists(path_img_original) or not os.path.exists(path_img_defaced):
            continue
            
        try:
            print(f"--- CRUZAMENTO: Defaced '{exam_id_defaced}' vs Original '{exam_id_original}' ---")
            
            # =======================================================
            # 1. CONTROLO ORIGINAL (Baseline do paciente B)
            # =======================================================
            image_original = face_recognition.load_image_file(path_img_original)
            locs_orig = face_recognition.face_locations(image_original)
            encodings_orig = face_recognition.face_encodings(image_original, known_face_locations=locs_orig)
            
            if len(encodings_orig) == 0:
                print(f"[Controlo Falhou] Nenhuma biometria extraída no original de {exam_id_original}.")
                exames_falha_controlo.append(exam_id_original)
                continue
                
            originais_com_face_detetada += 1
            paciente_target_encoding = encodings_orig[0] 
            
            # =======================================================
            # 2. AVALIAÇÃO DO DEFACING (Defaced do paciente A)
            # =======================================================
            image_defaced = face_recognition.load_image_file(path_img_defaced)
            locs_defaced = face_recognition.face_locations(image_defaced)
            
            if len(locs_defaced) == 0:
                defaced_sucesso += 1
                print(f"  -> Sucesso (Nível 1): Estrutura facial eliminada.")
                continue
                
            encodings_defaced = face_recognition.face_encodings(image_defaced, known_face_locations=locs_defaced)
            
            if len(encodings_defaced) == 0:
                defaced_sucesso += 1
                print(f"  -> Sucesso (Nível 2): Falso positivo HOG rejeitado.")
                continue
                
            # =======================================================
            # 3. COMPARAÇÃO MATEMÁTICA
            # =======================================================
            distancia = face_recognition.face_distance([paciente_target_encoding], encodings_defaced[0])[0]
            
            if distancia > TOLERANCE:
                defaced_sucesso += 1
                print(f"  -> Sucesso (Nível 3): Identidades provadas como diferentes (Distância: {distancia:.2f} > {TOLERANCE}).")
            else:
                defaced_falha += 1
                exames_falha_defacing.append(f"Defaced({exam_id_defaced}) parece Original({exam_id_original})")
                print(f"  -> FALHA CRÍTICA: Falso positivo biométrico genérico! (Distância: {distancia:.2f} <= {TOLERANCE})")
                
        except Exception as e:
            print(f"[Erro Interno] Ao cruzar {exam_id_defaced} com {exam_id_original}: {e}")

    # =======================================================
    # 4. RELATÓRIO DA MÉTRICA
    # =======================================================
    print("\n" + "=" * 55)
    print("RELATÓRIO FINAL: TESTE CRUZADO ESTREITO DA SPOT")
    print("=" * 55)
    print(f"Total de pares cruzados processados: {total_exames_testados}")
    print(f"Base de Validação (Originais lidos): {originais_com_face_detetada}")
    print("----" * 15)
    print(f"CERTO (Rejeitou identidades cruzadas): {defaced_sucesso}")
    print(f"ERRO (Falsos matches genéricos): {defaced_falha}")
    
    if originais_com_face_detetada > 0:
        defacing_score = (defaced_sucesso / originais_com_face_detetada) * 100
        print(f"\nSCORE DE DISTINÇÃO: {defacing_score:.2f}%")
    
    if len(exames_falha_defacing) > 0:
        print(f"\nAtenção: Ocorreram falsos positivos nos seguintes cruzamentos:")
        for falha in exames_falha_defacing:
            print(f" - {falha}")
            
    print("=" * 55)

if __name__ == "__main__":
    calculate_defacing_score_crossed_strict()