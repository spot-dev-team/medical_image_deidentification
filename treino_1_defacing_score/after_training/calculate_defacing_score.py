import os
import face_recognition

# ================= CONFIGURAÇÃO =================
DIR_PARES = r"E:\Tese\Datasets\Rempe\pares_exames_non_def_and_def"
# Distância máxima (0.0 a 1.0) para considerar que duas faces são a mesma pessoa.
# O Geitgey recomenda 0.6 para o modelo base.
TOLERANCE = 0.6 
# ================================================

def calculate_defacing_score():
    print("A iniciar o cálculo de Defacing Score Avançado (Biometria) para a Spot...\n")
    
    total_exames_testados = 0
    originais_com_face_detetada = 0
    originais_sem_face_detetada = 0
    defaced_sucesso = 0
    defaced_falha = 0

    exames_falha_controlo = []
    exames_falha_defacing = []

    for exam_id in os.listdir(DIR_PARES):
        exam_folder = os.path.join(DIR_PARES, exam_id)
        if not os.path.isdir(exam_folder):
            continue
            
        total_exames_testados += 1
        
        path_img_original = os.path.join(exam_folder, f"{exam_id}_original_image.png")
        path_img_defaced = os.path.join(exam_folder, f"{exam_id}_defaced_image.png")
        
        if not os.path.exists(path_img_original) or not os.path.exists(path_img_defaced):
            continue
            
        try:
            # =======================================================
            # 1. CONTROLO ORIGINAL (Baseline)
            # =======================================================
            image_original = face_recognition.load_image_file(path_img_original)
            # Extrai as localizações e imediatamente tenta gerar os vetores 128D
            locs_orig = face_recognition.face_locations(image_original)
            encodings_orig = face_recognition.face_encodings(image_original, known_face_locations=locs_orig)
            
            if len(encodings_orig) == 0:
                print(f"[Controlo Falhou] {exam_id}: Nenhuma assinatura biométrica extraída no original.")
                originais_sem_face_detetada += 1
                exames_falha_controlo.append(exam_id)
                continue
                
            originais_com_face_detetada += 1
            # Assumimos a primeira face detetada como o target do paciente
            paciente_target_encoding = encodings_orig[0] 
            print(f"[Green Light] {exam_id}: Biometria original guardada. A testar defacing...")
            
            # =======================================================
            # 2. AVALIAÇÃO DO DEFACING
            # =======================================================
            image_defaced = face_recognition.load_image_file(path_img_defaced)
            locs_defaced = face_recognition.face_locations(image_defaced)
            
            # Se nem sequer encontrar as bordas falsas, é um sucesso imediato
            if len(locs_defaced) == 0:
                defaced_sucesso += 1
                print(f"  -> Sucesso (Nível 1): Estrutura facial completamente eliminada.")
                continue
                
            # Se encontrou bordas (o tal falso positivo do "buraco"), tentamos extrair biometria
            encodings_defaced = face_recognition.face_encodings(image_defaced, known_face_locations=locs_defaced)
            
            if len(encodings_defaced) == 0:
                defaced_sucesso += 1
                print(f"  -> Sucesso (Nível 2): Falso positivo detetado pelo HOG, mas rejeitado por ausência de biometria.")
                continue
                
            # =======================================================
            # 3. COMPARAÇÃO MATEMÁTICA (O Último Recurso)
            # =======================================================
            # Se chegou aqui, encontrou "olhos/nariz" no ruído. Vamos calcular a distância Euclidiana
            # entre o vetor original e o vetor do defaced.
            distancia = face_recognition.face_distance([paciente_target_encoding], encodings_defaced[0])[0]
            
            if distancia > TOLERANCE:
                defaced_sucesso += 1
                print(f"  -> Sucesso (Nível 3): Face detetada, mas a identidade foi destruída (Distância: {distancia:.2f} > {TOLERANCE}).")
            else:
                defaced_falha += 1
                exames_falha_defacing.append(exam_id)
                print(f"  -> FALHA CRÍTICA: O modelo reconheceu o mesmo paciente! (Distância: {distancia:.2f} <= {TOLERANCE})")
                
        except Exception as e:
            print(f"[Erro Interno] {exam_id}: {e}")

    # =======================================================
    # 4. RELATÓRIO DA MÉTRICA
    # =======================================================
    print("\n" + "=" * 55)
    print("RELATÓRIO FINAL: DEFACING SCORE DA SPOT")
    print("=" * 55)
    print(f"Total de pares de exames processados: {total_exames_testados}")
    print(f"Base de Validação (Originais Validados): {originais_com_face_detetada}")
    print("----" * 15)
    print(f"CERTO: Defaced ocultado com sucesso: {defaced_sucesso}")
    print(f"ERRO: Defaced com identidade reconhecível: {defaced_falha}")
    
    if originais_com_face_detetada > 0:
        defacing_score = (defaced_sucesso / originais_com_face_detetada) * 100
        print(f"\nDEFACING SCORE: {defacing_score:.2f}%")
    else:
        print("\nErro Matemático: Nenhuma face original possuía qualidade biométrica suficiente.")
        
    print("=" * 55)

if __name__ == "__main__":
    calculate_defacing_score()