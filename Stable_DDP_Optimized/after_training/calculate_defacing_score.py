import os
import time
import face_recognition
import numpy as np

# ================= CONFIGURAÇÃO =================
DIR_PARES = r"/home/andresousa615/rempe/mede_code/after_training/pares_exames_non_def_and_def"
TOLERANCE = 0.6  # Limiar de similaridade (menor é mais rigoroso)

# Define o modelo de deteção: 'cnn' é mais preciso (requer GPU/CUDA), 
# 'hog' é mais rápido em CPU.
DETECTION_MODEL = 'cnn' 
# ================================================

def verificar_deteccao_real(image_path):
    """
    Valida se a deteção de face é real verificando a presença de 
    marcos anatómicos centrais (olhos, nariz, boca).
    """
    image = face_recognition.load_image_file(image_path)
    
    # 1. Deteção de localização
    face_locations = face_recognition.face_locations(image, model=DETECTION_MODEL)
    
    if not face_locations:
        return False, None

    # 2. Extração de Landmarks para ignorar a periferia (orelhas/contorno)
    landmarks_list = face_recognition.face_landmarks(image, face_locations, model='large')
    
    for i, landmarks in enumerate(landmarks_list):
        # Marcos que provam a existência de uma face estruturada
        caracteristicas_centrais = [
            'nose_bridge', 
            'nose_tip', 
            'left_eye', 
            'right_eye', 
            'top_lip'
        ]
        
        # Validação booleana: a face só é real se contiver todos os marcos centrais
        deteccao_valida = all(feature in landmarks for feature in caracteristicas_centrais)
        
        if deteccao_valida:
            # Gerar o vetor 128D apenas para a face validada
            encoding = face_recognition.face_encodings(image, known_face_locations=[face_locations[i]])[0]
            return True, encoding
            
    return False, None

def calculate_defacing_score():
    # Inicia a contagem de tempo
    start_time = time.time()
    print("A iniciar o cálculo de Defacing Score por Biometria Topológica...\n")
    
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
        
        print(f"\n[Processing] {exam_id}...")
        total_exames_testados += 1
        path_img_original = os.path.join(exam_folder, f"{exam_id}_original_image.png")
        path_img_defaced = os.path.join(exam_folder, f"{exam_id}_defaced_image.png")
        
        if not os.path.exists(path_img_original) or not os.path.exists(path_img_defaced):
            print(f"[Aviso] Imagens em falta para {exam_id}. Skipping...")
            continue
            
        try:
            # 1. VALIDAÇÃO DO ORIGINAL (Baseline)
            face_orig_existe, encoding_orig = verificar_deteccao_real(path_img_original)
            
            if not face_orig_existe:
                print(f"[Controlo Falhou] {exam_id}: Sem biometria central no original.")
                originais_sem_face_detetada += 1
                exames_falha_controlo.append(exam_id)
                continue
                
            originais_com_face_detetada += 1
            print(f"[Green Light] {exam_id}: Biometria validada. A testar anonimização...")
            
            # 2. AVALIAÇÃO DO DEFACED (Landmarks + Distância)
            face_def_existe, encoding_def = verificar_deteccao_real(path_img_defaced)
            
            if not face_def_existe:
                # SUCESSO: Ou não encontrou nada, ou o que encontrou não tinha marcos faciais
                defaced_sucesso += 1
                print(f"  -> Sucesso: Características faciais centrais eliminadas.")
            else:
                # Se biometria central persistir, comparamos a identidade
                distancia = face_recognition.face_distance([encoding_orig], encoding_def)[0]
                
                if distancia > TOLERANCE:
                    defaced_sucesso += 1
                    print(f"  -> Sucesso: Face detetada, mas identidade destruída (Dist: {distancia:.2f}).")
                else:
                    defaced_falha += 1
                    exames_falha_defacing.append(exam_id)
                    print(f"  -> FALHA CRÍTICA: Identidade reconhecida (Dist: {distancia:.2f}).")
                
        except Exception as e:
            print(f"[Erro] {exam_id}: {e}")

    # Calcula o tempo decorrido
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    tempo_formatado = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    # RELATÓRIO FINAL
    print("\n" + "=" * 55)
    print("RELATÓRIO FINAL: DEFACING SCORE")
    print("=" * 55)
    print(f"Pares processados: {total_exames_testados}")
    print(f"Originais Validados (Marcos Centrais): {originais_com_face_detetada}")
    print(f"Originais Rejeitados (Baixa Qualidade): {originais_sem_face_detetada}")
    print("----" * 15)
    print(f"CERTO: Defaced ocultado com sucesso: {defaced_sucesso}")
    print(f"ERRO: Defaced com identidade preservada: {defaced_falha}")
    
    if originais_com_face_detetada > 0:
        score = (defaced_sucesso / originais_com_face_detetada) * 100
        print(f"\nDEFACING SCORE: {score:.2f}%")
        
    print(f"\n⏱️ Tempo total de execução: {tempo_formatado}")
    print("=" * 55)

if __name__ == "__main__":
    calculate_defacing_score()