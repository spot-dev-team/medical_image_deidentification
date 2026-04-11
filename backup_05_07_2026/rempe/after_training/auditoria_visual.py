import os
import face_recognition
from PIL import Image, ImageDraw

def desenhar_landmarks(image_path, output_name):
    print(f"A analisar imagem: {image_path}")
    
    # 1. Carregar a matriz de pixéis
    image_array = face_recognition.load_image_file(image_path)
    
    # 2. Extrair a geometria completa
    # Usamos o modelo CNN para coerência com o resultado de 100% do cluster
    locs = face_recognition.face_locations(image_array, model='cnn')
    face_landmarks_list = face_recognition.face_landmarks(image_array, locs, model='large')
    
    # Converter o array numpy para uma imagem manipulável pelo Pillow
    pil_image = Image.fromarray(image_array)
    d = ImageDraw.Draw(pil_image)

    # 3. Iterar sobre a estrutura de dados e desenhar
    if not face_landmarks_list:
        print(" -> Nenhuma estrutura facial detetada pelo modelo.")
    else:
        for face_landmarks in face_landmarks_list:
            # face_landmarks é um dicionário: {'chin': [(x,y), ...], 'left_eye': [(x,y), ...], ...}
            
            # Percorrer cada característica anatómica e os seus pontos
            for facial_feature in face_landmarks.keys():
                # Ignorar o contorno do queixo/orelhas para vermos estritamente a face central
                if facial_feature == 'chin':
                    continue
                    
                # Desenhar uma linha verde a ligar as coordenadas anatómicas
                d.line(face_landmarks[facial_feature], width=3, fill='lime')

    # Guardar a prova visual
    pil_image.save(output_name)
    print(f" -> Auditoria guardada como: {output_name}\n")


if __name__ == "__main__":
    # Escolhe um ID de exame para testar
    DIR = "/home/andresousa615/rempe/mede_code/after_training/pares_exames_non_def_and_def"

    for exam_id in os.listdir(DIR):
        print(f"\n[Processando] {exam_id}...")
        if not os.path.isdir(os.path.join(DIR, exam_id)):
            continue

        #id= "032_S_0187__2006-03-14_10_43_340__I11954"
        #PASTA_BASE = f"/home/andresousa615/rempe/mede_code/after_training/pares_exames_non_def_and_def/{id}"
        # Ficheiros alvo
        DIR_EXAME = os.path.join(DIR, exam_id)
        path_original = os.path.join(DIR_EXAME, f"{exam_id}_original_image.png")
        path_defaced = os.path.join(DIR_EXAME, f"{exam_id}_defaced_image.png")
        
        # Onde guardar os resultados
        out_original = os.path.join(DIR_EXAME, "AUDITORIA_original.png")
        out_defaced = os.path.join(DIR_EXAME, "AUDITORIA_defaced.png")
        
        if os.path.exists(path_original) and os.path.exists(path_defaced):
            desenhar_landmarks(path_original, out_original)
            desenhar_landmarks(path_defaced, out_defaced)
            print("Auditoria concluída! Abre os ficheiros AUDITORIA_*.png para verificares o mapeamento.")
        else:
            print("Caminhos das imagens não encontrados. Verifica a configuração da PASTA_BASE.")