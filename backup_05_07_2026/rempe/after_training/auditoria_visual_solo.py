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
                if facial_feature == 'chin':
                    continue
                
                pontos = face_landmarks[facial_feature]
                
                # Fechar a geometria da lista de tuplos para estruturas circulares
                if facial_feature in ['left_eye', 'right_eye', 'top_lip', 'bottom_lip']:
                    pontos = pontos + [pontos[0]]
                    
                # Desenhar a linha com os pontos atualizados
                d.line(pontos, width=3, fill='lime')

    # Guardar a prova visual
    pil_image.save(output_name)
    print(f" -> Auditoria guardada como: {output_name}\n")


if __name__ == "__main__":
    # Escolhe um ID de exame para testar

    id= "033_S_1279__2008-02-29_15_01_380__I94980"
    PASTA_BASE = f"/projects/F202500001HPCVLABEPICURE/andresousa615/rempe/results/train_mednext_0.0005_mednext_aurora_16gb_vram_1120800/pares_2d/{id}"

    path_original = os.path.join(PASTA_BASE, f"{id}_original_image.png")
    path_defaced = os.path.join(PASTA_BASE, f"{id}_defaced_image.png")
    
    # Onde guardar os resultados
    out_original = os.path.join(PASTA_BASE, "AUDITORIA_original.png")
    out_defaced = os.path.join(PASTA_BASE, "AUDITORIA_defaced.png")
    
    if os.path.exists(path_original) and os.path.exists(path_defaced):
        desenhar_landmarks(path_original, out_original)
        desenhar_landmarks(path_defaced, out_defaced)
        print("Auditoria concluída! Abre os ficheiros AUDITORIA_*.png para verificares o mapeamento.")
    else:
        print("Caminhos das imagens não encontrados. Verifica a configuração da PASTA_BASE.")