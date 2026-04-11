import os
import time
import argparse
import face_recognition
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

# ================= CONFIGURAÇÃO =================
TOLERANCE = 0.6  # Limiar de similaridade (menor é mais rigoroso)
DETECTION_MODEL = 'cnn' 
# ================================================

def ddp_setup():
    """Inicializa o grupo de processos e força o contexto CUDA para o dlib"""
    if "WORLD_SIZE" not in os.environ:
        # Fallback para execução sequencial se não for usado o torchrun
        torch.cuda.set_device(0)
        return 0, 0, 1

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # MUITO IMPORTANTE: Isto garante que a CNN do dlib acorda na GPU correta
    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl")
    
    return local_rank, global_rank, world_size

def verificar_deteccao_real(image_path):
    image = face_recognition.load_image_file(image_path)
    
    # 1. Deteção de localização
    face_locations = face_recognition.face_locations(image, model=DETECTION_MODEL)
    
    if not face_locations:
        return False, None

    # 2. Extração de Landmarks para ignorar a periferia (orelhas/contorno)
    landmarks_list = face_recognition.face_landmarks(image, face_locations, model='large')
    
    for i, landmarks in enumerate(landmarks_list):
        caracteristicas_centrais = [
            'nose_bridge', 'nose_tip', 'left_eye', 'right_eye', 'top_lip'
        ]
        
        deteccao_valida = all(feature in landmarks for feature in caracteristicas_centrais)
        
        if deteccao_valida:
            encoding = face_recognition.face_encodings(image, known_face_locations=[face_locations[i]])[0]
            return True, encoding
            
    return False, None

def calculate_defacing_score(dir_pares, output_report):
    local_rank, global_rank, world_size = ddp_setup()
    
    if global_rank == 0:
        start_time = time.time()
        print("A iniciar o cálculo de Defacing Score Distribuído (CNN Biometrics)...\n")

    # 1. Obter e dividir a lista de exames
    todos_exames = sorted([d for d in os.listdir(dir_pares) if os.path.isdir(os.path.join(dir_pares, d))])
    chunks = np.array_split(todos_exames, world_size)
    meus_exames = chunks[global_rank]

    print(f"[Rank {global_rank}] Vai processar {len(meus_exames)} exames.")

    # 2. Variáveis de Contagem Locais (Para cada GPU localmente)
    local_stats = {
        "total_testados": 0,
        "originais_com_face": 0,
        "originais_sem_face": 0,
        "defaced_sucesso": 0,
        "defaced_falha": 0,
        "falha_controlo": [],
        "falha_defacing": []
    }

    # 3. Processamento Local
    for exam_id in meus_exames:
        exam_folder = os.path.join(dir_pares, exam_id)
        
        path_img_original = os.path.join(exam_folder, f"{exam_id}_original_image.png")
        path_img_defaced = os.path.join(exam_folder, f"{exam_id}_defaced_image.png")
        
        if not os.path.exists(path_img_original) or not os.path.exists(path_img_defaced):
            continue
            
        local_stats["total_testados"] += 1
            
        try:
            # VALIDAÇÃO DO ORIGINAL
            face_orig_existe, encoding_orig = verificar_deteccao_real(path_img_original)
            
            if not face_orig_existe:
                local_stats["originais_sem_face"] += 1
                local_stats["falha_controlo"].append(exam_id)
                continue
                
            local_stats["originais_com_face"] += 1
            
            # AVALIAÇÃO DO DEFACED
            face_def_existe, encoding_def = verificar_deteccao_real(path_img_defaced)
            
            if not face_def_existe:
                local_stats["defaced_sucesso"] += 1
            else:
                distancia = face_recognition.face_distance([encoding_orig], encoding_def)[0]
                if distancia > TOLERANCE:
                    local_stats["defaced_sucesso"] += 1
                else:
                    local_stats["defaced_falha"] += 1
                    local_stats["falha_defacing"].append(exam_id)
                    
        except Exception as e:
            print(f"[Rank {global_rank}][Erro] {exam_id}: {e}")

    # 4. Sincronização e Agregação (Gather) no rank 0
    if "WORLD_SIZE" in os.environ:
        gathered_stats = [None for _ in range(world_size)] #Se tivermos 4 GPUs (world_size = 4), esta linha cria uma lista com 4 espaços vazios: [None, None, None, None].
        dist.gather_object(local_stats, gathered_stats if global_rank == 0 else None, dst=0) #vai guardar na lista criada os resultados de cada gpu, guardado a gpu de rank 0 na posição 0, a gpu rank 1 na posição 1 etc.
    else:
        gathered_stats = [local_stats]

    # 5. Mestre Global processa o Relatório Final
    if global_rank == 0:
        final_stats = {
            "total_testados": sum(s["total_testados"] for s in gathered_stats),
            "originais_com_face": sum(s["originais_com_face"] for s in gathered_stats),
            "originais_sem_face": sum(s["originais_sem_face"] for s in gathered_stats),
            "defaced_sucesso": sum(s["defaced_sucesso"] for s in gathered_stats),
            "defaced_falha": sum(s["defaced_falha"] for s in gathered_stats),
            "falha_controlo": [item for s in gathered_stats for item in s["falha_controlo"]],
            "falha_defacing": [item for s in gathered_stats for item in s["falha_defacing"]]
        }

        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        tempo_formatado = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        tempo_segundos = f"Total de {int(elapsed_time)} segundos"
        score = 0.0
        if final_stats["originais_com_face"] > 0:
            score = (final_stats["defaced_sucesso"] / final_stats["originais_com_face"]) * 100

        relatorio = (
            f"=======================================================\n"
            f"RELATÓRIO FINAL: DEFACING SCORE (Processamento Paralelo)\n"
            f"=======================================================\n"
            f"Pares processados: {final_stats['total_testados']}\n"
            f"Originais Validados (Marcos Centrais): {final_stats['originais_com_face']}\n"
            f"Originais Rejeitados (Baixa Qualidade): {final_stats['originais_sem_face']}\n"
            f"Lista dos Originais Rejeitados: {final_stats['falha_controlo']}\n\n"
            f"-------------------------------------------------------\n"
            f"CERTO: Defaced ocultado com sucesso: {final_stats['defaced_sucesso']}\n"
            f"ERRO: Defaced com identidade preservada: {final_stats['defaced_falha']}\n\n"
            f"Lista dos Defaced que tiveram faces detetadas: {final_stats['falha_defacing']}\n\n"

            f"DEFACING SCORE: {score:.2f}%\n\n"


            f"⏱️ Tempo total de execução: {tempo_formatado}\n"
            f"Total de {int(elapsed_time)} segundos"
            f"=======================================================\n"
        )

        print("\n" + relatorio)
        
        # Guardar na pasta do Job
        with open(output_report, 'w', encoding='utf-8') as f:
            f.write(relatorio)
            # Guardar também a lista de falhas para análise forense
            if final_stats['falha_defacing']:
                f.write("\n\n--- LISTA DE FALHAS CRÍTICAS (Identidade Preservada) ---\n")
                for exam in final_stats['falha_defacing']:
                    f.write(f"- {exam}\n")

        print(f"✅ Relatório de auditoria guardado em: {output_report}")

    if "WORLD_SIZE" in os.environ:
        destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditoria Biométrica Distribuída")
    parser.add_argument("--dir_pares", type=str, required=True, help="Pasta com os exames emparelhados e PNGs gerados")
    parser.add_argument("--output_report", type=str, required=True, help="Caminho para guardar o txt final")
    args = parser.parse_args()

    calculate_defacing_score(args.dir_pares, args.output_report)