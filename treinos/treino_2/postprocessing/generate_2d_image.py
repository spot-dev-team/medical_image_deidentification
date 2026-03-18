import os
import random
import nibabel as nib
import numpy as np
import pyvista as pv

# ================= CONFIGURAÇÃO =================
# Substitui pelo caminho onde tens os exames anonimizados
INPUT_DIR = r"E:\Tese\Datasets\Rempe\processed_datasets_anon" 
# ================================================

def generate_face_render(base_path):
    print("A iniciar o pipeline de renderização 3D da Spot...")

    # 1. Obter exames e selecionar um aleatoriamente
    exams = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not exams:
        print("[Erro] Nenhuma pasta de exame encontrada no diretório.")
        return

    selected_exam = random.choice(exams)
    exam_dir = os.path.join(base_path, selected_exam)
    print(f"Exame sorteado: {selected_exam}")

    # 2. Localizar a matriz 3D do exame
    # (Procura por _anon.nii.gz ou adapta para o padrão de nomenclatura que estiveres a usar)
    nifti_path = None
    for f in os.listdir(exam_dir):
        if f.endswith(".nii.gz") and "anon" in f:
            nifti_path = os.path.join(exam_dir, f)
            break
            
    # Fallback caso os ficheiros não tenham a tag 'anon'
    if not nifti_path:
        for f in ["image.nii.gz", "raw.nii.gz"]:
            temp_path = os.path.join(exam_dir, f)
            if os.path.exists(temp_path):
                nifti_path = temp_path
                break

    if not nifti_path:
        print(f"[Erro] Ficheiro NIfTI não encontrado na pasta {selected_exam}.")
        return

    # 3. Carregar as estruturas NIfTI para a RAM
    nifti = nib.load(nifti_path)
    data = nifti.get_fdata()
    voxel_spacing = nifti.header.get_zooms()[:3]

    # 4. Normalização de Intensidades [0, 1]
    # Essencial para garantir que o isovalue da pele funciona independentemente da máquina de RM
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

    # 5. Instanciar a grelha estruturada (VTK ImageData)
    grid = pv.ImageData()
    grid.dimensions = np.array(data.shape)
    grid.spacing = voxel_spacing
    
    # O VTK em C++ lê os arrays indexados em Fortran (Coluna-maior).
    # O flatten(order="F") reorganiza os blocos de RAM do NumPy para evitar a corrupção da geometria.
    grid.point_data["intensities"] = data_norm.flatten(order="F")

    # 6. Extração Geométrica da Pele (Isosuperfície)
    # 0.05 a 0.15 geralmente captura a fronteira ar/tecido mole.
    print("A extrair malha poligonal da pele...")
    surface = grid.contour([0.08], scalars="intensities")

    # 7. Configuração do Motor de Renderização (Off-screen)
    plotter = pv.Plotter(off_screen=True)
    # Aplicar um material baço/mate à malha para simular pele e captar luzes
    plotter.add_mesh(surface, color='bisque', smooth_shading=True, specular=0.0)

    # 8. Matemática da Câmara (Formato RAS)
    center = surface.center
    
    # Colocar a câmara no eixo Y positivo (à frente do rosto), a olhar para o centro.
    # Aumentar a coordenada Y em 300mm cria um recuo de segurança para apanhar toda a face.
    camera_pos = (center[0], center[1] + 300, center[2]) 
    
    plotter.camera_position = [
        camera_pos,         # Origem da câmara
        center,             # Ponto focal
        (0, 0, 1)           # Vetor 'Cima' (Eixo Z no espaço RAS)
    ]

    # 9. Disparo e IO do disco
    output_filename = f"{selected_exam}_image.png"
    output_path = os.path.join(exam_dir, output_filename)
    
    plotter.screenshot(output_path)
    print(f"[{selected_exam}] Renderização concluída com sucesso!")
    print(f"Ficheiro gerado: {output_path}")

if __name__ == "__main__":
    generate_face_render(INPUT_DIR)