import os
import random
import nibabel as nib
import numpy as np
import pyvista as pv

# ================= CONFIGURAÇÃO =================
# Caminho para a tua pasta plana com todos os exames e máscaras
INPUT_DIR = r"E:\Tese\Datasets\Rempe\treino_3\resultados_inferencia" 
# ================================================

def generate_face_render(base_path):
    print("A iniciar o pipeline de renderização 3D da Spot...")

    # 1. Iterar sobre a pasta plana e filtrar apenas os ficheiros alvo
    all_files = os.listdir(base_path)
    valid_exams = [f for f in all_files if f.endswith("_anon.nii.gz")]
    
    if not valid_exams:
        print("[Erro] Nenhum exame com sufixo '_anon.nii.gz' encontrado no diretório.")
        return

    # Sorteio do ficheiro para teste
    #selected_file = random.choice(valid_exams)
    #nifti_path = os.path.join(base_path, selected_file)
    nifti_path = r"E:\Tese\Datasets\Rempe\treino_3\resultados_inferencia\002_S_0954__2007-05-03_08_00_340__I53479_raw_anon.nii.gz"

    # 2. Manipulação de String para o Output (XXXX_anon.nii.gz -> XXXX)
    #base_name = selected_file.replace("_anon.nii.gz", "")
    base_name = os.path.basename(nifti_path).replace("_anon.nii.gz", "")
    print(f"Exame sorteado: {base_name}")

    # 3. Carregar NIfTI e converter para RAM
    nifti = nib.load(nifti_path)
    data = nifti.get_fdata()
    voxel_spacing = nifti.header.get_zooms()[:3]

    # 4. Normalização [0, 1]
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

    # 5. Grelha Estruturada VTK
    grid = pv.UniformGrid()
    grid.dimensions = np.array(data.shape)
    grid.spacing = voxel_spacing
    grid.point_data["intensities"] = data_norm.flatten(order="F")

    # 6. Extração Geométrica (Isosuperfície)
    print("A extrair malha poligonal 3D...")
    surface = grid.contour([0.04], scalars="intensities") #apos testes 0.04 foi o melhor

    # 7. Motor Off-screen
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(surface, color='bisque', smooth_shading=True, specular=0.0)

    # 8. Câmara RAS
    center = surface.center
    camera_pos = (center[0]+200, center[1] + 600, center[2]) 
    
    plotter.camera_position = [
        camera_pos, 
        center,     
        (0, 0, 1)   
    ]

    # 9. Escrita no disco com a nova nomenclatura
    output_filename = f"{base_name}_image.png"
    output_path = os.path.join(base_path, output_filename)
    
    plotter.screenshot(output_path)
    print(f"[{base_name}] Renderização concluída com sucesso!")
    print(f"Ficheiro guardado em: {output_path}")

if __name__ == "__main__":
    # Garante que gravas este ficheiro (Ctrl+S) antes de executares!
    generate_face_render(INPUT_DIR)