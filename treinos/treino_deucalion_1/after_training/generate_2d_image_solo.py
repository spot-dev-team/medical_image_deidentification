# este código faz o memso que o outro mas recebe 1 exame da cada fez
import os
import nibabel as nib
import numpy as np
import pyvista as pv

# ================= CONFIGURAÇÃO MANUAL =================
# Define o caminho exato para a subpasta do paciente específico
TARGET_FOLDER = r"E:\Tese\Datasets\Rempe\pares_exames_non_def_and_def\infant_t1_11"

# PARÂMETROS DE AFINAÇÃO DA MATRIZ:
# Aumenta este valor (ex: 0.08, 0.12, 0.15) para "cortar" o ruído de fundo.
# Se a face começar a ficar esburacada, baixa ligeiramente.
ISO_VALUE = 0.18

# Ajuste do recuo geométrico da câmara (eixo Y)
CAMERA_Y_OFFSET = 500
# =======================================================

def renderizar_exame_isolado():
    print(f"A iniciar afinação isolada na pasta: {TARGET_FOLDER}")
    
    if not os.path.exists(TARGET_FOLDER):
        print("[Erro] O caminho especificado não existe.")
        return

    # Iterar apenas sobre os NIfTIs dentro desta subpasta
    for f in os.listdir(TARGET_FOLDER):
        if not f.endswith(".nii.gz"):
            continue
            
        nifti_path = os.path.join(TARGET_FOLDER, f)
        base_name = f.replace(".nii.gz", "")
        output_path = os.path.join(TARGET_FOLDER, f"{base_name}_image.png")
        
        print(f"A processar matriz de: {base_name}...")
        
        try:
            # 1. Carregar e Normalizar
            nifti = nib.load(nifti_path)
            data = nifti.get_fdata()
            voxel_spacing = nifti.header.get_zooms()[:3]
            
            # Normalização linear [0, 1]
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

            # 2. Mapeamento da Grelha
            grid = pv.UniformGrid()
            grid.dimensions = np.array(data.shape)
            grid.spacing = voxel_spacing
            grid.point_data["intensities"] = data_norm.flatten(order="F")

            # 3. Extração com Limiar Ajustado
            surface = grid.contour([ISO_VALUE], scalars="intensities")

            # 4. Motor Render
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(surface, color='bisque', smooth_shading=True, specular=0.0)

            # 5. Geometria da Câmara
            center = surface.center
            camera_pos = (center[0] + 50, center[1] + CAMERA_Y_OFFSET, center[2]) 
            
            plotter.camera_position = [
                camera_pos, 
                center,     
                (0, 0, 1)   
            ]

            # 6. Forçar a gravação por cima de imagens antigas
            if os.path.exists(output_path):
                os.remove(output_path)
                
            plotter.screenshot(output_path)
            plotter.close()
            
            print(f"[{base_name}] Renderizado com ISO={ISO_VALUE}.")

        except Exception as e:
            print(f"[Erro] Falha matricial em {f}: {e}")

    print("=" * 50)
    print("Processamento isolado concluído. Verifica os PNGs.")

if __name__ == "__main__":
    renderizar_exame_isolado()