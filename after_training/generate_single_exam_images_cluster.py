import nibabel as nib
import numpy as np
import pyvista as pv
import os
from skimage.filters import gaussian

# ================= CONFIGURAÇÃO LOCAL (PARA TESTE) =================
BASE_DIR = r"D:\Github Projects\medical_image_deidentification\treinos\teste_completo_tentativa_1\pares_exames_non_def_and_def\051_S_1131__2007-01-18_11_49_060__I36802"
FILE_ORIGINAL = os.path.join(BASE_DIR, "051_S_1131__2007-01-18_11_49_060__I36802_original.nii.gz")
FILE_DEFACED = os.path.join(BASE_DIR, "051_S_1131__2007-01-18_11_49_060__I36802_defaced.nii.gz")
OUTPUT_DIR = os.path.join(BASE_DIR, "renders_cluster_test")

# PARÂMETROS RÉPLICA DO CLUSTER
AMBIENT = 1.3646
DIFFUSE = 1.6875
SPECULAR = 1.3854
SPECULAR_POWER = 4.6094
MIN_OPACITY = 55.1615
MAX_OPACITY = 417.6510
OPACITY_UNIT = 2.0000
SAMPLE_DIST = 0.1070
SMOOTHING_SIGMA = 0.7396

# CONFIGURAÇÃO DA CÂMARA DO CLUSTER
CAMERA_POSITION = [
    (310.13371931613545, 856.894473217762, 84.39690110118934),
    (95.40000379085541, 119.5, 127.5),
    (-0.04003280809502822, 0.06992100315989472, 0.9967489290654598)
]

def render_cluster_logic(file_path, output_name):
    print(f"A processar com Lógica Cluster: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Erro: Ficheiro não encontrado: {file_path}")
        return

    # 1. Carregar volume
    nifti = nib.load(file_path)
    data = nifti.get_fdata()
    voxel_spacing = nifti.header.get_zooms()[:3]
    vol_min, vol_max = np.min(data), np.max(data)

    if SMOOTHING_SIGMA > 0.01:
        data = gaussian(data, sigma=SMOOTHING_SIGMA)

    # 2. Criar Grelha (ImageData nativo)
    grid = pv.ImageData()
    grid.dimensions = np.array(data.shape)
    grid.spacing = voxel_spacing
    grid.point_data["intensities"] = data.flatten(order="F")

    # 3. Configurar Plotter
    plotter = pv.Plotter(off_screen=True, window_size=[1200, 1200])
    plotter.set_background('black')
    
    volume = plotter.add_volume(
        grid, 
        scalars="intensities", 
        cmap="gray",          
        shade=True,           
        ambient=AMBIENT,         
        diffuse=DIFFUSE,          
        specular=SPECULAR,         
        specular_power=SPECULAR_POWER,
        show_scalar_bar=False
    )
    
    # Propriedades e Compatibilidade VTK
    volume.prop.interpolation_type = 'linear'
    if hasattr(volume.prop, 'SetScalarOpacityUnitDistance'):
        volume.prop.SetScalarOpacityUnitDistance(OPACITY_UNIT)
    if hasattr(volume.mapper, 'SetSampleDistance'):
        volume.mapper.SetSampleDistance(SAMPLE_DIST)

    # Curva de Opacidade
    pwf = volume.prop.GetScalarOpacity()
    pwf.RemoveAllPoints()
    pwf.AddPoint(vol_min, 0.0)
    pwf.AddPoint(MIN_OPACITY, 0.0)
    pwf.AddPoint(MAX_OPACITY, 1.0)
    pwf.AddPoint(vol_max, 1.0)
    
    # 4. Posicionamento da Câmara (Invertida a ordem para reset_camera vir primeiro)
    plotter.reset_camera()
    plotter.camera_position = CAMERA_POSITION
    
    # 5. Gravação
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    out_path = os.path.join(OUTPUT_DIR, f"{output_name}_cluster_style.png")
    plotter.screenshot(out_path)
    print(f"Imagem guardada: {out_path}")

    plotter.close()

def main():
    # Gerar para Original
    render_cluster_logic(FILE_ORIGINAL, "original")
    
    # Gerar para Defaced
    render_cluster_logic(FILE_DEFACED, "defaced")

if __name__ == "__main__":
    main()
