import os
import nibabel as nib
import numpy as np
import pyvista as pv
from skimage.filters import gaussian

# ================= CONFIGURAÇÃO FINAL (VALORES COPIADOS DO IMAGE_V2) =================
TARGET_FOLDER = r"D:\Github Projects\medical_image_deidentification\teste_completo_tentativa_1\pares_exames_non_def_and_def\051_S_1331__2007-04-26_09_25_480__I51349"

# Parâmetros de Material e Opacidade:
AMBIENT = 1.0000
DIFFUSE = 0.4796
SPECULAR = 1.0000
SPECULAR_POWER = 3.6751
MIN_OPACITY = 10.9184
MAX_OPACITY = 15.9029
OPACITY_UNIT = 0.5000
SAMPLE_DIST = 0.1000
SMOOTHING_SIGMA = 0.0000

# Configuração da Câmara (Posição exata extraída):
CAMERA_POSITION = [
    (170.18488548619595, 639.4618889691, 118.11055977286792),
 (95.40000379085541, 119.5, 127.5),
 (-0.012766183627301964, 0.019889025296620486, 0.9997206866061861)
 ]
# =====================================================================================

def renderizar_exames_com_volume_config():
    print(f"A iniciar renderização volumétrica (Modo Réplica) na pasta: {TARGET_FOLDER}")
    
    if not os.path.exists(TARGET_FOLDER):
        print(f"[Erro] O caminho não existe: {TARGET_FOLDER}")
        return

    for f in os.listdir(TARGET_FOLDER):
        if not f.endswith(".nii.gz") or "_mask" in f:
            continue
            
        nifti_path = os.path.join(TARGET_FOLDER, f)
        base_name = f.replace(".nii.gz", "")
        output_path = os.path.join(TARGET_FOLDER, f"{base_name}_v2_render.png")
        
        print(f"A processar: {f}...")
        
        try:
            # 1. Carregar Dados
            nifti = nib.load(nifti_path)
            data = nifti.get_fdata()
            voxel_spacing = nifti.header.get_zooms()[:3]
            vol_min, vol_max = np.min(data), np.max(data)

            # 2. Aplicar Suavização (Sigma={SMOOTHING_SIGMA})
            if SMOOTHING_SIGMA > 0.01:
                data = gaussian(data, sigma=SMOOTHING_SIGMA)

            # 3. Criar Grelha
            grid = pv.UniformGrid()
            grid.dimensions = np.array(data.shape)
            grid.spacing = voxel_spacing
            grid.point_data["intensities"] = data.flatten(order="F")

            # 4. Configurar Plotter
            plotter = pv.Plotter(off_screen=True, window_size=[1200, 1200])
            plotter.set_background('black')
            plotter.enable_anti_aliasing('fxaa')

            volume = plotter.add_volume(
                grid,
                scalars="intensities",
                cmap="gray",
                shade=True,
                ambient=AMBIENT,
                diffuse=DIFFUSE,
                specular=SPECULAR,
                specular_power=SPECULAR_POWER,
                show_scalar_bar=False # Garante que a barra não tapa a cara
            )
            
            volume.prop.interpolation_type = 'linear'
            if hasattr(volume.prop, 'SetScalarOpacityUnitDistance'):
                volume.prop.SetScalarOpacityUnitDistance(OPACITY_UNIT)
            
            if hasattr(volume.mapper, 'SetSampleDistance'):
                volume.mapper.SetSampleDistance(SAMPLE_DIST)

            # Configurar Curva de Opacidade
            pwf = volume.prop.GetScalarOpacity()
            pwf.RemoveAllPoints()
            pwf.AddPoint(vol_min, 0.0)
            pwf.AddPoint(MIN_OPACITY, 0.0)
            pwf.AddPoint(MAX_OPACITY, 1.0)
            pwf.AddPoint(vol_max, 1.0)

            # 5. Posicionar Câmara (RÉPLICA EXATA)
            plotter.camera_position = CAMERA_POSITION

            # 6. Salvar Imagem
            if os.path.exists(output_path):
                os.remove(output_path)
                
            plotter.screenshot(output_path)
            plotter.close()
            
            print(f" -> Guardado em: {output_path}")

        except Exception as e:
            print(f"[Erro] Falha ao processar {f}: {e}")

    print("\nProcessamento concluído com as definições de réplica.")

if __name__ == "__main__":
    renderizar_exames_com_volume_config()
