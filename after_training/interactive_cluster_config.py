import nibabel as nib
import numpy as np
import pyvista as pv
from skimage.filters import gaussian
import os

# ================= CONFIGURAÇÃO =================
TEST_FILE = r"D:\Github Projects\medical_image_deidentification\treinos\teste_completo_tentativa_1\pares_exames_non_def_and_def\002_S_0559__2006-12-12_09_59_240__I32918\002_S_0559__2006-12-12_09_59_240__I32918_original.nii.gz"
# ================================================

def interactive_cluster_config():
    print(f"A carregar volume para sintonização Cluster: {TEST_FILE}")
    
    # 1. Carregar matriz e geometria
    nifti = nib.load(TEST_FILE)
    original_data = nifti.get_fdata()
    voxel_spacing = nifti.header.get_zooms()[:3]
    vol_min, vol_max = np.min(original_data), np.max(original_data)

    # 2. Estruturar Grelha (ImageData nativo)
    grid = pv.ImageData()
    grid.dimensions = np.array(original_data.shape)
    grid.spacing = voxel_spacing
    grid.point_data["intensities"] = original_data.flatten(order="F")

    plotter = pv.Plotter(off_screen=False, window_size=[1200, 1200])
    plotter.set_background('black')
    
    # REMOVIDO: FXAA (para garantir que vês o mesmo "ruído" que o Cluster veria)
    
    volume = plotter.add_volume(
        grid, 
        scalars="intensities", 
        cmap="gray",          
        shade=True,           
        ambient=1.0,         
        diffuse=1.0,          
        specular=1.0,         
        specular_power=3.84,
        show_scalar_bar=False
    )
    volume.prop.interpolation_type = 'linear'
    
    # 3. Estado local para tracking dos valores (Réplica dos teus nomes do Cluster)
    state = {
        "min_opacity": 10.42,
        "max_opacity": 43.9,
        "smoothing": 0.7,
        "sample_dist": 0.1,
        "opacity_unit": 0.5
    }

    def atualizar_curva_opacidade():
        pwf = volume.prop.GetScalarOpacity()
        pwf.RemoveAllPoints()
        pwf.AddPoint(vol_min, 0.0)
        pwf.AddPoint(state["min_opacity"], 0.0)
        pwf.AddPoint(state["max_opacity"], 1.0)
        pwf.AddPoint(vol_max, 1.0)
        plotter.render()

    atualizar_curva_opacidade()

    # 4. Callbacks de Sintonização
    def update_ambient(value):
        volume.prop.ambient = value
        plotter.render()

    def update_min(value):
        state["min_opacity"] = value
        atualizar_curva_opacidade()

    def update_max(value):
        state["max_opacity"] = value
        atualizar_curva_opacidade()

    def update_diffuse(value):
        volume.prop.diffuse = value
        plotter.render()

    def update_specular(value):
        volume.prop.specular = value
        plotter.render()

    def update_specular_power(value):
        volume.prop.specular_power = value
        plotter.render()

    def update_opacity_unit(value):
        state["opacity_unit"] = value
        if hasattr(volume.prop, 'SetScalarOpacityUnitDistance'):
            volume.prop.SetScalarOpacityUnitDistance(value)
        plotter.render()

    def update_sample_dist(value):
        state["sample_dist"] = value
        if hasattr(volume.mapper, 'SetSampleDistance'):
            volume.mapper.SetSampleDistance(value)
        plotter.render()

    def apply_smoothing(value):
        state["smoothing"] = value
        if value > 0.01:
            smoothed = gaussian(original_data, sigma=value)
            grid.point_data["intensities"] = smoothed.flatten(order="F")
        else:
            grid.point_data["intensities"] = original_data.flatten(order="F")
        grid.Modified()
        plotter.render()

    def print_params():
        print("\n" + "="*40)
        print("CONFIGURAÇÃO IDEAL PARA O CLUSTER:")
        print("="*40)
        print(f"AMBIENT = {volume.prop.ambient:.4f}")
        print(f"DIFFUSE = {volume.prop.diffuse:.4f}")
        print(f"SPECULAR = {volume.prop.specular:.4f}")
        print(f"SPECULAR_POWER = {volume.prop.specular_power:.4f}")
        print(f"MIN_OPACITY = {state['min_opacity']:.4f}")
        print(f"MAX_OPACITY = {state['max_opacity']:.4f}")
        print(f"OPACITY_UNIT = {state['opacity_unit']:.4f}")
        print(f"SAMPLE_DIST = {state['sample_dist']:.4f}")
        print(f"SMOOTHING_SIGMA = {state['smoothing']:.4f}")
        print("\nCAMERA_POSITION = [")
        print(f"    {plotter.camera_position[0]},")
        print(f"    {plotter.camera_position[1]},")
        print(f"    {plotter.camera_position[2]}")
        print("]")
        print("="*40)
        print("NOTA: Lembra-te de usar reset_camera() antes da POSITION no Cluster.")

    plotter.add_key_event('p', print_params)

    # 5. Sliders (Posicionados para não tapar a cara)
    plotter.add_slider_widget(update_ambient, [0.0, 2.0], value=1.0, title="Ambient",
                              pointa=(0.02, 0.92), pointb=(0.22, 0.92), style='modern')
    plotter.add_slider_widget(update_diffuse, [0.0, 2.0], value=1.0, title="Diffuse",
                              pointa=(0.02, 0.78), pointb=(0.22, 0.78), style='modern')
    plotter.add_slider_widget(update_specular, [0.0, 2.0], value=1.0, title="Specular",
                              pointa=(0.02, 0.64), pointb=(0.22, 0.64), style='modern')
    plotter.add_slider_widget(update_specular_power, [1.0, 100.0], value=3.84, title="Spec Power",
                              pointa=(0.02, 0.50), pointb=(0.22, 0.50), style='modern')
    
    plotter.add_slider_widget(apply_smoothing, [0.0, 2.0], value=0.7, title="Smoothing",
                              pointa=(0.02, 0.36), pointb=(0.22, 0.36), style='modern')

    plotter.add_slider_widget(update_min, [vol_min, vol_max/2], value=state["min_opacity"], title="Min Opacity",
                              pointa=(0.77, 0.92), pointb=(0.97, 0.92), style='modern')
    plotter.add_slider_widget(update_max, [vol_min, vol_max], value=state["max_opacity"], title="Max Opacity",
                              pointa=(0.77, 0.78), pointb=(0.97, 0.78), style='modern')
    plotter.add_slider_widget(update_opacity_unit, [0.01, 2.0], value=state["opacity_unit"], title="Opacity Unit",
                              pointa=(0.77, 0.64), pointb=(0.97, 0.64), style='modern')
    plotter.add_slider_widget(update_sample_dist, [0.01, 0.5], value=state["sample_dist"], title="Sample Dist",
                              pointa=(0.77, 0.50), pointb=(0.97, 0.50), style='modern')

    plotter.show()

if __name__ == "__main__":
    interactive_cluster_config()
