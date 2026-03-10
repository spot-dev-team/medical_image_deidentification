import nibabel as nib
import numpy as np
import pyvista as pv
from skimage.filters import gaussian

# ================= CONFIGURAÇÃO =================
TEST_FILE = r"D:\Github Projects\medical_image_deidentification\teste_completo_tentativa_1\pares_exames_non_def_and_def\051_S_1331__2007-04-26_09_25_480__I51349\051_S_1331__2007-04-26_09_25_480__I51349_original.nii.gz"
# ================================================

def teste_interativo_raw_spot():
    print(f"A carregar volume RAW para a Spot: {TEST_FILE}")
    
    # 1. Carregar matriz e geometria
    nifti = nib.load(TEST_FILE)
    original_data = nifti.get_fdata()
    voxel_spacing = nifti.header.get_zooms()[:3]
    vol_min, vol_max = np.min(original_data), np.max(original_data)

    # 2. Estruturar Grelha
    grid = pv.UniformGrid()
    grid.dimensions = np.array(original_data.shape)
    grid.spacing = voxel_spacing
    grid.point_data["intensities"] = original_data.flatten(order="F")

    plotter = pv.Plotter(off_screen=False)
    plotter.set_background('black')
    plotter.enable_anti_aliasing('fxaa') 
    
    volume = plotter.add_volume(
        grid, 
        scalars="intensities", 
        cmap="gray",          
        shade=True,           
        ambient=0.32,         
        diffuse=0.8,          
        specular=0.5,         
        specular_power=15.0    
    )
    volume.prop.interpolation_type = 'linear'
    
    # 3. Estado local para tracking dos valores
    state = {
        "min_opacity": 15.0,
        "max_opacity": 70.0,
        "smoothing": 0.0,
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

    # 4. Callbacks
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
            print(f"A aplicar suavização (Sigma={value:.2f})...")
            smoothed = gaussian(original_data, sigma=value)
            grid.point_data["intensities"] = smoothed.flatten(order="F")
        else:
            grid.point_data["intensities"] = original_data.flatten(order="F")
        grid.Modified()
        plotter.render()

    def print_params():
        print("\n" + "="*40)
        print("COPIA ESTES VALORES PARA O CHAT:")
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
        print("\nCONFIGURACAO DA CAMARA (FUNDAMENTAL):")
        print(f"CAMERA_POSITION = {plotter.camera_position}")
        print("="*40)

    plotter.add_key_event('p', print_params)

    # 5. Sliders
    plotter.add_slider_widget(update_ambient, [0.0, 1.0], value=0.32, title="Brilho Base",
                              pointa=(0.02, 0.92), pointb=(0.22, 0.92), style='modern')
    plotter.add_slider_widget(update_diffuse, [0.0, 1.0], value=0.8, title="Luz Difusa",
                              pointa=(0.02, 0.78), pointb=(0.22, 0.78), style='modern')
    plotter.add_slider_widget(update_specular, [0.0, 1.0], value=0.5, title="Reflexo (Specular)",
                              pointa=(0.02, 0.64), pointb=(0.22, 0.64), style='modern')
    plotter.add_slider_widget(update_specular_power, [1.0, 50.0], value=15.0, title="Dureza Reflexo",
                              pointa=(0.02, 0.50), pointb=(0.22, 0.50), style='modern')
    plotter.add_slider_widget(apply_smoothing, [0.0, 2.0], value=0.0, title="Suavizacao (Gauss)",
                              pointa=(0.02, 0.36), pointb=(0.22, 0.36), style='modern')
    plotter.add_slider_widget(update_min, [vol_min, vol_max/4], value=state["min_opacity"], title="Transparencia",
                              pointa=(0.77, 0.92), pointb=(0.97, 0.92), style='modern')
    plotter.add_slider_widget(update_max, [vol_min, vol_max/2], value=state["max_opacity"], title="Solidez",
                              pointa=(0.77, 0.78), pointb=(0.97, 0.78), style='modern')
    plotter.add_slider_widget(update_opacity_unit, [0.01, 1.0], value=state["opacity_unit"], title="Densidade Volume",
                              pointa=(0.77, 0.64), pointb=(0.97, 0.64), style='modern')
    plotter.add_slider_widget(update_sample_dist, [0.05, 1.0], value=state["sample_dist"], title="Qlde Amostragem",
                              pointa=(0.77, 0.50), pointb=(0.97, 0.50), style='modern')

    plotter.camera_position = 'yz'
    plotter.show()

if __name__ == "__main__":
    teste_interativo_raw_spot()
