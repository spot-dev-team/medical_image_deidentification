import nibabel as nib
import numpy as np
import pyvista as pv
from skimage.filters import gaussian

# ================= CONFIGURAÇÃO =================
TEST_FILE = r"D:\Github Projects\medical_image_deidentification\teste_completo_tentativa_1\pares_exames_non_def_and_def\051_S_1331__2007-04-26_09_25_480__I51349\051_S_1331__2007-04-26_09_25_480__I51349_original.nii.gz"
# ================================================

def teste_interativo_raw_spot():
    print(f"A carregar volume RAW para a Spot: {TEST_FILE}")
    
    # 1. Carregar matriz e geometria (SEM NORMALIZAÇÃO)
    nifti = nib.load(TEST_FILE)
    original_data = nifti.get_fdata()
    voxel_spacing = nifti.header.get_zooms()[:3]
    
    vol_min = np.min(original_data)
    vol_max = np.max(original_data)
    print(f"Intensidades reais da matriz: Mínimo = {vol_min:.2f}, Máximo = {vol_max:.2f}")

    # 2. Estruturar Grelha
    grid = pv.UniformGrid()
    grid.dimensions = np.array(original_data.shape)
    grid.spacing = voxel_spacing
    grid.point_data["intensities"] = original_data.flatten(order="F")

    plotter = pv.Plotter(off_screen=False)
    plotter.set_background('black')
    plotter.enable_anti_aliasing('fxaa') # Suaviza as arestas no ecrã
    
    volume = plotter.add_volume(
        grid, 
        scalars="intensities", 
        cmap="gray",          
        shade=True,           
        ambient=0.32,         # Ponto de partida baseado no teu teste
        diffuse=0.8,          
        specular=0.5,         
        specular_power=15.0    
    )

    volume.prop.interpolation_type = 'linear'
    # Forçar qualidade de amostragem (pode ser pesado, mas fica liso)
    if hasattr(volume.mapper, 'sample_distance'):
        volume.mapper.sample_distance = 0.1 

    # 3. Lógica da Opacidade
    current_min = 15.0 # Baseado no teu screenshot
    current_max = 70.0 # Baseado no teu screenshot

    def atualizar_curva_opacidade():
        pwf = volume.prop.GetScalarOpacity()
        pwf.RemoveAllPoints()
        pwf.AddPoint(vol_min, 0.0)
        pwf.AddPoint(current_min, 0.0)
        pwf.AddPoint(current_max, 1.0)
        pwf.AddPoint(vol_max, 1.0)

    atualizar_curva_opacidade()

    # 4. Funções de Callback
    def update_ambient(value):
        volume.prop.ambient = value

    def update_min(value):
        nonlocal current_min
        current_min = value
        atualizar_curva_opacidade()

    def update_max(value):
        nonlocal current_max
        current_max = value
        atualizar_curva_opacidade()

    def update_diffuse(value):
        volume.prop.diffuse = value

    def update_specular(value):
        volume.prop.specular = value

    def update_specular_power(value):
        volume.prop.specular_power = value

    def update_opacity_unit(value):
        # NOTA: Baixar este valor remove as "riscas" (staircase artifact)
        volume.prop.scalar_opacity_unit_distance = value

    def update_sample_dist(value):
        # NOTA: Valores pequenos (ex: 0.1) = Máxima Qualidade
        if hasattr(volume.mapper, 'sample_distance'):
            volume.mapper.sample_distance = value

    def apply_smoothing(value):
        # Aplica um filtro de Gauss para remover o ruído "sal e pimenta"
        if value > 0:
            print(f"A aplicar suavização (Sigma={value:.2f})...")
            smoothed = gaussian(original_data, sigma=value)
            grid.point_data["intensities"] = smoothed.flatten(order="F")
            print("Suavização concluída.")
        else:
            grid.point_data["intensities"] = original_data.flatten(order="F")

    def print_params():
        print("\n--- VALORES ATUAIS ---")
        print(f"Ambient: {volume.prop.ambient:.4f}")
        print(f"Diffuse: {volume.prop.diffuse:.4f}")
        print(f"Specular: {volume.prop.specular:.4f}")
        print(f"Specular Power: {volume.prop.specular_power:.4f}")
        print(f"Min (Transparency): {current_min:.4f}")
        print(f"Max (Solidity): {current_max:.4f}")
        print(f"Opacity Unit Distance: {volume.prop.scalar_opacity_unit_distance:.4f}")
        if hasattr(volume.mapper, 'sample_distance'):
            print(f"Sample Distance: {volume.mapper.sample_distance:.4f}")
        print("----------------------")

    plotter.add_key_event('p', print_params)

    # 5. Sliders (Organizados em duas colunas para garantir visibilidade total)
    # Coluna 1 (Esquerda): Iluminação e Suavização
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

    # Coluna 2 (Direita): Transparência, Densidade e Qualidade
    plotter.add_slider_widget(update_min, [vol_min, vol_max/4], value=current_min, title="Transparencia",
                              pointa=(0.77, 0.92), pointb=(0.97, 0.92), style='modern')
                              
    plotter.add_slider_widget(update_max, [vol_min, vol_max/2], value=current_max, title="Solidez",
                              pointa=(0.77, 0.78), pointb=(0.97, 0.78), style='modern')

    plotter.add_slider_widget(update_opacity_unit, [0.01, 1.0], value=0.5, title="Densidade Volume",
                              pointa=(0.77, 0.64), pointb=(0.97, 0.64), style='modern')

    plotter.add_slider_widget(update_sample_dist, [0.05, 1.0], value=0.1, title="Qlde Amostragem",
                              pointa=(0.77, 0.50), pointb=(0.97, 0.50), style='modern')

    print("\nRECOMENDAÇÕES:")
    print("1. Suavização (Esquerda): Tenta subir para 0.6-1.0 para limpar o ruído.")
    print("2. Transparência/Solidez (Direita): Ajusta para definir os contornos da pele.")
    print("3. Pressiona 'p' para ver todos os parâmetros no terminal.")
    
    plotter.camera_position = 'yz'
    plotter.show()

if __name__ == "__main__":
    teste_interativo_raw_spot()
