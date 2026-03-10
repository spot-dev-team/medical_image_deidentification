import os
import shutil
import nibabel as nib
import numpy as np
import pyvista as pv
from skimage.filters import gaussian

# ================= CONFIGURAÇÃO DE CAMINHOS (ESTRUTURA CLUSTER) =================
DIR_DEFACED = "/home/andresousa615/rempe/mede_code/resultados_inferencia"
DIR_ORIGINAL = "/home/andresousa615/rempe/mede_code/processed_datasets/processed_datasets/teste"
DIR_PARES = "/home/andresousa615/rempe/mede_code/after_training/pares_exames_non_def_and_def"

# PARÂMETROS DE RENDERIZAÇÃO (RÉPLICA EXATA DOS TEUS VALORES DE SUCESSO)
AMBIENT = 1.0000
DIFFUSE = 0.4796
SPECULAR = 1.0000
SPECULAR_POWER = 3.6751
MIN_OPACITY = 10.9184
MAX_OPACITY = 15.9029
OPACITY_UNIT = 0.5000
SAMPLE_DIST = 0.1000
SMOOTHING_SIGMA = 0.0000

# CONFIGURAÇÃO DA CÂMARA (RÉPLICA EXATA DA TUA POSIÇÃO)
CAMERA_POSITION = [
    (192.4029043355724, 882.3921319231572, 110.03325486207237),
    (95.40000379085541, 119.5, 127.5),
    (-0.00623714731018042, 0.023681628614305828, 0.9997000942580756)
]
# ================================================================================

def preparar_pares_exames():
    print("A iniciar a organização dos pares (Original vs Defaced)...")
    os.makedirs(DIR_PARES, exist_ok=True)
    count_pares = 0
    
    for f in os.listdir(DIR_DEFACED):
        if not f.endswith("_anon.nii.gz"):
            continue
            
        path_defaced = os.path.join(DIR_DEFACED, f)
        
        # Parsing do ID do exame
        if "_raw_anon" in f: exam_id = f.split("_raw_anon")[0]
        elif "_imagem_anon" in f: exam_id = f.split("_imagem_anon")[0]
        elif "_image_anon" in f: exam_id = f.split("_image_anon")[0]
        else: exam_id = f.replace("_anon.nii.gz", "")
            
        path_orig_folder = os.path.join(DIR_ORIGINAL, exam_id)
        if not os.path.exists(path_orig_folder):
            continue
            
        path_orig_raw = os.path.join(path_orig_folder, "raw.nii.gz")
        path_orig_img = os.path.join(path_orig_folder, "image.nii.gz")
        path_orig = path_orig_raw if os.path.exists(path_orig_raw) else path_orig_img
        
        if not os.path.exists(path_orig):
            continue
            
        target_folder = os.path.join(DIR_PARES, exam_id)
        os.makedirs(target_folder, exist_ok=True)
        
        target_defaced = os.path.join(target_folder, f"{exam_id}_defaced.nii.gz")
        target_original = os.path.join(target_folder, f"{exam_id}_original.nii.gz")
        
        if not os.path.exists(target_defaced): shutil.copy2(path_defaced, target_defaced)
        if not os.path.exists(target_original): shutil.copy2(path_orig, target_original)
        count_pares += 1

    print(f"Organização concluída: {count_pares} pares prontos.\n" + "="*50)


def gerar_renders_lote():
    print("A iniciar o pipeline de renderização (Ray-Casting High Fidelity)...")
    
    for exam_id in os.listdir(DIR_PARES):
        exam_folder = os.path.join(DIR_PARES, exam_id)
        if not os.path.isdir(exam_folder): continue
            
        for f in os.listdir(exam_folder):
            if not f.endswith(".nii.gz") or "_mask" in f: continue
                
            nifti_path = os.path.join(exam_folder, f)
            output_path = os.path.join(exam_folder, f.replace(".nii.gz", "_image.png"))
            
            if os.path.exists(output_path): os.remove(output_path) 
            print(f"A processar: {f}")
            
            try:
                # 1. Carregar Dados (Sem normalização para usar os valores reais do image_v2)
                nifti = nib.load(nifti_path)
                data = nifti.get_fdata()
                voxel_spacing = nifti.header.get_zooms()[:3]
                vol_min, vol_max = np.min(data), np.max(data)

                # 2. Suavização (Opcional, definida como 0.0)
                if SMOOTHING_SIGMA > 0.01:
                    data = gaussian(data, sigma=SMOOTHING_SIGMA)

                # 3. Estruturar Grelha
                grid = pv.ImageData()
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
                    show_scalar_bar=False
                )

                # Propriedades e Compatibilidade VTK
                volume.prop.interpolation_type = 'linear'
                if hasattr(volume.prop, 'SetScalarOpacityUnitDistance'):
                    volume.prop.SetScalarOpacityUnitDistance(OPACITY_UNIT)
                if hasattr(volume.mapper, 'SetSampleDistance'):
                    volume.mapper.SetSampleDistance(SAMPLE_DIST)

                # Curva de Opacidade (Rampa cirúrgica baseada nos teus valores)
                pwf = volume.prop.GetScalarOpacity()
                pwf.RemoveAllPoints()
                pwf.AddPoint(vol_min, 0.0)
                pwf.AddPoint(MIN_OPACITY, 0.0)
                pwf.AddPoint(MAX_OPACITY, 1.0)
                pwf.AddPoint(vol_max, 1.0)

                # 5. Câmara (Réplica Exata da tua posição de sucesso)
                plotter.camera_position = CAMERA_POSITION

                # 6. Screenshot
                plotter.screenshot(output_path)
                plotter.close()

            except Exception as e:
                print(f"[Erro] Falha em {f}: {e}")

    print("Pipeline de lote concluído com sucesso!")

if __name__ == "__main__":
    preparar_pares_exames()
    gerar_renders_lote()
