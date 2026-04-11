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

# ================= PARÂMETROS DE RENDERIZAÇÃO (Vindos de generate_single_exam_auto_configs.py) =================
AMBIENT = 1.3646
DIFFUSE = 1.6875
SPECULAR = 1.3854
SPECULAR_POWER = 4.6094
SAMPLE_DIST = 0.1070
SMOOTHING_SIGMA = 0.4

# CONFIGURAÇÃO DA CÂMARA DO CLUSTER
CAMERA_POSITION = [
    (95.40000379085541, 887.52, 84.39690110118934),  # Posição da câmara ajustada
    (95.40000379085541, 119.5, 127.5),               # Ponto de foco inalterado
    (0.0, 0.0, 1.0)                                  # Vetor Up nivelado
]

def calcular_parametros_dinamicos(data_matrix, voxel_spacing, percentil_ar=20, percentil_pele=35):
    """
    percentil_ar: Ignora o fundo escuro (default 5). Se o exame for muito ruidoso, sobe para 8 ou 10.
    percentil_pele: Define o corte do min_opacity. Sobe de 10 para 15 ou 20 para eliminar a estática.
    """
    # 1. Isolar o tecido do paciente
    limiar_ar = np.percentile(data_matrix, percentil_ar)
    tecido_real = data_matrix[data_matrix > limiar_ar]
    
    # 2. Calcular Percentis para definir os limiares (Aqui atuamos no ruído)
    min_op = float(np.percentile(tecido_real, percentil_pele))
    max_op = float(np.percentile(tecido_real, 99))
    
    distancia_fisica = np.linalg.norm(voxel_spacing)
    op_unit = float(distancia_fisica * 0.5)
    
    return min_op, max_op, op_unit

    

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
    print("A iniciar o pipeline de renderização (Ray-Casting High Fidelity com Lógica Dinâmica)...")
    
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
                # 1. Carregar Dados
                nifti = nib.load(nifti_path)
                data = nifti.get_fdata().astype(np.float32)
                voxel_spacing = nifti.header.get_zooms()[:3]
                vol_min, vol_max = np.min(data), np.max(data)

                # 2. Calcular Parâmetros Dinâmicos
                min_op, max_op, op_unit = calcular_parametros_dinamicos(data, voxel_spacing)
                print(f" -> Dinâmico: MinOp {min_op:.2f} | MaxOp {max_op:.2f} | Unit {op_unit:.4f}")

                # 3. Suavização
                if SMOOTHING_SIGMA > 0.01:
                    data = gaussian(data, sigma=SMOOTHING_SIGMA)

                # 4. Estruturar Grelha
                grid = pv.ImageData()
                grid.dimensions = np.array(data.shape)
                grid.spacing = voxel_spacing
                grid.point_data["intensities"] = data.flatten(order="F")

                # 5. Configurar Plotter
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

                # Forçar RayCast (Software) para estabilidade no Cluster
                #if hasattr(volume.mapper, 'SetRequestedRenderModeToRayCast'):
                #    volume.mapper.SetRequestedRenderModeToRayCast()
                
                if hasattr(volume.mapper, 'SetAutoAdjustSampleDistances'):
                    volume.mapper.SetAutoAdjustSampleDistances(0)

                # Propriedades
                volume.prop.interpolation_type = 'linear'
                if hasattr(volume.prop, 'SetScalarOpacityUnitDistance'):
                    volume.prop.SetScalarOpacityUnitDistance(op_unit)
                if hasattr(volume.mapper, 'SetSampleDistance'):
                    volume.mapper.SetSampleDistance(SAMPLE_DIST)

                # Curva de Opacidade Dinâmica
                pwf = volume.prop.GetScalarOpacity()
                pwf.RemoveAllPoints()
                pwf.AddPoint(vol_min, 0.0)
                pwf.AddPoint(min_op, 0.0)
                pwf.AddPoint(max_op, 1.0)
                pwf.AddPoint(vol_max, 1.0)

                # 6. Câmara (reset_camera PRIMEIRO)
                plotter.reset_camera()
                plotter.camera_position = CAMERA_POSITION

                # 7. Screenshot
                plotter.screenshot(output_path)
                plotter.close()

            except Exception as e:
                print(f"[Erro] Falha em {f}: {e}")

    print("Pipeline de lote concluído com sucesso!")

if __name__ == "__main__":
    preparar_pares_exames()
    gerar_renders_lote()
