import os
import shutil
import nibabel as nib
import numpy as np
import pyvista as pv
from skimage.filters import gaussian
import argparse
import torch.distributed as dist

# ================= PARÂMETROS DE RENDERIZAÇÃO =================
AMBIENT = 1.3646
DIFFUSE = 1.6875
SPECULAR = 1.3854
SPECULAR_POWER = 4.6094
SAMPLE_DIST = 0.1070
SMOOTHING_SIGMA = 0.4

CAMERA_POSITION = [
    (95.40000379085541, 887.52, 84.39690110118934),  
    (95.40000379085541, 119.5, 127.5),               
    (0.0, 0.0, 1.0)                                  
]
# ==============================================================

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def calcular_parametros_dinamicos(data_matrix, voxel_spacing, percentil_ar=20, percentil_pele=35):
    limiar_ar = np.percentile(data_matrix, percentil_ar)
    tecido_real = data_matrix[data_matrix > limiar_ar]
    
    min_op = float(np.percentile(tecido_real, percentil_pele))
    max_op = float(np.percentile(tecido_real, 99))
    
    distancia_fisica = np.linalg.norm(voxel_spacing)
    op_unit = float(distancia_fisica * 0.5)
    
    return min_op, max_op, op_unit

def preparar_pares_exames(dir_defaced, dir_original, dir_pares):
    """
    Apenas o Rank 0 deve executar isto para evitar colisões na cópia de ficheiros.
    """
    print("A iniciar a organização dos pares (Original vs Predição)...")
    os.makedirs(dir_pares, exist_ok=True)
    count_pares = 0
    
    for f in os.listdir(dir_defaced):
        # LÓGICA MÉDICA ORIGINAL: Procurar APENAS o exame já anonimizado
        if not f.endswith("_anon.nii.gz"):
            continue
            
        path_defaced = os.path.join(dir_defaced, f)
        
        # Extracção do ID limpo
        exam_id = f.replace("_anon.nii.gz", "")
            
        # 1. TENTATIVA A: Formato Subpasta
        path_orig_folder = os.path.join(dir_original, exam_id)
        path_orig = None
        
        if os.path.exists(path_orig_folder) and os.path.isdir(path_orig_folder):
            if os.path.exists(os.path.join(path_orig_folder, "raw.nii.gz")):
                path_orig = os.path.join(path_orig_folder, "raw.nii.gz")
            elif os.path.exists(os.path.join(path_orig_folder, "image.nii.gz")):
                path_orig = os.path.join(path_orig_folder, "image.nii.gz")
                
        # 2. TENTATIVA B: Formato Ficheiros Soltos
        if path_orig is None:
            possibilidades = [
                os.path.join(dir_original, f"{exam_id}.nii.gz"),
                os.path.join(dir_original, f"{exam_id}_image.nii.gz"),
                os.path.join(dir_original, f"{exam_id}_0000.nii.gz"),
                os.path.join(dir_original, f"{exam_id}_raw.nii.gz")
            ]
            for p in possibilidades:
                if os.path.exists(p):
                    path_orig = p
                    break
        
        # 3. VALIDAÇÃO E CÓPIA DIRETAS (A Tua Lógica Original)
        if path_orig is None:
            print(f"[Aviso] Original não encontrado para o ID: {exam_id}. Ignorado!")
            continue
            
        target_folder = os.path.join(dir_pares, exam_id)
        os.makedirs(target_folder, exist_ok=True)
        
        target_defaced = os.path.join(target_folder, f"{exam_id}_defaced.nii.gz")
        target_original = os.path.join(target_folder, f"{exam_id}_original.nii.gz")
        
        if not os.path.exists(target_defaced): shutil.copy2(path_defaced, target_defaced)
        if not os.path.exists(target_original): shutil.copy2(path_orig, target_original)
        
        count_pares += 1

    print(f"Organização concluída: {count_pares} pares prontos.\n" + "="*50)


def gerar_renders_lote(dir_pares):
    rank = get_rank()
    world_size = get_world_size()
    
    if rank == 0:
        print("A iniciar o pipeline de renderização Paralelizado...")
    
    todos_exames = sorted([d for d in os.listdir(dir_pares) if os.path.isdir(os.path.join(dir_pares, d))])
    
    if len(todos_exames) == 0:
        if rank == 0:
            print("Nenhum exame encontrado para renderizar. A abortar Fase 4.")
        return

    chunks = np.array_split(todos_exames, world_size)
    meus_exames = chunks[rank]
    
    print(f"[Rank {rank}] Vai processar {len(meus_exames)} exames.")
    
    for exam_id in meus_exames:
        exam_folder = os.path.join(dir_pares, exam_id)
        
        for f in os.listdir(exam_folder):
            if not f.endswith(".nii.gz") or "_mask" in f: continue
                
            nifti_path = os.path.join(exam_folder, f)
            output_path = os.path.join(exam_folder, f.replace(".nii.gz", "_image.png"))
            
            if os.path.exists(output_path): os.remove(output_path) 
            
            try:
                nifti = nib.load(nifti_path)
                data = nifti.get_fdata().astype(np.float32)
                voxel_spacing = nifti.header.get_zooms()[:3]
                vol_min, vol_max = np.min(data), np.max(data)

                min_op, max_op, op_unit = calcular_parametros_dinamicos(data, voxel_spacing)

                if SMOOTHING_SIGMA > 0.01:
                    data = gaussian(data, sigma=SMOOTHING_SIGMA)

                grid = pv.ImageData()
                grid.dimensions = np.array(data.shape)
                grid.spacing = voxel_spacing
                grid.point_data["intensities"] = data.flatten(order="F")

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
                
                if hasattr(volume.mapper, 'SetAutoAdjustSampleDistances'):
                    volume.mapper.SetAutoAdjustSampleDistances(0)

                volume.prop.interpolation_type = 'linear'
                if hasattr(volume.prop, 'SetScalarOpacityUnitDistance'):
                    volume.prop.SetScalarOpacityUnitDistance(op_unit)
                if hasattr(volume.mapper, 'SetSampleDistance'):
                    volume.mapper.SetSampleDistance(SAMPLE_DIST)

                pwf = volume.prop.GetScalarOpacity()
                pwf.RemoveAllPoints()
                pwf.AddPoint(vol_min, 0.0)
                pwf.AddPoint(min_op, 0.0)
                pwf.AddPoint(max_op, 1.0)
                pwf.AddPoint(vol_max, 1.0)

                plotter.reset_camera()
                plotter.camera_position = CAMERA_POSITION

                plotter.screenshot(output_path)
                plotter.close()

            except Exception as e:
                print(f"[Rank {rank}][Erro] Falha em {f}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de renderização 2D distribuída")
    parser.add_argument("--dir_defaced", type=str, required=True, help="Pasta contendo os ficheiros anonimizados (_anon.nii.gz)")
    parser.add_argument("--dir_original", type=str, required=True, help="Pasta contendo os datasets originais completos")
    parser.add_argument("--dir_pares", type=str, required=True, help="Pasta de saída para organizar pares e PNGs")
    args = parser.parse_args()

    if "WORLD_SIZE" in os.environ:
        import torch.distributed as dist
        dist.init_process_group(backend="gloo")

    if get_rank() == 0:
        preparar_pares_exames(args.dir_defaced, args.dir_original, args.dir_pares)
    
    if is_dist_avail_and_initialized():
        dist.barrier()
        
    gerar_renders_lote(args.dir_pares)

    if is_dist_avail_and_initialized():
        dist.destroy_process_group()