import os
import shutil
import nibabel as nib
import numpy as np
import pyvista as pv

# ================= CONFIGURAÇÃO =================
DIR_DEFACED = r"E:\Tese\Datasets\Rempe\treino_3\resultados_inferencia"
DIR_ORIGINAL = r"E:\Tese\Datasets\Rempe\processed_datasets\teste"
DIR_PARES = r"E:\Tese\Datasets\Rempe\pares_exames_non_def_and_def"
# ================================================

def preparar_pares_exames():
    print("A iniciar a organização dos pares (Original vs Defaced)...")
    os.makedirs(DIR_PARES, exist_ok=True)
    
    count_pares = 0
    
    for f in os.listdir(DIR_DEFACED):
        if not f.endswith("_anon.nii.gz"):
            continue
            
        path_defaced = os.path.join(DIR_DEFACED, f)
        
        # 1. Parsing robusto do nome do paciente (XXX)
        if "_raw_anon" in f:
            exam_id = f.split("_raw_anon")[0]
        elif "_imagem_anon" in f:
            exam_id = f.split("_imagem_anon")[0]
        elif "_image_anon" in f:
            exam_id = f.split("_image_anon")[0]
        else:
            exam_id = f.replace("_anon.nii.gz", "")
            
        # 2. Localizar o exame original
        path_orig_folder = os.path.join(DIR_ORIGINAL, exam_id)
        if not os.path.exists(path_orig_folder):
            print(f"[Aviso] Pasta original não encontrada para: {exam_id}")
            continue
            
        path_orig_raw = os.path.join(path_orig_folder, "raw.nii.gz")
        path_orig_img = os.path.join(path_orig_folder, "image.nii.gz")
        
        # Assume o ficheiro que existir na pasta original
        path_orig = path_orig_raw if os.path.exists(path_orig_raw) else path_orig_img
        
        if not os.path.exists(path_orig):
            print(f"[Aviso] Ficheiro NIfTI original não encontrado em: {path_orig_folder}")
            continue
            
        # 3. Criar a nova subpasta para o par
        target_folder = os.path.join(DIR_PARES, exam_id)
        os.makedirs(target_folder, exist_ok=True)
        
        # 4. Copiar os ficheiros com nomes normalizados para facilitar o tracking
        target_defaced = os.path.join(target_folder, f"{exam_id}_defaced.nii.gz")
        target_original = os.path.join(target_folder, f"{exam_id}_original.nii.gz")
        
        if not os.path.exists(target_defaced):
            shutil.copy2(path_defaced, target_defaced)
        if not os.path.exists(target_original):
            shutil.copy2(path_orig, target_original)
            
        count_pares += 1

    print(f"Organização concluída: {count_pares} pares criados.\n" + "="*50)


def gerar_renders_lote():
    print("A iniciar o pipeline de renderização em lote...")
    
    # Percorrer a nova estrutura de pastas
    for exam_id in os.listdir(DIR_PARES):
        exam_folder = os.path.join(DIR_PARES, exam_id)
        
        if not os.path.isdir(exam_folder):
            continue
            
        for f in os.listdir(exam_folder):
            if not f.endswith(".nii.gz"):
                continue
                
            nifti_path = os.path.join(exam_folder, f)
            base_name = f.replace(".nii.gz", "")
            output_path = os.path.join(exam_folder, f"{base_name}_image.png")
            
            # Se a imagem já existir, avança para o próximo
            if os.path.exists(output_path):
                os.remove(output_path) # Força a re-renderização por cima da antiga
                
            print(f"A renderizar: {base_name}")
            
            try:
                # 1. Carregar e Normalizar
                nifti = nib.load(nifti_path)
                data = nifti.get_fdata()
                voxel_spacing = nifti.header.get_zooms()[:3]
                data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

                # 2. Estruturar Grelha
                grid = pv.ImageData()
                grid.dimensions = np.array(data.shape)
                grid.spacing = voxel_spacing
                grid.point_data["intensities"] = data_norm.flatten(order="F")

                # 3. Extração Geométrica (o teu valor otimizado)
                surface = grid.contour([0.06], scalars="intensities") #0.04

                # 4. Renderização Off-screen
                plotter = pv.Plotter(off_screen=True)
                plotter.add_mesh(surface, color='bisque', smooth_shading=True, specular=0.0)

                # 5. Câmara RAS
                center = surface.center
                camera_pos = (center[0] + 50, center[1] + 500, center[2]) 
                
                plotter.camera_position = [
                    camera_pos, 
                    center,     
                    (0, 0, 1)   
                ]

                # 6. Gravar e libertar memória (crítico para loops)
                plotter.screenshot(output_path)
                plotter.close()

            except Exception as e:
                print(f"[Erro] Falha ao processar {f}: {e}")

    print("Renderização de lote concluída com sucesso!")

if __name__ == "__main__":
    #preparar_pares_exames()
    gerar_renders_lote()