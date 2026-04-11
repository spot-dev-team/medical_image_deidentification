import os
import numpy as np
import nibabel as nib
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from skimage.transform import resize
from matplotlib.colors import ListedColormap

# ================= CONFIGURAÇÃO =================
INFERENCE_DIR_OLD = r"E:\Tese\Datasets\Rempe\processed_datasets\processed_datasets\inference_test_masks"
INFERENCE_DIR_NEW = r"D:\Github Projects\medical_image_deidentification\treinos\treino_bs2_new_dataset2\inference_test_masks"
BASE_DIR_ORIGINAL = r"E:\Tese\Datasets\Rempe\processed_datasets\processed_datasets\teste_noise_cleanse_v2"
# ================================================

class DualInferenceReviewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Inference Reviewer - Diagnóstico V2 (Antiga vs Nova)")
        self.root.geometry("1400x900")
        
        self.exam_pairs = []
        self.current_index = 0
        
        # Modos: "ALL", "GT_VS_NEW", "GT_VS_OLD"
        self.view_mode = "ALL" 
        
        # Paletas de cores (Otimizadas para contraste)
        self.cmap_gt = ListedColormap(['red'])
        self.cmap_old = ListedColormap(['#00BFFF']) # Azul Celeste
        self.cmap_new = ListedColormap(['#39FF14']) # Verde Neon
        
        # UI Elements
        self.lbl_info = tk.Label(root, text="A carregar dataset...", font=("Arial", 14, "bold"))
        self.lbl_info.pack(pady=5)

        self.lbl_diff_status = tk.Label(root, text="A calcular diferenças...", font=("Arial", 12, "bold"))
        self.lbl_diff_status.pack(pady=2)

        self.btn_toggle = tk.Button(root, text="Mudar Visualização (Atual: TODOS (GT + Antiga + Nova))", 
                                    font=("Arial", 12), bg="#444444", fg="white", command=self.toggle_mode)
        self.btn_toggle.pack(pady=5)
        
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 7))
        self.fig.patch.set_facecolor('#000000') 
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        
        self.lbl_legend = tk.Label(root, text="LEGENDA: VERMELHO = GT | AZUL = Antiga | VERDE = Nova", 
                                  font=("Arial", 12, "bold"), fg="white", bg="#333333")
        self.lbl_legend.pack(pady=5, fill=tk.X)

        self.root.bind("<Left>", self.prev_exam)
        self.root.bind("<Right>", self.next_exam)
        self.root.bind("<space>", lambda e: self.toggle_mode()) 
        
        self.load_dataset()
        self.show_current()

    def toggle_mode(self):
        # Alternar ciclicamente entre os 3 modos
        if self.view_mode == "ALL":
            self.view_mode = "GT_VS_NEW"
            self.btn_toggle.config(text="Mudar Visualização (Atual: GT vs NOVA)")
            self.lbl_legend.config(text="LEGENDA: VERMELHO = GT | VERDE NEON = Inferência Nova")
        
        elif self.view_mode == "GT_VS_NEW":
            self.view_mode = "GT_VS_OLD"
            self.btn_toggle.config(text="Mudar Visualização (Atual: GT vs ANTIGA)")
            self.lbl_legend.config(text="LEGENDA: VERMELHO = GT | AZUL CELESTE = Inferência Antiga")
            
        else:
            self.view_mode = "ALL"
            self.btn_toggle.config(text="Mudar Visualização (Atual: TODOS (GT + Antiga + Nova))")
            self.lbl_legend.config(text="LEGENDA: VERMELHO = GT | AZUL = Antiga | VERDE = Nova")
        
        self.show_current()

    def load_dataset(self):
        print(f"--- A carregar e a cruzar dados ---")
        
        if not os.path.exists(INFERENCE_DIR_NEW) or not os.path.exists(INFERENCE_DIR_OLD):
            print("ERRO: Verifica se os caminhos das duas pastas de inferência estão corretos!")
            return

        for filename in os.listdir(INFERENCE_DIR_NEW):
            if filename.endswith("_pred_mask.nii") or filename.endswith("_pred_mask.nii.gz"):
                
                # Extrair o exam_id
                if filename.endswith("_pred_mask.nii"):
                    exam_id = filename.replace("_pred_mask.nii", "")
                else:
                    exam_id = filename.replace("_pred_mask.nii.gz", "")

                # Definir caminhos Inferência
                path_new_pred = os.path.join(INFERENCE_DIR_NEW, filename)
                path_old_pred = os.path.join(INFERENCE_DIR_OLD, filename)
                
                # Definir caminhos Originais
                path_orig_folder = os.path.join(BASE_DIR_ORIGINAL, exam_id)
                path_orig_gt = os.path.join(path_orig_folder, "mask_GT.nii.gz")
                if not os.path.exists(path_orig_gt):
                    path_orig_gt = os.path.join(path_orig_folder, "mask_GT.nii") 

                path_orig_img = None
                if os.path.exists(path_orig_folder):
                    for img_name in ["image.nii.gz", "raw.nii.gz", "image.nii", "raw.nii"]:
                        p = os.path.join(path_orig_folder, img_name)
                        if os.path.exists(p): 
                            path_orig_img = p
                            break
                
                # Só adiciona se a previsão nova, velha, GT e imagem existirem
                if path_orig_img and os.path.exists(path_orig_gt) and os.path.exists(path_new_pred) and os.path.exists(path_old_pred):
                    self.exam_pairs.append({
                        "name": exam_id,
                        "orig_img": path_orig_img,
                        "orig_gt": path_orig_gt,
                        "old_pred": path_old_pred,
                        "new_pred": path_new_pred
                    })

        print(f"Encontrados {len(self.exam_pairs)} exames em comum entre as três pastas.")

    def plot_mask_overlay(self, ax, mask_slice, cmap, alpha):
        if np.any(mask_slice > 0):
            masked_data = np.ma.masked_where(mask_slice == 0, mask_slice)
            ax.imshow(masked_data, cmap=cmap, alpha=alpha, interpolation='nearest', vmin=0, vmax=1)

    def show_current(self):
        if not self.exam_pairs: return
        data = self.exam_pairs[self.current_index]
        self.lbl_info.config(text=f"Exame [{self.current_index + 1}/{len(self.exam_pairs)}]: {data['name']}")
        
        try:
            # Carregar ficheiros
            orig_img = nib.load(data["orig_img"]).get_fdata()
            orig_gt_raw = nib.load(data["orig_gt"]).get_fdata()
            old_pred_raw = nib.load(data["old_pred"]).get_fdata()
            new_pred_raw = nib.load(data["new_pred"]).get_fdata()
            
            target_shape = orig_img.shape

            def safe_resize(mask, target):
                if mask.shape != target:
                    return resize(mask, target, order=0, preserve_range=True, anti_aliasing=False)
                return mask

            orig_gt_mask = safe_resize(orig_gt_raw, target_shape)
            old_pred_mask = safe_resize(old_pred_raw, target_shape)
            new_pred_mask = safe_resize(new_pred_raw, target_shape)

            orig_gt_bin = (orig_gt_mask > 0).astype(np.uint8)
            old_pred_bin = (old_pred_mask > 0).astype(np.uint8)
            new_pred_bin = (new_pred_mask > 0).astype(np.uint8)

            # === ESTATÍSTICA: ANTIGO VS NOVO ===
            diff = np.sum(old_pred_bin != new_pred_bin)
            if diff == 0:
                self.lbl_diff_status.config(text="Comparação: Inferência Nova e Antiga são EXATAMENTE IGUAIS", fg="white")
            else:
                self.lbl_diff_status.config(text=f"Comparação: Modelo Novo difere do Antigo em {diff} voxels", fg="yellow")

            # Preparar a imagem de fundo
            p2, p98 = np.percentile(orig_img, (2, 98))
            orig_img = np.clip(orig_img, p2, p98)
            orig_img = (orig_img - p2) / (p98 - p2 + 1e-8)
            
            # Centro de massa (Focus)
            if np.sum(orig_gt_bin) > 0:
                cx, cy, cz = [int(c) for c in center_of_mass(orig_gt_bin)]
            else:
                cx, cy, cz = [s // 2 for s in target_shape]

            # Decidir que máscaras desenhar de acordo com o modo
            masks_to_draw = []
            if self.view_mode == "ALL":
                masks_to_draw.append((orig_gt_bin, self.cmap_gt))
                masks_to_draw.append((old_pred_bin, self.cmap_old))
                masks_to_draw.append((new_pred_bin, self.cmap_new))
            elif self.view_mode == "GT_VS_NEW":
                masks_to_draw.append((orig_gt_bin, self.cmap_gt))
                masks_to_draw.append((new_pred_bin, self.cmap_new))
            elif self.view_mode == "GT_VS_OLD":
                masks_to_draw.append((orig_gt_bin, self.cmap_gt))
                masks_to_draw.append((old_pred_bin, self.cmap_old))

            for ax in self.axes:
                ax.clear()
                ax.axis('off')
                
            # AXIAL
            self.axes[0].imshow(np.rot90(orig_img[:, :, cz]), cmap='gray')
            for m, cmap in masks_to_draw:
                self.plot_mask_overlay(self.axes[0], np.rot90(m[:, :, cz]), cmap, 0.45)
            
            # CORONAL
            self.axes[1].imshow(np.rot90(orig_img[cx, :, :]), cmap='gray')
            for m, cmap in masks_to_draw:
                self.plot_mask_overlay(self.axes[1], np.rot90(m[cx, :, :]), cmap, 0.45)

            # SAGITAL
            self.axes[2].imshow(np.rot90(orig_img[:, cy, :]), cmap='gray')
            for m, cmap in masks_to_draw:
                self.plot_mask_overlay(self.axes[2], np.rot90(m[:, cy, :]), cmap, 0.45)

            self.canvas.draw()
            
        except Exception as e:
            print(f"Erro ao processar {data['name']}: {e}")

    def next_exam(self, event=None):
        self.current_index = (self.current_index + 1) % len(self.exam_pairs)
        self.show_current()

    def prev_exam(self, event=None):
        self.current_index = (self.current_index - 1) % len(self.exam_pairs)
        self.show_current()

if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg='#000000')
    app = DualInferenceReviewer(root)
    root.mainloop()