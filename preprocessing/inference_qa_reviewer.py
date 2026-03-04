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
INFERENCE_DIR = r"D:\\Github Projects\\medical_image_deidentification\\teste_completo_tentativa_1\\inference_test_masks"
BASE_DIR_ORIGINAL = r"E:\\Tese\\Datasets\\Rempe\\teste"
# ================================================

class InferenceMaskReviewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Inference Reviewer - Diagnóstico de Máscara")
        self.root.geometry("1400x850")
        
        self.exam_pairs = []
        self.current_index = 0
        
        self.cmap_gt = ListedColormap(['red'])
        self.cmap_pred = ListedColormap(['#00FF00']) 
        
        self.lbl_info = tk.Label(root, text="A carregar dataset...", font=("Arial", 14, "bold"))
        self.lbl_info.pack(pady=5)
        
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 7))
        self.fig.patch.set_facecolor('#000000') 
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        
        self.lbl_legend = tk.Label(root, text="LEGENDA: VERMELHO = GT | VERDE NEON = Predição", 
                                  font=("Arial", 12, "bold"), fg="white", bg="#333333")
        self.lbl_legend.pack(pady=5, fill=tk.X)

        self.root.bind("<Left>", self.prev_exam)
        self.root.bind("<Right>", self.next_exam)
        
        self.load_dataset()
        self.show_current()

    def load_dataset(self):
        print(f"--- A procurar inferências em: {INFERENCE_DIR} ---")
        suffixes = ["_pred_mask.nii.gz", "_raw_mask.nii.gz", "_image_mask.nii.gz", "_imagem_mask.nii.gz", "_mask.nii.gz"]

        for filename in os.listdir(INFERENCE_DIR):
            if filename.endswith(".nii.gz"):
                exam_id = filename
                matched = False
                for suffix in suffixes:
                    if exam_id.endswith(suffix):
                        exam_id = exam_id.rsplit(suffix, 1)[0]
                        matched = True
                        break
                
                if not matched: continue

                path_pred_mask = os.path.join(INFERENCE_DIR, filename)
                path_orig_folder = os.path.join(BASE_DIR_ORIGINAL, exam_id)
                
                if os.path.exists(path_orig_folder):
                    path_gt_mask = os.path.join(path_orig_folder, "mask_GT.nii.gz")
                    path_orig_img = None
                    for f in ["image.nii.gz", "raw.nii.gz"]:
                        p = os.path.join(path_orig_folder, f)
                        if os.path.exists(p): path_orig_img = p; break
                    
                    if path_orig_img and os.path.exists(path_gt_mask):
                        self.exam_pairs.append({
                            "name": exam_id,
                            "orig_path": path_orig_img,
                            "gt_mask_path": path_gt_mask,
                            "pred_mask_path": path_pred_mask
                        })

        print(f"Encontrados {len(self.exam_pairs)} pares válidos.")

    def plot_mask_overlay(self, ax, mask_slice, cmap, alpha):
        if np.any(mask_slice > 0):
            masked_data = np.ma.masked_where(mask_slice == 0, mask_slice)
            ax.imshow(masked_data, cmap=cmap, alpha=alpha, interpolation='nearest', vmin=0, vmax=1)
            return True
        return False

    def show_current(self):
        if not self.exam_pairs: return
        data = self.exam_pairs[self.current_index]
        self.lbl_info.config(text=f"Exame [{self.current_index + 1}/{len(self.exam_pairs)}]: {data['name']}")
        
        try:
            orig_img = nib.load(data["orig_path"]).get_fdata()
            gt_mask_raw = nib.load(data["gt_mask_path"]).get_fdata()
            pred_mask_raw = nib.load(data["pred_mask_path"]).get_fdata()
            
            # DIAGNÓSTICO: Ver o que está dentro do ficheiro de predição
            print(f"\n[DEBUG] Exame: {data['name']}")
            print(f"  > Pred Mask - Max: {np.max(pred_mask_raw)}, Soma: {np.sum(pred_mask_raw > 0)}")
            print(f"  > GT Mask   - Max: {np.max(gt_mask_raw)}, Soma: {np.sum(gt_mask_raw > 0)}")

            p2, p98 = np.percentile(orig_img, (2, 98))
            orig_img = np.clip(orig_img, p2, p98)
            orig_img = (orig_img - p2) / (p98 - p2 + 1e-8)
            
            target_shape = orig_img.shape
            
            # Resize
            gt_mask = resize(gt_mask_raw, target_shape, order=0, preserve_range=True, anti_aliasing=False) if gt_mask_raw.shape != target_shape else gt_mask_raw
            pred_mask = resize(pred_mask_raw, target_shape, order=0, preserve_range=True, anti_aliasing=False) if pred_mask_raw.shape != target_shape else pred_mask_raw

            gt_bin = (gt_mask > 0).astype(np.uint8)
            pred_bin = (pred_mask > 0).astype(np.uint8)
            
            if np.sum(gt_bin) > 0:
                cx, cy, cz = [int(c) for c in center_of_mass(gt_bin)]
            elif np.sum(pred_bin) > 0:
                cx, cy, cz = [int(c) for c in center_of_mass(pred_bin)]
            else:
                cx, cy, cz = [s // 2 for s in target_shape]

            for ax in self.axes:
                ax.clear()
                ax.axis('off')
                
            # Renderização com opacidade reforçada para o verde
            self.axes[0].imshow(np.rot90(orig_img[:, :, cz]), cmap='gray')
            self.plot_mask_overlay(self.axes[0], np.rot90(gt_bin[:, :, cz]), self.cmap_gt, 0.7)
            self.plot_mask_overlay(self.axes[0], np.rot90(pred_bin[:, :, cz]), self.cmap_pred, 0.7)
            
            self.axes[1].imshow(np.rot90(orig_img[cx, :, :]), cmap='gray')
            self.plot_mask_overlay(self.axes[1], np.rot90(gt_bin[cx, :, :]), self.cmap_gt, 0.7)
            self.plot_mask_overlay(self.axes[1], np.rot90(pred_bin[cx, :, :]), self.cmap_pred, 0.7)

            self.axes[2].imshow(np.rot90(orig_img[:, cy, :]), cmap='gray')
            self.plot_mask_overlay(self.axes[2], np.rot90(gt_bin[:, cy, :]), self.cmap_gt, 0.7)
            self.plot_mask_overlay(self.axes[2], np.rot90(pred_bin[:, cy, :]), self.cmap_pred, 0.7)

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
    app = InferenceMaskReviewer(root)
    root.mainloop()
