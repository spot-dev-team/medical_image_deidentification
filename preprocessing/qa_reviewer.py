import os
import shutil
import numpy as np
import nibabel as nib
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass

# ================= CONFIGURAÇÃO =================
MASK_DIR = r"E:\Tese\Datasets\Rempe\CC_359\cc_359_GT_Masks\ge_final"
# MASK_DIR = r#r"E:\Tese\Datasets\Rempe\CC_359\cc_359_GT_Masks\philips"
ORIGINAL_DIR = r"E:\Tese\Datasets\Rempe\Original\Original\ge"

# A pasta de quarentena será criada dentro da pasta das máscaras
REJECT_DIR = os.path.join(MASK_DIR, "_REJECTED")
# ================================================

class SpotMaskReviewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Spot - Controlo de Qualidade de GTs (3 Vistas)")
        self.root.geometry("1200x600")
        
        if not os.path.exists(REJECT_DIR):
            os.makedirs(REJECT_DIR)

        self.exam_folders = []
        self.current_index = 0
        
        # Elementos da Interface
        self.lbl_info = tk.Label(root, text="A mapear exames NIfTI...", font=("Arial", 14, "bold"))
        self.lbl_info.pack(pady=5)
        
        # Motor de Renderização 3D (Matplotlib embutido no Tkinter)
        self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 4))
        self.fig.patch.set_facecolor('#f0f0f0')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        
        self.lbl_instructions = tk.Label(root, text="[<- REJEITAR e Mover]   |   [MANTER e Avançar ->]", font=("Arial", 12))
        self.lbl_instructions.pack(pady=10, side=tk.BOTTOM)

        # Atalhos de Teclado
        self.root.bind("<Left>", self.reject_exam)
        self.root.bind("<Right>", self.keep_exam)
        
        self.load_dataset()
        self.show_current()

    def load_dataset(self):
        print("A procurar pares de Imagem e Máscara...")
        
        if not os.path.exists(MASK_DIR):
            print(f"Erro: Pasta de máscaras não encontrada - {MASK_DIR}")
            return  
            
        for entry in os.scandir(MASK_DIR):
            if entry.is_file() and entry.name.endswith('_mask.nii.gz'): #_shoulderless  
                mask_path = entry.path
                
                # CORREÇÃO: Limpar tanto o "_shoulderless" como o "_mask" do nome!
                # Ex: "CC001_mask_shoulderless.nii.gz" -> "CC001.nii.gz"
                orig_name = entry.name.replace('.nii.gz', '').replace('_mask', '') + '.nii.gz' #_shoulderless  
                
                orig_path = os.path.join(ORIGINAL_DIR, orig_name)
                
                if os.path.exists(orig_path):
                    self.exam_folders.append({
                        "name": orig_name,
                        "orig_path": orig_path,
                        "mask_path": mask_path,
                        "mask_name": entry.name
                    })
                    
        print(f"Encontrados {len(self.exam_folders)} exames válidos.")

    def show_current(self):
        if self.current_index >= len(self.exam_folders):
            messagebox.showinfo("Spot QA", "Revisão concluída! O teu dataset está limpo.")
            self.root.quit()
            return

        data = self.exam_folders[self.current_index]
        self.lbl_info.config(text=f"[{self.current_index + 1}/{len(self.exam_folders)}] Exame: {data['name']}")
        
        try:
            # Carregar matrizes NIfTI
            orig_img = nib.load(data["orig_path"]).get_fdata()
            mask_img = nib.load(data["mask_path"]).get_fdata()
            
            # Encontrar o "Centro de Gravidade" da máscara para cortar a fatia perfeita
            if np.sum(mask_img) > 0:
                cx, cy, cz = [int(c) for c in center_of_mass(mask_img)]
            else:
                cx, cy, cz = [s // 2 for s in orig_img.shape] # Se a máscara estiver vazia, vai para o meio do cérebro
                
            # Limpar ecrãs anteriores
            for ax in self.axes:
                ax.clear()
                ax.axis('off')
                
            # O rot90() ajusta a orientação genérica das matrizes para ficar de pé no ecrã
            # 1. Vista Axial
            self.axes[0].imshow(np.rot90(orig_img[:, :, cz]), cmap='gray')
            self.axes[0].imshow(np.ma.masked_where(np.rot90(mask_img[:, :, cz]) == 0, np.rot90(mask_img[:, :, cz])), cmap='autumn', alpha=0.5)
            self.axes[0].set_title("Axial (Z)")

            # 2. Vista Sagital
            self.axes[1].imshow(np.rot90(orig_img[cx, :, :]), cmap='gray')
            self.axes[1].imshow(np.ma.masked_where(np.rot90(mask_img[cx, :, :]) == 0, np.rot90(mask_img[cx, :, :])), cmap='autumn', alpha=0.5)
            self.axes[1].set_title("Sagital (X)")

            # 3. Vista Coronal
            self.axes[2].imshow(np.rot90(orig_img[:, cy, :]), cmap='gray')
            self.axes[2].imshow(np.ma.masked_where(np.rot90(mask_img[:, cy, :]) == 0, np.rot90(mask_img[:, cy, :])), cmap='autumn', alpha=0.5)
            self.axes[2].set_title("Coronal (Y)")

            self.canvas.draw()
            
        except Exception as e:
            print(f"Erro ao renderizar {data['name']}: {e}")
            self.current_index += 1
            self.show_current()

    def keep_exam(self, event=None):
        self.current_index += 1
        self.show_current()

    def reject_exam(self, event=None):
        data = self.exam_folders[self.current_index]
        
        target_mask = os.path.join(REJECT_DIR, data["mask_name"])
        target_orig = os.path.join(REJECT_DIR, data["name"])
        
        try:
            # Move os DOIS ficheiros para a Quarentena
            shutil.move(data["mask_path"], target_mask)
            shutil.move(data["orig_path"], target_orig)
            print(f"❌ REJEITADO: {data['name']} (Par movido para Quarentena)")
        except Exception as e:
            print(f"Erro ao isolar os ficheiros: {e}")
            
        self.current_index += 1
        self.show_current()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpotMaskReviewer(root)
    root.mainloop()