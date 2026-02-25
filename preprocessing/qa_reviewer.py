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
# Substitui pelo caminho onde tens as tuas pastas de treino/teste no teu PC
BASE_DIR = r"E:\Tese\Datasets\Rempe\test"
REJECT_DIR = os.path.join(BASE_DIR, "_REJECTED")
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
        for entry in os.scandir(BASE_DIR):
            if entry.is_dir() and entry.name != "_REJECTED":
                mask_path = os.path.join(entry.path, "mask_GT.nii.gz") #colocar nome da segmentação
                
                # Procura a imagem original (suporta os dois nomes que usas)
                orig_path = None
                for f in ["image.nii.gz", "raw.nii.gz"]:
                    if os.path.exists(os.path.join(entry.path, f)):
                        orig_path = os.path.join(entry.path, f)
                        break
                        
                if orig_path and os.path.exists(mask_path):
                    self.exam_folders.append({
                        "folder": entry.path,
                        "name": entry.name,
                        "orig_path": orig_path,
                        "mask_path": mask_path
                    })
                    
        print(f"Encontrados {len(self.exam_folders)} exames válidos.")

    def show_current(self):
        if self.current_index >= len(self.exam_folders):
            messagebox.showinfo("Spot QA", "Revisão concluída! O teu dataset está limpo.")
            self.root.quit()
            return

        data = self.exam_folders[self.current_index]
        self.lbl_info.config(text=f"[{self.current_index + 1}/{len(self.exam_folders)}] Paciente: {data['name']}")
        
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
        target = os.path.join(REJECT_DIR, data["name"])
        try:
            shutil.move(data["folder"], target)
            print(f"❌ REJEITADO: {data['name']} (Movido para Quarentena)")
        except Exception as e:
            print(f"Erro ao isolar a pasta: {e}")
        self.current_index += 1
        self.show_current()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpotMaskReviewer(root)
    root.mainloop()