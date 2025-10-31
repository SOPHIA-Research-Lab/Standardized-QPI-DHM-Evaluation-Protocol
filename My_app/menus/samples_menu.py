import wx
import wx.aui
import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
from analysis.module1 import utilitiesRBPV as utRBPV
from core.file_manager import process_complex_field



# ============================================
# 🔍 Buscar archivo dentro de Samples
# ============================================
def find_sample_file(subfolder, filename):
    """
    Busca un archivo dentro de ./Samples/<subfolder>/<filename>
    sin depender de rutas absolutas.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))  # raíz de My_app
    samples_dir = os.path.join(base_dir, "Samples", subfolder)
    file_path = os.path.join(samples_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found:\nSearched in:\n{samples_dir}\nExpected file: {filename}"
        )

    return file_path


# ============================================
# 🔍 Crear menú de Samples
# ============================================
def create(parent_frame, notebook=None):
    samples_menu = wx.Menu()

    # ===== Submenús =====
    usaf_submenu = wx.Menu()
    star_submenu = wx.Menu()

    parent_frame.usaf_150nm_mat_id = wx.NewIdRef()
    parent_frame.usaf_150nm_img_id = wx.NewIdRef()
    parent_frame.star_150nm_mat_id = wx.NewIdRef()
    parent_frame.star_150nm_img_id = wx.NewIdRef()

    usaf_submenu.Append(parent_frame.usaf_150nm_mat_id, "USAF - 150 nm (.mat)")
    usaf_submenu.Append(parent_frame.usaf_150nm_img_id, "USAF - 150 nm (image)")
    star_submenu.Append(parent_frame.star_150nm_mat_id, "Star - 150 nm (.mat)")
    star_submenu.Append(parent_frame.star_150nm_img_id, "Star - 150 nm (image)")

    samples_menu.AppendSubMenu(usaf_submenu, "USAF Target")
    samples_menu.AppendSubMenu(star_submenu, "Star Target")

    # ===== Enlazar eventos =====
    parent_frame.Bind(wx.EVT_MENU,
                      lambda evt: load_sample(parent_frame, notebook, "usaf", "usaf_150nm.mat", "USAF 150nm (.mat)"),
                      id=parent_frame.usaf_150nm_mat_id)
    parent_frame.Bind(wx.EVT_MENU,
                      lambda evt: load_sample(parent_frame, notebook, "usaf", "usaf_150nm.png", "USAF 150nm (image)"),
                      id=parent_frame.usaf_150nm_img_id)
    parent_frame.Bind(wx.EVT_MENU,
                      lambda evt: load_sample(parent_frame, notebook, "star", "star_150nm.mat", "Star 150nm (.mat)"),
                      id=parent_frame.star_150nm_mat_id)
    parent_frame.Bind(wx.EVT_MENU,
                      lambda evt: load_sample(parent_frame, notebook, "star", "star_150nm.png", "Star 150nm (image)"),
                      id=parent_frame.star_150nm_img_id)

    return samples_menu


# ============================================
# 📂 Cargar una muestra (usando process_complex_field)
# ============================================
def load_sample(parent_frame, notebook, subfolder, filename, display_name):
    try:
        file_path = find_sample_file(subfolder, filename)
    except FileNotFoundError as e:
        wx.MessageBox(str(e), "Error", wx.ICON_ERROR)
        return

    ext = os.path.splitext(file_path)[1].lower()

    try:
        # --------------------------
        # 📊 Caso 1: archivo .mat
        # --------------------------
        if ext == ".mat":
            mat_data = loadmat(file_path)
            data_keys = [k for k in mat_data.keys() if not k.startswith("__")]

            if not data_keys:
                wx.MessageBox(f"No valid data found in {filename}", "Error", wx.ICON_ERROR)
                return

            # Tomamos el primer campo válido
            data = mat_data[data_keys[0]]

            # Si es complejo, usar tal cual; si no, convertir a complejo con fase artificial
            if np.iscomplexobj(data):
                complex_field = data
            else:
                amplitude = np.abs(data)
                phase = utRBPV.grayscaleToPhase(amplitude)
                complex_field = amplitude * np.exp(1j * phase)

            # ✅ Usar tu función reutilizable
            process_complex_field(notebook, complex_field, filename, data_keys[0])

        # --------------------------
        # 📷 Caso 2: imagen normal
        # --------------------------
        elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            img = Image.open(file_path).convert("L")
            img_array = np.array(img, dtype=float)
            phase = utRBPV.grayscaleToPhase(img_array)
            complex_field = img_array * np.exp(1j * phase)

            # ✅ Llamar también a tu función
            process_complex_field(notebook, complex_field, filename, "Grayscale Image")

        else:
            wx.MessageBox(f"Unsupported file format: {ext}", "Error", wx.ICON_ERROR)
            return

        # 🔒 Marcar como imagen de muestra
        parent_frame.mark_as_sample_image(True)

    except Exception as e:
        wx.MessageBox(f"Error loading {filename}:\n{str(e)}", "Error", wx.ICON_ERROR)
