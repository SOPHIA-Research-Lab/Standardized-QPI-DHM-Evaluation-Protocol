import wx
import wx.aui
import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
from analysis.module1 import utilitiesRBPV as utRBPV
from core.file_manager import process_complex_field




def find_sample_file(subfolder, filename):

    base_dir = os.path.dirname(os.path.dirname(__file__))  # raíz de My_app
    samples_dir = os.path.join(base_dir, "Samples Hologram", subfolder)
    file_path = os.path.join(samples_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found:\nSearched in:\n{samples_dir}\nExpected file: {filename}"
        )

    return file_path


# ============================================
#  Sample Menu Creation
# ============================================
def create(parent_frame, notebook=None):
    samples_menu = wx.Menu()

    # ===== Submenus =====
    NTelecentric_submenu = wx.Menu()
    Telecentric_submenu = wx.Menu()

    parent_frame.usaf_150nm_mat_id = wx.NewIdRef()
    parent_frame.usaf_150nm_img_id = wx.NewIdRef()
    parent_frame.star_150nm_mat_id = wx.NewIdRef()
    parent_frame.star_150nm_img_id = wx.NewIdRef()
    parent_frame.Nstar_mat_id = wx.NewIdRef()
    parent_frame.probiotics_mat_id = wx.NewIdRef()

    #Telecentric_submenu.Append(parent_frame.usaf_150nm_mat_id, "USAF - 150 nm (.mat)")
    Telecentric_submenu.Append(parent_frame.usaf_150nm_img_id, "USAF (.bmp)")
    #Telecentric_submenu.Append(parent_frame.star_150nm_mat_id, "Star - 150 nm (.mat)")
    Telecentric_submenu.Append(parent_frame.star_150nm_img_id, "Star (.bmp)")
    NTelecentric_submenu.Append(parent_frame.Nstar_mat_id, "No telecentric Star (.tiff)")
    NTelecentric_submenu.Append(parent_frame.Nstar_mat_id, "Red blood cells (.tiff)")
    Telecentric_submenu.Append(parent_frame.probiotics_mat_id, "Probiotics (.bmp)")

    samples_menu.AppendSubMenu(Telecentric_submenu, "Telecentric")
    samples_menu.AppendSubMenu(NTelecentric_submenu, "No Telecentric")
   

    # ===== Events =====
    # parent_frame.Bind(wx.EVT_MENU,
    #                   lambda evt: load_sample(parent_frame, notebook, "Telecentric", "usaf_150nm.mat", "USAF 150nm (.mat)"),
    #                   id=parent_frame.usaf_150nm_mat_id)
    parent_frame.Bind(wx.EVT_MENU,
                      lambda evt: load_sample(parent_frame, notebook, "Telecentric", "T_Usaf_20x_632_3.75.bmp", "USAF (.bmp)"),
                      id=parent_frame.usaf_150nm_img_id)
    # parent_frame.Bind(wx.EVT_MENU,
    #                   lambda evt: load_sample(parent_frame, notebook, "Telecentric", "star_150nm.mat", "Star 150nm (.mat)"),                      
    #                   id=parent_frame.star_150nm_mat_id)
    parent_frame.Bind(wx.EVT_MENU,
                      lambda evt: load_sample(parent_frame, notebook, "Telecentric", "T_Star_10x_632_3.75.bmp", "Star (.bmp)"),
                      id=parent_frame.star_150nm_img_id)
    parent_frame.Bind(wx.EVT_MENU,
                      lambda evt: load_sample(parent_frame, notebook, "No Telecentric", "N_Star_20x_532_5.86_-4cm.tiff", "No telecentric Star target (.tiff)"),
                      id=parent_frame.Nstar_mat_id)
    parent_frame.Bind(wx.EVT_MENU,
                          lambda evt: load_sample(parent_frame, notebook, "No Telecentric", "NredBlood_40x_632_4.65_30mm.tiff", "Red Blood Cells (.tiff)"),
                          id=parent_frame.Nstar_mat_id)
    parent_frame.Bind(wx.EVT_MENU,
                          lambda evt: load_sample(parent_frame, notebook, "Telecentric", "T_Probiotics_20x_632_3.75.bmp", "Probiotics (.bmp)"),
                          id=parent_frame.probiotics_mat_id)
    return samples_menu


# ============================================
# Load Sample
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
        #  Case 1: archivo .mat
        # --------------------------
        if ext == ".mat":
            mat_data = loadmat(file_path)
            data_keys = [k for k in mat_data.keys() if not k.startswith("__")]

            if not data_keys:
                wx.MessageBox(f"No valid data found in {filename}", "Error", wx.ICON_ERROR)
                return

            # Take the first valid key
            data = mat_data[data_keys[0]]

            # Evaluate if the data is complex or needs to be converted
            if np.iscomplexobj(data):
                complex_field = data
            else:
                amplitude = np.abs(data)
                phase = utRBPV.grayscaleToPhase(amplitude)
                complex_field = amplitude * np.exp(1j * phase)

            # Call your function to process the complex field
            process_complex_field(notebook, complex_field, filename, data_keys[0])

        # --------------------------
        # Case 2: file (png, jpg, jpeg, bmp)
        # --------------------------
        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]:
            img = Image.open(file_path).convert("L")
            img_array = np.array(img, dtype=float)
            phase = utRBPV.grayscaleToPhase(img_array)
            complex_field = img_array * np.exp(1j * phase)

            # Call your function to process the complex field
            process_complex_field(notebook, complex_field, filename, "Grayscale Image")

        else:
            wx.MessageBox(f"Unsupported file format: {ext}", "Error", wx.ICON_ERROR)
            return


        parent_frame.mark_as_sample_image(True)

    except Exception as e:
        wx.MessageBox(f"Error loading {filename}:\n{str(e)}", "Error", wx.ICON_ERROR)
