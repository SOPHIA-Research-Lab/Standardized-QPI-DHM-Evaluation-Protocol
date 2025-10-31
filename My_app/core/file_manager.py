import wx
from PIL import Image
import numpy as np
import os
from scipy.io import loadmat
from analysis.module1 import utilitiesRBPV as utRBPV

def open_image(parent, notebook):
    """Open image or .mat file, convert to phase if needed, and display in a new notebook tab."""
    
    wildcard = (
        "Supported files (*.png;*.jpg;*.jpeg;*.bmp;*.mat)|*.png;*.jpg;*.jpeg;*.bmp;*.mat|"
        "Image files (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp|"
        "MATLAB files (*.mat)|*.mat"
    )

    with wx.FileDialog(parent, "Open Image or .mat File",
                       wildcard=wildcard,
                       style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE) as fileDialog:
        
        if fileDialog.ShowModal() == wx.ID_CANCEL:
            return

        paths = fileDialog.GetPaths()

        for path in paths:
            ext = os.path.splitext(path)[1].lower()

            try:
                # -------------------------------------------------------
                # CASE 1: Standard image (PNG, JPG, BMP, etc.)
                # -------------------------------------------------------
                if ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                    img = Image.open(path).convert("L")
                    img_array = np.array(img, dtype=float)

                    # Convert grayscale → phase [-π, π]
                    phase = utRBPV.grayscaleToPhase(img_array)

                    amp_img = Image.fromarray(img_array.astype(np.uint8), mode="L")

                    page = notebook.add_image_tab(amp_img, os.path.basename(path))
                    page.phase = phase
                    page.is_mat_complex = False

                # -------------------------------------------------------
                # CASE 2: MATLAB .mat file - user selects variable
                # -------------------------------------------------------
                elif ext == ".mat":
                    mat_data = loadmat(path)
                    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                    dlg = wx.SingleChoiceDialog(parent,
                                                "Select the variable to open:",
                                                "File .MAT",
                                                data_keys)
                    if dlg.ShowModal() == wx.ID_OK:
                        complex_key = dlg.GetStringSelection()
                        complex_field = mat_data[complex_key]

                        if not isinstance(complex_field, np.ndarray):
                            raise TypeError(f"The variable '{complex_key}' is not a valid array.")

                        # Process the complex field as before
                        process_complex_field(notebook, complex_field, os.path.basename(path), complex_key)

                    dlg.Destroy()

            except Exception as e:
                wx.MessageBox(f"Error {os.path.basename(path)}:\n{e}", "Error", wx.ICON_ERROR)


def process_complex_field(notebook, complex_field, filename, field_name):
    """Process a complex field: extract amplitude and phase, display amplitude normalized for viewing."""
    
    # Calculate amplitude and phase
    amplitude = np.abs(complex_field)
    phase = np.angle(complex_field)

    amp_min = np.min(amplitude)
    amp_max = np.max(amplitude)

    # --------------------------------------------
    # Normalize amplitude for correct image display
    # --------------------------------------------
    amplitude_norm = amplitude - amp_min
    if amp_max > amp_min:
        amplitude_norm = amplitude_norm / (amp_max - amp_min)
    amplitude_norm = (amplitude_norm * 255).astype(np.uint8)

    # Create PIL image from normalized amplitude
    amp_img = Image.fromarray(amplitude_norm, mode='L')

    # Add image to notebook
    display_name = f"{filename} [{field_name}]"
    page = notebook.add_image_tab(amp_img, display_name)

    # Store original data for further analysis
    page.amplitude = amplitude
    page.phase = phase
    page.is_mat_complex = True
    page.complex_field_name = field_name

    # Show information message
    info_msg = (
        f"✅ Complex field loaded successfully!\n\n"
        f"File: {filename}\n"
        f"Field: {field_name}\n"
        f"Size: {complex_field.shape}\n\n"
        f"Displaying: Amplitude (normalized for visualization)\n"
        f"Analysis will use: Original Phase data\n\n"
        f"Amplitude range: [{amp_min:.4f}, {amp_max:.4f}]\n"
        f"Phase range: [{np.min(phase):.4f}, {np.max(phase):.4f}] rad"
    )

    wx.MessageBox(info_msg, "Complex Field Loaded", wx.ICON_INFORMATION)
