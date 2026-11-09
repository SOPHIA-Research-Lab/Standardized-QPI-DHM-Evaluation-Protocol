import wx
from PIL import Image
import numpy as np
import os
from scipy.io import loadmat
from analysis.module1 import utilitiesRBPV as utRBPV


def open_image(parent, notebook):
    """Open images or .mat files and show each one in a new notebook tab."""
    wildcard = (
        "Supported files (*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.mat)|"
        "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.mat|"
        "Image files (*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff)|"
        "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff|"
        "MATLAB files (*.mat)|*.mat"
    )

    with wx.FileDialog(parent, "Open Image or .mat File",
                       wildcard=wildcard,
                       style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE) as fileDialog:

        if fileDialog.ShowModal() == wx.ID_CANCEL:
            return

        for path in fileDialog.GetPaths():
            extension = os.path.splitext(path)[1].lower()

            try:
                if extension in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                    # 8-bit: 256 phase levels, remapped linearly onto [-pi, pi].
                    grayscale = Image.open(path).convert("L")
                    grayLevels = np.array(grayscale, dtype=float)
                    phase = utRBPV.gray_to_phase(grayLevels)

                    displayImage = Image.fromarray(grayLevels.astype(np.uint8), mode="L")
                    page = notebook.add_image_tab(displayImage, os.path.basename(path))
                    page.phase = phase
                    page.is_mat_complex = False

                elif extension == ".mat":
                    matData = loadmat(path)
                    fieldNames = [matKey for matKey in matData.keys() if not matKey.startswith('__')]
                    dialog = wx.SingleChoiceDialog(parent,
                                                   "Select the variable to open:",
                                                   "File .MAT",
                                                   fieldNames)
                    if dialog.ShowModal() == wx.ID_OK:
                        fieldName = dialog.GetStringSelection()
                        variable = matData[fieldName]

                        if not isinstance(variable, np.ndarray):
                            raise TypeError(f"The variable '{fieldName}' is not a valid array.")

                        # A real variable is already the phase, in radians. It
                        # must not go through np.angle, which would collapse it
                        # onto 0 and pi according to its sign alone.
                        if np.iscomplexobj(variable):
                            process_complex_field(notebook, variable,
                                                  os.path.basename(path), fieldName)
                        else:
                            process_phase_array(notebook, variable,
                                                os.path.basename(path), fieldName)

                    dialog.Destroy()

            except Exception as error:
                wx.MessageBox(f"Error {os.path.basename(path)}:\n{error}", "Error", wx.ICON_ERROR)


def process_complex_field(notebook, complex_field, filename, field_name):
    """Keep the true phase of a complex field and display its amplitude."""
    amplitude = np.abs(complex_field)
    phase = np.angle(complex_field)

    amplitudeMin = np.min(amplitude)
    amplitudeMax = np.max(amplitude)

    # The amplitude is normalised for DISPLAY only; the phase is kept as is.
    normalisedAmplitude = amplitude - amplitudeMin
    if amplitudeMax > amplitudeMin:
        normalisedAmplitude = normalisedAmplitude / (amplitudeMax - amplitudeMin)
    normalisedAmplitude = (normalisedAmplitude * 255).astype(np.uint8)

    displayImage = Image.fromarray(normalisedAmplitude, mode='L')
    page = notebook.add_image_tab(displayImage, f"{filename} [{field_name}]")

    page.amplitude = amplitude
    page.phase = phase
    page.is_mat_complex = True
    page.complex_field_name = field_name

    wx.MessageBox(
        f"Complex field loaded successfully!\n\n"
        f"File: {filename}\n"
        f"Field: {field_name}\n"
        f"Size: {complex_field.shape}\n\n"
        f"Displaying: Amplitude (normalized for visualization)\n"
        f"Analysis will use: Original Phase data\n\n"
        f"Amplitude range: [{amplitudeMin:.4f}, {amplitudeMax:.4f}]\n"
        f"Phase range: [{np.min(phase):.4f}, {np.max(phase):.4f}] rad",
        "Complex Field Loaded", wx.ICON_INFORMATION)

    return page


def process_phase_array(notebook, phase_array, filename, field_name):
    """Open a real .mat variable, which is already the phase in radians.

    Nothing is converted: the array is handed to the metrics as it stands, and
    only the picture on screen is rescaled so that it can be displayed at all."""
    phase = np.asarray(phase_array, dtype=float)

    phaseMin = float(np.min(phase))
    phaseMax = float(np.max(phase))

    # Normalised for DISPLAY only; the phase itself is kept untouched.
    displayLevels = phase - phaseMin
    if phaseMax > phaseMin:
        displayLevels = displayLevels / (phaseMax - phaseMin)
    displayLevels = (displayLevels * 255).astype(np.uint8)

    displayImage = Image.fromarray(displayLevels, mode='L')
    page = notebook.add_image_tab(displayImage, f"{filename} [{field_name}]")

    page.amplitude = None
    page.phase = phase
    # True means "the phase is stored on the page", not "the file held a
    # complex field": it is what tells the caller not to re-read the pixels.
    page.is_mat_complex = True
    page.complex_field_name = field_name

    wx.MessageBox(
        f"Phase map loaded successfully!\n\n"
        f"File: {filename}\n"
        f"Variable: {field_name}\n"
        f"Size: {phase.shape}\n\n"
        f"The variable is real, so it is taken to be the phase in radians "
        f"and is used as it stands.\n"
        f"Displaying: the phase, rescaled for visualization only.\n\n"
        f"Phase range: [{phaseMin:.4f}, {phaseMax:.4f}] rad",
        "Phase Map Loaded", wx.ICON_INFORMATION)

    return page
