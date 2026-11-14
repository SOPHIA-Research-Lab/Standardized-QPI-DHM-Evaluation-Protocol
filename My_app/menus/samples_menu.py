import wx
import wx.aui
import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
from analysis.module1 import utilitiesRBPV as utRBPV
from core.file_manager import process_complex_field, process_phase_array


def find_sample_file(subfolder, filename):
    """Full path of a bundled sample, or FileNotFoundError saying where it looked."""
    rootDir = os.path.dirname(os.path.dirname(__file__))
    samplesDir = os.path.join(rootDir, "Samples Hologram", subfolder)
    samplePath = os.path.join(samplesDir, filename)

    if not os.path.exists(samplePath):
        raise FileNotFoundError(
            f"File not found:\nSearched in:\n{samplesDir}\nExpected file: {filename}"
        )

    return samplePath


def create(parent_frame, notebook=None):
    """Build the Samples menu, grouped by telecentric and non-telecentric setup."""
    samplesMenu = wx.Menu()

    telecentricSubmenu = wx.Menu()
    nonTelecentricSubmenu = wx.Menu()

    parent_frame.usaf_150nm_img_id = wx.NewIdRef()
    parent_frame.star_150nm_img_id = wx.NewIdRef()
    parent_frame.Nstar_mat_id = wx.NewIdRef()
    parent_frame.NsredBlood_mat_id = wx.NewIdRef()
    parent_frame.probiotics_mat_id = wx.NewIdRef()

    telecentricSubmenu.Append(parent_frame.usaf_150nm_img_id, "USAF (.bmp)")
    telecentricSubmenu.Append(parent_frame.star_150nm_img_id, "Star (.bmp)")
    telecentricSubmenu.Append(parent_frame.probiotics_mat_id, "Probiotics (.bmp)")
    nonTelecentricSubmenu.Append(parent_frame.Nstar_mat_id, "No telecentric Star (.tiff)")
    nonTelecentricSubmenu.Append(parent_frame.NsredBlood_mat_id, "Red blood cells (.tiff)")

    samplesMenu.AppendSubMenu(telecentricSubmenu, "Telecentric")
    samplesMenu.AppendSubMenu(nonTelecentricSubmenu, "Non-telecentric")

    parent_frame.Bind(wx.EVT_MENU,
                      lambda evt: load_sample(parent_frame, notebook, "Telecentric",
                                              "T_Usaf_20x_632_3.75.bmp"),
                      id=parent_frame.usaf_150nm_img_id)
    parent_frame.Bind(wx.EVT_MENU,
                      lambda evt: load_sample(parent_frame, notebook, "Telecentric",
                                              "T_Star_10x_632_3.75.bmp"),
                      id=parent_frame.star_150nm_img_id)
    parent_frame.Bind(wx.EVT_MENU,
                      lambda evt: load_sample(parent_frame, notebook, "No Telecentric",
                                              "N_Star_20x_532_5.86_-4cm.tiff"),
                      id=parent_frame.Nstar_mat_id)
    parent_frame.Bind(wx.EVT_MENU,
                      lambda evt: load_sample(parent_frame, notebook, "No Telecentric",
                                              "NredBlood_40x_632_4.65_30mm.tiff"),
                      id=parent_frame.NsredBlood_mat_id)
    parent_frame.Bind(wx.EVT_MENU,
                      lambda evt: load_sample(parent_frame, notebook, "Telecentric",
                                              "T_Probiotics_20x_632_3.75.bmp"),
                      id=parent_frame.probiotics_mat_id)
    return samplesMenu


def load_sample(parent_frame, notebook, subfolder, filename):
    """Open one bundled sample and mark the frame as showing sample data."""
    try:
        samplePath = find_sample_file(subfolder, filename)
    except FileNotFoundError as error:
        wx.MessageBox(str(error), "Error", wx.ICON_ERROR)
        return

    extension = os.path.splitext(samplePath)[1].lower()

    try:
        # Exactly the routes File > Open takes, so that a bundled sample and
        # the same file opened by hand give the same phase. They used not to:
        # an image was wrapped into a fake complex field and the phase read
        # back with np.angle, which returns 0 wherever the amplitude is 0, so
        # every pure black pixel lost its -pi.
        if extension == ".mat":
            matData = loadmat(samplePath)
            fieldNames = [matKey for matKey in matData.keys() if not matKey.startswith("__")]

            if not fieldNames:
                wx.MessageBox(f"No valid data found in {filename}", "Error", wx.ICON_ERROR)
                return

            fieldData = matData[fieldNames[0]]

            if np.iscomplexobj(fieldData):
                page = process_complex_field(notebook, fieldData, filename, fieldNames[0])
            else:
                # Real: already the phase, in radians.
                page = process_phase_array(notebook, fieldData, filename, fieldNames[0])

        elif extension in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]:
            grayscale = Image.open(samplePath).convert("L")
            grayLevels = np.array(grayscale, dtype=float)

            displayImage = Image.fromarray(grayLevels.astype(np.uint8), mode="L")
            page = notebook.add_image_tab(displayImage, filename)
            page.phase = utRBPV.gray_to_phase(grayLevels)
            page.is_mat_complex = False

        else:
            wx.MessageBox(f"Unsupported file format: {extension}", "Error", wx.ICON_ERROR)
            return

        # Carried by the page, so that the restriction follows the tab the user
        # is looking at instead of the last file that happened to be opened.
        page.is_sample = True
        # The sample ships inside the application, so File > Save Image As is
        # how a copy gets out to a reconstruction program.
        page.source_path = samplePath
        parent_frame.mark_as_sample_image(True)

        wx.MessageBox(
            f"{filename} is a hologram, not a compensated phase map.\n\n"
            "It is provided so that a reconstruction and compensation algorithm "
            "can be run on it, and the quality of the result then evaluated.",
            "Hologram loaded", wx.OK | wx.ICON_INFORMATION)

    except Exception as error:
        wx.MessageBox(f"Error loading {filename}:\n{str(error)}", "Error", wx.ICON_ERROR)
