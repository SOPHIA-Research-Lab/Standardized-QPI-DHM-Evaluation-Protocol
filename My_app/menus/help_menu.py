import wx


def create(parent):
    help_menu = wx.Menu()

    help_id = wx.NewIdRef()
    parent.Bind(wx.EVT_MENU, lambda e: show_help(parent), id=help_id)
    help_menu.Append(help_id, "&View Help\tF1", "View application help")

    return help_menu


def show_help(parent):
    help_text = (
        "Standardized QPI-DHM Evaluation Protocol – Help\n\n"
        "This application implements a standardized evaluation protocol for quantitative phase imaging (QPI) "
        "in Digital Holographic Microscopy (DHM), using four core modules:\n\n"

        "🔹 Module I – Residual Background Phase Variance:\n"
        "   -  Standard Deviation (STD): Low values indicate uniform background (good quality), "
        "high values indicate noise or excessive variation (poor quality).\n"
        "   -  Mean Absolute Deviation (MAD): Low MAD indicates greater uniformity and less dispersion "
        "(less noise, better quality). High MAD indicates greater variability.\n"
        "   -  Root Mean Square (RMS): Low RMS indicates values close to mean, more uniform background, "
        "less noise (better quality). High RMS shows greater dispersion and roughness (worse quality).\n"
        "   -  Peak-to-Valley (PV): Low PV indicates uniform phase and stable background (good quality). "
        "High PV means greater phase differences, more distortion or optical errors.\n"
        "   -  Full Width at Half Maximum (FWHM): Low FWHM represents sharper image, higher resolution, "
        "better focus. High FWHM represents blurrier or degraded image, lower resolution.\n"
        "   -  Entropy: High entropy is associated with more details, textures and variety (better perceived quality). "
        "Low entropy indicates uniformity, which may occur in unfocused or featureless images.\n\n"

        "🔹 Module II – Global Phase Distortion Metrics:\n"
        "   -  Maximum-Minus-Minimum (PV): Low PV indicates uniform phase (good quality), "
        "high PV indicates greater distortion or optical noise.\n"
        "   -  Global Phase Gradient: Values close to zero imply uniform background (good optical quality). "
        "High values indicate clear slope, reflecting severity of artifacts or optical aberrations.\n"
        "   -  TSM (Tenengrad Sharpness Metric): Low TSM values indicate flat and stable phase image "
        "(desirable, artifact-free background). High TSM reflects many abrupt changes or inclinations.\n"
        "   -  Laplacian Energy: Low values represent smooth homogeneous background (few details, defocused image). "
        "High values indicate many edges, sharp details and marked texture (may represent noise if background should be uniform).\n"
        "   -  Spatial Frequency: High spatial frequency indicates many small details and rapid intensity changes "
        "(sharp images, high resolution). Low spatial frequency relates to smooth changes, homogeneous areas (poor detail, low quality).\n"
        "   -  Generalized Sharpness Metric (GSM): High GSM indicates clearly defined zones, sharp edges and high contrast. "
        "Low GSM reflects blurry or out-of-focus images with little useful information.\n\n"

        "🔹 Module III – Ground-Truth Comparisons:\n"
        "   -  Structural Similarity (SSIM): SSIM close to 1 = high structural similarity, high visual quality. "
        "SSIM around 0 = little or no similarity, low quality.\n"
        "   -  Mean Squared Error (MSE): MSE = 0 means signals/images are identical. "
        "Low MSE = very similar values (better quality). High MSE = large differences (low quality).\n"
        "   -  Peak Signal-to-Noise Ratio (PSNR): High PSNR (30-50 dB for 8-bit images, >60 dB for higher bit depth) "
        "indicates reconstructed image is very similar to original (less distortion, better quality). "
        "Low PSNR reflects greater error and worse quality. PSNR infinite = MSE is 0 (identical images, perfect quality).\n\n"

        "🔹 Module IV – Computational Complexity:\n"
        "   -  Operation counts\n"
        "   -  Execution time and memory profiling\n"
        "   -  Hardware-aware benchmarking\n\n"

        " Test Holograms:\n"
        "   A diverse collection of holograms will be available for benchmarking across different scenarios.\n\n"

        " Features of the App:\n"
        "   -  Image zoom and detachment\n"
        "   -  Metric table with export (CSV/XLS)\n"
        "   -  Analyze multiple images at once\n"
        "   -  Contextual actions via right-click\n\n"

        " Shortcuts:\n"
        "   -  Ctrl+O – Open image\n"
        "   -  Ctrl+S – Save image\n"
        "   -  Ctrl+Q – Exit\n"
        "   -  F1 – Help\n"
        "   -  Alt + key – Menu access\n\n"

        " Setup:\n"
        "   1. Install Python and requirements:\n"
        "      pip install -r requirements.txt\n"
        "   2. Run the app:\n"
        "      python app.py\n\n"

        "Thank you for using the QPI Evaluation App! ✨"
    )

    dialog = wx.Dialog(parent, title="Help", size=(650, 520))
    dialog.SetBackgroundColour(wx.Colour(255, 255, 255))

    panel = wx.Panel(dialog)
    panel.SetCursor(wx.Cursor(wx.CURSOR_BLANK))  # Hide cursor in help window

    sizer = wx.BoxSizer(wx.VERTICAL)

    text_ctrl = wx.TextCtrl(panel,
                            value=help_text,
                            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL | wx.VSCROLL)
    sizer.Add(text_ctrl, 1, wx.EXPAND | wx.ALL, 10)

    close_btn = wx.Button(panel, label="Close")
    close_btn.Bind(wx.EVT_BUTTON, lambda evt: dialog.Close())
    sizer.Add(close_btn, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

    panel.SetSizer(sizer)

    dialog.ShowModal()
    dialog.Destroy()
