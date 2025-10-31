import wx

def create(parent):
    help_menu = wx.Menu()

    help_id = wx.NewIdRef()
    parent.Bind(wx.EVT_MENU, lambda e: show_help(parent), id=help_id)
    help_menu.Append(help_id, "&View Help\tF1", "View application help")

    return help_menu

def show_help(parent):
    help_text = (
        "📘 Standardized QPI-DHM Evaluation Protocol – Help\n\n"
        "This application implements a standardized evaluation protocol for quantitative phase imaging (QPI) "
        "in Digital Holographic Microscopy (DHM), using four core modules:\n\n"

        "🔹 Nucleo I – Residual Background Phase Variance:\n"
        "   • STD or MAD in object-free regions\n"
        "   • RMS and Peak-to-Valley (P–V)\n"
        "   • Tilt/Curvature (Legendre/Zernike)\n"
        "   • FWHM of phase histogram\n"
        "   • Background entropy and spatial frequencies\n\n"

        "🔹 Nucleo II – Global Phase Distortion Metrics:\n"
        "   • Max–Min and α-STD variations\n"
        "   • Global gradient and curvature\n"
        "   • TSM (Sharpness), Laplacian energy\n"
        "   • Entropy and contrast metrics\n\n"
        "🔹 Nucleo III – Ground-Truth Comparisons:\n"
        "   • Percent Error (PE) against reference values\n"
        "   • SSIM, pSNR\n"
        "   • Double-exposure, conjugate wavefront techniques\n"
        "   • Complex-field subtraction/multiplication\n\n"
        "🔹 Nucleo IV – Computational Complexity:\n"
        "   • Operation counts\n"
        "   • Execution time and memory profiling\n"
        "   • Hardware-aware benchmarking\n\n"

        "🧪 Test Holograms:\n"
        "   A diverse collection of holograms will be available for benchmarking across different scenarios.\n\n"

        "💡 Features of the App:\n"
        "   • Image zoom and detachment\n"
        "   • Metric table with export (CSV/XLS)\n"
        "   • Analyze multiple images at once\n"
        "   • Contextual actions via right-click\n\n"

        "🧭 Shortcuts:\n"
        "   • Ctrl+O – Open image\n"
        "   • Ctrl+S – Save image\n"
        "   • Ctrl+Q – Exit\n"
        "   • F1 – Help\n"
        "   • Alt + key – Menu access\n\n"

        "📦 Setup:\n"
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
