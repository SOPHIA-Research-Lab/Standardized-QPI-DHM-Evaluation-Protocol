"""
Viewer for the Legendre background reconstruction.

This was the tail of reconstruction_background() in the analysis layer: the
function reconstructed the surface AND popped up a window showing it, and the
only caller threw the return value away. Computing and displaying are now two
separate things, and only this half knows about wx.
"""
import matplotlib.pyplot as plt
import numpy as np
import wx
from matplotlib.backends.backend_agg import FigureCanvasAgg


def coefficient_figure(coefficients):
    """
    Build the coefficient bar chart as a matplotlib figure.

    Separate from the dialog so the chart can be rendered and inspected
    without opening a window.
    """
    SUPERFICIE = "#fcfcfb"
    TINTA = "#0b0b0b"
    TINTA_SUAVE = "#52514e"
    REJILLA = "#e5e4e0"
    POSITIVO = "#2a78d6"
    NEGATIVO = "#e34948"

    valores = np.asarray(coefficients, dtype=float)
    indices = np.arange(1, len(valores) + 1)

    fig = plt.figure(figsize=(5, 4), dpi=100)
    fig.patch.set_facecolor(SUPERFICIE)
    ax = fig.add_subplot(111)
    ax.set_facecolor(SUPERFICIE)

    # The sign of a coefficient is part of the reading, so the two poles of a
    # diverging pair carry it and the zero line stays visible. The bars wear a
    # surface-coloured edge, which is what keeps adjacent ones from touching.
    colores = [POSITIVO if v >= 0 else NEGATIVO for v in valores]
    ax.bar(indices, valores, width=0.68, color=colores,
           edgecolor=SUPERFICIE, linewidth=2)
    ax.axhline(0, color=TINTA_SUAVE, linewidth=1.0)

    # Few bars, and reading the exact number is the reason this panel exists,
    # so every bar is labelled. Labels wear text ink, never the bar colour.
    for x, v in zip(indices, valores):
        ax.annotate("%.4f" % v, (x, v),
                    xytext=(0, 4 if v >= 0 else -4), textcoords="offset points",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8, color=TINTA)

    ax.set_xticks(indices)
    ax.set_xlabel("Legendre index", color=TINTA_SUAVE)
    ax.set_ylabel("Coefficient value [rad]", color=TINTA_SUAVE)
    ax.tick_params(colors=TINTA_SUAVE, labelsize=9, length=0)
    ax.grid(True, axis="y", color=REJILLA, linewidth=1.0)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(REJILLA)
    ax.margins(y=0.18)
    fig.tight_layout()
    return fig


def show_legendre_reconstruction(superficie, coefficients):
    """
    Show the reconstructed background next to the coefficient plot.

    Parameters
    ----------
    superficie : ndarray
        Surface rebuilt from the Legendre coefficients.
    coefficients : sequence of float
        The coefficients themselves, plotted on the right-hand panel.
    """
    # Create wx.App if needed
    app_created = False
    app = wx.GetApp()
    if app is None:
        app = wx.App(False)
        app_created = True

    # Create dialog
    dlg = wx.Dialog(None, title="Legendre Reconstruction Results",
                    style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
                    size=(1100, 550))
    dlg.Centre()
    
    main_sizer = wx.BoxSizer(wx.VERTICAL)
    content_sizer = wx.BoxSizer(wx.HORIZONTAL)
    
    # Left panel: Surface
    left_panel = wx.Panel(dlg)
    left_sizer = wx.BoxSizer(wx.VERTICAL)
    label_surface = wx.StaticText(left_panel, label="Legendre-reconstructed background")
    left_sizer.Add(label_surface, 0, wx.ALIGN_CENTER | wx.ALL, 5)
    
    # Convert surface to colored image
    from matplotlib import cm
    viridis = cm.get_cmap('viridis')
    surf_norm = (superficie - superficie.min()) / (superficie.max() - superficie.min())
    colored = (viridis(surf_norm)[:, :, :3] * 255).astype(np.uint8)
    
    h, w = colored.shape[:2]
    img_surface = wx.Image(w, h)
    img_surface.SetData(colored.tobytes())
    img_surface = img_surface.Scale(500, 400, wx.IMAGE_QUALITY_HIGH)
    
    bmp_surface = wx.StaticBitmap(left_panel, bitmap=wx.Bitmap(img_surface))
    left_sizer.Add(bmp_surface, 1, wx.ALL | wx.EXPAND, 5)
    left_panel.SetSizer(left_sizer)
    content_sizer.Add(left_panel, 1, wx.EXPAND | wx.ALL, 5)
    
    # Right panel: Coefficients
    right_panel = wx.Panel(dlg)
    right_sizer = wx.BoxSizer(wx.VERTICAL)
    label_coeff = wx.StaticText(right_panel, label="Legendre coefficients")
    right_sizer.Add(label_coeff, 0, wx.ALIGN_CENTER | wx.ALL, 5)
    
    # Create plot
    fig = coefficient_figure(coefficients)
    
    # Convert plot to bitmap
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    w_fig, h_fig = fig.canvas.get_width_height()
    plot_array = np.frombuffer(buf, dtype=np.uint8).reshape(h_fig, w_fig, 4)[:, :, :3]
    
    img_plot = wx.Image(w_fig, h_fig)
    img_plot.SetData(plot_array.tobytes())
    img_plot = img_plot.Scale(500, 400, wx.IMAGE_QUALITY_HIGH)
    
    bmp_plot = wx.StaticBitmap(right_panel, bitmap=wx.Bitmap(img_plot))
    right_sizer.Add(bmp_plot, 1, wx.ALL | wx.EXPAND, 5)
    right_panel.SetSizer(right_sizer)
    content_sizer.Add(right_panel, 1, wx.EXPAND | wx.ALL, 5)
    
    plt.close(fig)
    
    main_sizer.Add(content_sizer, 1, wx.EXPAND | wx.ALL, 10)
    
    # Close button
    btn_close = wx.Button(dlg, wx.ID_CLOSE, "Close")
    btn_close.Bind(wx.EVT_BUTTON, lambda evt: dlg.Close())
    main_sizer.Add(btn_close, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)
    
    dlg.SetSizer(main_sizer)
    dlg.ShowModal()
    dlg.Destroy()

    if app_created:
        try:
            app.Destroy()
        except:
            pass

