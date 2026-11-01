"""
Viewer for the Legendre background reconstruction.

This was the tail of reconstruction_background() in the analysis layer: the
function reconstructed the surface AND popped up a window showing it, and the
only caller threw the return value away. Computing and displaying are now two
separate things, and only this half knows about wx.
"""
import matplotlib
import numpy as np
import wx
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ui import hidpi

#: Same palette as the TSM histogram and the toolbar icons.
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SUAVE = "#52514e"
REJILLA = "#e5e4e0"
POSITIVO = "#2a78d6"
NEGATIVO = "#e34948"


def coefficient_figure(coefficients, escala=1.0):
    """
    Build the coefficient bar chart as a matplotlib figure.

    Separate from the dialog so the chart can be rendered and inspected
    without opening a window. `escala` is the screen's pixel-per-unit factor:
    the chart measures 5 x 4 inches whatever it is, but it is drawn with that
    many more pixels.

    The figure is built directly instead of through pyplot. pyplot attaches
    the figure to whichever interactive backend is active, and the wx backend
    doubles the dpi behind your back on a scaled screen: a chart asked for at
    1000x800 came back 2000x1521 and overflowed the window it had to sit in.
    """
    valores = np.asarray(coefficients, dtype=float)
    indices = np.arange(1, len(valores) + 1)

    fig = Figure(figsize=(5, 4), dpi=100 * escala)
    FigureCanvasAgg(fig)
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


def surface_image(superficie):
    """The reconstructed surface as a wx.Image, in viridis."""
    mapa = matplotlib.colormaps['viridis']
    rango = float(superficie.max() - superficie.min())
    norm = (superficie - superficie.min()) / (rango if rango > 1e-12 else 1.0)
    rgb = (mapa(norm)[:, :, :3] * 255).astype(np.uint8)

    alto, ancho = rgb.shape[:2]
    imagen = wx.Image(ancho, alto)
    imagen.SetData(rgb.tobytes())
    return imagen


def show_legendre_reconstruction(superficie, coefficients, parent=None):
    """
    Show the reconstructed background next to the coefficient plot.

    Parameters
    ----------
    superficie : ndarray
        Surface rebuilt from the Legendre coefficients.
    coefficients : sequence of float
        The coefficients themselves, plotted on the right-hand panel.
    parent : wx.Window, optional
        Window the dialog belongs to.
    """
    app_creada = False
    app = wx.GetApp()
    if app is None:
        app = wx.App(False)
        app_creada = True

    dlg = wx.Dialog(parent, title="Legendre Reconstruction Results",
                    style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
    escala = hidpi.scale(dlg)

    principal = wx.BoxSizer(wx.VERTICAL)
    contenido = wx.BoxSizer(wx.HORIZONTAL)

    # ---------------------------------------------------- the surface
    panel_izq = wx.Panel(dlg)
    sizer_izq = wx.BoxSizer(wx.VERTICAL)
    sizer_izq.Add(wx.StaticText(panel_izq,
                                label="Legendre-reconstructed background"),
                  0, wx.ALIGN_CENTER | wx.ALL, 5)

    imagen = surface_image(superficie)

    # The reconstruction is square -- legendre_background works on a grid the
    # height of the image -- and it used to be squashed into a fixed 500x400,
    # which distorted it. It is fitted inside the box instead, keeping its
    # proportions.
    caja = dlg.FromDIP(wx.Size(400, 360))
    factor = min(caja.width / imagen.GetWidth(), caja.height / imagen.GetHeight())
    imagen = imagen.Scale(max(1, int(imagen.GetWidth() * factor)),
                          max(1, int(imagen.GetHeight() * factor)),
                          wx.IMAGE_QUALITY_HIGH)

    sizer_izq.Add(wx.StaticBitmap(panel_izq, bitmap=hidpi.bitmap(imagen, dlg)),
                  0, wx.ALIGN_CENTER | wx.ALL, 5)
    panel_izq.SetSizer(sizer_izq)
    contenido.Add(panel_izq, 0, wx.ALL, 5)

    # ---------------------------------------------------- the coefficients
    panel_der = wx.Panel(dlg)
    sizer_der = wx.BoxSizer(wx.VERTICAL)
    sizer_der.Add(wx.StaticText(panel_der, label="Legendre coefficients"),
                  0, wx.ALIGN_CENTER | wx.ALL, 5)

    fig = coefficient_figure(coefficients, escala)
    sizer_der.Add(wx.StaticBitmap(panel_der, bitmap=hidpi.figure_bitmap(fig, dlg)),
                  0, wx.ALIGN_CENTER | wx.ALL, 5)
    panel_der.SetSizer(sizer_der)
    contenido.Add(panel_der, 0, wx.ALL, 5)

    principal.Add(contenido, 1, wx.EXPAND | wx.ALL, 10)

    cerrar = wx.Button(dlg, wx.ID_CLOSE, "Close")
    cerrar.Bind(wx.EVT_BUTTON, lambda evt: dlg.Close())
    principal.Add(cerrar, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

    # The window takes the size its contents need, rather than a fixed one
    # that was right on the screen it was written on. Clamped so a very wide
    # reconstruction cannot push it off the display.
    dlg.SetSizer(principal)
    principal.Fit(dlg)
    pantalla = wx.GetClientDisplayRect()
    dlg.SetSize(wx.Size(min(dlg.GetSize().width, pantalla.width),
                        min(dlg.GetSize().height, pantalla.height)))
    dlg.Centre()

    dlg.ShowModal()
    dlg.Destroy()

    if app_creada:
        try:
            app.Destroy()
        except Exception:
            pass
