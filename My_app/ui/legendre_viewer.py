"""Viewer for the Legendre background reconstruction."""
# Split off from the analysis layer, which used to compute AND display.
import matplotlib
import numpy as np
import wx
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ui import hidpi

#: Same palette as the TSM histogram and the toolbar icons.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e5e4e0"
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"


def coefficient_figure(coefficients, scaleFactor=1.0):
    """Build the coefficient bar chart as a matplotlib figure."""
    # Not through pyplot: its wx backend doubles the dpi on a scaled screen.
    values = np.asarray(coefficients, dtype=float)
    indices = np.arange(1, len(values) + 1)

    fig = Figure(figsize=(5, 4), dpi=100 * scaleFactor)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(SURFACE)
    axes = fig.add_subplot(111)
    axes.set_facecolor(SURFACE)

    # The sign is part of the reading, so the two poles of a pair carry it.
    barColors = [POSITIVE if coefficient >= 0 else NEGATIVE for coefficient in values]
    axes.bar(indices, values, width=0.68, color=barColors,
             edgecolor=SURFACE, linewidth=2)
    axes.axhline(0, color=INK_SOFT, linewidth=1.0)

    # Every bar is labeled: the exact number is why this panel exists.
    for index, value in zip(indices, values):
        axes.annotate("%.4f" % value, (index, value),
                      xytext=(0, 4 if value >= 0 else -4),
                      textcoords="offset points", ha="center",
                      va="bottom" if value >= 0 else "top",
                      fontsize=8, color=INK)

    axes.set_xticks(indices)
    axes.set_xlabel("Legendre index", color=INK_SOFT)
    axes.set_ylabel("Coefficient value [rad]", color=INK_SOFT)
    axes.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    axes.grid(True, axis="y", color=GRID, linewidth=1.0)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.margins(y=0.18)
    fig.tight_layout()
    return fig


def surface_image(surface):
    """The reconstructed surface as a wx.Image, in viridis."""
    colorMap = matplotlib.colormaps['viridis']
    surfaceRange = float(surface.max() - surface.min())
    normalized = ((surface - surface.min())
                  / (surfaceRange if surfaceRange > 1e-12 else 1.0))
    rgbPixels = (colorMap(normalized)[:, :, :3] * 255).astype(np.uint8)

    height, width = rgbPixels.shape[:2]
    image = wx.Image(width, height)
    image.SetData(rgbPixels.tobytes())
    return image


def show_legendre_reconstruction(surface, coefficients, parent=None):
    """Show the reconstructed background next to the coefficient plot."""
    createdApp = False
    app = wx.GetApp()
    if app is None:
        app = wx.App(False)
        createdApp = True

    dialog = wx.Dialog(parent, title="Legendre Reconstruction Results",
                       style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
    scaleFactor = hidpi.scale(dialog)

    mainSizer = wx.BoxSizer(wx.VERTICAL)
    contentSizer = wx.BoxSizer(wx.HORIZONTAL)

    # ---------------------------------------------------- the surface
    leftPanel = wx.Panel(dialog)
    leftSizer = wx.BoxSizer(wx.VERTICAL)
    leftSizer.Add(wx.StaticText(leftPanel,
                                label="Legendre-reconstructed background"),
                  0, wx.ALIGN_CENTER | wx.ALL, 5)

    image = surface_image(surface)

    # The reconstruction is square and must not be squashed into the box.
    box = dialog.FromDIP(wx.Size(400, 360))
    fitFactor = min(box.width / image.GetWidth(), box.height / image.GetHeight())
    image = image.Scale(max(1, int(image.GetWidth() * fitFactor)),
                        max(1, int(image.GetHeight() * fitFactor)),
                        wx.IMAGE_QUALITY_HIGH)

    leftSizer.Add(wx.StaticBitmap(leftPanel, bitmap=hidpi.bitmap(image, dialog)),
                  0, wx.ALIGN_CENTER | wx.ALL, 5)
    leftPanel.SetSizer(leftSizer)
    contentSizer.Add(leftPanel, 0, wx.ALL, 5)

    # ---------------------------------------------------- the coefficients
    rightPanel = wx.Panel(dialog)
    rightSizer = wx.BoxSizer(wx.VERTICAL)
    rightSizer.Add(wx.StaticText(rightPanel, label="Legendre coefficients"),
                   0, wx.ALIGN_CENTER | wx.ALL, 5)

    fig = coefficient_figure(coefficients, scaleFactor)
    rightSizer.Add(wx.StaticBitmap(rightPanel,
                                   bitmap=hidpi.figure_bitmap(fig, dialog)),
                   0, wx.ALIGN_CENTER | wx.ALL, 5)
    rightPanel.SetSizer(rightSizer)
    contentSizer.Add(rightPanel, 0, wx.ALL, 5)

    mainSizer.Add(contentSizer, 1, wx.EXPAND | wx.ALL, 10)

    closeButton = wx.Button(dialog, wx.ID_CLOSE, "Close")
    closeButton.Bind(wx.EVT_BUTTON, lambda evt: dialog.Close())
    mainSizer.Add(closeButton, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

    # The size its contents need, clamped to the desktop.
    dialog.SetSizer(mainSizer)
    mainSizer.Fit(dialog)
    desktop = wx.GetClientDisplayRect()
    dialog.SetSize(wx.Size(min(dialog.GetSize().width, desktop.width),
                           min(dialog.GetSize().height, desktop.height)))
    dialog.Center()

    dialog.ShowModal()
    dialog.Destroy()

    if createdApp:
        try:
            app.Destroy()
        except Exception:
            pass
