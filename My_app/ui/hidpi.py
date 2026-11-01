"""
One rule for every bitmap the application puts on screen.

wx reads a bitmap's own pixels as LOGICAL units. Hand it a 400x300 bitmap on a
200 % screen and the widget comes out 800x600, with the image stretched to
fill it: twice the size it was meant to be, and soft, because the extra pixels
were invented. That is why enabling DPI awareness made the toolbar icons grow,
the Legendre window overflow and the threshold previews crush the controls
underneath them.

Everything here is drawn at the screen's real pixel count -- the phase images,
the mask previews, the charts, the icons -- so each bitmap has to say so. A
bitmap that carries its scale factor is laid out at the size it was drawn for
and reaches the screen pixel for pixel.
"""
import numpy as np
import wx


def scale(window=None):
    """The screen's pixel-per-unit factor: 1.0 at 100 %, 2.0 at 200 %."""
    if window is not None:
        return window.GetDPIScaleFactor()
    # No window yet: fall back to the primary display.
    return wx.Window.FromDIP(100, None) / 100.0


def bitmap(source, window=None):
    """
    A wx.Bitmap that draws 1:1 on screen.

    `source` is a wx.Image or wx.Bitmap already rendered at the physical size
    it should occupy.
    """
    bmp = source if isinstance(source, wx.Bitmap) else wx.Bitmap(source)
    factor = scale(window)
    if factor != 1.0:
        bmp.SetScaleFactor(factor)
    return bmp


def figure_bitmap(fig, window=None):
    """
    Render a matplotlib figure into a bitmap of the right size.

    The figure must have been built with an Agg canvas at the screen's dpi;
    see the note in legendre_viewer.coefficient_figure about why pyplot is not
    used to make one.
    """
    fig.canvas.draw()
    datos = np.asarray(fig.canvas.buffer_rgba())
    alto, ancho = datos.shape[:2]
    imagen = wx.Image(ancho, alto)
    imagen.SetData(datos[:, :, :3].tobytes())
    return bitmap(imagen, window)
