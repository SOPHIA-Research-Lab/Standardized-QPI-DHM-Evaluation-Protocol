"""One rule for every bitmap the application puts on screen."""
# wx reads a bitmap's pixels as LOGICAL units, so each one declares its scale.
import numpy as np
import wx


def scale(window=None):
    """The screen's pixel-per-unit factor: 1.0 at 100 %, 2.0 at 200 %."""
    if window is not None:
        return window.GetDPIScaleFactor()
    # No window yet: fall back to the primary display.
    return wx.Window.FromDIP(100, None) / 100.0


def bitmap(source, window=None):
    """A wx.Bitmap from an image already rendered at its physical size."""
    scaledBitmap = source if isinstance(source, wx.Bitmap) else wx.Bitmap(source)
    scaleFactor = scale(window)
    if scaleFactor != 1.0:
        scaledBitmap.SetScaleFactor(scaleFactor)
    return scaledBitmap


def figure_bitmap(fig, window=None):
    """Render a matplotlib figure into a bitmap that draws 1:1 on screen."""
    # The figure must carry an Agg canvas at the screen's dpi, not a pyplot one.
    fig.canvas.draw()
    rgbaPixels = np.asarray(fig.canvas.buffer_rgba())
    height, width = rgbaPixels.shape[:2]
    image = wx.Image(width, height)
    image.SetData(rgbaPixels[:, :, :3].tobytes())
    return bitmap(image, window)
