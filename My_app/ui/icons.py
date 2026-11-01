"""
Toolbar icons, from Tabler Icons.

The SVG files live in ui/icons/ and are rendered at whatever size the toolbar
asks for, so they stay sharp at any DPI and there is no set of bitmaps to keep
in three sizes.

Source: https://github.com/tabler/tabler-icons -- MIT licence, no attribution
required. The files are the untouched `outline` set; only the colour is
substituted at load time, so they can be refreshed by dropping in newer ones.

Tabler names behind each action, for whoever has to change one:

    open         folder-open         save          file-spreadsheet
    background   border-outer        global        photo
    ground_truth columns-2           complexity    chart-line
    clear        trash               chevron_up    chevron-up
                                     chevron_down  chevron-down
"""
import os

import wx
import wx.svg

from ui import hidpi

#: One accent for the actions, a neutral for the plain ones, red for the
#: destructive one. The accent matches the Legendre coefficient chart.
ACCENT = "#2a78d6"
NEUTRAL = "#52514e"
DANGER = "#d03b3b"

#: Which colour each icon is drawn in.
_COLOUR = {
    'open': ACCENT,
    'save': ACCENT,
    'background': ACCENT,
    'global': ACCENT,
    'ground_truth': ACCENT,
    'complexity': ACCENT,
    'clear': DANGER,
    'chevron_up': NEUTRAL,
    'chevron_down': NEUTRAL,
}

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons')

#: Rendering an SVG is cheap but not free, and the toolbar asks for the same
#: handful over and over on every redraw of a detached window. Keyed by screen
#: scale too: the same icon is a different number of pixels on each monitor.
_CACHE = {}


def _load(nombre):
    with open(os.path.join(_DIR, '%s.svg' % nombre), encoding='utf-8') as f:
        return f.read()


def icon(nombre, size=24, window=None):
    """
    The named icon as a wx.Bitmap, `size` in logical units.

    It is DRAWN at the physical size the screen needs and then told its own
    scale factor. Without that last step wx reads the bitmap's pixels as
    logical units: a 48 px icon on a 200 % screen is taken to mean 48 units,
    wx asks the SVG for 96 physical pixels, and the toolbar ends up with icons
    twice the size they should be. This is the whole reason the icons are SVG
    -- at any scale they are rendered, never stretched.
    """
    factor = hidpi.scale(window)
    clave = (nombre, size, factor)
    if clave in _CACHE:
        return _CACHE[clave]

    texto = _load(nombre)
    # Tabler ships stroke="currentColor"; wx.svg has no CSS context to resolve
    # it, so the colour is substituted here.
    texto = texto.replace('currentColor', _COLOUR.get(nombre, NEUTRAL))

    lado = max(1, int(round(size * factor)))
    svg = wx.svg.SVGimage.CreateFromBytes(texto.encode('utf-8'))
    bmp = svg.ConvertToScaledBitmap(wx.Size(lado, lado))
    if factor != 1.0:
        bmp.SetScaleFactor(factor)
    _CACHE[clave] = bmp
    return bmp


def names():
    return sorted(n[:-4] for n in os.listdir(_DIR) if n.endswith('.svg'))
