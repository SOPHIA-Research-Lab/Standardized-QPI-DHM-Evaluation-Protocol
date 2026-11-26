"""Toolbar icons, rendered from the Tabler Icons SVG set at any size."""
# The untouched Tabler `outline` set, MIT license: only the color is changed.
import os

import wx
import wx.svg

from ui import hidpi

#: An accent for the actions, a neutral for the plain ones, red to destroy.
ACCENT = "#2a78d6"
NEUTRAL = "#52514e"
DANGER = "#d03b3b"

#: Which color each icon is drawn in.
_COLOR = {
    'open': ACCENT,
    'save': ACCENT,
    'background': ACCENT,
    'global': ACCENT,
    'ground_truth': ACCENT,
    'complexity': ACCENT,
    'imagej': ACCENT,
    'help': NEUTRAL,
    'clear': DANGER,
    'chevron_up': NEUTRAL,
    'chevron_down': NEUTRAL,
}

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons')

#: Rendered icons, keyed by name, size and screen scale.
_CACHE = {}


def _load(name):
    """Read one SVG file as text."""
    with open(os.path.join(_DIR, "%s.svg" % name), encoding='utf-8') as svgFile:
        return svgFile.read()


def icon(name, size=24, window=None):
    """The named icon as a wx.Bitmap, `size` in logical units."""
    # Drawn at the physical size and told its scale, or wx doubles it again.
    scaleFactor = hidpi.scale(window)
    cacheKey = (name, size, scaleFactor)
    if cacheKey in _CACHE:
        return _CACHE[cacheKey]

    # wx.svg has no CSS context, so Tabler's currentColor is substituted here.
    markup = _load(name).replace('currentColor', _COLOR.get(name, NEUTRAL))

    pixelSize = max(1, int(round(size * scaleFactor)))
    svgImage = wx.svg.SVGimage.CreateFromBytes(markup.encode('utf-8'))
    iconBitmap = svgImage.ConvertToScaledBitmap(wx.Size(pixelSize, pixelSize))
    if scaleFactor != 1.0:
        iconBitmap.SetScaleFactor(scaleFactor)
    _CACHE[cacheKey] = iconBitmap
    return iconBitmap


def names():
    """Every icon name available in ui/icons/."""
    return sorted(fileName[:-4] for fileName in os.listdir(_DIR) if fileName.endswith('.svg'))


#: The PhaseMark logo, one level up from the toolbar icons.
_LOGO_DIR = os.path.dirname(os.path.abspath(__file__))


def logo(height, window=None):
    """The full PhaseMark logo (ui/logo.svg) as a wx.Bitmap, `height` logical px tall."""
    scaleFactor = hidpi.scale(window)
    cacheKey = ('__logo__', height, scaleFactor)
    if cacheKey in _CACHE:
        return _CACHE[cacheKey]

    svgImage = wx.svg.SVGimage.CreateFromFile(os.path.join(_LOGO_DIR, 'logo.svg'))
    pixelHeight = max(1, int(round(height * scaleFactor)))
    pixelWidth = max(1, int(round(pixelHeight * svgImage.width / svgImage.height)))
    logoBitmap = svgImage.ConvertToScaledBitmap(wx.Size(pixelWidth, pixelHeight))
    if scaleFactor != 1.0:
        logoBitmap.SetScaleFactor(scaleFactor)
    _CACHE[cacheKey] = logoBitmap
    return logoBitmap


def app_icons():
    """The square logo (ui/logo_icon.svg) at every size Windows asks for."""
    svgImage = wx.svg.SVGimage.CreateFromFile(os.path.join(_LOGO_DIR, 'logo_icon.svg'))
    bundle = wx.IconBundle()
    for side in (16, 20, 24, 32, 40, 48, 64, 128, 256):
        bundle.AddIcon(wx.Icon(svgImage.ConvertToScaledBitmap(wx.Size(side, side))))
    return bundle
