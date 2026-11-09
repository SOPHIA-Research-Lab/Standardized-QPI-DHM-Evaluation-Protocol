import wx
import wx.aui
import numpy as np
from PIL import Image

from ui import hidpi


def fit_bitmap(bitmap_ctrl, pil_img, zoom):
    """Draw the image at the panel's real pixel count."""
    # Resized to PHYSICAL pixels; the scale factor keeps wx from doubling it.
    panel = bitmap_ctrl.Parent
    panelSize = panel.GetClientSize()
    imageWidth, imageHeight = pil_img.size

    if panelSize[0] > 0 and panelSize[1] > 0:
        autoScale = min(panelSize[0] / imageWidth, panelSize[1] / imageHeight)
    else:
        autoScale = 1.0

    totalScale = autoScale * zoom
    targetSize = (max(1, int(imageWidth * totalScale)), max(1, int(imageHeight * totalScale)))

    resizedImage = pil_img.resize(targetSize, Image.LANCZOS)
    rgbPixels = np.array(resizedImage.convert("RGB"))
    scaledBitmap = wx.Bitmap.FromBuffer(rgbPixels.shape[1], rgbPixels.shape[0], rgbPixels)
    bitmap_ctrl.SetBitmap(hidpi.bitmap(scaledBitmap, panel))
    panel.Layout()

class DetachedFrame(wx.Frame):
    """Independent window for images detached from the notebook."""
    
    def __init__(self, title, pil_img, notebook, page_data=None):
        super().__init__(parent=None, title=title,
                         size=wx.Window.FromDIP(wx.Size(400, 300), None))
        self.notebook = notebook
        self.pil_img = pil_img
        self.scale = 1.0
        self.page_data = page_data or {}

        panel = wx.Panel(self)
        self.bitmap_ctrl = wx.StaticBitmap(panel)
        panelSizer = wx.BoxSizer(wx.VERTICAL)
        panelSizer.Add(self.bitmap_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        panel.SetSizer(panelSizer)

        panel.Bind(wx.EVT_MOUSEWHEEL, self.on_zoom)
        self.bitmap_ctrl.Bind(wx.EVT_MOUSEWHEEL, self.on_zoom)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Bind(wx.EVT_LEFT_UP, self.check_reattach)
        self.Bind(wx.EVT_MOVE_END, self.check_reattach)

        self.update_bitmap()

    def update_bitmap(self):
        """Update bitmap with current scale."""
        self._update_bitmap(self.bitmap_ctrl, self.pil_img, self.scale)

    def _update_bitmap(self, bitmap_ctrl, pil_img, scale):
        """Update bitmap control with specified zoom level or auto-fit to panel."""
        fit_bitmap(bitmap_ctrl, pil_img, scale)

    def on_zoom(self, event):
        """Handle mouse wheel zoom."""
        self.scale *= 1.1 if event.GetWheelRotation() > 0 else 0.9
        self.scale = max(0.1, min(10.0, self.scale))
        self.update_bitmap()

    def on_close(self, event):
        """Reattach to notebook when closing."""
        self.notebook.add_image_tab(self.pil_img, self.GetTitle(), self.page_data)
        self.Destroy()

    def check_reattach(self, event):
        """Check if window should reattach to notebook on drag."""
        frameRect = self.GetScreenRect()
        mainFrame = self.notebook.GetTopLevelParent()
        mainRect = mainFrame.GetScreenRect()

        if mainRect.Intersects(frameRect):
            self.notebook.add_image_tab(self.pil_img, self.GetTitle(), self.page_data)
            self.Destroy()
        else:
            event.Skip()


class ImageNotebook(wx.aui.AuiNotebook):
    """Custom notebook for managing images with support for complex data."""
    
    def __init__(self, parent):
        style = wx.aui.AUI_NB_DEFAULT_STYLE | wx.aui.AUI_NB_TAB_EXTERNAL_MOVE
        super().__init__(parent, style=style)
        self.Bind(wx.aui.EVT_AUINOTEBOOK_END_DRAG, self.on_end_drag)
        self.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CLOSE, self.on_page_close)

    def _unique_title(self, title):
        """Return `title`, or `title (2)`, `title (3)`... if it is already taken."""
        takenTitles = {self.GetPageText(pageIndex) for pageIndex in range(self.GetPageCount())}
        if title not in takenTitles:
            return title
        suffix = 2
        while "%s (%d)" % (title, suffix) in takenTitles:
            suffix += 1
        return "%s (%d)" % (title, suffix)

    def add_image_tab(self, pil_img, title, page_data=None):
        """Add image tab with optional complex data (amplitude, phase)."""
        panel = wx.Panel(self)
        panel.pil_img = pil_img
        panel.scale_factor = 1.0

        bitmapCtrl = wx.StaticBitmap(panel)
        panel.bitmap_ctrl = bitmapCtrl

        panelSizer = wx.BoxSizer(wx.VERTICAL)
        panelSizer.Add(bitmapCtrl, 1, wx.EXPAND | wx.ALL, 5)
        panel.SetSizer(panelSizer)

        panel.Bind(wx.EVT_SIZE, lambda evt: self._update_bitmap(bitmapCtrl, pil_img, panel.scale_factor))
        wx.CallAfter(self._update_bitmap, bitmapCtrl, pil_img, panel.scale_factor)

        panel.Bind(wx.EVT_MOUSEWHEEL, self.on_zoom)
        bitmapCtrl.Bind(wx.EVT_MOUSEWHEEL, self.on_zoom)

        if page_data:
            for key, value in page_data.items():
                setattr(panel, key, value)
        else:
            panel.is_mat_complex = False
            panel.amplitude = None
            panel.phase = None

        # The tab text is the row key in the results table, so it must be unique.
        self.AddPage(panel, self._unique_title(title))
        self.SetSelection(self.GetPageCount() - 1)
        return panel

    def _update_bitmap(self, bitmap_ctrl, pil_img, scale):
        """Update bitmap control with specified zoom level or auto-fit to panel."""
        fit_bitmap(bitmap_ctrl, pil_img, scale)

    def on_zoom(self, event):
        """Handle mouse wheel zoom for specific panel."""
        eventObject = event.GetEventObject()
        panel = eventObject
        while panel and not isinstance(panel, wx.Panel):
            panel = panel.GetParent()
        
        if not panel or not hasattr(panel, 'pil_img'):
            event.Skip()
            return

        panel.scale_factor *= 1.1 if event.GetWheelRotation() > 0 else 0.9
        panel.scale_factor = max(0.1, min(10.0, panel.scale_factor))
        self._update_bitmap(panel.bitmap_ctrl, panel.pil_img, panel.scale_factor)

    def on_page_close(self, event):
        """Handle tab closing."""
        event.Skip()

    def on_end_drag(self, event):
        """Handle tab dragging outside notebook."""
        mousePosition = wx.GetMousePosition()
        topFrame = self.GetTopLevelParent()
        
        if not topFrame.GetScreenRect().Contains(mousePosition):
            pageIndex = event.GetSelection()
            if pageIndex != wx.NOT_FOUND:
                pageTitle = self.GetPageText(pageIndex)
                panel = self.GetPage(pageIndex)
                pilImage = panel.pil_img
                pageData = self._extract_page_data(panel)
                
                self.DeletePage(pageIndex)
                detachedFrame = DetachedFrame(pageTitle, pilImage, self, pageData)
                detachedFrame.Show()
        event.Skip()

    def _extract_page_data(self, panel):
        """Extract additional data from panel."""
        preserved = {}
        attributesToPreserve = ['is_mat_complex', 'amplitude', 'phase',
                                'complex_field_name', 'is_sample']
        for attributeName in attributesToPreserve:
            if hasattr(panel, attributeName):
                preserved[attributeName] = getattr(panel, attributeName)
        return preserved

    def get_current_image(self):
        """Get current PIL image."""
        pageIndex = self.GetSelection()
        if pageIndex == wx.NOT_FOUND:
            return None
        panel = self.GetPage(pageIndex)
        return getattr(panel, 'pil_img', None)

    def get_current_image_name(self):
        """Get current image name (tab title)."""
        pageIndex = self.GetSelection()
        return None if pageIndex == wx.NOT_FOUND else self.GetPageText(pageIndex)

    def get_current_page(self):
        """Get current panel with all attributes."""
        pageIndex = self.GetSelection()
        return None if pageIndex == wx.NOT_FOUND else self.GetPage(pageIndex)
