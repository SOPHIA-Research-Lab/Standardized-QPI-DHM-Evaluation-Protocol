import wx
import wx.aui
import numpy as np
from PIL import Image

class DetachedFrame(wx.Frame):
    """Independent window for images detached from the notebook."""
    
    def __init__(self, parent, title, pil_img, notebook, page_data=None):
        super().__init__(parent=None, title=title, size=(400, 300))
        self.notebook = notebook
        self.pil_img = pil_img
        self.scale = 1.0
        self.page_data = page_data or {}

        panel = wx.Panel(self)
        self.bitmap_ctrl = wx.StaticBitmap(panel)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.bitmap_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        panel.SetSizer(sizer)

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
        panel = bitmap_ctrl.Parent
        panel_size = panel.GetClientSize()
        img_w, img_h = pil_img.size

        # Si el panel tiene tamaño válido, ajustar automáticamente
        if panel_size[0] > 0 and panel_size[1] > 0:
            # Calcular escala automática inicial
            auto_scale = min(panel_size[0] / img_w, panel_size[1] / img_h)
        else:
            auto_scale = 1.0

        # Aplicar el zoom del usuario (rueda del mouse)
        total_scale = auto_scale * scale
        new_size = (max(1, int(img_w * total_scale)), max(1, int(img_h * total_scale)))

        resized = pil_img.resize(new_size, Image.LANCZOS)
        arr = np.array(resized.convert("RGB"))
        bitmap = wx.Bitmap.FromBuffer(arr.shape[1], arr.shape[0], arr)
        bitmap_ctrl.SetBitmap(bitmap)
        panel.Layout()

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
        frame_rect = self.GetScreenRect()
        main_frame = self.notebook.GetTopLevelParent()
        main_rect = main_frame.GetScreenRect()

        if main_rect.Intersects(frame_rect):
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

    def add_image_tab(self, pil_img, title, page_data=None):
        """Add image tab with optional complex data (amplitude, phase)."""
        panel = wx.Panel(self)
        panel.pil_img = pil_img
        panel.scale_factor = 1.0  # User zoom (mouse wheel)

        bitmap_ctrl = wx.StaticBitmap(panel)
        panel.bitmap_ctrl = bitmap_ctrl

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(bitmap_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        panel.SetSizer(sizer)

        # Adjust image to fit panel size when creating tab
        panel.Bind(wx.EVT_SIZE, lambda evt: self._update_bitmap(bitmap_ctrl, pil_img, panel.scale_factor))
        wx.CallAfter(self._update_bitmap, bitmap_ctrl, pil_img, panel.scale_factor)

        # Allow zoom with mouse wheel
        panel.Bind(wx.EVT_MOUSEWHEEL, self.on_zoom)
        bitmap_ctrl.Bind(wx.EVT_MOUSEWHEEL, self.on_zoom)

        # Save additional data (amplitude, phase, etc.)
        if page_data:
            for key, value in page_data.items():
                setattr(panel, key, value)
        else:
            panel.is_mat_complex = False
            panel.amplitude = None
            panel.phase = None

        self.AddPage(panel, title)
        self.SetSelection(self.GetPageCount() - 1)
        return panel

    def _update_bitmap(self, bitmap_ctrl, pil_img, scale):
        """Update bitmap control with specified zoom level or auto-fit to panel."""
        panel = bitmap_ctrl.Parent
        panel_size = panel.GetClientSize()
        img_w, img_h = pil_img.size

        # If the panel has a valid size, calculate auto-fit
        if panel_size[0] > 0 and panel_size[1] > 0:
            auto_scale = min(panel_size[0] / img_w, panel_size[1] / img_h)
        else:
            auto_scale = 1.0

        total_scale = auto_scale * scale
        new_size = (max(1, int(img_w * total_scale)), max(1, int(img_h * total_scale)))

        resized = pil_img.resize(new_size, Image.LANCZOS)
        arr = np.array(resized.convert("RGB"))
        bitmap = wx.Bitmap.FromBuffer(arr.shape[1], arr.shape[0], arr)
        bitmap_ctrl.SetBitmap(bitmap)
        panel.Layout()

    def on_zoom(self, event):
        """Handle mouse wheel zoom for specific panel."""
        event_obj = event.GetEventObject()
        panel = event_obj
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
        mouse = wx.GetMousePosition()
        frame = self.GetTopLevelParent()
        
        if not frame.GetScreenRect().Contains(mouse):
            idx = event.GetSelection()
            if idx != wx.NOT_FOUND:
                title = self.GetPageText(idx)
                panel = self.GetPage(idx)
                pil_img = panel.pil_img
                page_data = self._extract_page_data(panel)
                
                self.DeletePage(idx)
                detached = DetachedFrame(self.GetTopLevelParent(), title, pil_img, self, page_data)
                detached.Show()
        event.Skip()

    def _extract_page_data(self, panel):
        """Extract additional data from panel."""
        data = {}
        attrs_to_preserve = ['is_mat_complex', 'amplitude', 'phase', 'complex_field_name']
        for attr in attrs_to_preserve:
            if hasattr(panel, attr):
                data[attr] = getattr(panel, attr)
        return data

    def get_current_image(self):
        """Get current PIL image."""
        idx = self.GetSelection()
        if idx == wx.NOT_FOUND:
            return None
        panel = self.GetPage(idx)
        return getattr(panel, 'pil_img', None)

    def get_current_image_name(self):
        """Get current image name (tab title)."""
        idx = self.GetSelection()
        return None if idx == wx.NOT_FOUND else self.GetPageText(idx)

    def get_current_page(self):
        """Get current panel with all attributes."""
        idx = self.GetSelection()
        return None if idx == wx.NOT_FOUND else self.GetPage(idx)

    def get_current_phase_data(self):
        """Get phase data from current image if available."""
        page = self.get_current_page()
        if page is None:
            return None
        if hasattr(page, 'is_mat_complex') and page.is_mat_complex:
            return getattr(page, 'phase', None)
        return None