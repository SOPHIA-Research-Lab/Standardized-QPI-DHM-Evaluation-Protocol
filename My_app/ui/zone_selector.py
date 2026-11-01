"""
Interactive selection of rectangular zones on a phase image.

This lived inside analysis/module1/residual_background.py, which meant that
importing a metric pulled wx in with it and made the analysis layer impossible
to drive from a script or a test. The maths stayed there; the window came here.

The entry point is select_manual_zones(): it shows the dialog and returns the
rectangles as (xmin, xmax, ymin, ymax) tuples in pixel coordinates.
"""
from typing import List, Tuple

import cv2
import numpy as np
import wx

from ui import hidpi


class ManualRectangleSelector:
    """Interactive rectangle selector using wxPython (replaces matplotlib version)."""
    
    def __init__(self, img: np.ndarray, num_zones: int):
        self.img = np.array(img, dtype=float)
        self.num_zones = num_zones
        self.rectangles = []
        self.finished = False
        
        # Create wx.App if needed
        self.app_created = False
        app = wx.GetApp()
        if app is None:
            self.app = wx.App(False)
            self.app_created = True
        else:
            self.app = app
        
        # Create and show dialog
        self.dialog = WxRectangleSelectorDialog(None, self.img, self.num_zones)
    
    def show(self):
        """Show the selector dialog."""
        result = self.dialog.ShowModal()
        self.rectangles = self.dialog.get_rectangles() if result == wx.ID_OK else []
        self.finished = True
        self.dialog.Destroy()
        
        # Clean up app if we created it
        if self.app_created:
            try:
                self.app.Destroy()
            except Exception:
                pass
    
    def get_rectangles(self) -> List[Tuple[int, int, int, int]]:
        """Return selected rectangles."""
        return self.rectangles


class WxRectangleSelectorDialog(wx.Dialog):
    """Interactive rectangle selector dialog using wxPython."""
    
    def __init__(self, parent, image: np.ndarray, num_zones: int):
        super().__init__(parent, title="Zone Selection", 
                        style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        
        self.image = np.array(image, dtype=float)
        self.num_zones = num_zones
        self.rectangles = []  # List of (xmin, xmax, ymin, ymax)
        self.current_rect = None  # (start_x, start_y, end_x, end_y)
        self.is_drawing = False
        self.finished = False
        
        # Convert image to uint8 for display
        img_min, img_max = self.image.min(), self.image.max()
        if img_max - img_min > 1e-12:
            img_normalized = (self.image - img_min) / (img_max - img_min)
        else:
            img_normalized = np.zeros_like(self.image)
        self.img_u8 = (img_normalized * 255).astype(np.uint8)
        
        # Calculate display size
        display_size = wx.GetDisplaySize()
        max_width = int(display_size[0] * 0.8)
        max_height = int(display_size[1] * 0.8)
        
        img_h, img_w = self.img_u8.shape
        scale_w = max_width / img_w
        scale_h = max_height / img_h
        self.scale = min(scale_w, scale_h, 1.5)  # Don't scale up too much
        
        self.display_w = int(img_w * self.scale)
        self.display_h = int(img_h * self.scale)
        
        # The margins are the title and the button row, which are text: they
        # grow with the screen scale, the image does not. Clamped to the
        # desktop, which is shorter than the screen by the taskbar.
        util = wx.GetClientDisplayRect()
        self.SetSize((min(self.display_w + self.FromDIP(40), util.width),
                      min(self.display_h + self.FromDIP(120), util.height)))
        self.Centre()
        
        self._build_ui()
        self._update_display()
    
    def _build_ui(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title and instructions
        self.title_text = wx.StaticText(
            self, 
            label=f"Select {self.num_zones} zones - Current: 0/{self.num_zones}\nClick and drag to draw rectangles"
        )
        font = self.title_text.GetFont()
        font.PointSize += 2
        font = font.Bold()
        self.title_text.SetFont(font)
        main_sizer.Add(self.title_text, 0, wx.ALL | wx.ALIGN_CENTER, 10)
        
        # Create panel for image display
        self.image_panel = wx.Panel(self, size=(self.display_w, self.display_h))
        self.image_panel.SetBackgroundColour(wx.Colour(50, 50, 50))
        
        # StaticBitmap for image
        self.bitmap_widget = wx.StaticBitmap(self.image_panel)
        self.bitmap_widget.SetPosition((0, 0))
        
        # Bind mouse events
        self.bitmap_widget.Bind(wx.EVT_LEFT_DOWN, self._on_mouse_down)
        self.bitmap_widget.Bind(wx.EVT_LEFT_UP, self._on_mouse_up)
        self.bitmap_widget.Bind(wx.EVT_MOTION, self._on_mouse_move)
        
        main_sizer.Add(self.image_panel, 1, wx.ALL | wx.EXPAND, 5)
        
        # Control buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.btn_undo = wx.Button(self, label="Undo Last")
        self.btn_clear = wx.Button(self, label="Clear All")
        self.btn_finish = wx.Button(self, label="Finish")
        self.btn_cancel = wx.Button(self, label="Cancel")
        
        self.btn_undo.Enable(False)
        self.btn_clear.Enable(False)
        
        btn_sizer.Add(self.btn_undo, 0, wx.ALL, 5)
        btn_sizer.Add(self.btn_clear, 0, wx.ALL, 5)
        btn_sizer.AddStretchSpacer()
        btn_sizer.Add(self.btn_finish, 0, wx.ALL, 5)
        btn_sizer.Add(self.btn_cancel, 0, wx.ALL, 5)
        
        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        self.SetSizer(main_sizer)
        
        # Bind button events
        self.btn_undo.Bind(wx.EVT_BUTTON, self._on_undo)
        self.btn_clear.Bind(wx.EVT_BUTTON, self._on_clear)
        self.btn_finish.Bind(wx.EVT_BUTTON, self._on_finish)
        self.btn_cancel.Bind(wx.EVT_BUTTON, self._on_cancel)
    
    def _update_display(self):
        """Redraw the image with all rectangles."""
        # Create RGB image from grayscale
        rgb_array = np.dstack([self.img_u8, self.img_u8, self.img_u8]).copy()
        
        # Draw saved rectangles in blue
        for i, (xmin, xmax, ymin, ymax) in enumerate(self.rectangles):
            cv2.rectangle(rgb_array, (xmin, ymin), (xmax, ymax), (0, 0, 255), 4)
            cv2.putText(rgb_array, f"{i+1}", (xmin + 5, ymin + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        
        # Draw current rectangle being drawn in red
        if self.current_rect is not None:
            x0, y0, x1, y1 = self.current_rect
            xmin, xmax = min(x0, x1), max(x0, x1)
            ymin, ymax = min(y0, y1), max(y0, y1)
            cv2.rectangle(rgb_array, (xmin, ymin), (xmax, ymax), (255, 0, 0), 4)
        
        # Scale image for display
        if self.scale != 1.0:
            rgb_display = cv2.resize(rgb_array, (self.display_w, self.display_h), 
                                    interpolation=cv2.INTER_LINEAR)
        else:
            rgb_display = rgb_array
        
        # Convert to wx.Bitmap
        height, width = rgb_display.shape[:2]
        wx_image = wx.Image(width, height)
        wx_image.SetData(rgb_display.tobytes())
        # The widget is sized to the bitmap's own pixels just below, so the
        # bitmap has to carry the screen's scale factor: otherwise wx draws it
        # at double size and the control shows a corner of the image, with
        # every mouse coordinate off by the same factor.
        bitmap = hidpi.bitmap(wx_image, self)

        self.bitmap_widget.SetBitmap(bitmap)
        self.bitmap_widget.SetSize((width, height))
        
        # Update title
        self.title_text.SetLabel(
            f"Select {self.num_zones} zones - Current: {len(self.rectangles)}/{self.num_zones}\n"
            f"Click and drag to draw rectangles"
        )
        
        # Update button states
        self.btn_undo.Enable(len(self.rectangles) > 0)
        self.btn_clear.Enable(len(self.rectangles) > 0)
        
        self.Refresh()
    
    def _screen_to_image_coords(self, screen_x, screen_y):
        """Convert screen coordinates to image coordinates."""
        img_x = int(screen_x / self.scale)
        img_y = int(screen_y / self.scale)
        
        # Clamp to image boundaries
        img_h, img_w = self.img_u8.shape
        img_x = max(0, min(img_x, img_w - 1))
        img_y = max(0, min(img_y, img_h - 1))
        
        return img_x, img_y
    
    def _on_mouse_down(self, event):
        """Handle mouse button press."""
        if self.finished:
            return
        
        pos = event.GetPosition()
        img_x, img_y = self._screen_to_image_coords(pos.x, pos.y)
        
        self.is_drawing = True
        self.current_rect = (img_x, img_y, img_x, img_y)
        self._update_display()
    
    def _on_mouse_move(self, event):
        """Handle mouse movement."""
        if not self.is_drawing or self.finished:
            return
        
        pos = event.GetPosition()
        img_x, img_y = self._screen_to_image_coords(pos.x, pos.y)
        
        if self.current_rect is not None:
            x0, y0 = self.current_rect[:2]
            self.current_rect = (x0, y0, img_x, img_y)
            self._update_display()
    
    def _on_mouse_up(self, event):
        """Handle mouse button release."""
        if not self.is_drawing or self.finished:
            return
        
        pos = event.GetPosition()
        img_x, img_y = self._screen_to_image_coords(pos.x, pos.y)
        
        if self.current_rect is not None:
            x0, y0 = self.current_rect[:2]
            x1, y1 = img_x, img_y
            
            xmin, xmax = min(x0, x1), max(x0, x1)
            ymin, ymax = min(y0, y1), max(y0, y1)
            
            # Only add if rectangle has some area
            if xmax > xmin + 2 and ymax > ymin + 2:
                self.rectangles.append((xmin, xmax, ymin, ymax))                
                # Auto-finish if we reached target number
                if len(self.rectangles) >= self.num_zones:
                    wx.CallAfter(self._auto_finish)
        
        self.is_drawing = False
        self.current_rect = None
        self._update_display()
    
    def _on_undo(self, event):
        """Remove last rectangle."""
        if self.rectangles:
            removed = self.rectangles.pop()
            self._update_display()
    
    def _on_clear(self, event):
        """Clear all rectangles."""
        if self.rectangles:
            dlg = wx.MessageDialog(
                self,
                f"Clear all {len(self.rectangles)} zones?",
                "Confirm Clear",
                wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION
            )
            if dlg.ShowModal() == wx.ID_YES:
                self.rectangles.clear()
                self._update_display()
            dlg.Destroy()
    
    def _auto_finish(self):
        """Auto-finish when target reached."""
        self.finished = True
        self.EndModal(wx.ID_OK)
    
    def _on_finish(self, event):
        """Manual finish via button."""
        if len(self.rectangles) == 0:
            dlg = wx.MessageDialog(
                self,
                "No zones selected. Do you want to cancel?",
                "No Zones",
                wx.YES_NO | wx.ICON_WARNING
            )
            if dlg.ShowModal() == wx.ID_YES:
                dlg.Destroy()
                self._on_cancel(None)
                return
            dlg.Destroy()
            return
        
        self.finished = True
        self.EndModal(wx.ID_OK)
    
    def _on_cancel(self, event):
        """Cancel selection."""
        if self.rectangles:
            dlg = wx.MessageDialog(
                self,
                f"Cancel and discard {len(self.rectangles)} zones?",
                "Confirm Cancel",
                wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION
            )
            result = dlg.ShowModal()
            dlg.Destroy()
            if result != wx.ID_YES:
                return
        
        self.rectangles.clear()
        self.finished = False
        self.EndModal(wx.ID_CANCEL)
    
    def get_rectangles(self):
        """Return selected rectangles."""
        return self.rectangles


def select_manual_zones(img: np.ndarray, num_zones: int) -> List[Tuple[int, int, int, int]]:
    """Select rectangular areas manually using wxPython. Auto-finishes when zones completed."""
    selector = ManualRectangleSelector(img, num_zones)
    selector.show()
    
    zones = selector.get_rectangles()
    # if len(zones) == 0:
    #     print("WARNING: User did not select any zones.")
    return zones
