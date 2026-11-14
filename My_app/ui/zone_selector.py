"""Interactive selection of rectangular zones on a phase image."""
# Split off from residual_background, which pulled wx into the analysis layer.
from typing import List, Tuple

import cv2
import numpy as np
import wx

from ui import hidpi


class ManualRectangleSelector:
    """Owns the dialog and hands back the rectangles the user drew."""
    
    def __init__(self, img: np.ndarray, num_zones: int):
        self.img = np.array(img, dtype=float)
        self.num_zones = num_zones
        self.rectangles = []
        self.finished = False
        
        self.app_created = False
        app = wx.GetApp()
        if app is None:
            self.app = wx.App(False)
            self.app_created = True
        else:
            self.app = app
        
        self.dialog = WxRectangleSelectorDialog(None, self.img, self.num_zones)
    
    def show(self):
        """Show the selector dialog."""
        answer = self.dialog.ShowModal()
        self.rectangles = self.dialog.get_rectangles() if answer == wx.ID_OK else []
        self.finished = True
        self.dialog.Destroy()
        
        if self.app_created:
            try:
                self.app.Destroy()
            except Exception:
                pass
    
    def get_rectangles(self) -> List[Tuple[int, int, int, int]]:
        """The rectangles the user drew, in image pixel coordinates."""
        return self.rectangles


class WxRectangleSelectorDialog(wx.Dialog):
    """Draw rectangles on the image; coordinates come back in image pixels."""
    
    def __init__(self, parent, image: np.ndarray, num_zones: int):
        super().__init__(parent, title="Zone Selection", 
                        style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        
        self.image = np.array(image, dtype=float)
        self.num_zones = num_zones
        self.rectangles = []      # (xmin, xmax, ymin, ymax) each
        self.current_rect = None  # (startX, startY, endX, endY)
        self.is_drawing = False
        self.finished = False
        
        phaseMin, phaseMax = self.image.min(), self.image.max()
        if phaseMax - phaseMin > 1e-12:
            normalizedPhase = (self.image - phaseMin) / (phaseMax - phaseMin)
        else:
            normalizedPhase = np.zeros_like(self.image)
        self.img_u8 = (normalizedPhase * 255).astype(np.uint8)
        
        screenSize = wx.GetDisplaySize()
        maxWidth = int(screenSize[0] * 0.8)
        maxHeight = int(screenSize[1] * 0.8)
        
        imageHeight, imageWidth = self.img_u8.shape
        widthScale = maxWidth / imageWidth
        heightScale = maxHeight / imageHeight
        # Capped at 1.5 so a small image is not blown up past recognition.
        self.scale = min(widthScale, heightScale, 1.5)
        
        self.display_w = int(imageWidth * self.scale)
        self.display_h = int(imageHeight * self.scale)
        
        # The margins are text, so they grow with the scale; the image does not.
        desktop = wx.GetClientDisplayRect()
        self.SetSize((min(self.display_w + self.FromDIP(40), desktop.width),
                      min(self.display_h + self.FromDIP(120), desktop.height)))
        self.Center()
        
        self._build_ui()
        self._update_display()
    
    def _build_ui(self):
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        
        self.title_text = wx.StaticText(
            self, 
            label=f"Select {self.num_zones} zones - Current: 0/{self.num_zones}\nClick and drag to draw rectangles"
        )
        titleFont = self.title_text.GetFont()
        titleFont.PointSize += 2
        titleFont = titleFont.Bold()
        self.title_text.SetFont(titleFont)
        mainSizer.Add(self.title_text, 0, wx.ALL | wx.ALIGN_CENTER, 10)
        
        self.image_panel = wx.Panel(self, size=(self.display_w, self.display_h))
        self.image_panel.SetBackgroundColour(wx.Colour(50, 50, 50))
        
        self.bitmap_widget = wx.StaticBitmap(self.image_panel)
        self.bitmap_widget.SetPosition((0, 0))
        
        self.bitmap_widget.Bind(wx.EVT_LEFT_DOWN, self._on_mouse_down)
        self.bitmap_widget.Bind(wx.EVT_LEFT_UP, self._on_mouse_up)
        self.bitmap_widget.Bind(wx.EVT_MOTION, self._on_mouse_move)
        
        mainSizer.Add(self.image_panel, 1, wx.ALL | wx.EXPAND, 5)
        
        buttonRow = wx.BoxSizer(wx.HORIZONTAL)
        
        self.btn_undo = wx.Button(self, label="Undo Last")
        self.btn_clear = wx.Button(self, label="Clear All")
        self.btn_finish = wx.Button(self, label="Finish")
        self.btn_cancel = wx.Button(self, label="Cancel")
        
        self.btn_undo.Enable(False)
        self.btn_clear.Enable(False)
        
        buttonRow.Add(self.btn_undo, 0, wx.ALL, 5)
        buttonRow.Add(self.btn_clear, 0, wx.ALL, 5)
        buttonRow.AddStretchSpacer()
        buttonRow.Add(self.btn_finish, 0, wx.ALL, 5)
        buttonRow.Add(self.btn_cancel, 0, wx.ALL, 5)
        
        mainSizer.Add(buttonRow, 0, wx.EXPAND | wx.ALL, 5)
        
        self.SetSizer(mainSizer)
        
        # Enter finishes, Escape cancels. The buttons carry their own ids, so
        # the dialog has to be told which one each key stands for.
        self.btn_finish.SetDefault()
        self.SetAffirmativeId(self.btn_finish.GetId())
        self.SetEscapeId(self.btn_cancel.GetId())

        self.btn_undo.Bind(wx.EVT_BUTTON, self._on_undo)
        self.btn_clear.Bind(wx.EVT_BUTTON, self._on_clear)
        self.btn_finish.Bind(wx.EVT_BUTTON, self._on_finish)
        self.btn_cancel.Bind(wx.EVT_BUTTON, self._on_cancel)
    
    def _update_display(self):
        """Redraw the image with every rectangle drawn so far."""
        rgbPixels = np.dstack([self.img_u8, self.img_u8, self.img_u8]).copy()
        
        # Saved zones in blue, the one being dragged in red.
        for zoneNumber, (xMin, xMax, yMin, yMax) in enumerate(self.rectangles):
            cv2.rectangle(rgbPixels, (xMin, yMin), (xMax, yMax), (0, 0, 255), 4)
            cv2.putText(rgbPixels, f"{zoneNumber+1}", (xMin + 5, yMin + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        
        if self.current_rect is not None:
            startX, startY, endX, endY = self.current_rect
            xMin, xMax = min(startX, endX), max(startX, endX)
            yMin, yMax = min(startY, endY), max(startY, endY)
            cv2.rectangle(rgbPixels, (xMin, yMin), (xMax, yMax), (255, 0, 0), 4)
        
        if self.scale != 1.0:
            displayPixels = cv2.resize(rgbPixels, (self.display_w, self.display_h), 
                                    interpolation=cv2.INTER_LINEAR)
        else:
            displayPixels = rgbPixels
        
        bitmapHeight, bitmapWidth = displayPixels.shape[:2]
        image = wx.Image(bitmapWidth, bitmapHeight)
        image.SetData(displayPixels.tobytes())
        # Without the scale factor wx draws it double and the clicks miss.
        scaledBitmap = hidpi.bitmap(image, self)

        self.bitmap_widget.SetBitmap(scaledBitmap)
        self.bitmap_widget.SetSize((bitmapWidth, bitmapHeight))
        
        self.title_text.SetLabel(
            f"Select {self.num_zones} zones - Current: {len(self.rectangles)}/{self.num_zones}\n"
            f"Click and drag to draw rectangles"
        )
        
        self.btn_undo.Enable(len(self.rectangles) > 0)
        self.btn_clear.Enable(len(self.rectangles) > 0)
        
        self.Refresh()
    
    def _screen_to_image_coords(self, screen_x, screen_y):
        """Convert a position inside the widget to image pixel coordinates."""
        imageX = int(screen_x / self.scale)
        imageY = int(screen_y / self.scale)

        imageHeight, imageWidth = self.img_u8.shape
        imageX = max(0, min(imageX, imageWidth - 1))
        imageY = max(0, min(imageY, imageHeight - 1))
        
        return imageX, imageY
    
    def _on_mouse_down(self, event):
        """Start a new rectangle where the button went down."""
        if self.finished:
            return
        
        position = event.GetPosition()
        imageX, imageY = self._screen_to_image_coords(position.x, position.y)
        
        self.is_drawing = True
        self.current_rect = (imageX, imageY, imageX, imageY)
        self._update_display()
    
    def _on_mouse_move(self, event):
        """Stretch the rectangle being drawn to follow the pointer."""
        if not self.is_drawing or self.finished:
            return
        
        position = event.GetPosition()
        imageX, imageY = self._screen_to_image_coords(position.x, position.y)
        
        if self.current_rect is not None:
            startX, startY = self.current_rect[:2]
            self.current_rect = (startX, startY, imageX, imageY)
            self._update_display()
    
    def _on_mouse_up(self, event):
        """Close the rectangle, keeping it only if it has some area."""
        if not self.is_drawing or self.finished:
            return
        
        position = event.GetPosition()
        imageX, imageY = self._screen_to_image_coords(position.x, position.y)
        
        if self.current_rect is not None:
            startX, startY = self.current_rect[:2]
            endX, endY = imageX, imageY
            
            xMin, xMax = min(startX, endX), max(startX, endX)
            yMin, yMax = min(startY, endY), max(startY, endY)
            
            if xMax > xMin + 2 and yMax > yMin + 2:
                self.rectangles.append((xMin, xMax, yMin, yMax))
                # Closing on its own once the asked-for zones are there.
                if len(self.rectangles) >= self.num_zones:
                    wx.CallAfter(self._auto_finish)
        
        self.is_drawing = False
        self.current_rect = None
        self._update_display()
    
    def _on_undo(self, event):
        """Remove the last rectangle."""
        if self.rectangles:
            self.rectangles.pop()
            self._update_display()
    
    def _on_clear(self, event):
        """Drop every rectangle, after confirming."""
        if self.rectangles:
            dialog = wx.MessageDialog(
                self,
                f"Clear all {len(self.rectangles)} zones?",
                "Confirm Clear",
                wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION
            )
            if dialog.ShowModal() == wx.ID_YES:
                self.rectangles.clear()
                self._update_display()
            dialog.Destroy()
    
    def _auto_finish(self):
        """Close as accepted once the asked-for number of zones is drawn."""
        self.finished = True
        self.EndModal(wx.ID_OK)
    
    def _on_finish(self, event):
        """Accept what has been drawn, warning if nothing has."""
        if len(self.rectangles) == 0:
            dialog = wx.MessageDialog(
                self,
                "No zones selected. Do you want to cancel?",
                "No Zones",
                wx.YES_NO | wx.ICON_WARNING
            )
            if dialog.ShowModal() == wx.ID_YES:
                dialog.Destroy()
                self._on_cancel(None)
                return
            dialog.Destroy()
            return
        
        self.finished = True
        self.EndModal(wx.ID_OK)
    
    def _on_cancel(self, event):
        """Discard the selection, after confirming if there is anything to lose."""
        if self.rectangles:
            dialog = wx.MessageDialog(
                self,
                f"Cancel and discard {len(self.rectangles)} zones?",
                "Confirm Cancel",
                wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION
            )
            answer = dialog.ShowModal()
            dialog.Destroy()
            if answer != wx.ID_YES:
                return
        
        self.rectangles.clear()
        self.finished = False
        self.EndModal(wx.ID_CANCEL)
    
    def get_rectangles(self):
        """The rectangles drawn so far."""
        return self.rectangles


def select_manual_zones(img: np.ndarray, num_zones: int) -> List[Tuple[int, int, int, int]]:
    """Ask the user to draw `num_zones` rectangles; returns them, or an empty list."""
    selector = ManualRectangleSelector(img, num_zones)
    selector.show()
    
    return selector.get_rectangles()
