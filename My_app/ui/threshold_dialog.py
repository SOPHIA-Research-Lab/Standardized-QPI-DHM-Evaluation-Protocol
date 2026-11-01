"""
Interactive threshold selection for the background mask.

This lived inside analysis/module1/utilitiesRBPV.py, so importing anything from
the analysis layer pulled wx in with it. The dialog moved here; deciding what
the chosen threshold means for the mask stayed there, in build_background_mask.

The entry point is create_background_mask(): it asks the user for a threshold,
then hands it to the analysis layer. Returns (background_mask, threshold), or
(None, None) if the dialog was cancelled.
"""
import cv2
import numpy as np
import wx

from analysis.module1.utilitiesRBPV import build_background_mask
from ui import hidpi


#: The welcome text explains the tool, not this particular image, so it is shown
#: once per session. Segmentation is asked per image -- each one gets its own
#: threshold -- which used to mean dismissing the same instructions N times.
_WELCOME_SHOWN = False


def numpy_to_wx_bitmap(arr: np.ndarray, target_size=(400, 400), window=None):
    """
    Convert numpy array (H,W) or (H,W,3) to wx.Bitmap, fitting target_size.

    `target_size` is in physical pixels, and `window` is what tells the bitmap
    the screen's scale -- without it wx lays the result out at twice the size
    on a 200 % display. See ui/hidpi.py.
    """
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    
    if arr.ndim == 2:
        h, w = arr.shape
        rgb = np.dstack([arr, arr, arr])
    else:
        h, w, _ = arr.shape
        rgb = arr
    
    img = wx.Image(w, h)
    img.SetData(rgb.tobytes())

    # Calculate scaling to fit within target_size while maintaining aspect ratio
    target_w, target_h = target_size
    scale_w = target_w / w
    scale_h = target_h / h
    scale = min(scale_w, scale_h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Always scale to fit target size
    if scale != 1.0:
        img = img.Scale(new_w, new_h, wx.IMAGE_QUALITY_HIGH)

    return hidpi.bitmap(img, window)


# ------------------------------------------------
# wx Dialog for threshold
# ------------------------------------------------
class WxThresholdDialog(wx.Dialog):
    """Interactive threshold selector using wxPython."""

    def __init__(self, parent, image: np.ndarray, initial_threshold: float = None,
                 min_area: int = 100, hole_area: int = 50, title="Threshold Selector"):
        super().__init__(parent, title=title, style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        
        self.image = np.array(image, dtype=float)
        self.min_area = min_area
        self.hole_area = hole_area

        # Normalize image to 0-1 range
        img = self.image.copy()
        mi, ma = img.min(), img.max()
        if ma - mi < 1e-12:
            self.img_normalized = np.zeros_like(img, dtype=float)
        else:
            self.img_normalized = (img - mi) / (ma - mi)
        
        # Convert to uint8 for display
        self.img_u8 = (self.img_normalized * 255).astype(np.uint8)

        # Calculate Otsu threshold in 0-1 range
        thr_u8, _ = cv2.threshold(self.img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.otsu_thr = float(thr_u8) / 255.0
        
        if initial_threshold is None:
            self.current_threshold = self.otsu_thr
        else:
            self.current_threshold = float(np.clip(initial_threshold, 0.0, 1.0))

        # The preview is shown at about 450x350 px, so computing it at full
        # resolution is wasted work: 10.7 s per update on a 4096x4096 image
        # against 0.38 s on a 1024-px copy. Only the PREVIEW is subsampled --
        # the mask returned on OK is rebuilt at full resolution by
        # create_background_mask().
        self.preview_step = max(1, max(self.image.shape) // 1024)

        self.sample_is_white = True
        self.confirmed = False
        self.final_threshold = self.current_threshold
        
        # Preview size, in physical pixels: a share of the screen, capped at
        # a comfortable size expressed in logical units so the cap means the
        # same thing on every display.
        util = wx.GetClientDisplayRect()
        self.preview_w = min(self.FromDIP(420), int(util.width * 0.26))
        self.preview_h = min(self.FromDIP(330), int(util.height * 0.35))
        self.preview_size = (self.preview_w, self.preview_h)

        self._build_ui()
        self._update_previews()

        # The window asks its contents how big it has to be. It used to be
        # given the previews plus 280 px for everything underneath, which at
        # 200 % was half of what the controls needed: the threshold slider was
        # squeezed into a few pixels and its label cut off mid-word.
        self.GetSizer().Fit(self)
        self.SetSize(wx.Size(min(self.GetSize().width, util.width),
                             min(self.GetSize().height, util.height)))
        self.Centre()
        wx.CallAfter(self._show_welcome_message)

    def _show_welcome_message(self):
        global _WELCOME_SHOWN
        if _WELCOME_SHOWN:
            return
        _WELCOME_SHOWN = True

        dlg = wx.MessageDialog(
            self,
            "Welcome to the Threshold Selector.\n\n"
            "Instructions:\n"
            "Now adjust the binarization threshold that accurately \n" \
            "discriminates between background and sample regions.\n\n"
            "Verify the Binary Preview, Sample Mask and Background Mask.\n"
            "Press OK to continue.",
            "Threshold Selection",
            wx.OK | wx.ICON_INFORMATION
        )
        dlg.ShowModal()
        dlg.Destroy()

    def _build_ui(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Create grid with panels for better control
        grid = wx.GridSizer(rows=2, cols=2, gap=(10, 10))

        # Create panels for each preview with fixed size
        self.panel_original = self._create_preview_panel("Original Image")
        self.panel_binary = self._create_preview_panel("Binary Preview")
        self.panel_sample = self._create_preview_panel("Sample Mask")
        self.panel_background = self._create_preview_panel("Background Mask")
        
        grid.Add(self.panel_original, 1, wx.EXPAND)
        grid.Add(self.panel_binary, 1, wx.EXPAND)
        grid.Add(self.panel_sample, 1, wx.EXPAND)
        grid.Add(self.panel_background, 1, wx.EXPAND)
        
        main_sizer.Add(grid, 1, wx.ALL | wx.EXPAND, 10)

        ctrl_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Radio buttons for polarity
        radio_box = wx.StaticBox(self, label="Sample Polarity")
        radio_sizer = wx.StaticBoxSizer(radio_box, orient=wx.VERTICAL)
        self.radio_white = wx.RadioButton(self, label="White sample", style=wx.RB_GROUP)
        self.radio_black = wx.RadioButton(self, label="Black sample")
        self.radio_white.SetValue(True)
        radio_sizer.Add(self.radio_white, 0, wx.ALL, 5)
        radio_sizer.Add(self.radio_black, 0, wx.ALL, 5)
        ctrl_sizer.Add(radio_sizer, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 10)

        # Threshold controls
        mid_sizer = wx.BoxSizer(wx.VERTICAL)
        self.slider_label = wx.StaticText(self, label=f"Threshold (0–1): {self.current_threshold:.4f}")
        mid_sizer.Add(self.slider_label, 0, wx.LEFT | wx.TOP, 6)
        
        # Slider with 1000 steps for precision (0.001 increments)
        self.slider = wx.Slider(self, value=int(self.current_threshold * 1000),
                                minValue=0, maxValue=1000, style=wx.SL_HORIZONTAL)
        mid_sizer.Add(self.slider, 0, wx.EXPAND | wx.ALL, 6)
        
        otsu_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_reset_otsu = wx.Button(self, label=f"Reset to Otsu ({self.otsu_thr:.4f})")
        self.info_text = wx.StaticText(self, label="")
        otsu_sizer.Add(self.btn_reset_otsu, 0, wx.RIGHT, 8)
        otsu_sizer.Add(self.info_text, 1, wx.ALIGN_CENTER_VERTICAL)
        mid_sizer.Add(otsu_sizer, 0, wx.EXPAND | wx.ALL, 6)
        ctrl_sizer.Add(mid_sizer, 1, wx.EXPAND | wx.ALL, 6)

        # Cleanup sizes. These were hardcoded (min_area=100, hole_area=50)
        # and invisible, yet they decide whether small objects survive at all:
        # on the probiotics sample, min_area 100 vs 10 changes the detected
        # sample area by 25x. Small samples were being erased in silence.
        clean_box = wx.StaticBox(self, label="Cleanup")
        clean_sizer = wx.StaticBoxSizer(clean_box, orient=wx.VERTICAL)

        fila_obj = wx.BoxSizer(wx.HORIZONTAL)
        fila_obj.Add(wx.StaticText(self, label="Discard objects under"), 0,
                     wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.spin_min_area = wx.SpinCtrl(self, min=1, max=100000,
                                         initial=self.min_area,
                                         size=self.FromDIP(wx.Size(80, -1)))
        fila_obj.Add(self.spin_min_area, 0, wx.ALIGN_CENTER_VERTICAL)
        fila_obj.Add(wx.StaticText(self, label="px"), 0,
                     wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 4)
        clean_sizer.Add(fila_obj, 0, wx.ALL, 4)

        fila_hue = wx.BoxSizer(wx.HORIZONTAL)
        fila_hue.Add(wx.StaticText(self, label="Fill holes under"), 0,
                     wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.spin_hole_area = wx.SpinCtrl(self, min=1, max=100000,
                                          initial=self.hole_area,
                                          size=self.FromDIP(wx.Size(80, -1)))
        fila_hue.Add(self.spin_hole_area, 0, wx.ALIGN_CENTER_VERTICAL)
        fila_hue.Add(wx.StaticText(self, label="px"), 0,
                     wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 4)
        clean_sizer.Add(fila_hue, 0, wx.ALL, 4)

        ctrl_sizer.Add(clean_sizer, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 10)

        # OK/Cancel buttons
        btn_sizer = wx.BoxSizer(wx.VERTICAL)
        self.btn_ok = wx.Button(self, label="OK")
        self.btn_cancel = wx.Button(self, label="Cancel")
        btn_sizer.Add(self.btn_ok, 0, wx.ALL, 6)
        btn_sizer.Add(self.btn_cancel, 0, wx.ALL, 6)
        ctrl_sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        
        main_sizer.Add(ctrl_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
        self.SetSizer(main_sizer)

        # Bind events
        # Recomputing the masks costs ~0.12 s on a 1280x960 image, and the
        # slider has 1000 steps: dragging it used to queue dozens of those and
        # the dialog stopped responding. Events are coalesced instead, so a
        # drag recomputes once, when the hand stops.
        self._preview_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, lambda evt: self._update_previews(),
                  self._preview_timer)

        self.slider.Bind(wx.EVT_SLIDER, self._on_slider)
        self.radio_white.Bind(wx.EVT_RADIOBUTTON, self._on_radio)
        self.radio_black.Bind(wx.EVT_RADIOBUTTON, self._on_radio)
        self.spin_min_area.Bind(wx.EVT_SPINCTRL, self._on_cleanup)
        self.spin_hole_area.Bind(wx.EVT_SPINCTRL, self._on_cleanup)
        self.btn_reset_otsu.Bind(wx.EVT_BUTTON, self._on_reset_otsu)
        self.btn_ok.Bind(wx.EVT_BUTTON, self._on_ok)
        self.btn_cancel.Bind(wx.EVT_BUTTON, self._on_cancel)

    def _create_preview_panel(self, label_text):
        """Create a panel with label and StaticBitmap for preview."""
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Label
        label = wx.StaticText(panel, label=label_text)
        sizer.Add(label, 0, wx.ALIGN_CENTER | wx.TOP, 2)
        
        # StaticBitmap with fixed minimum size
        bitmap = wx.StaticBitmap(panel)
        bitmap.SetMinSize((self.preview_w, self.preview_h))
        sizer.Add(bitmap, 1, wx.ALL, 5)
        
        panel.SetSizer(sizer)
        
        # Store reference to bitmap widget
        setattr(self, f"bmp_{label_text.lower().replace(' ', '_')}", bitmap)
        
        return panel

    #: Idle time before the preview is recomputed, in milliseconds.
    PREVIEW_DELAY_MS = 150

    def _schedule_preview(self):
        """Ask for a preview refresh, coalescing bursts of events into one."""
        self._preview_timer.Start(self.PREVIEW_DELAY_MS, oneShot=True)

    def _on_slider(self, event):
        val = self.slider.GetValue()
        self.current_threshold = float(val) / 1000.0
        # The number follows the hand immediately; only the heavy part waits.
        self.slider_label.SetLabel(f"Threshold (0–1): {self.current_threshold:.4f}")
        self._schedule_preview()

    def _on_cleanup(self, event):
        """Cleanup sizes changed: re-run the preview with the new values."""
        self.min_area = self.spin_min_area.GetValue()
        self.hole_area = self.spin_hole_area.GetValue()
        self._schedule_preview()

    def _on_radio(self, event):
        self.sample_is_white = self.radio_white.GetValue()
        self._update_previews()

    def _on_reset_otsu(self, event):
        self.current_threshold = self.otsu_thr
        self.slider.SetValue(int(self.otsu_thr * 1000))
        self.slider_label.SetLabel(f"Threshold (0–1): {self.current_threshold:.4f}")
        self._update_previews()

    def _on_ok(self, event):
        self._preview_timer.Stop()
        self.final_threshold = self.current_threshold
        self.confirmed = True
        self.EndModal(wx.ID_OK)

    def _on_cancel(self, event):
        self._preview_timer.Stop()
        self.confirmed = False
        self.EndModal(wx.ID_CANCEL)

    def _compute_masks(self, threshold_value):
        """
        Masks for the preview, from the SAME function that builds the result.

        This used to reimplement the watershed and the morphological cleanup
        alongside build_background_mask, so what the preview showed and what
        the metrics actually used could drift apart unnoticed.

        On large images the masks are computed on a subsampled copy; see
        preview_step. Shapes, proportions and the reported percentages hold;
        single-pixel detail may not.
        """
        paso = self.preview_step
        vista = self.image[::paso, ::paso] if paso > 1 else self.image
        vista_norm = (self.img_normalized[::paso, ::paso] if paso > 1
                      else self.img_normalized)

        binary = (vista_norm > threshold_value).astype(np.uint8) * 255

        # The cleanup sizes are areas, so they scale with the square of the
        # step: "discard objects under 100 px" must mean the same thing on the
        # reduced copy.
        background_mask, _ = build_background_mask(
            vista, threshold_value, self.sample_is_white,
            min_area=max(1, self.min_area // (paso * paso)),
            hole_area=max(1, self.hole_area // (paso * paso)),
        )
        return binary, ~background_mask, background_mask

    def _update_previews(self):
        bin_u8, final_sample_mask, background_mask = self._compute_masks(self.current_threshold)
        
        # Create bitmaps with proper sizing
        bmp_orig = numpy_to_wx_bitmap(self.img_u8, self.preview_size, self)
        bmp_bin = numpy_to_wx_bitmap(bin_u8, self.preview_size, self)
        bmp_sample = numpy_to_wx_bitmap(
            final_sample_mask.astype(np.uint8) * 255, self.preview_size, self)
        bmp_background = numpy_to_wx_bitmap(
            background_mask.astype(np.uint8) * 255, self.preview_size, self)
        
        # Update bitmaps
        self.bmp_original_image.SetBitmap(bmp_orig)
        self.bmp_binary_preview.SetBitmap(bmp_bin)
        self.bmp_sample_mask.SetBitmap(bmp_sample)
        self.bmp_background_mask.SetBitmap(bmp_background)
        
        # Counts come from the preview, so they are scaled back up to the
        # full image and flagged as approximate when it was subsampled.
        bg_frac = float(np.mean(background_mask))
        sample_frac = float(np.mean(final_sample_mask))
        bg_pixels = int(round(bg_frac * self.image.size))
        sample_pixels = int(round(sample_frac * self.image.size))

        nota = ("  (preview at 1/%d scale)" % self.preview_step
                if self.preview_step > 1 else "")
        self.info_text.SetLabel(
            f"Background: {bg_pixels} px ({100*bg_frac:.1f}%)   "
            f"Sample: {sample_pixels} px ({100*sample_frac:.1f}%){nota}")
        
        # Force layout update
        self.Layout()

    def get_result(self):
        """
        Show the dialog and return what the user chose.

        Returns
        -------
        tuple
            (threshold, sample_is_white, min_area, hole_area), or
            (None, None, None, None) if cancelled. The cleanup sizes travel
            with the threshold because they are now the user's choice, and the
            caller must rebuild the mask with the same ones the preview used.
        """
        self.ShowModal()
        if not self.confirmed:
            return None, None, None, None
        return (self.final_threshold, self.sample_is_white,
                self.min_area, self.hole_area)


def create_background_mask(image: np.ndarray, method: str = 'otsu',
                           manual_threshold: float = 0.5, min_area: int = 100,
                           hole_area: int = 50, parent=None):
    """
    Ask the user for a threshold and build the background mask from it.

    Returns
    -------
    tuple
        (background_mask, threshold_value), or (None, None) if cancelled.
    """
    image = np.array(image, dtype=float)

    # Create wx.App if needed
    app_created = False
    app = wx.GetApp()
    if app is None:
        app = wx.App(False)
        app_created = True

    dlg = WxThresholdDialog(parent, image, initial_threshold=manual_threshold,
                            min_area=min_area, hole_area=hole_area)
    threshold_value, sample_is_white, min_area, hole_area = dlg.get_result()
    dlg.Destroy()

    if app_created:
        try:
            app.Destroy()
        except Exception:
            pass

    if threshold_value is None or sample_is_white is None:
        return None, None

    # min_area and hole_area come back FROM the dialog: the user may have
    # changed them, and the mask has to match the preview they approved.
    return build_background_mask(image, threshold_value, sample_is_white,
                                 min_area=min_area, hole_area=hole_area)
