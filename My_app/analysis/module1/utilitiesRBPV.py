from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt
from skimage import measure, morphology
from skimage.segmentation import watershed
import numpy as np
import cv2
import wx


# ----------------------------
# IO / helpers
# ----------------------------
def imageRead(namefile):
    """Read image from file and return grayscale PIL Image."""
    imagen = Image.open(namefile)
    loadImage = ImageOps.grayscale(imagen)
    return loadImage


def grayscaleToPhase(image: np.ndarray) -> np.ndarray:
    """Convert grayscale image (0-255 or 0-1) to phase map [-π, π]."""
    image = np.array(image)
    img = image.astype(float)
    if img.max() > 1.0:
        img = img / 255.0
    phase = img * (2 * np.pi) - np.pi
    return phase


def separate_touching_samples(binary_mask):
    """Separate touching samples using watershed segmentation."""
    distance = distance_transform_edt(binary_mask)
    from scipy import ndimage
    local_maxima = ndimage.maximum_filter(distance, size=20) == distance
    local_maxima = local_maxima & (distance > 5)
    markers = measure.label(local_maxima)
    
    if np.max(markers) > 1:
        labels = watershed(-distance, markers, mask=binary_mask)
        separated_mask = labels > 0
        
        # Print component statistics like matplotlib version
        original_components = len(measure.regionprops(measure.label(binary_mask)))
        new_components = len(measure.regionprops(measure.label(separated_mask)))
        
        return separated_mask
    else:
        return binary_mask


# ----------------------------
# Utilities to convert numpy -> wx.Bitmap
# ----------------------------
def numpy_to_wx_bitmap(arr: np.ndarray, target_size=(400, 400)):
    """Convert numpy array (H,W) or (H,W,3) to wx.Bitmap, fitting within target_size."""
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

    return wx.Bitmap(img)


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

        self.sample_is_white = True
        self.confirmed = False
        self.final_threshold = self.current_threshold
        
        # Calculate preview size based on screen (leaving margin for taskbar and title bar)
        display_size = wx.GetDisplaySize()
        available_height = display_size[1] - 150  # Reserve space for taskbar and title bar
        available_width = display_size[0] - 100
        
        # Calculate preview sizes
        self.preview_w = min(450, int(available_width * 0.22))
        self.preview_h = min(350, int(available_height * 0.35))
        self.preview_size = (self.preview_w, self.preview_h)

        # Set dialog size based on preview sizes (ensure it fits on screen)
        dialog_width = min(self.preview_w * 2 + 80, available_width)
        dialog_height = min(self.preview_h * 2 + 280, available_height)
        self.SetSize((dialog_width, dialog_height))
        self.Centre()

        self._build_ui()
        self._update_previews()

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
        self.slider.Bind(wx.EVT_SLIDER, self._on_slider)
        self.radio_white.Bind(wx.EVT_RADIOBUTTON, self._on_radio)
        self.radio_black.Bind(wx.EVT_RADIOBUTTON, self._on_radio)
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

    def _on_slider(self, event):
        val = self.slider.GetValue()
        self.current_threshold = float(val) / 1000.0
        self.slider_label.SetLabel(f"Threshold (0–1): {self.current_threshold:.4f}")
        self._update_previews()

    def _on_radio(self, event):
        self.sample_is_white = self.radio_white.GetValue()
        self._update_previews()

    def _on_reset_otsu(self, event):
        self.current_threshold = self.otsu_thr
        self.slider.SetValue(int(self.otsu_thr * 1000))
        self.slider_label.SetLabel(f"Threshold (0–1): {self.current_threshold:.4f}")
        self._update_previews()

    def _on_ok(self, event):
        self.final_threshold = self.current_threshold
        self.confirmed = True
        self.EndModal(wx.ID_OK)

    def _on_cancel(self, event):
        self.confirmed = False
        self.EndModal(wx.ID_CANCEL)

    def _compute_masks(self, threshold_value):
        """Compute masks using normalized threshold (0-1 range)."""
        binary = (self.img_normalized > threshold_value).astype(np.uint8) * 255
        
        preview_mask = (binary > 0)
        sample_mask = preview_mask.copy() if self.sample_is_white else ~preview_mask
        
        # Separate touching samples if needed
        try:
            if np.any(sample_mask):
                sample_mask = separate_touching_samples(sample_mask)
        except Exception:
            pass
        
        # Clean up mask
        if self.sample_is_white:
            cleaned = morphology.remove_small_objects(sample_mask, min_size=self.min_area)
        else:
            cleaned = ~morphology.remove_small_objects(~sample_mask, min_size=self.min_area)
        
        final_sample_mask = morphology.remove_small_holes(cleaned, area_threshold=self.hole_area)
        background_mask = ~final_sample_mask
        
        return (binary, final_sample_mask, background_mask)

    def _update_previews(self):
        bin_u8, final_sample_mask, background_mask = self._compute_masks(self.current_threshold)
        
        # Create bitmaps with proper sizing
        bmp_orig = numpy_to_wx_bitmap(self.img_u8, target_size=self.preview_size)
        bmp_bin = numpy_to_wx_bitmap(bin_u8, target_size=self.preview_size)
        bmp_sample = numpy_to_wx_bitmap(final_sample_mask.astype(np.uint8) * 255, target_size=self.preview_size)
        bmp_background = numpy_to_wx_bitmap(background_mask.astype(np.uint8) * 255, target_size=self.preview_size)
        
        # Update bitmaps
        self.bmp_original_image.SetBitmap(bmp_orig)
        self.bmp_binary_preview.SetBitmap(bmp_bin)
        self.bmp_sample_mask.SetBitmap(bmp_sample)
        self.bmp_background_mask.SetBitmap(bmp_background)
        
        # Update statistics
        total = self.image.size
        bg_pixels = int(np.sum(background_mask))
        sample_pixels = int(np.sum(final_sample_mask))
        
        self.info_text.SetLabel(f"Background: {bg_pixels} px ({100*bg_pixels/total:.1f}%)   "
                                f"Sample: {sample_pixels} px ({100*sample_pixels/total:.1f}%)")
        
        # Force layout update
        self.Layout()

    def get_result(self):
        """Show dialog and return results. Returns (None, None) if cancelled."""
        res = self.ShowModal()
        if res == wx.ID_OK and self.confirmed:
            return float(self.current_threshold), bool(self.sample_is_white)
        else:
            return None, None


# ------------------------------------------------
# create_background_mask_interactive + wrapper
# ------------------------------------------------
def create_background_mask_interactive(image: np.ndarray, method: str = 'otsu',
                                       manual_threshold: float = 0.5, min_area: int = 100,
                                       hole_area: int = 50, parent=None):
    """
    Create background mask interactively with wxPython dialog.
    
    Returns:
        tuple: (background_mask, background_values, threshold_value) or (None, None, None) if cancelled
    """
    image = np.array(image, dtype=float)
    
    # Create wx.App if needed
    app_created = False
    app = wx.GetApp()
    if app is None:
        app = wx.App(False)
        app_created = True

    # Show dialog
    dlg = WxThresholdDialog(parent, image, initial_threshold=manual_threshold,
                            min_area=min_area, hole_area=hole_area)
    threshold_value, sample_is_white = dlg.get_result()
    dlg.Destroy()

    # Clean up app if we created it
    if app_created:
        try:
            app.Destroy()
        except Exception:
            pass

    # Handle cancellation
    if threshold_value is None or sample_is_white is None:
        return None, None, None
    
    # Normalize image to 0-1 range
    img_min, img_max = np.min(image), np.max(image)
    if img_max - img_min < 1e-12:
        img_normalized = np.zeros_like(image, dtype=float)
    else:
        img_normalized = (image - img_min) / (img_max - img_min)

    # Apply threshold
    binary = (img_normalized > threshold_value).astype(np.uint8) * 255
    preview_mask = (binary > 0)
    sample_mask = preview_mask.copy() if sample_is_white else ~preview_mask

    # Separate touching samples
    try:
        if np.any(sample_mask):
            sample_mask = separate_touching_samples(sample_mask)
    except Exception:
        pass

    # Clean up mask
    if sample_is_white:
        cleaned = morphology.remove_small_objects(sample_mask, min_size=min_area)
    else:
        cleaned = ~morphology.remove_small_objects(~sample_mask, min_size=min_area)

    final_sample_mask = morphology.remove_small_holes(cleaned, area_threshold=hole_area)
    background_mask = ~final_sample_mask
    
    # Extract background values (like matplotlib version)
    #background_values = image[background_mask]

    total_pixels = image.size
    background_pixels = int(np.sum(background_mask))
    sample_pixels = int(np.sum(final_sample_mask))
    
    # print("\n✓ Mask created successfully!")
    # print("\nMask Statistics:")
    # print(f"Total pixels: {total_pixels}")
    # print(f"Background pixels: {background_pixels} ({100 * background_pixels / total_pixels:.1f}%)")
    # print(f"Sample pixels: {sample_pixels} ({100 * sample_pixels / total_pixels:.1f}%)")
    # #print(f"Background values extracted: {len(background_values)}")
    # print(f"Threshold value: {threshold_value:.4f}")

    return background_mask, float(threshold_value)


def create_background_mask(image: np.ndarray, method: str = 'otsu',
                          manual_threshold: float = 0.5, min_area: int = 100,
                          hole_area: int = 50, parent=None):
    """
    Wrapper function for create_background_mask_interactive.
    
    Returns:
        tuple: (background_mask, background_values, threshold_value) or (None, None, None) if cancelled
    """
    return create_background_mask_interactive(
        image, method, manual_threshold, min_area, hole_area, parent
    )