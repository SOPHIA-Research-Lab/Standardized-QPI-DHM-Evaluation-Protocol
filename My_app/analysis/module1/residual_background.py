import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional
from matplotlib.widgets import Button
from skimage.restoration import unwrap_phase
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds
import cv2
from typing import List, Tuple
import wx
from matplotlib.backends.backend_agg import FigureCanvasAgg




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
        
        self.SetSize((self.display_w + 40, self.display_h + 120))
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
        bitmap = wx.Bitmap(wx_image)
        
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

def unwrap_with_scikit(wrapped_phase: np.ndarray) -> np.ndarray:
    """Unwrap phase images using scikit-image."""
    return unwrap_phase(wrapped_phase)


def std_background(img: np.ndarray, mask: Optional[np.ndarray] = None, 
                   manual: bool = False, num_zones: int = 2):
    """Calculate standard deviation of background."""
    if manual:
        zones = select_manual_zones(img, num_zones)
        if not zones:
            return np.nan, []

        zone_stats = []
        std_values = []
        
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone_img = img[ymin:ymax, xmin:xmax]
            zone_std = float(np.std(zone_img))
            std_values.append(zone_std)
            zone_stats.append({'zone': i, 'std': zone_std, 'coords': (xmin, xmax, ymin, ymax)})

        std_mean = float(np.mean(std_values))
        return std_mean, zone_stats

    values = img[mask] if mask is not None else img.flatten()
    std_val = float(np.std(values))
    return std_val


def mean_absolute_deviation_background(img: np.ndarray, mask: Optional[np.ndarray] = None,
                                       manual: bool = False, num_zones: int = 3):
    """Calculate Mean Absolute Deviation (MAD) of background."""
    if manual:
        zones = select_manual_zones(img, num_zones)
        if not zones:
            return np.nan, []

        zone_stats = []
        mad_values = []
        
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone_img = img[ymin:ymax, xmin:xmax]
            zone_mean = np.mean(zone_img)
            zone_mad = float(np.mean(np.abs(zone_img - zone_mean)))
            mad_values.append(zone_mad)
            zone_stats.append({'zone': i, 'mad': zone_mad, 'coords': (xmin, xmax, ymin, ymax)})

        mad_mean = float(np.mean(mad_values))
        return mad_mean, zone_stats

    values = img[mask] if mask is not None else img.flatten()
    mean_val = np.mean(values)
    mad = float(np.mean(np.abs(values - mean_val)))
    return mad


def rms_background(img: np.ndarray, mask: Optional[np.ndarray] = None,
                   manual: bool = False, num_zones: int = 3):
    """Calculate Root Mean Square (RMS) of background."""
    if manual:
        zones = select_manual_zones(img, num_zones)
        if not zones:
            return np.nan, []

        zone_stats = []
        rms_values = []
        
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone_img = img[ymin:ymax, xmin:xmax]
            zone_rms = float(np.sqrt(np.mean((zone_img - np.mean(zone_img)) ** 2)))
            rms_values.append(zone_rms)
            zone_stats.append({'zone': i, 'rms': zone_rms, 'coords': (xmin, xmax, ymin, ymax)})

        rms_mean = float(np.mean(rms_values))
        return rms_mean, zone_stats

    values = img[mask] if mask is not None else img.flatten()
    rms_val = float(np.sqrt(np.mean((values - np.mean(values)) ** 2)))
    return rms_val


def pv_background(img: np.ndarray, mask: Optional[np.ndarray] = None,
                  manual: bool = False, num_zones: int = 3):
    """Calculate Peak-to-Valley (PV) of background."""
    if manual:
        zones = select_manual_zones(img, num_zones)
        if not zones:
            return np.nan, []

        zone_stats = []
        pv_values = []
        
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone_img = img[ymin:ymax, xmin:xmax]
            zone_pv = float(np.max(zone_img) - np.min(zone_img))
            pv_values.append(zone_pv)
            zone_stats.append({'zone': i, 'pv': zone_pv, 'coords': (xmin, xmax, ymin, ymax)})

        pv_mean = float(np.mean(pv_values))
        return pv_mean, zone_stats

    values = img[mask] if mask is not None else img.flatten()
    pv_val = float(np.max(values) - np.min(values))
    return pv_val


def _fwhm_from_values(values: np.ndarray, bins: int = 100) -> float:
    """Compute FWHM via histogram half-maximum with linear interpolation."""
    vals = np.asarray(values).ravel()
    vals = vals[~np.isnan(vals)]
    if vals.size < 5:
        return np.nan

    hist, bin_edges = np.histogram(vals, bins=bins)
    if not np.any(hist):
        return np.nan

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    peak_idx = int(np.argmax(hist))
    peak = float(hist[peak_idx])
    if peak <= 0:
        return np.nan
    half = 0.5 * peak

    left_idx = None
    for i in range(peak_idx, 0, -1):
        if hist[i-1] <= half <= hist[i]:
            x1, y1 = centers[i-1], hist[i-1]
            x2, y2 = centers[i], hist[i]
            t = (half - y1) / (y2 - y1 + 1e-12)
            left_idx = x1 + t * (x2 - x1)
            break

    right_idx = None
    for i in range(peak_idx, len(hist)-1):
        if hist[i] >= half >= hist[i+1]:
            x1, y1 = centers[i], hist[i]
            x2, y2 = centers[i+1], hist[i+1]
            t = (half - y1) / (y2 - y1 + 1e-12)
            right_idx = x1 + t * (x2 - x1)
            break

    if left_idx is None or right_idx is None:
        return np.nan
    return float(right_idx - left_idx)


def fwhm_background(img: np.ndarray, mask: Optional[np.ndarray] = None,
                    manual: bool = False, num_zones: int = 3):
    """Calculate FWHM of background phase noise using histogram method."""
    if manual:
        zones = select_manual_zones(img, num_zones)
        if not zones:
            return np.nan, []

        fwhm_values = []
        zone_stats = []
        
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone = img[ymin:ymax, xmin:xmax]
            fwhm = _fwhm_from_values(zone, bins=100)
            fwhm_values.append(fwhm)
            zone_stats.append({'zone': i, 'fwhm': fwhm, 'coords': (xmin, xmax, ymin, ymax)})

        fwhm_mean = float(np.mean(fwhm_values))
        return fwhm_mean, zone_stats

    values = img[mask] if mask is not None else img.ravel()
    fwhm_val = _fwhm_from_values(values, bins=100)
    return fwhm_val


def _shannon_entropy(values: np.ndarray, bins: int = 256, base: float = 2.0) -> float:
    """Calculate Shannon entropy (bits if base=2) from histogram."""
    vals = np.asarray(values).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float('nan')

    hist, _ = np.histogram(vals, bins=bins)
    total = hist.sum()
    if total == 0:
        return float('nan')

    p = hist.astype(float) / total
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0

    if base == 2.0:
        return float(-np.sum(p * np.log2(p)))
    else:
        return float(-np.sum(p * (np.log(p) / np.log(base))))


def entropy_background(img: np.ndarray, mask: Optional[np.ndarray] = None,
                       manual: bool = False, num_zones: int = 3,
                       bins: int = 256, base: float = 2.0):
    """Calculate Shannon entropy (histogram-based) of background phase values."""
    if manual:
        zones = select_manual_zones(img, num_zones)
        if not zones:
            return np.nan, []

        entropy_values = []
        zone_stats = []
        
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone = img[ymin:ymax, xmin:xmax]
            H = _shannon_entropy(zone, bins=bins, base=base)
            entropy_values.append(H)
            zone_stats.append({'zone': i, 'entropy': H, 'coords': (xmin, xmax, ymin, ymax)})

        entropy_mean = float(np.mean(entropy_values))
        return entropy_mean, zone_stats

    values = img[mask] if mask is not None else img.ravel()
    H = _shannon_entropy(values, bins=bins, base=base)
    return H


def spatial_frequency(img: np.ndarray, mask: Optional[np.ndarray] = None,
                      normalize: bool = True, return_components: bool = False):
    """Calculate Spatial Frequency (SF) metric."""
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("img must be 2D")

    if normalize:
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin)

    M, N = arr.shape
    if M < 2 and N < 2:
        return (np.nan, np.nan, np.nan) if return_components else np.nan

    dx = arr[:, 1:] - arr[:, :-1]
    dy = arr[1:, :] - arr[:-1, :]

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != arr.shape:
            raise ValueError("Mask shape must match image shape")

        hmask = m[:, 1:] & m[:, :-1]
        vmask = m[1:, :] & m[:-1, :]

        nh = int(np.count_nonzero(hmask))
        nv = int(np.count_nonzero(vmask))

        RF = np.sqrt(np.sum((dx[hmask])**2) / nh) if nh > 0 else np.nan
        CF = np.sqrt(np.sum((dy[vmask])**2) / nv) if nv > 0 else np.nan
    else:
        denom_h = M * max(N - 1, 0)
        denom_v = max(M - 1, 0) * N
        RF = np.sqrt(np.sum(dx**2) / denom_h) if denom_h > 0 else np.nan
        CF = np.sqrt(np.sum(dy**2) / denom_v) if denom_v > 0 else np.nan

    SF = np.sqrt(RF**2 + CF**2) if np.isfinite(RF) and np.isfinite(CF) else np.nan
    return (SF, RF, CF) if return_components else SF


def legendre_background(complex_field, mask=None, manual=False, num_zones=2, 
                       limit=64, order_max=5, NoPistonCompensation=True, UsePCA=False, zones=None):
    """Calculate Legendre coefficients for background analysis."""
    if manual:
        # Use provided zones if available, otherwise select manually
        if zones is None:
            zones = select_manual_zones(
                complex_field if not np.iscomplexobj(complex_field) else np.angle(complex_field), 
                num_zones
            )
        
        if not zones:
            return np.full(order_max, np.nan), []

        zone_stats = []
        all_coefficients = []
        
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone_field = complex_field[ymin:ymax, xmin:xmax]
            coeffs = _process_legendre_zone(zone_field, limit, order_max, UsePCA)
            all_coefficients.append(coeffs)
            zone_stats.append({'zone': i, 'legendre': coeffs, 'coords': (xmin, xmax, ymin, ymax)})

        mean_coeffs = np.mean(all_coefficients, axis=0)
        return mean_coeffs, zone_stats

    coeffs = _process_legendre_zone(complex_field, limit, order_max, UsePCA)

    if not NoPistonCompensation:
        coeffs = _optimize_piston(complex_field, coeffs, limit, order_max, UsePCA)
    
    gridSize = complex_field.shape[0]
    coords = np.linspace(-1, 1 - 2 / gridSize, gridSize)
    X_recon, Y_recon = np.meshgrid(coords, coords)
    orders = np.arange(1, len(coeffs) + 1)
    
    
    reconstruction_background(coeffs, X_recon, Y_recon, orders)

    return coeffs


def _process_legendre_zone(zone_field, limit, order_max, UsePCA):
    """Process single zone for Legendre coefficient extraction."""
    fftField = fftshift(fft2(ifftshift(zone_field)))
    A, B = fftField.shape
    center_A, center_B = int(round(A / 2)), int(round(B / 2))
    
    fftField = fftField[center_A - limit:center_A + limit, center_B - limit:center_B + limit]
    square = ifftshift(ifft2(fftshift(fftField)))

    if UsePCA:
        u, s, vt = svds(square, k=1, which='LM')
        dominant = u[:, :1] @ np.diag(s[:1]) @ vt[:1, :]
        dominant = unwrap_phase(np.angle(dominant))
    else:
        dominant = unwrap_phase(np.angle(square))

    gridSize = dominant.shape[0]
    coords = np.linspace(-1, 1 - 2 / gridSize, gridSize)
    X, Y = np.meshgrid(coords, coords)
    dA = (2 / gridSize) ** 2
    order = np.arange(1, order_max + 1)

    polynomials = square_legendre_fitting(order, X, Y)
    ny, nx, n_terms = polynomials.shape
    Legendres = polynomials.reshape(ny * nx, n_terms)

    zProds = Legendres.T @ Legendres * dA
    Legendres = Legendres / np.sqrt(np.diag(zProds))
    phaseVector = dominant.reshape(-1, 1)
    
    # Projection onto Legendre basis
    Legendre_Coefficients = np.sum(Legendres * phaseVector, axis=0) * dA
    
    return Legendre_Coefficients


def _optimize_piston(complex_field, coeffs, limit, order_max, UsePCA):
    """Optimize piston coefficient by searching for minimum variance."""
    fftField = fftshift(fft2(ifftshift(complex_field)))
    A, B = fftField.shape
    center_A, center_B = int(round(A / 2)), int(round(B / 2))
    
    fftField = fftField[center_A - limit:center_A + limit, center_B - limit:center_B + limit]
    square = ifftshift(ifft2(fftshift(fftField)))

    gridSize = square.shape[0]
    coords = np.linspace(-1, 1 - 2 / gridSize, gridSize)
    X, Y = np.meshgrid(coords, coords)
    dA = (2 / gridSize) ** 2
    order = np.arange(1, order_max + 1)

    polynomials = square_legendre_fitting(order, X, Y)
    ny, nx, n_terms = polynomials.shape
    Legendres = polynomials.reshape(ny * nx, n_terms)

    zProds = Legendres.T @ Legendres * dA
    Legendres = Legendres / np.sqrt(np.diag(zProds))
    Legendres_norm_const = np.sum(Legendres ** 2, axis=0) * dA

    # Search for the optimal piston value
    values = np.arange(-np.pi, np.pi + np.pi / 6, np.pi / 6)
    variances = []

    for val in values:
        temp_coeffs = coeffs.copy()
        temp_coeffs[0] = val
        coeffs_norm = temp_coeffs / np.sqrt(Legendres_norm_const)
        wavefront = np.sum((coeffs_norm[:, np.newaxis]) * Legendres.T, axis=0)
        temp_holo = np.exp(1j * np.angle(square)) / np.exp(1j * wavefront.reshape(ny, nx))
        variances.append(np.var(np.angle(temp_holo)))

    # Update piston coefficient with best value
    best = values[np.argmin(variances)]
    coeffs[0] = best
    
    return coeffs


def square_legendre_fitting(order, X, Y):
    """Generate Legendre polynomials for fitting."""
    polynomials = []
    for i in order:
        if i == 1:
            polynomials.append(np.ones_like(X))
        elif i == 2:
            polynomials.append(X)
        elif i == 3:
            polynomials.append(Y)
        elif i == 4:
            polynomials.append((3 * X**2 - 1) / 2)
        elif i == 5:
            polynomials.append(X * Y)
        elif i == 6:
            polynomials.append((3 * Y**2 - 1) / 2)
        elif i == 7:
            polynomials.append((X * (5 * X**2 - 3)) / 2)
        elif i == 8:
            polynomials.append((Y * (3 * X**2 - 1)) / 2)
        elif i == 9:
            polynomials.append((X * (3 * Y**2 - 1)) / 2)
        elif i == 10:
            polynomials.append((Y * (5 * Y**2 - 3)) / 2)
        elif i == 11:
            polynomials.append((35 * X**4 - 30 * X**2 + 3) / 8)
        elif i == 12:
            polynomials.append((X * Y * (5 * X**2 - 3)) / 2)
        elif i == 13:
            polynomials.append(((3 * Y**2 - 1) * (3 * X**2 - 1)) / 4)
        elif i == 14:
            polynomials.append((X * Y * (5 * Y**2 - 3)) / 2)
        elif i == 15:
            polynomials.append((35 * Y**4 - 30 * Y**2 + 3) / 8)
    return np.stack(polynomials, axis=-1)


def reconstruction_background(coefficients, X, Y, orders):
    """Reconstruct phase surface using Legendre coefficients."""
    polynomials = square_legendre_fitting(orders, X, Y)
    ny, nx, n_terms = polynomials.shape
    coeffs_used = coefficients[:n_terms]

    # Reconstruction background
    superficie = np.zeros((ny, nx))
    for i, coeff in enumerate(coeffs_used):
        superficie += coeff * polynomials[:, :, i]

    # Create wx.App if needed
    app_created = False
    app = wx.GetApp()
    if app is None:
        app = wx.App(False)
        app_created = True

    # Create dialog
    dlg = wx.Dialog(None, title="Legendre Reconstruction Results",
                    style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
                    size=(1100, 550))
    dlg.Centre()
    
    main_sizer = wx.BoxSizer(wx.VERTICAL)
    content_sizer = wx.BoxSizer(wx.HORIZONTAL)
    
    # Left panel: Surface
    left_panel = wx.Panel(dlg)
    left_sizer = wx.BoxSizer(wx.VERTICAL)
    label_surface = wx.StaticText(left_panel, label="Legendre-reconstructed background")
    left_sizer.Add(label_surface, 0, wx.ALIGN_CENTER | wx.ALL, 5)
    
    # Convert surface to colored image
    from matplotlib import cm
    viridis = cm.get_cmap('viridis')
    surf_norm = (superficie - superficie.min()) / (superficie.max() - superficie.min())
    colored = (viridis(surf_norm)[:, :, :3] * 255).astype(np.uint8)
    
    h, w = colored.shape[:2]
    img_surface = wx.Image(w, h)
    img_surface.SetData(colored.tobytes())
    img_surface = img_surface.Scale(500, 400, wx.IMAGE_QUALITY_HIGH)
    
    bmp_surface = wx.StaticBitmap(left_panel, bitmap=wx.Bitmap(img_surface))
    left_sizer.Add(bmp_surface, 1, wx.ALL | wx.EXPAND, 5)
    left_panel.SetSizer(left_sizer)
    content_sizer.Add(left_panel, 1, wx.EXPAND | wx.ALL, 5)
    
    # Right panel: Coefficients
    right_panel = wx.Panel(dlg)
    right_sizer = wx.BoxSizer(wx.VERTICAL)
    label_coeff = wx.StaticText(right_panel, label="Legendre coefficients")
    right_sizer.Add(label_coeff, 0, wx.ALIGN_CENTER | wx.ALL, 5)
    
    # Create plot
    fig = plt.figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(coefficients, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Legendre Index')
    ax.set_ylabel('Coefficient value')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    # Convert plot to bitmap
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    w_fig, h_fig = fig.canvas.get_width_height()
    plot_array = np.frombuffer(buf, dtype=np.uint8).reshape(h_fig, w_fig, 4)[:, :, :3]
    
    img_plot = wx.Image(w_fig, h_fig)
    img_plot.SetData(plot_array.tobytes())
    img_plot = img_plot.Scale(500, 400, wx.IMAGE_QUALITY_HIGH)
    
    bmp_plot = wx.StaticBitmap(right_panel, bitmap=wx.Bitmap(img_plot))
    right_sizer.Add(bmp_plot, 1, wx.ALL | wx.EXPAND, 5)
    right_panel.SetSizer(right_sizer)
    content_sizer.Add(right_panel, 1, wx.EXPAND | wx.ALL, 5)
    
    plt.close(fig)
    
    main_sizer.Add(content_sizer, 1, wx.EXPAND | wx.ALL, 10)
    
    # Close button
    btn_close = wx.Button(dlg, wx.ID_CLOSE, "Close")
    btn_close.Bind(wx.EVT_BUTTON, lambda evt: dlg.Close())
    main_sizer.Add(btn_close, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)
    
    dlg.SetSizer(main_sizer)
    dlg.ShowModal()
    dlg.Destroy()

    if app_created:
        try:
            app.Destroy()
        except:
            pass

    return superficie
