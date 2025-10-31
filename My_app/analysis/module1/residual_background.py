import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional
from matplotlib.widgets import Button
from skimage.restoration import unwrap_phase
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds


class ManualRectangleSelector:
    """Interactive rectangle selector for manual zone selection."""
    
    def __init__(self, img: np.ndarray, num_zones: int):
        self.img = img
        self.num_zones = num_zones
        self.rectangles = []
        self.current_rect = None
        self.start_point = None
        self.finished = False
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.setup_plot()

    def setup_plot(self):
        """Setup the interactive plot."""
        self.ax.imshow(self.img, cmap='viridis')
        self.ax.set_title(f'Select {self.num_zones} rectangles. Current: 0/{self.num_zones}')

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        ax_button = plt.axes([0.81, 0.01, 0.1, 0.05])
        self.button = Button(ax_button, 'Finish')
        self.button.on_clicked(self.finish_selection)

    def on_press(self, event):
        """Handle mouse press event."""
        if event.inaxes != self.ax or self.finished:
            return
        self.start_point = (event.xdata, event.ydata)

    def on_motion(self, event):
        """Handle mouse motion event."""
        if self.start_point is None or event.inaxes != self.ax or self.finished:
            return

        if self.current_rect:
            self.current_rect.remove()

        x0, y0 = self.start_point
        width = event.xdata - x0
        height = event.ydata - y0

        self.current_rect = patches.Rectangle(
            (x0, y0), width, height,
            linewidth=2, edgecolor='red', facecolor='none', alpha=0.7
        )
        self.ax.add_patch(self.current_rect)
        self.fig.canvas.draw()

    def on_release(self, event):
        """Handle mouse release event."""
        if self.start_point is None or event.inaxes != self.ax or self.finished:
            return

        x0, y0 = self.start_point
        x1, y1 = event.xdata, event.ydata

        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)

        xmin = int(max(0, min(xmin, self.img.shape[1] - 1)))
        xmax = int(max(0, min(xmax, self.img.shape[1] - 1)))
        ymin = int(max(0, min(ymin, self.img.shape[0] - 1)))
        ymax = int(max(0, min(ymax, self.img.shape[0] - 1)))

        if xmax > xmin and ymax > ymin:
            rect_coords = (xmin, xmax, ymin, ymax)
            self.rectangles.append(rect_coords)

            if self.current_rect:
                self.current_rect.remove()

            final_rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor='blue', facecolor='none', alpha=0.8
            )
            self.ax.add_patch(final_rect)
            self.ax.text(xmin + 5, ymin + 15, f'{len(self.rectangles)}',
                         color='blue', fontsize=12, fontweight='bold')

        self.current_rect = None
        self.start_point = None
        self.ax.set_title(f'Select {self.num_zones} rectangles. Current: {len(self.rectangles)}/{self.num_zones}')
        self.fig.canvas.draw()

        if len(self.rectangles) >= self.num_zones:
            print(f"Target number of zones reached ({self.num_zones}). Auto-finishing...")
            self.auto_finish()

    def auto_finish(self):
        """Automatic completion when number of zones is reached."""
        self.finished = True
        print(f"Complete! {len(self.rectangles)} zones selected. Closing interface...")
        plt.close(self.fig)

    def finish_selection(self, event):
        """Manual completion via button."""
        print(f"User clicked Finish. Selected {len(self.rectangles)} zones.")
        self.finished = True
        plt.close(self.fig)

    def get_rectangles(self) -> List[Tuple[int, int, int, int]]:
        """Return selected rectangles."""
        return self.rectangles


def select_manual_zones(img: np.ndarray, num_zones: int) -> List[Tuple[int, int, int, int]]:
    """Select rectangular areas manually. Auto-finishes when zones completed."""
    print(f"Opening zone selection interface for {num_zones} zones...")
    selector = ManualRectangleSelector(img, num_zones)
    plt.show(block=False)
    
    while plt.fignum_exists(selector.fig.number) and not selector.finished:
        plt.pause(0.1)
    
    zones = selector.get_rectangles()
    print(f"Zone selection completed. Selected {len(zones)} zones.")
    if len(zones) == 0:
        print("WARNING: User did not select any zones.")
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
            print("No zones selected.")
            return np.nan, []

        zone_stats = []
        std_values = []
        print("\n=== STD per Zone ===")
        
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone_img = img[ymin:ymax, xmin:xmax]
            zone_std = float(np.std(zone_img))
            std_values.append(zone_std)
            zone_stats.append({'zone': i, 'std': zone_std, 'coords': (xmin, xmax, ymin, ymax)})
            print(f"Zone {i}: STD = {zone_std:.4f}")

        std_mean = float(np.mean(std_values))
        print(f"Mean STD: {std_mean:.4f}")
        return std_mean, zone_stats

    values = img[mask] if mask is not None else img.flatten()
    std_val = float(np.std(values))
    print(f"STD whole background: {std_val:.4f}")
    return std_val


def mean_absolute_deviation_background(img: np.ndarray, mask: Optional[np.ndarray] = None,
                                       manual: bool = False, num_zones: int = 3):
    """Calculate Mean Absolute Deviation (MAD) of background."""
    if manual:
        zones = select_manual_zones(img, num_zones)
        if not zones:
            print("No zones selected.")
            return np.nan, []

        zone_stats = []
        mad_values = []
        print("\n=== MAD per Zone ===")
        
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone_img = img[ymin:ymax, xmin:xmax]
            zone_mean = np.mean(zone_img)
            zone_mad = float(np.mean(np.abs(zone_img - zone_mean)))
            mad_values.append(zone_mad)
            zone_stats.append({'zone': i, 'mad': zone_mad, 'coords': (xmin, xmax, ymin, ymax)})
            print(f"Zone {i}: MAD = {zone_mad:.4f}")

        mad_mean = float(np.mean(mad_values))
        print(f"Mean MAD: {mad_mean:.4f}")
        return mad_mean, zone_stats

    values = img[mask] if mask is not None else img.flatten()
    mean_val = np.mean(values)
    mad = float(np.mean(np.abs(values - mean_val)))
    print(f"MAD whole background: {mad:.4f}")
    return mad


def rms_background(img: np.ndarray, mask: Optional[np.ndarray] = None,
                   manual: bool = False, num_zones: int = 3):
    """Calculate Root Mean Square (RMS) of background."""
    if manual:
        zones = select_manual_zones(img, num_zones)
        if not zones:
            print("No zones selected.")
            return np.nan, []

        zone_stats = []
        rms_values = []
        print("\n=== RMS per Zone ===")
        
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone_img = img[ymin:ymax, xmin:xmax]
            zone_rms = float(np.sqrt(np.mean((zone_img - np.mean(zone_img)) ** 2)))
            rms_values.append(zone_rms)
            zone_stats.append({'zone': i, 'rms': zone_rms, 'coords': (xmin, xmax, ymin, ymax)})
            print(f"Zone {i}: RMS = {zone_rms:.4f}")

        rms_mean = float(np.mean(rms_values))
        print(f"Mean RMS: {rms_mean:.4f}")
        return rms_mean, zone_stats

    values = img[mask] if mask is not None else img.flatten()
    rms_val = float(np.sqrt(np.mean((values - np.mean(values)) ** 2)))
    print(f"RMS whole background: {rms_val:.4f}")
    return rms_val


def pv_background(img: np.ndarray, mask: Optional[np.ndarray] = None,
                  manual: bool = False, num_zones: int = 3):
    """Calculate Peak-to-Valley (PV) of background."""
    if manual:
        zones = select_manual_zones(img, num_zones)
        if not zones:
            print("No zones selected.")
            return np.nan, []

        zone_stats = []
        pv_values = []
        print("\n=== PV per Zone ===")
        
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone_img = img[ymin:ymax, xmin:xmax]
            zone_pv = float(np.max(zone_img) - np.min(zone_img))
            pv_values.append(zone_pv)
            zone_stats.append({'zone': i, 'pv': zone_pv, 'coords': (xmin, xmax, ymin, ymax)})
            print(f"Zone {i}: PV = {zone_pv:.4f}")

        pv_mean = float(np.mean(pv_values))
        print(f"Mean PV: {pv_mean:.4f}")
        return pv_mean, zone_stats

    values = img[mask] if mask is not None else img.flatten()
    pv_val = float(np.max(values) - np.min(values))
    print(f"PV whole background: {pv_val:.4f}")
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
            print("No zones selected.")
            return np.nan, []

        fwhm_values = []
        zone_stats = []
        print("\n=== FWHM per Zone (background) ===")
        
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone = img[ymin:ymax, xmin:xmax]
            fwhm = _fwhm_from_values(zone, bins=100)
            fwhm_values.append(fwhm)
            zone_stats.append({'zone': i, 'fwhm': fwhm, 'coords': (xmin, xmax, ymin, ymax)})
            print(f"Zone {i}: FWHM = {fwhm:.4f}" if np.isfinite(fwhm) else f"Zone {i}: FWHM = nan")

        fwhm_mean = float(np.mean(fwhm_values))
        print(f"Mean FWHM: {fwhm_mean:.4f}")
        return fwhm_mean, zone_stats

    values = img[mask] if mask is not None else img.ravel()
    fwhm_val = _fwhm_from_values(values, bins=100)
    print(f"FWHM (background): {fwhm_val:.4f}" if np.isfinite(fwhm_val) else "FWHM (background): nan")
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
            print("No zones selected.")
            return np.nan, []

        entropy_values = []
        zone_stats = []
        print("\n=== Entropy per Zone (background) ===")
        
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone = img[ymin:ymax, xmin:xmax]
            H = _shannon_entropy(zone, bins=bins, base=base)
            entropy_values.append(H)
            zone_stats.append({'zone': i, 'entropy': H, 'coords': (xmin, xmax, ymin, ymax)})
            print(f"Zone {i}: Entropy = {H:.4f} bits" if np.isfinite(H) else f"Zone {i}: Entropy = nan")

        entropy_mean = float(np.mean(entropy_values))
        print(f"Mean Entropy: {entropy_mean:.4f} bits")
        return entropy_mean, zone_stats

    values = img[mask] if mask is not None else img.ravel()
    H = _shannon_entropy(values, bins=bins, base=base)
    print(f"Entropy (background): {H:.4f} bits" if np.isfinite(H) else "Entropy (background): nan")
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
    print(f"SF whole background: {SF:.4f}")
    return (SF, RF, CF) if return_components else SF


def legendre_background(complex_field, mask=None, manual=False, num_zones=2, 
                       limit=64, order_max=10, NoPistonCompensation=True, UsePCA=False):
    """Calculate Legendre coefficients for background analysis."""
    if manual:
        zones = select_manual_zones(
            complex_field if not np.iscomplexobj(complex_field) else np.angle(complex_field), 
            num_zones
        )
        if not zones:
            print("No zones selected.")
            return np.full(10, np.nan), []

        zone_stats = []
        all_coefficients = []
        print("\n=== Legendre Coefficients per Zone ===")
        
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone_field = complex_field[ymin:ymax, xmin:xmax]
            coeffs = _process_legendre_zone(zone_field, limit, order_max, UsePCA)
            all_coefficients.append(coeffs)
            zone_stats.append({'zone': i, 'legendre': coeffs, 'coords': (xmin, xmax, ymin, ymax)})
            print(f"Zone {i}: Coefficients computed (shape: {coeffs.shape})")

        mean_coeffs = np.mean(all_coefficients, axis=0)
        print(f"Mean Legendre Coefficients: {mean_coeffs}")
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
    
    return np.sum(Legendres * phaseVector, axis=0) * dA


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

    values = np.arange(-np.pi, np.pi + np.pi / 6, np.pi / 6)
    variances = []

    for val in values:
        temp_coeffs = coeffs.copy()
        temp_coeffs[0] = val
        coeffs_norm = temp_coeffs / np.sqrt(Legendres_norm_const)
        wavefront = np.sum((coeffs_norm[:, np.newaxis]) * Legendres.T, axis=0)
        temp_holo = np.exp(1j * np.angle(square)) / np.exp(1j * wavefront.reshape(ny, nx))
        variances.append(np.var(np.angle(temp_holo)))

    coeffs[0] = values[np.argmin(variances)]
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

    superficie = np.zeros((ny, nx))
    for i, coeff in enumerate(coeffs_used):
        superficie += coeff * polynomials[:, :, i]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(superficie, cmap='viridis')
    plt.title('Legendre-reconstructed background')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.plot(coefficients, 'o-')
    plt.title('Legendre coefficients')
    plt.xlabel('Legendre Index')
    plt.ylabel('Coefficient value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return superficie