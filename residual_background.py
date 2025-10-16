

'''
Núcleo 1 - Residual background phase variance
- Peak-to-Valley (P–V) Value in Background.
- Background phase tilt/Curvature residuals (esta involucra medir los coeficientes de legenre/zernike para el mapa de fase resultante, y con ellos, saber cuáles son los residuales)
'''

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List
from matplotlib.widgets import Button
import unwrapping as uw
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds


class ManualRectangleSelector:
    def __init__(self, img, num_zones):
        self.img = img
        self.num_zones = num_zones
        self.rectangles = []
        self.current_rect = None
        self.start_point = None
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.setup_plot()

    def setup_plot(self):
        self.ax.imshow(self.img, cmap='viridis')
        self.ax.set_title(f'Selecciona {self.num_zones} rectángulos. Actual: 0/{self.num_zones}')

        # Mouse conections
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Button to end
        ax_button = plt.axes([0.81, 0.01, 0.1, 0.05])
        self.button = Button(ax_button, 'Finalizar')
        self.button.on_clicked(self.finish_selection)

    def on_press(self, event):
        if event.inaxes != self.ax or len(self.rectangles) >= self.num_zones:
            return

        self.start_point = (event.xdata, event.ydata)

    def on_motion(self, event):
        if self.start_point is None or event.inaxes != self.ax:
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
        if self.start_point is None or event.inaxes != self.ax:
            return

        if len(self.rectangles) >= self.num_zones:
            return

        x0, y0 = self.start_point
        x1, y1 = event.xdata, event.ydata

        # guarantee well-coordinates
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)

        xmin = int(max(0, min(xmin, self.img.shape[1] - 1)))
        xmax = int(max(0, min(xmax, self.img.shape[1] - 1)))
        ymin = int(max(0, min(ymin, self.img.shape[0] - 1)))
        ymax = int(max(0, min(ymax, self.img.shape[0] - 1)))

        if xmax > xmin and ymax > ymin:
            rect_coords = (xmin, xmax, ymin, ymax)
            self.rectangles.append(rect_coords)

            # Draw rectangles
            if self.current_rect:
                self.current_rect.remove()

            final_rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor='blue', facecolor='none', alpha=0.8
            )
            self.ax.add_patch(final_rect)

            # add zones
            self.ax.text(xmin + 5, ymin + 15, f'{len(self.rectangles)}',
                         color='blue', fontsize=12, fontweight='bold')

        self.current_rect = None
        self.start_point = None

        # Select ROI
        self.ax.set_title(f'Select {self.num_zones} Rectangles. Actual: {len(self.rectangles)}/{self.num_zones}')
        self.fig.canvas.draw()

    def finish_selection(self, event):
        plt.close(self.fig)

    def get_rectangles(self):
        return self.rectangles


def select_manual_zones(img, num_zones):
    """
    function to select rectangular areas manually.
    """
    selector = ManualRectangleSelector(img, num_zones)
    plt.show()

    zones = selector.get_rectangles()

    if len(zones) == 0:
        print("No zone has been selected")
        return None

    print(f"Number of zones selected are {len(zones)}")
    return zones


def std_background(img,  mask: np.ndarray = None, manual=False, num_zones=3) -> float:
    if manual:
        zones = select_manual_zones(img, num_zones)
        if len(zones) == 0:
            print("No zones selected.")
            return np.nan, []

        zone_stats = []
        std_values = []

        print("\n=== STD per Zone ===")
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone_img = img[ymin:ymax, xmin:xmax]
            zone_std = float(np.std(zone_img))
            std_values.append(zone_std)
            zone_stats.append({
                'zone': i,
                'std': zone_std,
                'coords': (xmin, xmax, ymin, ymax),
            })
            print(f"Zone {i}: STD = {zone_std:.4f}")

        std_mean = float(np.mean(std_values))
        print(f"Mean STD: {std_mean:.4f}")
        return std_mean, zone_stats
    else:
        if mask is not None:
            values = img[mask]
        else:
            values = img.flatten()

        std_val = np.std(values)
        print(f" STD whole background: {std_val:.4f}")

        return std_val


def mean_absolute_deviation_background(img,  mask: np.ndarray = None, manual=False, num_zones=3) -> float:
    """
    Compute the Mean Absolute Deviation (MAD) of an image or array.
    """
    if manual:
        zones = select_manual_zones(img, num_zones)
        if len(zones) == 0:
            print("No zones selected.")
            return np.nan, []

        zone_stats = []
        mad_values = []

        print("\n=== MAD per Zone ===")
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone_img = img[ymin:ymax, xmin:xmax]
            zone_mean = np.mean(zone_img)
            zone_mad = np.mean(np.abs(zone_img - zone_mean))
            mad_values.append(zone_mad)
            zone_stats.append({
                'zone': i,
                'mad': zone_mad,
                'coords': (xmin, xmax, ymin, ymax),
            })
            print(f"Zone {i}: MAD = {zone_mad:.4f}")

        mad_mean = float(np.mean(mad_values))
        print(f"Mean MAD: {mad_mean:.4f}")

        return mad_values, zone_stats
    else:
        if mask is not None:
            values = img[mask]
        else:
            values = img.flatten()

        mean_val = np.mean(values)
        mad = np.mean(np.abs(values - mean_val))
        print(f" MAD whole background: {mad:.4f}")

    return mad


def rms_background(img,  mask: np.ndarray = None, manual=False, num_zones=3) -> float:
    """
    Compute the Root Mean Square (RMS) of an image or array.
    """
    if manual:
        zones = select_manual_zones(img, num_zones)
        if len(zones) == 0:
            print("No zones selected.")
            return np.nan, []

        zone_stats = []
        rms_values = []

        print("\n=== RMS per Zone ===")
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone_img = img[ymin:ymax, xmin:xmax]
            zone_rms = np.sqrt(np.mean((zone_img - np.mean(zone_img)) ** 2))
            rms_values.append(zone_rms)
            zone_stats.append({
                'zone': i,
                'rms': zone_rms,
                'coords': (xmin, xmax, ymin, ymax),
            })
            print(f"Zone {i}: MAD = {zone_rms:.4f}")

        rms_mean = float(np.mean(rms_values))
        print(f"Mean MAD: {rms_mean:.4f}")

        return rms_values, zone_stats
    else:
        if mask is not None:
            values = img[mask]
        else:
            values = img.flatten()

    rms_val = np.sqrt(np.mean((values - np.mean(values)) ** 2))
    print(f" MAD whole background: {rms_val:.4f}")

    return rms_val


def _fwhm_from_values(values: np.ndarray, bins: int = 100) -> float:
    """Compute FWHM of a 1D sample via histogram half-maximum with linear interpolation."""
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

    # Left crossing (from peak to the left)
    left_idx = None
    for i in range(peak_idx, 0, -1):
        if hist[i-1] <= half <= hist[i]:
            x1, y1 = centers[i-1], hist[i-1]
            x2, y2 = centers[i],   hist[i]
            t = (half - y1) / (y2 - y1 + 1e-12)
            x_half_left = x1 + t * (x2 - x1)
            left_idx = x_half_left
            break

    # Right crossing (from peak to the right)
    right_idx = None
    for i in range(peak_idx, len(hist)-1):
        if hist[i] >= half >= hist[i+1]:
            x1, y1 = centers[i],   hist[i]
            x2, y2 = centers[i+1], hist[i+1]
            t = (half - y1) / (y2 - y1 + 1e-12)
            x_half_right = x1 + t * (x2 - x1)
            right_idx = x_half_right
            break

    if left_idx is None or right_idx is None:
        return np.nan
    return float(right_idx - left_idx)


def fwhm_background(img: np.ndarray, mask: np.ndarray = None, manual=False, num_zones=3)-> float:
    """
    FWHM of background phase noise (histogram method).
    """
    if manual:
        zones = select_manual_zones(img, num_zones)
        if len(zones) == 0:
            print("No zones selected.")
            return [], []

        fwhm_values = []
        zone_stats = []
        print("\n=== FWHM per Zone (background) ===")
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone = img[ymin:ymax, xmin:xmax]
            fwhm = _fwhm_from_values(zone, bins=100)
            fwhm_values.append(fwhm)
            zone_stats.append({
                'zone': i,
                'fwhm': fwhm,
                'coords': (xmin, xmax, ymin, ymax),
            })
            print(f"Zone {i}: FWHM = {fwhm:.4f}" if np.isfinite(fwhm) else f"Zone {i}: FWHM = nan")

        return fwhm_values, zone_stats

    # Non-manual: use provided background mask or the whole image
    values = img[mask] if (mask is not None) else img.ravel()
    fwhm_val = _fwhm_from_values(values, bins=100)
    print(f"FWHM (background): {fwhm_val:.4f}" if np.isfinite(fwhm_val) else "FWHM (background): nan")

    return fwhm_val


def _shannon_entropy(values: np.ndarray, bins: int = 256, base: float = 2.0) -> float:
    """Shannon entropy (bits if base=2) estimated from a histogram of values."""
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


def entropy_background(img: np.ndarray, mask: np.ndarray = None, manual=False, num_zones: int = 3,
                       bins: int = 256,
                       base: float = 2.0):
    """
    Shannon entropy (histogram-based) of background phase values.
    """
    if manual:
        zones = select_manual_zones(img, num_zones)
        if len(zones) == 0:
            print("No zones selected.")
            return [], []

        entropy_values: List[float] = []
        zone_stats: List[dict] = []

        print("\n=== Entropy per Zone (background) ===")
        for i, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zone = img[ymin:ymax, xmin:xmax]
            H = _shannon_entropy(zone, bins=bins, base=base)
            entropy_values.append(H)
            zone_stats.append({
                'zone': i,
                'entropy': H,
                'coords': (xmin, xmax, ymin, ymax),
            })
            if np.isfinite(H):
                print(f"Zone {i}: Entropy = {H:.4f} bits")
            else:
                print(f"Zone {i}: Entropy = nan")

        return entropy_values, zone_stats

    values = img[mask] if (mask is not None) else img.ravel()
    H = _shannon_entropy(values, bins=bins, base=base)
    if np.isfinite(H):
        print(f"Entropy (background): {H:.4f} bits")
    else:
        print("Entropy (background): nan")
    return H


def spatial_frequency(img: np.ndarray, mask: np.ndarray | None = None,
                      normalize=True, return_components: bool = False):
    """
    Spatial Frequency (SF).
    """
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("img must be 2d")

    if normalize:
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin)

    M, N = arr.shape
    if M < 2 and N < 2:
        return (np.nan, np.nan, np.nan) if return_components else np.nan

    dx = arr[:, 1:] - arr[:, :-1]     # M x (N-1)
    dy = arr[1:, :] - arr[:-1, :]     # (M-1) x N

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != arr.shape:
            raise ValueError("mask")

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
    print(f" SF whole background: {SF:.4f}")
    return (SF, RF, CF) if return_components else SF


def legendre(complex_field, limit, NoPistonCompensation=True, UsePCA=False):
    """
    Legendre coefficient for background.
    """

    # Centered Fourier transform
    fftField = fftshift(fft2(ifftshift(complex_field)))

    A, B = fftField.shape
    center_A = int(round(A / 2))
    center_B = int(round(B / 2))

    start_A = int(center_A - limit)
    end_A = int(center_A + limit)
    start_B = int(center_B - limit)
    end_B = int(center_B + limit)

    fftField = fftField[start_A:end_A, start_B:end_B]
    square = ifftshift(ifft2(fftshift(fftField)))

    # Extract dominant wavefront
    if UsePCA:
        u, s, vt = svds(square, k=1, which='LM')
        dominant = u[:, :1] @ np.diag(s[:1]) @ vt[:1, :]
        #dominant = unwrap_phase(np.angle(dominant))
        dominant = uw.phase_unwrap(np.angle(dominant))
    else:
        #dominant = unwrap_phase(np.angle(square))
        dominant = uw.phase_unwrap(np.angle(square))

    # Normalized spatial grid
    gridSize = dominant.shape[0]
    coords = np.linspace(-1, 1 - 2 / gridSize, gridSize)
    X, Y = np.meshgrid(coords, coords)

    dA = (2 / gridSize) ** 2
    order = np.arange(1, 11)

    # Get orthonormal Legendre polynomial basis
    polynomials = square_legendre_fitting(order, X, Y)
    ny, nx, n_terms = polynomials.shape
    Legendres = polynomials.reshape(ny * nx, n_terms)

    zProds = Legendres.T @ Legendres * dA
    Legendres = Legendres / np.sqrt(np.diag(zProds))

    Legendres_norm_const = np.sum(Legendres ** 2, axis=0) * dA
    phaseVector = dominant.reshape(-1, 1)

    # Projection onto Legendre basis
    Legendre_Coefficients = np.sum(Legendres * phaseVector, axis=0) * dA

    if NoPistonCompensation:
        # Piston compensation disabled
        # Optionally skip the first coefficient in the reconstruction
        pass
    else:
        # Search for the optimal piston value
        values = np.arange(-np.pi, np.pi + np.pi / 6, np.pi / 6)
        variances = []

        for val in values:
            temp_coeffs = Legendre_Coefficients.copy()
            temp_coeffs[0] = val
            coeffs_norm = temp_coeffs / np.sqrt(Legendres_norm_const)
            wavefront = np.sum((coeffs_norm[:, np.newaxis]) * Legendres.T, axis=0)
            temp_holo = np.exp(1j * np.angle(square)) / np.exp(1j * wavefront.reshape(ny, nx))
            variances.append(np.var(np.angle(temp_holo)))

        best = values[np.argmin(variances)]
        Legendre_Coefficients[0] = best
        #coeffs_norm = Legendre_Coefficients / np.sqrt(Legendres_norm_const)

        gridSize = complex_field.shape[0]
        coords = np.linspace(-1, 1 - 2 / gridSize, gridSize)
        X_recon, Y_recon = np.meshgrid(coords, coords)
        orders = np.arange(1, len(Legendre_Coefficients) + 1)

        reconstruction_backgroud(Legendre_Coefficients, X_recon, Y_recon, orders)

    return Legendre_Coefficients


def square_legendre_fitting(order, X, Y):
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


def reconstruction_backgroud(coefficients, X, Y, orders):
    """
    Reconstructs the phase surface using Legendre coefficients
    """
    polynomials = square_legendre_fitting(orders, X, Y)
    ny, nx, n_terms = polynomials.shape

    # select important coefficients
    coeffs_used = coefficients[:n_terms]

    # Reconstruction background
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
    plt.title('Legendre coefficient')
    plt.xlabel('Legendre Index')
    plt.ylabel('Coefficient value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return superficie