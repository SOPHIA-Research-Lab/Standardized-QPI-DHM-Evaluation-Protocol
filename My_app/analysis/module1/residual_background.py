import numpy as np
from typing import Optional
from skimage.restoration import unwrap_phase
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds


def unwrap_with_scikit(wrapped_phase: np.ndarray) -> np.ndarray:
    """Unwrap phase images using scikit-image."""
    return unwrap_phase(wrapped_phase)


def std_background(img: np.ndarray, mask: Optional[np.ndarray] = None):
    """Calculate standard deviation of background."""

    values = img[mask] if mask is not None else img.flatten()
    # Returned at full precision: this was the only metric in the module that
    # rounded (to 4 decimals), which hid differences below 1e-4 and did not
    # even match its own zone branch above, where the mean is not rounded.
    return float(np.std(values))


def mean_absolute_deviation_background(img: np.ndarray, mask: Optional[np.ndarray] = None):
    """Calculate Mean Absolute Deviation (MAD) of background."""

    values = img[mask] if mask is not None else img.flatten()
    mean_val = np.mean(values)
    mad = float(np.mean(np.abs(values - mean_val)))
    return mad


def mean_background(img: np.ndarray, mask: Optional[np.ndarray] = None):
    """
    Mean phase level of the background (the piston term).

    Reported alongside STD and RMS because the three are tied by an exact
    identity:

        RMS^2 = mean^2 + STD^2

    RMS is therefore not independent of STD: the only thing it adds is the
    mean. Measuring that mean explicitly makes the piston visible instead of
    leaving it buried inside RMS, and lets the three figures be checked
    against each other.

    Interpretation depends on how the image was loaded:

    - From a complex .mat field the phase is physical, so this is the
      residual piston left by the compensation. A well compensated
      background sits near zero.
    - From an 8-bit image file the phase comes from grayscaleToPhase, a
      linear remap of intensity to [-pi, pi], so this value tracks the mean
      brightness of the hologram rather than anything optical.

    Parameters
    ----------
    img : ndarray (float)
        Phase map in radians.
    mask : ndarray (bool), optional
        True on background pixels. Whole image when omitted.
    manual : bool, optional
        Select rectangular zones interactively (default: False).
    num_zones : int, optional
        Number of zones to request when manual is True.

    Returns
    -------
    float
        Mean background phase, in radians, when manual is False.
    tuple(float, list)
        Mean over the zones and the per-zone statistics, when manual is True.
    """

    values = img[mask] if mask is not None else img.flatten()
    return float(np.mean(values))


def rms_background(img: np.ndarray, mask: Optional[np.ndarray] = None):
    """Calculate Root Mean Square (RMS) of background."""


    values = img[mask] if mask is not None else img.flatten()

    # RMS = sqrt(mean(x^2))
    rms_val = float(np.sqrt(np.mean(values ** 2)))

    return rms_val


def pv_background(img: np.ndarray, mask: Optional[np.ndarray] = None):
    """Calculate Peak-to-Valley (PV) of background."""

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


def fwhm_background(img: np.ndarray, mask: Optional[np.ndarray] = None):
    """Calculate FWHM of background phase noise using histogram method."""

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
                       bins: int = 256, base: float = 2.0):
    """Calculate Shannon entropy (histogram-based) of background phase values."""

    values = img[mask] if mask is not None else img.ravel()
    H = _shannon_entropy(values, bins=bins, base=base)
    return H


def legendre_background(complex_field, mask=None, manual=False, num_zones=2,
                        limit=64, order_max=5, NoPistonCompensation=True,
                        UsePCA=False, zones=None, show_reconstruction=None):
    """Calculate Legendre coefficients for background analysis."""
    if manual:
        # The caller selects the zones and passes them in; this function
        # does not open dialogs. That is what keeps this module free of wx.
        
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

    # The caller decides whether the reconstruction is displayed, and with
    # what. This module does not import a viewer.
    if show_reconstruction is not None:
        show_reconstruction(
            reconstruction_background(coeffs, X_recon, Y_recon, orders), coeffs
        )

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
    """
    Rebuild the background surface from its Legendre coefficients.

    Pure computation. Displaying the result is ui/legendre_viewer.py's job;
    this function used to do both and discard what it returned.
    """
    polynomials = square_legendre_fitting(orders, X, Y)
    ny, nx, n_terms = polynomials.shape
    coeffs_used = coefficients[:n_terms]

    superficie = np.zeros((ny, nx))
    for i, coeff in enumerate(coeffs_used):
        superficie += coeff * polynomials[:, :, i]

    return superficie


