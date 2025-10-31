import numpy as np
from scipy.ndimage import laplace, convolve
from skimage.restoration import unwrap_phase
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from skimage.filters import threshold_local


def global_phase_gradient(phase: np.ndarray, use_unwrap: bool = False):
    """
    Calculates the global phase gradient (GPG) of a phase image.

    Parameters
    ----------
    phase : ndarray (float)
        2D phase map in radians [-π, π].
    use_unwrap : bool, optional
        If True, applies 2π phase unwrapping before fitting (default: False).

    Returns
    -------
    alpha : float
        Phase gradient along X direction (radians per pixel).
    beta : float
        Phase gradient along Y direction (radians per pixel).
    phi0 : float
        Global phase offset.
    GPGLin : float
        Average magnitude of the global phase gradient (radians/pixel).
    phase_corrected : ndarray
        Phase map with the fitted plane removed.
    """
    # Ensure input is a NumPy array
    phase = np.asarray(phase, dtype=np.float64)

    # Optional unwrapping
    if use_unwrap:
        phase = unwrap_phase(phase)

    # Image size
    ny, nx = phase.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    # Flatten arrays for least squares
    X = X.ravel()
    Y = Y.ravel()
    Z = phase.ravel()

    # Fit a plane: Z = alpha*X + beta*Y + phi0
    G = np.c_[X, Y, np.ones_like(X)]
    coeffs, _, _, _ = np.linalg.lstsq(G, Z, rcond=None)
    alpha, beta, phi0 = coeffs

    # Reconstruct fitted plane
    Xf, Yf = np.meshgrid(np.arange(nx), np.arange(ny))
    plane = alpha * Xf + beta * Yf + phi0

    # Compute global phase gradient metric
    GPGLin = np.sqrt(alpha**2 + beta**2)

    # Remove plane from phase (flatten phase map)
    phase_corrected = phase - plane

    return alpha, beta, phi0, GPGLin, phase_corrected



def phase_gradient_prewitt(E=None, phase=None, usePhaseUnwrap=False):
    """
    Computes the local phase gradient using Prewitt filters.

    Parameters
    ----------
    E : ndarray (complex), optional
        Reconstructed complex field.
    phase : ndarray (real), optional
        Reconstructed phase (in radians).
        Ignored if E is provided.
    usePhaseUnwrap : bool
        Whether to unwrap phase before computing gradient.

    Returns
    -------
    grad_x : ndarray
        Local phase gradient in x-direction (horizontal Prewitt).
    grad_y : ndarray
        Local phase gradient in y-direction (vertical Prewitt).
    grad_mag : ndarray
        Gradient magnitude.
    GPGPrw : float
        Global phase gradient magnitude (Prewitt).
    """
    if E is not None:
        phase = np.angle(E)
    elif phase is None:
        raise ValueError("You must provide either the complex field E or the phase.")

    if usePhaseUnwrap:
        phase_unwrapped = unwrap_phase(phase)
    else:
        phase_unwrapped = phase

    # Prewitt operators
    prewitt_x = np.array([[ -1, 0, 1],
                          [ -1, 0, 1],
                          [ -1, 0, 1]], dtype=float)

    prewitt_y = np.array([[  1,  1,  1],
                          [  0,  0,  0],
                          [ -1, -1, -1]], dtype=float)

    # Convolutions
    grad_x = convolve(phase_unwrapped, prewitt_x, mode="reflect")
    grad_y = convolve(phase_unwrapped, prewitt_y, mode="reflect")

    # Gradient magnitude
    grad_mag = np.hypot(grad_x, grad_y)
    N, M = grad_mag.shape
    GPGPrw = np.sum(grad_x ** 2 + grad_y ** 2) / (N * M)

    return grad_x, grad_y, grad_mag, GPGPrw



def laplacian_energy(phase: np.ndarray, use_unwrap: bool = False):
    """
    Calculates the Laplacian energy of a phase image.

    Parameters
    ----------
    phase : ndarray (float)
        2D phase map in radians [-π, π].
    use_unwrap : bool, optional
        If True, applies 2π phase unwrapping before Laplacian computation (default: False).

    Returns
    -------
    energy : float
        Mean Laplacian energy (average of squared Laplacian values).
    lap_map : ndarray
        Laplacian map of the phase image.
    mean_lap : float
        Mean value of the Laplacian map (for reference, should be near zero).
    std_lap : float
        Standard deviation of the Laplacian map (contrast or sharpness indicator).
    """
    # Ensure input is a NumPy array
    phase = np.asarray(phase, dtype=np.float64)

    # Optional phase unwrapping
    if use_unwrap:
        phase = unwrap_phase(phase)

    # Replace invalid values
    phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute Laplacian
    lap_map = laplace(phase, mode="reflect")

    # Compute energy and statistics
    energy = float(np.mean(lap_map ** 2))
    mean_lap = float(np.mean(lap_map))
    std_lap = float(np.std(lap_map))

    return energy, lap_map, mean_lap, std_lap


def maximum_minus_minimum(phase: np.ndarray, use_unwrap: bool = False):
    print(f"[DEBUG] use_unwrap received: {use_unwrap} (type: {type(use_unwrap)})")

    """
    Calculates the Maximum - Minimum difference of a phase image.

    Parameters
    ----------
    phase : ndarray (real)
        Phase image in radians.
    use_unwrap : bool
        If True, applies unwrapping before the calculation.

    Returns
    -------
    max_minus_min : float
        Difference between the maximum and the minimum of the phase.
    """
    if use_unwrap:
        phase = unwrap_phase(phase)

    max_minus_min = np.max(phase) - np.min(phase)
    return float(max_minus_min)




def spatial_frequency_global(phase: np.ndarray, use_unwrap: bool = False):
    """
    Computes the Global Spatial Frequency (SF) of a phase image.

    Parameters
    ----------
    phase : ndarray (float)
        2D phase image in radians.
    use_unwrap : bool, optional
        If True, applies 2π unwrapping before computation (default: False).

    Returns
    -------
    SF : float
        Total spatial frequency (radians/pixel).
    RF : float
        Row frequency component.
    CF : float
        Column frequency component.
    """
    # Input validation
    phase = np.asarray(phase, dtype=np.float64)
    print(phase.ndim)
    print(phase.shape)
    if phase.ndim != 2:
        raise ValueError("Phase image must be 2D.")

    # Unwrap if requested
    if use_unwrap:
        phase = unwrap_phase(phase)

    # Normalization (0–1)
    vmin, vmax = np.nanmin(phase), np.nanmax(phase)
    if vmax > vmin:
        phase = (phase - vmin) / (vmax - vmin)

    # Image size
    M, N = phase.shape
    if M < 2 or N < 2:
        return {"SF": np.nan, "RF": np.nan, "CF": np.nan}

    # Spatial differences
    dx = phase[:, 1:] - phase[:, :-1]   # horizontal changes
    dy = phase[1:, :] - phase[:-1, :]   # vertical changes

    # Frequencies by direction
    RF = np.sqrt(np.sum(dx**2) / (M * (N - 1)))
    CF = np.sqrt(np.sum(dy**2) / ((M - 1) * N))

    # Total spatial frequency
    SF = np.sqrt(RF**2 + CF**2)

    return SF, RF, CF

def global_entropy_global(phase: np.ndarray, use_unwrap: bool = False, bins: int = 256, base: float = 2.0):
    """
    Computes the global Shannon entropy of a phase image.

    Parameters
    ----------
    phase : ndarray (float)
        2D phase image in radians [-π, π] or normalized values.
    use_unwrap : bool, optional
        If True, applies 2π unwrapping before computation (default: False).
    bins : int, optional
        Number of histogram bins (default: 256).
    base : float, optional
        Logarithm base (2 for bits, e for nats).

    Returns
    -------
    H : float
        Global Shannon entropy (in bits if base=2).
    """

    # Validate type and dimensions
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("Phase image must be 2D.")

    # Optional phase unwrapping
    if use_unwrap:
        phase = unwrap_phase(phase)

    # Normalize to 0–1 for consistency
    vmin, vmax = np.nanmin(phase), np.nanmax(phase)
    if vmax > vmin:
        phase = (phase - vmin) / (vmax - vmin)

    # Flatten and remove NaN/Inf
    vals = phase.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float('nan')

    # Compute histogram and probabilities
    hist, _ = np.histogram(vals, bins=bins, range=(0, 1))
    total = hist.sum()
    if total == 0:
        return float('nan')

    p = hist.astype(float) / total
    p = p[p > 0.0]

    # Compute Shannon entropy
    if p.size == 0:
        return 0.0

    if base == 2.0:
        H = -np.sum(p * np.log2(p))
    else:
        H = -np.sum(p * (np.log(p) / np.log(base)))

    return float(H)


def tsm_global(phase: np.ndarray, use_unwrap: bool = False, use_adaptive: bool = True):
    """
    Calculates the global dark-area metric (percentage of black pixels)
    from a phase image after binarization.

    Parameters
    ----------
    phase : ndarray (float)
        2D phase map in radians [-π, π] or any grayscale image.
    use_unwrap : bool, optional
        If True, applies 2π phase unwrapping before processing (default: False).
    use_adaptive : bool, optional
        If True, applies adaptive (local) thresholding. If False, uses global threshold (default: True).

    Returns
    -------
    TSM : float
        Percentage of black pixels (0–100%).
    """
    # --- Input validation ---
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("Phase image must be 2D.")

    # --- Optional unwrapping ---
    if use_unwrap:
        phase = unwrap_phase(phase)

    # --- Normalization (0–1) ---
    vmin, vmax = np.nanmin(phase), np.nanmax(phase)
    if vmax > vmin:
        phase = (phase - vmin) / (vmax - vmin)
    else:
        raise ValueError("Phase image has zero dynamic range.")

    # --- Binarization ---
    if use_adaptive:
        block_size = 35  # can be tuned
        local_thresh = threshold_local(phase, block_size)
        binary = phase > local_thresh
    else:
        global_thresh = np.mean(phase)
        binary = phase > global_thresh

    # --- Metric computation ---
    M, N = phase.shape
    num_black = np.sum(~binary)             # pixels classified as "dark" (False)
    TSM = (num_black / (M * N)) * 100.0  # % of dark area
    return TSM


def sharpness_global(phase: np.ndarray, use_unwrap: bool = False, beta: float = 2.0):
    """
    Calculates the Generalized Sharpness Metric (GSM) for a phase image.
    Based on:
        - J. R. Fienup & J. J. Miller, "Aberration correction by maximizing generalized sharpness metrics,"
          J. Opt. Soc. Am. A 20, 609–620 (2003).
        - M. T. Banet et al., "3D multi-plane sharpness metric maximization with variable corrective
          phase screens," Applied Optics 60(25), G243–G252 (2021).

    Parameters
    ----------
    phase : ndarray (float)
        2D phase map in radians [-π, π] or grayscale image.
    use_unwrap : bool, optional
        If True, applies 2π phase unwrapping before computing the metric. (default: False)
    beta : float, optional
        Exponent parameter of the generalized sharpness metric (default: 2.0).
        Typical range: 1.5–3.0

    Returns
    -------
    results : dict
        Dictionary with:
            - 'GSM' : float
                Generalized sharpness metric value.
            - 'Phase' : ndarray (float)
                Processed phase image (unwrapped if selected).
            - 'Intensity' : ndarray (float)
                Normalized intensity image used for metric computation.

    Notes
    -----
    The generalized sharpness metric is defined as:
        GSM = Σ [ I(x, y) ^ β ]
    where I(x, y) is the normalized image intensity.
    For β > 1, the metric favors sharper (high-contrast) images.
    """

    # --- Input validation ---
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("Phase image must be 2D.")

    # --- Optional unwrapping ---
    if use_unwrap:
        phase = unwrap_phase(phase)

    # --- Convert to normalized intensity (0–1) ---
    vmin, vmax = np.nanmin(phase), np.nanmax(phase)
    if vmax > vmin:
        intensity = (phase - vmin) / (vmax - vmin)
    else:
        raise ValueError("Phase image has zero dynamic range.")

    # --- Compute the generalized sharpness metric ---
    GSM = np.sum(intensity ** beta)

    return GSM


def legendre_background(
    complex_field: np.ndarray,
    limit: int,
    use_unwrap: bool = False,
    no_piston_compensation: bool = True,
    use_pca: bool = False
) -> dict:
    """
    Computes the Legendre polynomial coefficients describing the background
    phase of a complex field.

    Parameters
    ----------
    complex_field : ndarray (complex)
        2D complex field input (typically the reconstructed holographic field).
    limit : int
        Frequency-domain cropping radius around the Fourier center.
    use_unwrap : bool, optional
        If True, performs 2π phase unwrapping before Legendre fitting. (default: True)
    no_piston_compensation : bool, optional
        If True, skips piston optimization. (default: True)
    use_pca : bool, optional
        If True, extracts dominant component using PCA (SVD). (default: False)

    Returns
    -------
    results : dict
        Dictionary with:
            - 'Coefficients' : ndarray (float)
                Computed Legendre coefficients.
            - 'DominantPhase' : ndarray (float)
                Processed phase map (unwrapped if selected).
            - 'Reconstruction' : ndarray (float)
                Reconstructed background surface.
    """

    # --- Input validation ---
    complex_field = np.asarray(complex_field, dtype=np.complex128)
    if complex_field.ndim != 2:
        raise ValueError("Input complex_field must be a 2D array.")

    # --- Centered Fourier transform ---
    fft_field = fftshift(fft2(ifftshift(complex_field)))

    A, B = fft_field.shape
    center_A, center_B = A // 2, B // 2
    start_A, end_A = center_A - limit, center_A + limit
    start_B, end_B = center_B - limit, center_B + limit

    cropped_fft = fft_field[start_A:end_A, start_B:end_B]
    square = ifftshift(ifft2(fftshift(cropped_fft)))

    # --- Extract dominant phase component ---
    if use_pca:
        u, s, vt = svds(square, k=1, which='LM')
        dominant = u[:, :1] @ np.diag(s[:1]) @ vt[:1, :]
    else:
        dominant = square

    phase = np.angle(dominant)
    if use_unwrap:
        #phase = unwrap_phase(phase)  # or your custom unwrapping
        phase = unwrap_phase(phase)

    # --- Create normalized grid ---
    grid_size = phase.shape[0]
    coords = np.linspace(-1, 1 - 2 / grid_size, grid_size)
    X, Y = np.meshgrid(coords, coords)

    dA = (2 / grid_size) ** 2
    order = np.arange(1, 11)

    # --- Build orthonormal Legendre basis ---
    polynomials = square_legendre_fitting(order, X, Y)
    ny, nx, n_terms = polynomials.shape
    Legendres = polynomials.reshape(ny * nx, n_terms)

    zProds = Legendres.T @ Legendres * dA
    Legendres = Legendres / np.sqrt(np.diag(zProds))
    Legendres_norm_const = np.sum(Legendres ** 2, axis=0) * dA

    phase_vector = phase.reshape(-1, 1)
    Legendre_coeffs = np.sum(Legendres * phase_vector, axis=0) * dA

    # --- Optional piston compensation ---
    if not no_piston_compensation:
        values = np.arange(-np.pi, np.pi + np.pi / 6, np.pi / 6)
        variances = []

        for val in values:
            temp_coeffs = Legendre_coeffs.copy()
            temp_coeffs[0] = val
            coeffs_norm = temp_coeffs / np.sqrt(Legendres_norm_const)
            wavefront = np.sum((coeffs_norm[:, np.newaxis]) * Legendres.T, axis=0)
            temp_holo = np.exp(1j * np.angle(square)) / np.exp(1j * wavefront.reshape(ny, nx))
            variances.append(np.var(np.angle(temp_holo)))

        best = values[np.argmin(variances)]
        Legendre_coeffs[0] = best

    # --- Background reconstruction ---
    reconstruction = reconstruction_background(Legendre_coeffs, X, Y, order)

    return {
        'Coefficients': Legendre_coeffs,
        'DominantPhase': phase,
        'Reconstruction': reconstruction
    }


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


def reconstruction_background(coefficients, X, Y, orders):
    """Reconstructs phase background from Legendre coefficients."""
    polynomials = square_legendre_fitting(orders, X, Y)
    ny, nx, n_terms = polynomials.shape

    coeffs_used = coefficients[:n_terms]
    superficie = np.sum([c * p for c, p in zip(coeffs_used, np.moveaxis(polynomials, -1, 0))], axis=0)

    # Visualization (optional)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(superficie, cmap='viridis')
    plt.title('Legendre-reconstructed background')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.plot(coefficients, 'o-')
    plt.title('Legendre coefficients')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return superficie
