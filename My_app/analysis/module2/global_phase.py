import numpy as np
from scipy.ndimage import laplace, convolve
from skimage.restoration import unwrap_phase
from skimage.filters import threshold_local


def phase_gradient_prewitt(phase: np.ndarray, use_unwrap: bool = False):
    """
    Global Phase Gradient (GPG) computed with Prewitt kernels.

    Follows Yang et al., Opt. Express 31(22), 36188-36201 (2023), Eqs. (8)-(10):

        G1 = phi * T1,   G2 = phi * T2
        GPG = (1 / (N*M)) * sum( sqrt(G1^2 + G2^2) )

    where * is discrete convolution and T1, T2 are the 3x3 horizontal and
    vertical Prewitt kernels. The 1/(N*M) normalisation is what makes the
    value comparable between reconstructions of different image sizes.

    The kernels are deliberately left unnormalised, as in the paper: T1 sums
    three rows of a difference taken over two pixels, so the result carries a
    gain of 6 relative to a plain rad/pixel derivative.

    Unlike a least-squares plane fit, which only sees the global tilt, this
    averages the LOCAL gradient magnitude, so it also responds to background
    noise and texture.

    Parameters
    ----------
    phase : ndarray (float)
        2D phase map in radians.
    use_unwrap : bool, optional
        Apply 2*pi phase unwrapping before differentiating (default: False).

    Returns
    -------
    float
        Average local phase-gradient magnitude over the image.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("Phase image must be 2D.")

    if use_unwrap:
        phase = unwrap_phase(phase)

    # Eq. (10): Prewitt operators, horizontal and vertical
    T1 = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]], dtype=np.float64)
    T2 = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1]], dtype=np.float64)

    G1 = convolve(phase, T1, mode="reflect")
    G2 = convolve(phase, T2, mode="reflect")

    # Eq. (8): mean local gradient magnitude over the image
    return float(np.sum(np.hypot(G1, G2)) / phase.size)


def laplacian_energy(phase: np.ndarray, use_unwrap: bool = False):
    """
    Energy of the Laplacian of a phase image.

    Parameters
    ----------
    phase : ndarray (float)
        2D phase map in radians.
    use_unwrap : bool, optional
        Apply 2*pi phase unwrapping first (default: False).

    Returns
    -------
    float
        Mean squared Laplacian, sum(lap^2) / (N*M), in rad^2/px^4.
    """
    phase = np.asarray(phase, dtype=np.float64)

    if use_unwrap:
        phase = unwrap_phase(phase)

    phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)

    lap_map = laplace(phase, mode="reflect")

    N, M = lap_map.shape
    energy = float(np.sum(lap_map ** 2) / (N * M))

    return energy


def maximum_minus_minimum(phase: np.ndarray, use_unwrap: bool = False):
    """
    Peak-to-valley of a phase image: max(phase) - min(phase).

    Parameters
    ----------
    phase : ndarray (float)
        2D phase map in radians.
    use_unwrap : bool, optional
        Apply 2*pi phase unwrapping first (default: False).

    Returns
    -------
    float
        Difference between the maximum and the minimum of the phase, in
        radians. Being based on two single pixels, it is sensitive to
        outliers.
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
        If True, applies 2*pi unwrapping before computation (default: False).

    Returns
    -------
    float
        Total spatial frequency, SF = sqrt(RF^2 + CF^2).

        The value is DIMENSIONLESS, not rad/pixel: the phase is min-max
        normalised to [0, 1] before the differences are taken, so the
        original radian scale is divided out. Two images with identical
        texture but different phase amplitude give the same SF.

        RF (row frequency) and CF (column frequency) are computed
        internally but not returned.
    """
    # Input validation
    phase = np.asarray(phase, dtype=np.float64)
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
        # Every other path returns a float; returning a dict here gave
        # the caller an object it could not format into the table.
        return float('nan')

    # Spatial differences
    dx = phase[:, 1:] - phase[:, :-1]   # horizontal changes
    dy = phase[1:, :] - phase[:-1, :]   # vertical changes

    # Frequencies by direction
    RF = np.sqrt(np.sum(dx**2) / (M * (N - 1)))
    CF = np.sqrt(np.sum(dy**2) / ((M - 1) * N))

    # Total spatial frequency
    SF = np.sqrt(RF**2 + CF**2)

    return SF

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


def normalised_phase(phase, use_unwrap=False):
    """The phi_comp_nor of the TSM definition: the phase mapped onto [0, 1]."""
    phi = np.asarray(phase, dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError("Phase image must be 2D.")
    if use_unwrap:
        phi = unwrap_phase(phi)

    vmin, vmax = np.nanmin(phi), np.nanmax(phi)
    if vmax <= vmin:
        raise ValueError("Phase image has zero dynamic range.")
    return (phi - vmin) / (vmax - vmin)


def histogram_valley(phi_nor, bins=256, smooth=5):
    """
    Suggest the binarisation threshold: the valley between the two peaks.

    A compensated phase map is bimodal -- background on one side, sample on
    the other -- so the threshold that separates them sits in the dip between
    the two humps. The dip is found geometrically: draw the chord joining the
    two peaks and take the histogram bin lying furthest BELOW it, which is
    the d of the usual figure. That beats "the lowest bin between them",
    because a noisy histogram has many local minima while the distance to the
    chord has one clear maximum.

    The two peaks are picked by PROMINENCE, not by height within a fixed
    window: a first attempt used "the tallest bin at least bins/20 away",
    which on a wide hump simply found its own shoulder and put the threshold
    on top of the first peak.

    Parameters
    ----------
    phi_nor : ndarray
        Phase normalised to [0, 1].
    bins : int
        Histogram resolution.
    smooth : int
        Width of the moving average applied before looking for peaks.

    Returns
    -------
    dict
        threshold : float in [0, 1]
        centres, counts, smoothed : the histogram, for plotting
        peak_lo, peak_hi : bin indices of the two peaks
        valley : bin index of the chosen threshold
        ok : False when no second peak stands out, in which case the
             histogram is not bimodal and `threshold` falls back to the mean.
    """
    from scipy.signal import find_peaks

    valores = np.asarray(phi_nor, dtype=np.float64).ravel()
    valores = valores[np.isfinite(valores)]

    counts, edges = np.histogram(valores, bins=bins, range=(0.0, 1.0))
    centres = 0.5 * (edges[:-1] + edges[1:])

    suave = counts.astype(float)
    if smooth > 1:
        suave = np.convolve(suave, np.ones(smooth) / smooth, mode='same')

    base = {'centres': centres, 'counts': counts, 'smoothed': suave}

    if suave.max() <= 0:
        base.update({'threshold': 0.5, 'peak_lo': 0, 'peak_hi': 0,
                     'valley': bins // 2, 'ok': False})
        return base

    # A hump has to rise at least 5% of the tallest one above its surroundings
    # to count, and two peaks closer than a twentieth of the range are the
    # same hump.
    picos, props = find_peaks(suave,
                              prominence=0.05 * suave.max(),
                              distance=max(3, bins // 20))

    if len(picos) < 2:
        base.update({'threshold': float(valores.mean()),
                     'peak_lo': int(picos[0]) if len(picos) else 0,
                     'peak_hi': int(picos[0]) if len(picos) else 0,
                     'valley': bins // 2, 'ok': False})
        return base

    # The two most prominent ones, in histogram order.
    dos = picos[np.argsort(props['prominences'])[-2:]]
    izq, der = int(min(dos)), int(max(dos))

    x = np.arange(izq, der + 1, dtype=float)
    y = suave[izq:der + 1]
    x0, y0 = float(izq), float(suave[izq])
    dx, dy = float(der) - x0, float(suave[der]) - y0
    norma = np.hypot(dx, dy)

    if norma == 0:
        valle = (izq + der) // 2
    else:
        # Positive where the histogram runs BELOW the chord.
        distancia = ((x - x0) * dy - (y - y0) * dx) / norma
        valle = int(izq + np.argmax(distancia))

    base.update({'threshold': float(centres[valle]),
                 'peak_lo': izq, 'peak_hi': der,
                 'valley': valle, 'ok': True})
    return base


def tsm_global(phase: np.ndarray, use_unwrap: bool = False, use_adaptive: bool = True,
               threshold: float = None):
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
    if threshold is not None:
        # Chosen by the user on the histogram; see histogram_valley().
        binary = phase > float(threshold)
    elif use_adaptive:
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
        - J. R. Fienup & J. J. Miller, "Aberration correction by maximizing
          generalized sharpness metrics," J. Opt. Soc. Am. A 20, 609-620 (2003).
        - M. T. Banet et al., "3D multi-plane sharpness metric maximization with
          variable corrective phase screens," Applied Optics 60(25), G243-G252 (2021).

    Parameters
    ----------
    phase : ndarray (float)
        2D phase map in radians.
    use_unwrap : bool, optional
        Apply 2*pi phase unwrapping first (default: False).
    beta : float, optional
        Exponent of the generalized sharpness metric (default: 2.0).

    Returns
    -------
    float
        Generalized sharpness metric value. The processed phase and the
        normalised intensity are used internally and not returned.

    Notes
    -----
    The published metric is a plain sum, GSM = sum( I(x,y) ^ beta ), but this
    implementation divides by M*N, i.e. it returns the MEAN of I^beta, so that
    images of different sizes stay comparable.

    I(x, y) here is the phase min-max normalised to [0, 1], not an optical
    intensity, so the result is dimensionless and invariant to the phase
    amplitude. For beta > 1 the metric favours high-contrast images.
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
    M, N = intensity.shape
    GSM = np.sum(intensity ** beta) / (M * N)

    return GSM


import numpy as np
from skimage.restoration import unwrap_phase


def fwhm_global(phase: np.ndarray, use_unwrap: bool = False, bins: int = 100):
    """
    Global FWHM of the phase histogram.

    Same estimator as Module 1 (``residual_background.fwhm_background``),
    applied to the WHOLE image instead of the masked background. This mirrors
    how PV, entropy and spatial frequency already have a global counterpart in
    this module.

    It implements the textbook definition, FWHM = x2 - x1 with
    p(x1) = p(x2) = max(p)/2, where p is the histogram of the phase values.

    This replaced a radial-Fourier estimator that fitted a Gaussian to the
    power spectrum. That one carried no reference, returned a fraction of the
    image width rather than a physical unit, changed with the aspect ratio,
    and mixed background noise with resolution, so it could not be attributed
    to either.

    Parameters
    ----------
    phase : ndarray (float)
        2D phase map in radians.
    use_unwrap : bool, optional
        Apply 2*pi phase unwrapping first (default: False).
    bins : int, optional
        Histogram bins (default: 100, as in Module 1).

    Returns
    -------
    float
        Width of the histogram at half of its maximum, in radians. NaN when
        the histogram has no peak with a crossing on both sides, which is the
        definition saying it does not apply to that distribution.
    """
    from analysis.module1.residual_background import _fwhm_from_values

    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("Phase image must be 2D.")

    if use_unwrap:
        phase = unwrap_phase(phase)

    return _fwhm_from_values(phase, bins=bins)
