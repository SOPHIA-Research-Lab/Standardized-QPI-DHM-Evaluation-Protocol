
'''
Núcleo 1 - Residual background phase variance: Métricas convencionales. Se mide phase flatness in
background (object-free) regions of the reconstructed phase image. En estas, hay que hacer un
segmentador, para saber cuáles son las zonas de fondo en el objeto.
Métricas en este grupo:
- Standard deviation (STD) or Mean Absolute Deviation (MAD) (o las dos).
- RMS.
- Peak-to-Valley (P–V) Value in Background.
- Background phase tilt/Curvature residuals (esta involucra medir los coeficientes de legenre/zernike para el mapa de fase resultante, y con ellos, saber cuáles son los residuales)
- Full Width at Half Maximum (FWHM) of the phase histogram (esta es muy buena cuando hay bastante fondo, yo diría, la mejor. Me cuentas si no sabes cómo medirla.)
- Spatial frequency content of background (la misma que implementaste para el paper del VortexLegendre, pero para el fondo solamente).
- Entropy of background phase map (creo que ya la tienes, pero para el bakground).
'''

# Libraries
import numpy as np
from numpy.polynomial import legendre
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.restoration import unwrap_phase
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d


def legendre_background_correction(image: np.ndarray, order=3):
    """
    Fits 2D Legendre polynomials to model the background of the image.
    """
    image = np.array(image)
    h, w = image.shape
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)

    # Create 2D Legendre polynomial basis
    basis = []
    for i in range(order + 1):
        for j in range(order + 1):
            if i + j <= order:
                leg_i = legendre.legval(X, [0] * i + [1])
                leg_j = legendre.legval(Y, [0] * j + [1])
                basis.append((leg_i * leg_j).flatten())

    basis = np.array(basis).T

    # Fit coefficients
    image_flat = image.flatten()
    coeffs = np.linalg.lstsq(basis, image_flat, rcond=None)[0]

    # Reconstruct background
    background = (basis @ coeffs).reshape(image.shape)

    return image - background, background


def pca_background_separation(image: np.ndarray, n_components=5, patch_size=(8, 8), keep_components=2):
    """
    Estimate the background via PCA on overlapping patches and reconstruct the background image.
    """
    img = np.asarray(image, dtype=float)
    h, w = img.shape
    ph, pw = patch_size

    # Extract overlapping patches
    patches = extract_patches_2d(img, patch_size=patch_size)
    N = patches.shape[0]
    patches_flat = patches.reshape(N, -1)

    # Standardize and apply PCA
    scaler = StandardScaler()
    patches_scaled = scaler.fit_transform(patches_flat)

    pca = PCA(n_components=n_components, svd_solver="auto", whiten=False)
    Z = pca.fit_transform(patches_scaled)

    # Keep only the first components (background) and zero out the rest
    keep = max(1, min(keep_components, n_components))
    Z_bg = np.hstack([Z[:, :keep], np.zeros((N, n_components - keep))])

    # Reconstruct background patches
    patches_bg_scaled = pca.inverse_transform(Z_bg)
    patches_bg = scaler.inverse_transform(patches_bg_scaled).reshape(N, ph, pw)

    # Reconstruct background image from patches
    background = reconstruct_from_patches_2d(patches_bg, (h, w))

    # Corrected image
    corrected = img - background
    return corrected, background


def legendre_background_extraction(field_compensate, limit, max_order=4, return_coefficients=False):
    """
    Extracts the background phase using Legendre polynomial fitting.
    """

    # Handle both complex fields and phase images
    if np.iscomplexobj(field_compensate):
        # If input is complex, extract phase
        phase_input = np.angle(field_compensate)
    else:
        # If input is already phase/real image
        phase_input = field_compensate

    # Centered Fourier transform for complex fields, or direct processing for phase
    if np.iscomplexobj(field_compensate):
        fftField = np.fft.fftshift(np.fft.fft2(field_compensate))
        A, B = fftField.shape
        center_A = int(round(A / 2))
        center_B = int(round(B / 2))

        start_A = int(center_A - limit)
        end_A = int(center_A + limit)
        start_B = int(center_B - limit)
        end_B = int(center_B + limit)

        fftField = fftField[start_A:end_A, start_B:end_B]
        square = np.fft.ifft2(np.fft.ifftshift(fftField))
        phase_to_fit = unwrap_phase(np.angle(square))
        original_shape = field_compensate.shape
    else:
        # For direct phase images, work with the full image
        phase_to_fit = phase_input
        original_shape = phase_input.shape

    # Normalized spatial grid
    gridSize_y, gridSize_x = phase_to_fit.shape
    coords_x = np.linspace(-1, 1 - 2 / gridSize_x, gridSize_x)
    coords_y = np.linspace(-1, 1 - 2 / gridSize_y, gridSize_y)
    X, Y = np.meshgrid(coords_x, coords_y)

    # Area element for integration
    dA = (2 / gridSize_x) * (2 / gridSize_y)

    # Use only low-order polynomials for background (smooth variations)
    order = np.arange(1, max_order + 1)

    # Get Legendre polynomial basis (only low orders for background)
    polynomials = square_legendre_fitting_background(order, X, Y)
    ny, nx, n_terms = polynomials.shape
    Legendres = polynomials.reshape(ny * nx, n_terms)

    # Orthonormalization
    zProds = Legendres.T @ Legendres * dA
    Legendres = Legendres / np.sqrt(np.diag(zProds))

    # Normalization constants
    Legendres_norm_const = np.sum(Legendres ** 2, axis=0) * dA

    # Phase vector
    phaseVector = phase_to_fit.reshape(-1, 1)

    # Project onto Legendre basis
    Legendre_Coefficients = np.sum(Legendres * phaseVector, axis=0) * dA

    # Reconstruct background using fitted coefficients
    coeffs_norm = Legendre_Coefficients / np.sqrt(Legendres_norm_const)
    background_flat = np.sum(coeffs_norm[:, np.newaxis] * Legendres.T, axis=0)
    background = background_flat.reshape(ny, nx)

    # If input was complex field processed in frequency domain,
    # we need to map back to original size
    if np.iscomplexobj(field_compensate) and background.shape != original_shape:
        # Simple resize/interpolation to match original shape
        from scipy.ndimage import zoom
        zoom_factors = (original_shape[0] / background.shape[0],
                        original_shape[1] / background.shape[1])
        background = zoom(background, zoom_factors, order=1)

    if return_coefficients:
        return background, Legendre_Coefficients
    else:
        return background


def square_legendre_fitting_background(order, X, Y):
    """
    Generate low-order Legendre polynomials suitable for background fitting.
    Focus on smooth, low-frequency variations.
    """
    polynomials = []
    for i in order:
        if i == 1:
            # Constant term (piston)
            polynomials.append(np.ones_like(X))
        elif i == 2:
            # Linear X (tilt)
            polynomials.append(X)
        elif i == 3:
            # Linear Y (tilt)
            polynomials.append(Y)
        elif i == 4:
            # Quadratic X (defocus/astigmatism)
            polynomials.append((3 * X ** 2 - 1) / 2)
        elif i == 5:
            # Cross term XY
            polynomials.append(X * Y)
        elif i == 6:
            # Quadratic Y (defocus/astigmatism)
            polynomials.append((3 * Y ** 2 - 1) / 2)
        # Add higher orders if needed, but typically background needs only low orders
        elif i == 7:
            polynomials.append((X * (5 * X ** 2 - 3)) / 2)
        elif i == 8:
            polynomials.append((Y * (3 * X ** 2 - 1)) / 2)
        elif i == 9:
            polynomials.append((X * (3 * Y ** 2 - 1)) / 2)
        elif i == 10:
            polynomials.append((Y * (5 * Y ** 2 - 3)) / 2)

    return np.stack(polynomials, axis=-1)


def spatial_frequency(self, img):
    gray = np.array(img.convert("L")).astype(np.float64)
    M, N = gray.shape
    RF = np.sqrt(np.sum((gray[:, 1:] - gray[:, :-1]) ** 2) / (M * N))
    CF = np.sqrt(np.sum((gray[1:, :] - gray[:-1, :]) ** 2) / (M * N))
    SF = np.sqrt(RF ** 2 + CF ** 2)

    return SF


def entropy_background(self, img):
    gray_img = img.convert("L")
    hist = gray_img.histogram()
    hist = np.array(hist) / np.sum(hist)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


def std_background(img,  mask: np.ndarray = None):
    if mask is not None:
        values = img[mask]
    else:
        values = img.flatten()

    std_val = np.std(values)
    return std_val


def mean_absolute_deviation_background(image: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Compute the Mean Absolute Deviation (MAD) of an image or array.
    """
    if mask is not None:
        values = image[mask]
    else:
        values = image.flatten()

    mean_val = np.mean(values)
    mad = np.mean(np.abs(values - mean_val))
    return mad


def rms(image: np.ndarray, mask: np.ndarray = None, relative_to_mean: bool = True) -> float:
    """
    Compute the Root Mean Square (RMS) of an image or array.
    """
    if mask is not None:
        values = image[mask]
    else:
        values = image.flatten()

    if relative_to_mean:
        mean_val = np.mean(values)
        rms_val = np.sqrt(np.mean((values - mean_val) ** 2))
    else:
        rms_val = np.sqrt(np.mean(values ** 2))

    return rms_val


def calculate_background_fwhm(background):
    """
    Calculate FWHM of background phase noise using histogram method
    """
    # Create histogram of background values
    hist, bin_edges = np.histogram(background.flatten(), bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peak of histogram
    max_idx = np.argmax(hist)
    max_value = hist[max_idx]
    half_max = max_value / 2

    # Find left edge (where histogram crosses half maximum before peak)
    left_idx = np.where(hist[:max_idx] <= half_max)[0]

    # Find right edge (where histogram crosses half maximum after peak)
    right_idx = np.where(hist[max_idx:] <= half_max)[0]

    # Calculate FWHM if both edges are found
    if len(left_idx) > 0 and len(right_idx) > 0:
        left_edge = bin_centers[left_idx[-1]]
        right_edge = bin_centers[max_idx + right_idx[0]]
        fwhm = right_edge - left_edge
    else:
        fwhm = np.nan

    return fwhm


def calculate_phase_entropy_background(phase_background, n_bins=256, method='shannon'):
    """
    Calculate entropy of background phase map, using shannon
    """
    import numpy as np

    # Flatten the image and remove any NaN or infinite values
    phase_flat = phase_background.flatten()
    phase_flat = phase_flat[np.isfinite(phase_flat)]

    if len(phase_flat) == 0:
        return np.nan

    # Create histogram with specified number of bins
    hist, bin_edges = np.histogram(phase_flat, bins=n_bins, density=True)

    # Calculate bin width for probability normalization
    bin_width = bin_edges[1] - bin_edges[0]

    # Convert to probabilities (normalize histogram)
    probabilities = hist * bin_width

    # Remove zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]

    # Calculate Shannon entropy in bits
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy