import numpy as np
from typing import Optional
from skimage.restoration import unwrap_phase
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds


def unwrap_with_scikit(wrapped_phase: np.ndarray) -> np.ndarray:
    """Unwrap phase images using scikit-image."""
    return unwrap_phase(wrapped_phase)


def std_background(img: np.ndarray, mask: Optional[np.ndarray] = None):
    """Standard deviation of the background phase, in radians."""
    backgroundValues = img[mask] if mask is not None else img.flatten()
    # Full precision: rounding to 4 decimals here hid differences below 1e-4.
    return float(np.std(backgroundValues))


def mean_absolute_deviation_background(img: np.ndarray, mask: Optional[np.ndarray] = None):
    """Mean absolute deviation of the background phase, in radians."""
    backgroundValues = img[mask] if mask is not None else img.flatten()
    meanValue = np.mean(backgroundValues)
    return float(np.mean(np.abs(backgroundValues - meanValue)))


def mean_background(img: np.ndarray, mask: Optional[np.ndarray] = None):
    """Mean background phase level, the piston term, in radians."""
    # RMS^2 = mean^2 + STD^2, so the piston is what RMS adds over STD.
    backgroundValues = img[mask] if mask is not None else img.flatten()
    return float(np.mean(backgroundValues))


def rms_background(img: np.ndarray, mask: Optional[np.ndarray] = None):
    """Root mean square of the background phase: sqrt(mean(x^2)), in radians."""
    backgroundValues = img[mask] if mask is not None else img.flatten()
    return float(np.sqrt(np.mean(backgroundValues ** 2)))


def pv_background(img: np.ndarray, mask: Optional[np.ndarray] = None):
    """Peak-to-valley of the background phase: max minus min, in radians."""
    backgroundValues = img[mask] if mask is not None else img.flatten()
    return float(np.max(backgroundValues) - np.min(backgroundValues))


def _fwhm_from_values(values: np.ndarray, bins: int = 100) -> float:
    """Full width at half maximum of the histogram, interpolated linearly."""
    finiteValues = np.asarray(values).ravel()
    finiteValues = finiteValues[~np.isnan(finiteValues)]
    if finiteValues.size < 5:
        return np.nan

    counts, binEdges = np.histogram(finiteValues, bins=bins)
    if not np.any(counts):
        return np.nan

    binCentres = 0.5 * (binEdges[:-1] + binEdges[1:])
    peakIndex = int(np.argmax(counts))
    peakHeight = float(counts[peakIndex])
    if peakHeight <= 0:
        return np.nan
    halfHeight = 0.5 * peakHeight

    # Walk left from the peak to where the histogram crosses half maximum.
    leftCrossing = None
    for binIndex in range(peakIndex, 0, -1):
        if counts[binIndex - 1] <= halfHeight <= counts[binIndex]:
            xLow, yLow = binCentres[binIndex - 1], counts[binIndex - 1]
            xHigh, yHigh = binCentres[binIndex], counts[binIndex]
            fraction = (halfHeight - yLow) / (yHigh - yLow + 1e-12)
            leftCrossing = xLow + fraction * (xHigh - xLow)
            break

    rightCrossing = None
    for binIndex in range(peakIndex, len(counts) - 1):
        if counts[binIndex] >= halfHeight >= counts[binIndex + 1]:
            xLow, yLow = binCentres[binIndex], counts[binIndex]
            xHigh, yHigh = binCentres[binIndex + 1], counts[binIndex + 1]
            fraction = (halfHeight - yLow) / (yHigh - yLow + 1e-12)
            rightCrossing = xLow + fraction * (xHigh - xLow)
            break

    # No crossing on one of the sides: the peak sits against the edge.
    if leftCrossing is None or rightCrossing is None:
        return np.nan
    return float(rightCrossing - leftCrossing)


def fwhm_background(img: np.ndarray, mask: Optional[np.ndarray] = None):
    """FWHM of the background phase histogram: how spread out the noise is."""
    backgroundValues = img[mask] if mask is not None else img.ravel()
    return _fwhm_from_values(backgroundValues, bins=100)


def _shannon_entropy(values: np.ndarray, bins: int = 256, base: float = 2.0) -> float:
    """Shannon entropy of a histogram of the values, in bits when base is 2."""
    finiteValues = np.asarray(values).ravel()
    finiteValues = finiteValues[np.isfinite(finiteValues)]
    if finiteValues.size == 0:
        return float('nan')

    counts, _ = np.histogram(finiteValues, bins=bins)
    totalCount = counts.sum()
    if totalCount == 0:
        return float('nan')

    probabilities = counts.astype(float) / totalCount
    probabilities = probabilities[probabilities > 0.0]
    if probabilities.size == 0:
        return 0.0

    if base == 2.0:
        return float(-np.sum(probabilities * np.log2(probabilities)))
    return float(-np.sum(probabilities * (np.log(probabilities) / np.log(base))))


def entropy_background(img: np.ndarray, mask: Optional[np.ndarray] = None,
                       bins: int = 256, base: float = 2.0):
    """Shannon entropy of the background phase values, in bits."""
    backgroundValues = img[mask] if mask is not None else img.ravel()
    return _shannon_entropy(backgroundValues, bins=bins, base=base)


def wrap_jump_density(phase, mask=None):
    """Percentage of adjacent pixel pairs that cross a 2*pi cliff.

    Counted INSIDE the region only, so the hard edge of the sample is not taken
    for a wrap. A compensated background reads about zero; anything above a
    fraction of a percent means fringes are still in there, and it is the
    quantity to read next to a Legendre fit, which smooths across such cliffs."""
    phase = np.asarray(phase, dtype=np.float64)
    horizontalJump = np.abs(np.diff(phase, axis=1)) > np.pi
    verticalJump = np.abs(np.diff(phase, axis=0)) > np.pi

    if mask is None:
        pairCount = horizontalJump.size + verticalJump.size
    else:
        # Only pairs INSIDE the background, so a hard sample edge is not a wrap.
        horizontalPair = mask[:, 1:] & mask[:, :-1]
        verticalPair = mask[1:, :] & mask[:-1, :]
        horizontalJump = horizontalJump & horizontalPair
        verticalJump = verticalJump & verticalPair
        pairCount = int(np.count_nonzero(horizontalPair)
                        + np.count_nonzero(verticalPair))

    if pairCount == 0:
        return 0.0

    jumpCount = int(np.count_nonzero(horizontalJump) + np.count_nonzero(verticalJump))
    return 100.0 * jumpCount / pairCount


def _looks_wrapped(phase, mask=None, fraction=5e-4):
    """True when 2*pi cliffs cross the region: the low pass would smear them."""
    # Kept for callers that ask to abstain; the density itself is the metric.
    return wrap_jump_density(phase, mask) / 100.0 > fraction


def legendre_background(phase, mask=None, manual=False, num_zones=2,
                        limit=64, order_max=5, NoPistonCompensation=True,
                        UsePCA=False, zones=None, show_reconstruction=None,
                        skip_wrapped=False):
    """Legendre coefficients of the background: inside the mask, or per zone.

    The fit begins with a low pass, and a low pass averages across a 2*pi cliff
    as if it were real, so on a wrapped region the coefficients describe a
    surface that is not there. `skip_wrapped=True` returns NaN in that case
    instead; by default the fit is returned as it stands and the caller is
    expected to report wrap_jump_density() beside it, so the reading can be
    judged rather than withheld."""
    if manual:
        # The caller selects the zones: no dialog here, so no wx in this layer.
        if not zones:
            return np.full(order_max, np.nan), []

        zoneStats = []
        zoneCoefficients = []

        for zoneNumber, (xmin, xmax, ymin, ymax) in enumerate(zones, start=1):
            zoneField = phase[ymin:ymax, xmin:xmax]
            # Wrapping in a zone poisons the mean of the coefficients too, so
            # the abstention is per zone when it is asked for at all.
            if skip_wrapped and _looks_wrapped(zoneField):
                coefficients = np.full(order_max, np.nan)
            else:
                coefficients = _process_legendre_zone(zoneField, limit, order_max,
                                                      UsePCA)
            zoneCoefficients.append(coefficients)
            zoneStats.append({'zone': zoneNumber, 'legendre': coefficients,
                              'coords': (xmin, xmax, ymin, ymax)})

        return np.mean(zoneCoefficients, axis=0), zoneStats

    if skip_wrapped and _looks_wrapped(phase, mask):
        return np.full(order_max, np.nan)

    coefficients = _process_legendre_zone(phase, limit, order_max,
                                          UsePCA, mask)

    if not NoPistonCompensation:
        coefficients = _optimize_piston(phase, coefficients, limit,
                                        order_max, UsePCA, mask)

    gridSize = phase.shape[0]
    gridCoords = np.linspace(-1, 1 - 2 / gridSize, gridSize)
    gridX, gridY = np.meshgrid(gridCoords, gridCoords)
    orders = np.arange(1, len(coefficients) + 1)

    # The caller decides how to display it: this module imports no viewer.
    if show_reconstruction is not None:
        show_reconstruction(
            reconstruction_background(coefficients, gridX, gridY, orders),
            coefficients
        )

    return coefficients


def _mask_on_grid(mask, grid_rows, grid_cols):
    """The background mask sampled onto the coarse grid the low-pass step leaves."""
    # Nearest neighbour at each block centre: no interpolation, so it stays boolean.
    rowIndex = np.minimum(((np.arange(grid_rows) + 0.5) * mask.shape[0]
                           / grid_rows).astype(int), mask.shape[0] - 1)
    colIndex = np.minimum(((np.arange(grid_cols) + 0.5) * mask.shape[1]
                           / grid_cols).astype(int), mask.shape[1] - 1)
    return mask[np.ix_(rowIndex, colIndex)]


def _low_pass_phase(phase, limit):
    """The smooth part of a phase map: its spectrum cropped around zero frequency."""
    spectrum = fftshift(fft2(ifftshift(phase)))
    fftRows, fftCols = spectrum.shape
    centreRow, centreCol = int(round(fftRows / 2)), int(round(fftCols / 2))

    spectrum = spectrum[centreRow - limit:centreRow + limit,
                        centreCol - limit:centreCol + limit]
    # ifft2 normalises by the CROPPED size, so the crop scales the phase up.
    cropScale = (spectrum.shape[0] / fftRows) * (spectrum.shape[1] / fftCols)
    return ifftshift(ifft2(fftshift(spectrum))).real * cropScale


def _process_legendre_zone(zone_phase, limit, order_max, UsePCA, mask=None):
    """Project one zone of the phase map onto the normalised Legendre basis."""
    # In phase throughout: the app converts to phase on load and never wraps.
    smoothPhase = _low_pass_phase(zone_phase, limit)

    if UsePCA:
        leftVectors, singularValues, rightVectors = svds(smoothPhase, k=1, which='LM')
        smoothPhase = (leftVectors[:, :1] @ np.diag(singularValues[:1])
                       @ rightVectors[:1, :])

    # The low pass leaves a grid of its own shape, and that shape is square only
    # where the crop reached the limit on BOTH axes. A region narrower than
    # 2*limit keeps that side's length, so the grid is rectangular -- and a
    # square grid then held one row per column, which could not even be formed:
    # a 42x36 zone raised "shapes (1764,5) (1512,1)".
    gridRows, gridCols = smoothPhase.shape
    rowCoords = np.linspace(-1, 1 - 2 / gridRows, gridRows)
    colCoords = np.linspace(-1, 1 - 2 / gridCols, gridCols)
    gridX, gridY = np.meshgrid(colCoords, rowCoords)
    cellArea = (2 / gridRows) * (2 / gridCols)
    orders = np.arange(1, order_max + 1)

    polynomials = square_legendre_fitting(orders, gridX, gridY)
    rowCount, colCount, termCount = polynomials.shape
    basis = polynomials.reshape(rowCount * colCount, termCount)

    # Normalise the basis so each term has unit norm over the grid.
    innerProducts = basis.T @ basis * cellArea
    basis = basis / np.sqrt(np.diag(innerProducts))
    phaseVector = smoothPhase.reshape(-1, 1)

    if mask is None:
        # Orthonormal over the whole grid, so the projection already is the fit.
        return np.sum(basis * phaseVector, axis=0) * cellArea

    # Over a subset the basis loses its orthonormality: project and you get noise.
    backgroundPixels = _mask_on_grid(mask, gridRows, gridCols).reshape(-1)
    if int(np.count_nonzero(backgroundPixels)) < termCount:
        return np.full(termCount, np.nan)
    coefficients, *_ = np.linalg.lstsq(basis[backgroundPixels],
                                       phaseVector[backgroundPixels], rcond=None)
    return coefficients.ravel()


def _optimize_piston(phase, coeffs, limit, order_max, UsePCA, mask=None):
    """Pick the piston coefficient that leaves the least residual variance."""
    smoothPhase = _low_pass_phase(phase, limit)

    # Rectangular-aware, like the fit it judges; see _process_legendre_zone.
    gridRows, gridCols = smoothPhase.shape
    rowCoords = np.linspace(-1, 1 - 2 / gridRows, gridRows)
    colCoords = np.linspace(-1, 1 - 2 / gridCols, gridCols)
    gridX, gridY = np.meshgrid(colCoords, rowCoords)
    cellArea = (2 / gridRows) * (2 / gridCols)
    orders = np.arange(1, order_max + 1)

    polynomials = square_legendre_fitting(orders, gridX, gridY)
    rowCount, colCount, termCount = polynomials.shape
    basis = polynomials.reshape(rowCount * colCount, termCount)

    innerProducts = basis.T @ basis * cellArea
    basis = basis / np.sqrt(np.diag(innerProducts))
    basisNorms = np.sum(basis ** 2, axis=0) * cellArea

    # The residual is judged where the background is, like the fit itself.
    backgroundPixels = (None if mask is None
                        else _mask_on_grid(mask, gridRows, gridCols))

    # Sweep the piston over a full turn in steps of pi/6.
    candidatePistons = np.arange(-np.pi, np.pi + np.pi / 6, np.pi / 6)
    residualVariances = []

    for piston in candidatePistons:
        trialCoeffs = coeffs.copy()
        trialCoeffs[0] = piston
        normalisedCoeffs = trialCoeffs / np.sqrt(basisNorms)
        wavefront = np.sum((normalisedCoeffs[:, np.newaxis]) * basis.T, axis=0)
        # Plain subtraction: both sides are phase, so there is nothing to unwrap.
        residualPhase = smoothPhase - wavefront.reshape(rowCount, colCount)
        residualVariances.append(np.var(residualPhase if backgroundPixels is None
                                        else residualPhase[backgroundPixels]))

    coeffs[0] = candidatePistons[np.argmin(residualVariances)]
    return coeffs


def square_legendre_fitting(order, X, Y):
    """Legendre polynomials 1 to 15 over a square grid, stacked on the last axis."""
    polynomials = []
    for termOrder in order:
        if termOrder == 1:
            polynomials.append(np.ones_like(X))
        elif termOrder == 2:
            polynomials.append(X)
        elif termOrder == 3:
            polynomials.append(Y)
        elif termOrder == 4:
            polynomials.append((3 * X**2 - 1) / 2)
        elif termOrder == 5:
            polynomials.append(X * Y)
        elif termOrder == 6:
            polynomials.append((3 * Y**2 - 1) / 2)
        elif termOrder == 7:
            polynomials.append((X * (5 * X**2 - 3)) / 2)
        elif termOrder == 8:
            polynomials.append((Y * (3 * X**2 - 1)) / 2)
        elif termOrder == 9:
            polynomials.append((X * (3 * Y**2 - 1)) / 2)
        elif termOrder == 10:
            polynomials.append((Y * (5 * Y**2 - 3)) / 2)
        elif termOrder == 11:
            polynomials.append((35 * X**4 - 30 * X**2 + 3) / 8)
        elif termOrder == 12:
            polynomials.append((X * Y * (5 * X**2 - 3)) / 2)
        elif termOrder == 13:
            polynomials.append(((3 * Y**2 - 1) * (3 * X**2 - 1)) / 4)
        elif termOrder == 14:
            polynomials.append((X * Y * (5 * Y**2 - 3)) / 2)
        elif termOrder == 15:
            polynomials.append((35 * Y**4 - 30 * Y**2 + 3) / 8)
    return np.stack(polynomials, axis=-1)


def reconstruction_background(coefficients, X, Y, orders):
    """Rebuild the background surface from its Legendre coefficients."""
    # Pure computation: displaying the result is ui/legendre_viewer.py's job.
    polynomials = square_legendre_fitting(orders, X, Y)
    rowCount, colCount, termCount = polynomials.shape
    usedCoefficients = coefficients[:termCount]

    surface = np.zeros((rowCount, colCount))
    for termIndex, coefficient in enumerate(usedCoefficients):
        surface += coefficient * polynomials[:, :, termIndex]

    return surface
