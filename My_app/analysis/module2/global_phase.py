import numpy as np
from scipy.ndimage import laplace, convolve
from skimage.restoration import unwrap_phase
from skimage.filters import threshold_local


def phase_gradient_prewitt(phase: np.ndarray, use_unwrap: bool = False):
    """Global Phase Gradient: mean local gradient magnitude, from Prewitt kernels."""
    # Yang et al., Opt. Express 31(22), 36188-36201 (2023), Eqs. (8)-(10).
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("Phase image must be 2D.")

    if use_unwrap:
        phase = unwrap_phase(phase)

    # Divided by 6: three rows averaged, over a central difference of two pixels.
    # Yang et al. leave it out, so this reads 6 times smaller than their figures.
    horizontalKernel = np.array([[1, 0, -1],
                                 [1, 0, -1],
                                 [1, 0, -1]], dtype=np.float64) / 6.0
    verticalKernel = np.array([[1, 1, 1],
                               [0, 0, 0],
                               [-1, -1, -1]], dtype=np.float64) / 6.0

    horizontalGradient = convolve(phase, horizontalKernel, mode="reflect")
    verticalGradient = convolve(phase, verticalKernel, mode="reflect")

    # The LOCAL magnitude, so it responds to noise as well as to global tilt.
    return float(np.sum(np.hypot(horizontalGradient, verticalGradient)) / phase.size)


def phase_tilt_plane(phase: np.ndarray, use_unwrap: bool = False):
    """Linear phase gradient: the slope of the least-squares plane, in rad/px."""
    # Blind to noise, unlike phase_gradient_prewitt: it fits, it does not differentiate.
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("Phase image must be 2D.")

    if use_unwrap:
        phase = unwrap_phase(phase)

    rowCount, colCount = phase.shape
    if rowCount < 2 or colCount < 2:
        return float('nan')

    rowIndex, colIndex = np.mgrid[0:rowCount, 0:colCount]
    design = np.column_stack((colIndex.ravel(), rowIndex.ravel(),
                              np.ones(phase.size)))
    finiteValues = np.isfinite(phase.ravel())
    if int(np.count_nonzero(finiteValues)) < 3:
        return float('nan')

    # alpha along x, beta along y, the offset discarded: only the slope is the metric.
    (alpha, beta, _), *_ = np.linalg.lstsq(design[finiteValues],
                                           phase.ravel()[finiteValues], rcond=None)
    return float(np.hypot(alpha, beta))


def laplacian_energy(phase: np.ndarray, use_unwrap: bool = False):
    """Mean squared Laplacian of the phase, in rad^2/px^4: high means many edges."""
    phase = np.asarray(phase, dtype=np.float64)

    if use_unwrap:
        phase = unwrap_phase(phase)

    phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)

    laplacianMap = laplace(phase, mode="reflect")
    rowCount, colCount = laplacianMap.shape
    return float(np.sum(laplacianMap ** 2) / (rowCount * colCount))


def maximum_minus_minimum(phase: np.ndarray, use_unwrap: bool = False):
    """Peak-to-valley of the phase in radians: based on two pixels, so outlier-prone."""
    if use_unwrap:
        phase = unwrap_phase(phase)

    return float(np.max(phase) - np.min(phase))


def spatial_frequency_global(phase: np.ndarray, use_unwrap: bool = False):
    """Spatial frequency sqrt(RF^2 + CF^2) of the phase normalised to [0, 1]."""
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("Phase image must be 2D.")

    if use_unwrap:
        phase = unwrap_phase(phase)

    # Normalising divides the radian scale out: the result is DIMENSIONLESS.
    phaseMin, phaseMax = np.nanmin(phase), np.nanmax(phase)
    if phaseMax > phaseMin:
        phase = (phase - phaseMin) / (phaseMax - phaseMin)

    rowCount, colCount = phase.shape
    if rowCount < 2 or colCount < 2:
        # Every other path returns a float, and the table can only format one.
        return float('nan')

    horizontalDiff = phase[:, 1:] - phase[:, :-1]
    verticalDiff = phase[1:, :] - phase[:-1, :]

    rowFrequency = np.sqrt(np.sum(horizontalDiff ** 2) / (rowCount * (colCount - 1)))
    colFrequency = np.sqrt(np.sum(verticalDiff ** 2) / ((rowCount - 1) * colCount))

    return np.sqrt(rowFrequency ** 2 + colFrequency ** 2)


def global_entropy_global(phase: np.ndarray, use_unwrap: bool = False,
                          bins: int = 256, base: float = 2.0):
    """Shannon entropy of the whole phase map, in bits when base is 2."""
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("Phase image must be 2D.")

    if use_unwrap:
        phase = unwrap_phase(phase)

    phaseMin, phaseMax = np.nanmin(phase), np.nanmax(phase)
    if phaseMax > phaseMin:
        phase = (phase - phaseMin) / (phaseMax - phaseMin)

    finiteValues = phase.ravel()
    finiteValues = finiteValues[np.isfinite(finiteValues)]
    if finiteValues.size == 0:
        return float('nan')

    counts, _ = np.histogram(finiteValues, bins=bins, range=(0, 1))
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


def normalised_phase(phase, use_unwrap=False):
    """The phi_comp_nor of the TSM definition: the phase mapped onto [0, 1]."""
    normalised = np.asarray(phase, dtype=np.float64)
    if normalised.ndim != 2:
        raise ValueError("Phase image must be 2D.")
    if use_unwrap:
        normalised = unwrap_phase(normalised)

    phaseMin, phaseMax = np.nanmin(normalised), np.nanmax(normalised)
    if phaseMax <= phaseMin:
        raise ValueError("Phase image has zero dynamic range.")
    return (normalised - phaseMin) / (phaseMax - phaseMin)


def histogram_valley(phi_nor, bins=256, smooth=5):
    """Suggest the TSM threshold: the histogram valley between the two peaks."""
    # The dip between the humps: the bin furthest BELOW the chord joining them.
    from scipy.signal import find_peaks

    finiteValues = np.asarray(phi_nor, dtype=np.float64).ravel()
    finiteValues = finiteValues[np.isfinite(finiteValues)]

    counts, edges = np.histogram(finiteValues, bins=bins, range=(0.0, 1.0))
    centres = 0.5 * (edges[:-1] + edges[1:])

    smoothed = counts.astype(float)
    if smooth > 1:
        smoothed = np.convolve(smoothed, np.ones(smooth) / smooth, mode='same')

    result = {'centres': centres, 'counts': counts, 'smoothed': smoothed}

    if smoothed.max() <= 0:
        result.update({'threshold': 0.5, 'peak_lo': 0, 'peak_hi': 0,
                       'valley': bins // 2, 'ok': False})
        return result

    # Picked by PROMINENCE: by height, a wide hump found its own shoulder.
    peaks, peakProperties = find_peaks(smoothed,
                                       prominence=0.05 * smoothed.max(),
                                       distance=max(3, bins // 20))

    if len(peaks) < 2:
        result.update({'threshold': float(finiteValues.mean()),
                       'peak_lo': int(peaks[0]) if len(peaks) else 0,
                       'peak_hi': int(peaks[0]) if len(peaks) else 0,
                       'valley': bins // 2, 'ok': False})
        return result

    # The two most prominent ones, in histogram order.
    topTwoPeaks = peaks[np.argsort(peakProperties['prominences'])[-2:]]
    leftPeak, rightPeak = int(min(topTwoPeaks)), int(max(topTwoPeaks))

    binIndices = np.arange(leftPeak, rightPeak + 1, dtype=float)
    binHeights = smoothed[leftPeak:rightPeak + 1]
    chordX, chordY = float(leftPeak), float(smoothed[leftPeak])
    chordRun, chordRise = float(rightPeak) - chordX, float(smoothed[rightPeak]) - chordY
    chordLength = np.hypot(chordRun, chordRise)

    if chordLength == 0:
        valleyIndex = (leftPeak + rightPeak) // 2
    else:
        # Positive where the histogram runs BELOW the chord.
        distanceToChord = (((binIndices - chordX) * chordRise
                            - (binHeights - chordY) * chordRun) / chordLength)
        valleyIndex = int(leftPeak + np.argmax(distanceToChord))

    result.update({'threshold': float(centres[valleyIndex]),
                   'peak_lo': leftPeak, 'peak_hi': rightPeak,
                   'valley': valleyIndex, 'ok': True})
    return result


def tsm_global(phase: np.ndarray, use_unwrap: bool = False, use_adaptive: bool = True,
               threshold: float = None):
    """Percentage of dark pixels after binarising the normalised phase."""
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("Phase image must be 2D.")

    if use_unwrap:
        phase = unwrap_phase(phase)

    phaseMin, phaseMax = np.nanmin(phase), np.nanmax(phase)
    if phaseMax > phaseMin:
        phase = (phase - phaseMin) / (phaseMax - phaseMin)
    else:
        raise ValueError("Phase image has zero dynamic range.")

    if threshold is not None:
        # Chosen by the user on the histogram; see histogram_valley().
        binarised = phase > float(threshold)
    elif use_adaptive:
        # Both split a symmetric distribution in half, hence the old ~50 %.
        localThreshold = threshold_local(phase, 35)
        binarised = phase > localThreshold
    else:
        binarised = phase > np.mean(phase)

    rowCount, colCount = phase.shape
    darkPixels = np.sum(~binarised)
    return (darkPixels / (rowCount * colCount)) * 100.0


def sharpness_global(phase: np.ndarray, use_unwrap: bool = False, beta: float = 2.0):
    """Generalized Sharpness Metric: mean of the normalised phase raised to beta."""
    # Fienup & Miller, J. Opt. Soc. Am. A 20, 609-620 (2003), as a mean.
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("Phase image must be 2D.")

    if use_unwrap:
        phase = unwrap_phase(phase)

    # Normalised phase, not an optical intensity: blind to the amplitude.
    phaseMin, phaseMax = np.nanmin(phase), np.nanmax(phase)
    if phaseMax > phaseMin:
        intensity = (phase - phaseMin) / (phaseMax - phaseMin)
    else:
        raise ValueError("Phase image has zero dynamic range.")

    rowCount, colCount = intensity.shape
    return np.sum(intensity ** beta) / (rowCount * colCount)


def fwhm_global(phase: np.ndarray, use_unwrap: bool = False, bins: int = 100):
    """FWHM of the whole-image phase histogram, in radians, NaN when undefined."""
    # The Module 1 estimator applied to the whole field.
    from analysis.module1.residual_background import _fwhm_from_values

    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("Phase image must be 2D.")

    if use_unwrap:
        phase = unwrap_phase(phase)

    return _fwhm_from_values(phase, bins=bins)
