import numpy as np
from scipy.ndimage import laplace, convolve
from skimage.restoration import unwrap_phase
from skimage.filters import threshold_local

# Which side of a threshold holds the sample is the same question Module I
# answers for the background mask, so the same rule is used here.
from analysis.module1.segmentation import bright_sample


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


def std_global(phase: np.ndarray, use_unwrap: bool = False):
    """Standard deviation of the whole phase map, in radians.

    The companion of the PV above: that one rests on two pixels, this one on
    all of them, so a single outlier moves PV and leaves STD where it was.
    No mask is involved -- Module II measures the map as it stands, sample
    included, which is what separates this from STD_B in Module I.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("Phase image must be 2D.")

    if use_unwrap:
        phase = unwrap_phase(phase)

    # One NaN would poison the whole reading, and every other path here
    # returns a float, which is all the table can format.
    finiteValues = phase[np.isfinite(phase)]
    if finiteValues.size == 0:
        return float('nan')

    return float(np.std(finiteValues))


def spatial_frequency_global(phase: np.ndarray, use_unwrap: bool = False):
    """Spatial frequency sqrt(RF^2 + CF^2) of the phase normalized to [0, 1]."""
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("Phase image must be 2D.")

    if use_unwrap:
        phase = unwrap_phase(phase)

    # Normalizing divides the radian scale out: the result is DIMENSIONLESS.
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


def normalized_phase(phase, use_unwrap=False):
    """The phi_comp_nor of the TSM definition: the phase mapped onto [0, 1]."""
    normalized = np.asarray(phase, dtype=np.float64)
    if normalized.ndim != 2:
        raise ValueError("Phase image must be 2D.")
    if use_unwrap:
        normalized = unwrap_phase(normalized)

    phaseMin, phaseMax = np.nanmin(normalized), np.nanmax(normalized)
    if phaseMax <= phaseMin:
        raise ValueError("Phase image has zero dynamic range.")
    return (normalized - phaseMin) / (phaseMax - phaseMin)


def _furthest_below_chord(smoothed, lo, hi):
    """The bin that lies furthest BELOW the chord joining two bins."""
    # The cross product of the chord with the offset: lo < hi, so the run is
    # positive and the sign is the same whichever way the chord slopes.
    if hi <= lo:
        return lo

    binIndices = np.arange(lo, hi + 1, dtype=float)
    binHeights = smoothed[lo:hi + 1]
    chordX, chordY = float(lo), float(smoothed[lo])
    chordRun, chordRise = float(hi) - chordX, float(smoothed[hi]) - chordY
    chordLength = np.hypot(chordRun, chordRise)
    if chordLength == 0:
        return (lo + hi) // 2

    distanceToChord = (((binIndices - chordX) * chordRise
                        - (binHeights - chordY) * chordRun) / chordLength)
    return int(lo + np.argmax(distanceToChord))


#: How strong the second hump must be, as a fraction of the first, before a
#: histogram counts as bimodal.
#
# A sparse sample is a few percent of the field and cannot raise a hump that
# rivals the background's, so a second "hump" far weaker than the first is a
# ripple -- an uncompensated map is nearly uniform and its fringes leave
# ripples everywhere. Measured over 66 phase maps: the old bar (5 % of the
# TALLEST BIN, which is low whenever the histogram has little contrast) called
# 42 of them bimodal, and only 5 had a second hump above half the first. On
# those 5 the valley and the triangle construction agree within 5 points of
# TSM; on the other 37 they differed by up to 72 points. So the bar belongs on
# how much the second population competes with the first, not on how tall it
# stands next to the busiest bin.
SECOND_HUMP_RATIO = 0.5


def histogram_valley(phi_nor, bins=256, smooth=5):
    """Suggest the TSM threshold from the shape of the phase histogram.

    Always the bin furthest below a chord; what the chord joins is what 'mode'
    reports:
      'valley'   two humps of comparable weight: the chord joins them, and the
                 answer is the dip between the two. The second has to carry at
                 least SECOND_HUMP_RATIO of the first's prominence, or it is a
                 ripple and the triangle below is used instead;
      'triangle' one hump: the chord runs from it to the far end of the
                 occupied range, which is the classic triangle construction.
                 A single hump with a tail that reaches the edge is the usual
                 QPI case -- background hump, sample tail -- and there the mean
                 lands inside the hump itself;
      'flat'     no hump at all: the mean is all that is left to offer.
    """
    from scipy.signal import find_peaks

    finiteValues = np.asarray(phi_nor, dtype=np.float64).ravel()
    finiteValues = finiteValues[np.isfinite(finiteValues)]

    counts, edges = np.histogram(finiteValues, bins=bins, range=(0.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])

    smoothed = counts.astype(float)
    if smooth > 1:
        # Padded with the edge value, NOT with zeros. np.convolve's 'same' pads
        # with zeros, which hollows out the first and last bins and so creates a
        # local maximum a couple of bins in; a histogram with one hump and a
        # tail that reaches the edge then passed as bimodal, on a peak that the
        # smoothing itself had invented.
        leftPad = smooth // 2
        rightPad = smooth - 1 - leftPad
        padded = np.pad(smoothed, (leftPad, rightPad), mode='edge')
        smoothed = np.convolve(padded, np.ones(smooth) / smooth, mode='valid')

    result = {'centers': centers, 'counts': counts, 'smoothed': smoothed}

    def flat(level):
        result.update({'threshold': float(level), 'chord_lo': 0, 'chord_hi': 0,
                       'valley': bins // 2, 'mode': 'flat', 'ok': False,
                       'peak': None})
        return result

    if smoothed.max() <= 0:
        return flat(0.5)

    # Picked by PROMINENCE: by height, a wide hump found its own shoulder.
    peaks, peakProperties = find_peaks(smoothed,
                                       prominence=0.05 * smoothed.max(),
                                       distance=max(3, bins // 20))

    if len(peaks) == 0:
        return flat(finiteValues.mean())

    byProminence = np.argsort(peakProperties['prominences'])
    strongestPeak = int(peaks[byProminence[-1]])
    bimodal = False
    if len(peaks) >= 2:
        firstProminence = float(peakProperties['prominences'][byProminence[-1]])
        secondProminence = float(peakProperties['prominences'][byProminence[-2]])
        bimodal = secondProminence >= SECOND_HUMP_RATIO * firstProminence

    if bimodal:
        # The two most prominent ones, in histogram order.
        topTwoPeaks = peaks[byProminence[-2:]]
        chordLo, chordHi = int(min(topTwoPeaks)), int(max(topTwoPeaks))
        mode = 'valley'
    else:
        occupied = np.flatnonzero(smoothed > 0)
        firstBin, lastBin = int(occupied[0]), int(occupied[-1])
        # The longer side of the hump: the tail is the side worth spanning.
        chordLo, chordHi = ((strongestPeak, lastBin)
                            if lastBin - strongestPeak >= strongestPeak - firstBin
                            else (firstBin, strongestPeak))
        mode = 'triangle'

    valleyIndex = _furthest_below_chord(smoothed, chordLo, chordHi)
    result.update({'threshold': float(centers[valleyIndex]),
                   'chord_lo': chordLo, 'chord_hi': chordHi,
                   'valley': valleyIndex, 'mode': mode,
                   # The busiest hump IS the background: it is the one phase
                   # level the field keeps coming back to. Which side of the
                   # threshold it falls on therefore says which side is NOT
                   # the sample, and that is a sturdier answer than the skew
                   # of the whole field, which a residual surface can flip.
                   'peak': float(centers[strongestPeak]),
                   'ok': mode == 'valley'})
    return result


def sample_is_bright_side(phi_nor, threshold, bins=256, smooth=5):
    """Which side of `threshold` holds the sample, read from the histogram.

    The background is the most populated phase level, because it is flat and
    piles up, while the sample spreads over many levels. So the hump is the
    background and the sample is whatever lies on the other side of the cut.
    That is also the premise of the triangle construction -- the chord runs
    from the hump to the tail and the cut goes between them -- so the level
    and the side come from one geometry instead of two statistics that can
    contradict each other.

    Measured over 132 readings of 66 phase maps: this agrees with the skew of
    the field on 113 of them, and where they differ it is the one that stays
    sane, dropping readings of 99 % (the background reported as the sample)
    to under 2 %. Unlike "the sample is the smaller side" it still allows an
    object that really does cover most of the field."""
    hump = histogram_valley(phi_nor, bins=bins, smooth=smooth).get('peak')
    if hump is None:
        return bright_sample(phi_nor)
    return bool(hump <= threshold)


def tsm_global(phase: np.ndarray, use_unwrap: bool = False, use_adaptive: bool = True,
               threshold: float = None, sample_is_bright: bool = None):
    """Percentage of SAMPLE pixels after binarizing the normalized phase.

    Which side of the threshold the sample lies on is read from the skew of the
    histogram, as the background mask of Module I does. Counting the dark side
    unconditionally reported the background instead: a bright sample pins the
    maximum of the normalization, the background settles near zero, and every
    threshold above its own noise left the whole background in the dark class,
    so the metric returned 100 % minus the sample area for the usual polarity
    and the sample area for the other one.

    The skew is read from the whole field, so a map whose residual surface
    dominates the histogram can flip it: an uncompensated map unwraps into a
    ramp that buries the sample's tail, the skew changes sign, and the reading
    comes back as the BACKGROUND's share. `sample_is_bright` lets the caller
    state the side instead, which is what the dialog does -- there the operator
    sees what share each side takes before the number reaches the table."""
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
        binarized = phase > float(threshold)
    elif use_adaptive:
        # Both split a symmetric distribution in half, hence the old ~50 %.
        localThreshold = threshold_local(phase, 35)
        binarized = phase > localThreshold
    else:
        binarized = phase > np.mean(phase)

    rowCount, colCount = phase.shape
    if sample_is_bright is None:
        # With a level to compare against, the histogram answers it; the
        # adaptive path has no single level, so there the skew is all there is.
        sample_is_bright = (sample_is_bright_side(phase, float(threshold))
                            if threshold is not None else bright_sample(phase))
    samplePixels = np.sum(binarized if sample_is_bright else ~binarized)
    return (samplePixels / (rowCount * colCount)) * 100.0


def sharpness_global(phase: np.ndarray, use_unwrap: bool = False, beta: float = 2.0):
    """Generalized Sharpness Metric: mean of the normalized phase raised to beta."""
    # Fienup & Miller, J. Opt. Soc. Am. A 20, 609-620 (2003), as a mean.
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("Phase image must be 2D.")

    if use_unwrap:
        phase = unwrap_phase(phase)

    # Normalized phase, not an optical intensity: blind to the amplitude.
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
