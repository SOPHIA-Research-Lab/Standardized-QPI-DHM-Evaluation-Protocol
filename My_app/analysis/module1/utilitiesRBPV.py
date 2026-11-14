from scipy.ndimage import distance_transform_edt
from skimage import measure, morphology
from skimage.segmentation import watershed
import numpy as np


# The file convention for a raster phase map: an 8-bit gray level spans the
# whole principal interval, so phase zero sits at mid gray. The endpoints are
# named rather than inlined because they are what makes a published number
# reproducible from the stored image.
GRAY_MIN, GRAY_MAX = 0.0, 255.0
PHASE_MIN, PHASE_MAX = -np.pi, np.pi


def gray_to_phase(values: np.ndarray) -> np.ndarray:
    """Map 8-bit gray levels onto the phase interval [-pi, pi], linearly.

    The scale is fixed, never inferred from the data: a previous version
    divided by 255 only when the image's maximum exceeded 1, so a nearly black
    image was read on a different scale from every other one, and the same gray
    level could come out as -pi or +pi depending on what else the image held.
    Values outside the gray range saturate rather than extrapolate."""
    values = np.clip(np.asarray(values, dtype=float), GRAY_MIN, GRAY_MAX)
    return PHASE_MIN + (values - GRAY_MIN) * (PHASE_MAX - PHASE_MIN) / (GRAY_MAX - GRAY_MIN)


def separate_touching_samples(binary_mask):
    """Split samples that touch each other, by watershed on the distance map."""
    distanceMap = distance_transform_edt(binary_mask)
    from scipy import ndimage
    localMaxima = ndimage.maximum_filter(distanceMap, size=20) == distanceMap
    localMaxima = localMaxima & (distanceMap > 5)
    markers = measure.label(localMaxima)

    # One marker means one sample: nothing to separate.
    if np.max(markers) > 1:
        labels = watershed(-distanceMap, markers, mask=binary_mask)
        return labels > 0
    return binary_mask


def build_background_mask(image: np.ndarray, threshold_value: float,
                          sample_is_white: bool, min_area: int = 100,
                          hole_area: int = 50):
    """Turn a chosen threshold into (background_mask, threshold_value)."""
    # Pure computation: ui/threshold_dialog.py is what asks for the threshold.
    image = np.array(image, dtype=float)

    phaseMin, phaseMax = np.min(image), np.max(image)
    if phaseMax - phaseMin < 1e-12:
        normalizedPhase = np.zeros_like(image, dtype=float)
    else:
        normalizedPhase = (image - phaseMin) / (phaseMax - phaseMin)

    aboveThreshold = normalizedPhase > threshold_value
    sampleMask = aboveThreshold.copy() if sample_is_white else ~aboveThreshold

    try:
        if np.any(sampleMask):
            sampleMask = separate_touching_samples(sampleMask)
    except Exception:
        pass

    # remove_small_objects works on the foreground, so a dark sample inverts.
    if sample_is_white:
        cleanedMask = morphology.remove_small_objects(sampleMask, min_size=min_area)
    else:
        cleanedMask = ~morphology.remove_small_objects(~sampleMask, min_size=min_area)

    finalSampleMask = morphology.remove_small_holes(cleanedMask,
                                                    area_threshold=hole_area)
    return ~finalSampleMask, float(threshold_value)
