from scipy.ndimage import distance_transform_edt
from skimage import measure, morphology
from skimage.segmentation import watershed
import numpy as np


# ----------------------------
# IO / helpers
# ----------------------------


def grayscaleToPhase(image: np.ndarray) -> np.ndarray:
    """Convert grayscale image (0-255 or 0-1) to phase map [-π, π]."""
    image = np.array(image)
    img = image.astype(float)
    if img.max() > 1.0:
        img = img / 255.0
    phase = img * (2 * np.pi) - np.pi
    return phase


def separate_touching_samples(binary_mask):
    """Separate touching samples using watershed segmentation."""
    distance = distance_transform_edt(binary_mask)
    from scipy import ndimage
    local_maxima = ndimage.maximum_filter(distance, size=20) == distance
    local_maxima = local_maxima & (distance > 5)
    markers = measure.label(local_maxima)
    
    if np.max(markers) > 1:
        labels = watershed(-distance, markers, mask=binary_mask)
        separated_mask = labels > 0
        
        # Print component statistics like matplotlib version
        original_components = len(measure.regionprops(measure.label(binary_mask)))
        new_components = len(measure.regionprops(measure.label(separated_mask)))
        
        return separated_mask
    else:
        return binary_mask


# ----------------------------
# Utilities to convert numpy -> wx.Bitmap
# ----------------------------
def build_background_mask(image: np.ndarray, threshold_value: float,
                          sample_is_white: bool, min_area: int = 100,
                          hole_area: int = 50):
    """
    Turn a chosen threshold into a background mask.

    Pure computation: no dialogs, no wx. Asking the user for the threshold is
    the UI layer's job (see ui/threshold_dialog.py); what the threshold means
    for the mask is decided here.

    Parameters
    ----------
    image : ndarray
        Phase map.
    threshold_value : float
        Threshold on the image normalised to [0, 1].
    sample_is_white : bool
        True when the sample is the bright side of the threshold.
    min_area, hole_area : int
        Morphological cleanup sizes.

    Returns
    -------
    tuple
        (background_mask, threshold_value)
    """
    image = np.array(image, dtype=float)

    
    # Normalize image to 0-1 range
    img_min, img_max = np.min(image), np.max(image)
    if img_max - img_min < 1e-12:
        img_normalized = np.zeros_like(image, dtype=float)
    else:
        img_normalized = (image - img_min) / (img_max - img_min)

    # Apply threshold
    binary = (img_normalized > threshold_value).astype(np.uint8) * 255
    preview_mask = (binary > 0)
    sample_mask = preview_mask.copy() if sample_is_white else ~preview_mask

    # Separate touching samples
    try:
        if np.any(sample_mask):
            sample_mask = separate_touching_samples(sample_mask)
    except Exception:
        pass

    # Clean up mask
    if sample_is_white:
        cleaned = morphology.remove_small_objects(sample_mask, min_size=min_area)
    else:
        cleaned = ~morphology.remove_small_objects(~sample_mask, min_size=min_area)

    final_sample_mask = morphology.remove_small_holes(cleaned, area_threshold=hole_area)
    background_mask = ~final_sample_mask
    
    # Extract background values (like matplotlib version)
    #background_values = image[background_mask]

    total_pixels = image.size
    background_pixels = int(np.sum(background_mask))
    sample_pixels = int(np.sum(final_sample_mask))
    
    # print("\n✓ Mask created successfully!")
    # print("\nMask Statistics:")
    # print(f"Total pixels: {total_pixels}")
    # print(f"Background pixels: {background_pixels} ({100 * background_pixels / total_pixels:.1f}%)")
    # print(f"Sample pixels: {sample_pixels} ({100 * sample_pixels / total_pixels:.1f}%)")
    # #print(f"Background values extracted: {len(background_values)}")
    # print(f"Threshold value: {threshold_value:.4f}")

    return background_mask, float(threshold_value)
