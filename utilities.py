
# import libraries
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from skimage.segmentation import watershed
from skimage import measure, morphology
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from tkinter import ttk
from tkinter import simpledialog


# Function to read an image file from the disk
def imageRead(namefile):
    imagen = Image.open(namefile)
    loadImage = ImageOps.grayscale(imagen)
    return loadImage


# Function to display an Image
def imageShow(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()


def separate_touching_samples(binary_mask):
    """Separate touching samples using watershed segmentation"""
    # Calculate distance transform
    distance = distance_transform_edt(binary_mask)

    # Find local maxima
    from scipy import ndimage
    local_maxima = ndimage.maximum_filter(distance, size=20) == distance
    local_maxima = local_maxima & (distance > 5)  # Minimum distance threshold

    # Create markers for watershed
    markers = measure.label(local_maxima)

    if np.max(markers) > 1:
        print(f"Applying watershed separation: found {np.max(markers)} potential sample centers")

        # Apply watershed
        labels = watershed(-distance, markers, mask=binary_mask)

        # Convert back to binary (any labeled region becomes True)
        separated_mask = labels > 0

        # Compare results
        original_components = len(measure.regionprops(measure.label(binary_mask)))
        new_components = len(measure.regionprops(measure.label(separated_mask)))
        print(f"Watershed result: {original_components} → {new_components} components")

        return separated_mask
    else:
        print("No watershed separation needed (single component or no clear centers)")
        return binary_mask


def create_binary_mask(image: np.ndarray, method: str = 'otsu', manual_threshold: float = 0.5) -> np.ndarray:
    """
    Create a binary mask from an image using the selected thresholding method.

    Parameters:
        image (np.ndarray): Input grayscale image.
        method (str): Thresholding method: 'otsu', 'manual'.
        manual_threshold (float): Manual threshold value in range [0.0, 1.0].

    Returns:
        np.ndarray: Binary mask (boolean array).
    """

    if method == 'otsu':
        threshold_value, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary)

    elif method == 'manual':
        _, binary = cv2.threshold(image, manual_threshold, 255, cv2.THRESH_BINARY)
        threshold_value = manual_threshold
        binary = cv2.bitwise_not(binary)

    else:
        raise ValueError(f"Unknown thresholding method: {method}")

    # Convert to boolean mask
    binary_mask = binary > 0

    # Additional processing to separate touching samples using watershed
    if np.sum(binary_mask) > 0:
        binary_mask = separate_touching_samples(binary_mask)

    # Show result
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Binary Mask - Is the sample white or black?")
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=True)

    # Ask for sample polarity
    answer = simpledialog.askstring(
        "Confirm Sample Polarity",
        "Does the sample appear white (w) or black (b)?\nPlease type: w or b"
        #parent=parent
    )
    sample_is_white = True
    if answer:
        answer = answer.strip().lower()
        sample_is_white = (answer == 'w')
    print(f"User selected {'WHITE' if sample_is_white else 'BLACK'} sample.")

    # Clean samples outside regions
    if sample_is_white:
        print("Cleaning WHITE regions...")
        cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_area)
    else:
        print("Cleaning BLACK regions...")
        cleaned_mask = ~morphology.remove_small_objects(~binary_mask, min_size=min_area)

    final_mask = morphology.remove_small_holes(cleaned_mask, area_threshold=50)

    return binary_mask, threshold_value

'''
def process_particles(image: np.ndarray, method: str, threshold=None, min_area=100, max_area=10000, μm_per_px=1.0, parent=None):
    """
    Core particle processing function that handles thresholding, cleaning, and detection.
    Returns: final_mask, samples_circles, threshold_value, sample_is_white
    """
    # Create binary mask
    binary_mask, threshold_value = create_binary_mask(
        image=image,
        method=method,
        manual_threshold=threshold if threshold is not None else 0.5
    )

    # Ask for sample polarity
    answer = simpledialog.askstring(
        "Confirm Sample Polarity",
        "Does the sample appear white (w) or black (b)?\nPlease type: w or b",
        parent=parent
    )
    sample_is_white = True
    if answer:
        answer = answer.strip().lower()
        sample_is_white = (answer == 'w')
    print(f"User selected {'WHITE' if sample_is_white else 'BLACK'} sample.")

    # Clean samples outside regions
    if sample_is_white:
        print("Cleaning WHITE regions...")
        cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_area)
    else:
        print("Cleaning BLACK regions...")
        cleaned_mask = ~morphology.remove_small_objects(~binary_mask, min_size=min_area)

    final_mask = morphology.remove_small_holes(cleaned_mask, area_threshold=50)

    if sample_is_white:
        mask_for_analysis = final_mask
    else:
        mask_for_analysis = ~final_mask

    # Label connected components
    labeled_image = measure.label(mask_for_analysis)
    regions = measure.regionprops(labeled_image)

    print(f"\nConnected component analysis:")
    print(f"  - Total components found: {len(regions)}")

    if len(regions) == 0:
        print("ERROR: No connected components found!")
        return final_mask, [], threshold_value, sample_is_white

    # Show area distribution of all components
    all_areas = [region.area for region in regions]
    print(f"  - Component areas: min={min(all_areas)}, max={max(all_areas)}, mean={np.mean(all_areas):.1f}")
    print(f"  - Area filter range: {min_area} - {max_area}")

    samples_circles = []

    for i, region in enumerate(regions):
        area = region.area

        # Filter by area
        if min_area <= area <= max_area:
            # Calculate circle parameters
            y_center, x_center = region.centroid

            # Estimate diameter from area (assuming circular samples)
            diameter_from_area = 2 * np.sqrt(area / np.pi)

            # Alternative: use equivalent diameter
            equivalent_diameter = region.equivalent_diameter

            # Use the larger of the two estimates for safety
            diameter = max(diameter_from_area, equivalent_diameter)

            samples_circles.append({
                'center_x': x_center,
                'center_y': y_center,
                'diameter': diameter,
                'diameter_um': diameter * μm_per_px,
                'area': area,
                'area_um2': area * (μm_per_px ** 2),
                'label': region.label,
                'bbox': region.bbox
            })

    print(f"\nFinal result: {len(samples_circles)} samples candidates accepted")

    if len(samples_circles) == 0:
        print("\nTROUBLESHOOTING SUGGESTIONS:")
        print(f"1. Adjust area limits:")
        print(f"   - Current min_area: {min_area}")
        print(f"   - Current max_area: {max_area}")
        print(f"   - Suggested min_area: {max(10, min(all_areas))}")
        print(f"   - Suggested max_area: {max(all_areas)}")

    # Visualize detection step (if function exists)
    try:
        visualize_detection_step(labeled_image, regions, samples_circles)
    except NameError:
        print("Visualization function not available")

    return final_mask, samples_circles, threshold_value, sample_is_white
'''