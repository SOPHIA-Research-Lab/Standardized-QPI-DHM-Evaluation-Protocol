
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
    """
    Read image
    """
    imagen = Image.open(namefile)
    loadImage = ImageOps.grayscale(imagen)
    return loadImage


# Function to display an Image
def imageShow(image, title):
    """
    Showing image using plt
    """
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()


# Function to convert a grayscale image to phase [-π, π]
def grayscaleToPhase(image: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale image (0–255 or 0–1) into a phase map in radians [-π, π].
    """
    image = np.array(image)
    img = image.astype(float)

    # Normalize to [0, 1] if values are in [0, 255]
    if img.max() > 1.0:
        img = img / 255.0

    # Scale to [-π, π]
    phase = img * (2 * np.pi) - np.pi
    return phase


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


def create_background_mask(image: np.ndarray, method: str = 'otsu', manual_threshold: float = 0.5):
    """
    Create a background mask that identifies regions WITHOUT sample (background only)

    Parameters:
    -----------
    image : np.ndarray
        Input phase image (original values we want to preserve)
    method : str
        Thresholding method: 'otsu', 'manual'
    manual_threshold : float
        Manual threshold value in range [0.0, 1.0]

    Returns:
    --------
    background_mask : np.ndarray (boolean)
        True where there is background (no sample), False where there is sample
    background_values : np.ndarray (1D)
        Original image values only from background regions
    threshold_value : float
        Threshold value used for binarization
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from tkinter import simpledialog
    from scipy.ndimage import distance_transform_edt
    from skimage import measure, morphology
    from skimage.segmentation import watershed

    # Convert image to uint8 for OpenCV processing
    image_normalized = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)

    # Step 1: Create initial binary mask
    if method == 'otsu':
        threshold_value, binary = cv2.threshold(image_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary)
    elif method == 'manual':
        manual_thresh_uint8 = int(manual_threshold * 255)
        _, binary = cv2.threshold(image_normalized, manual_thresh_uint8, 255, cv2.THRESH_BINARY)
        threshold_value = manual_threshold
        binary = cv2.bitwise_not(binary)
    else:
        raise ValueError(f"Unknown thresholding method: {method}")

    # Convert to boolean mask
    binary_mask = binary > 0

    # Step 2: Separate touching samples if needed
    if np.sum(binary_mask) > 0:
        binary_mask = separate_touching_samples(binary_mask)

    # Step 3: Show binary mask and ask user about sample polarity
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='jet')
    plt.title('Original Phase Image')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(binary_mask, cmap='gray')
    plt.title('Binary Mask - White regions detected')
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=True)

    # Ask user if sample appears white or black
    answer = simpledialog.askstring(
        "Sample Polarity",
        "In the binary mask, does the SAMPLE appear as:\n\nWhite regions (w) or Black regions (b)?\n\nType: w or b"
    )

    sample_is_white = True
    if answer:
        answer = answer.strip().lower()
        sample_is_white = (answer == 'w')

    print(f"User selected: Sample appears as {'WHITE' if sample_is_white else 'BLACK'} in binary mask")

    # Step 4: Clean the mask
    min_area = 100  # Minimum area for objects

    if sample_is_white:
        print("Cleaning WHITE sample regions...")
        cleaned_sample_mask = morphology.remove_small_objects(binary_mask, min_size=min_area)
    else:
        print("Cleaning BLACK sample regions...")
        cleaned_sample_mask = ~morphology.remove_small_objects(~binary_mask, min_size=min_area)

    # Remove small holes
    final_sample_mask = morphology.remove_small_holes(cleaned_sample_mask, area_threshold=50)

    # Step 5: Create BACKGROUND mask (inverse of sample mask)
    background_mask = ~final_sample_mask

    # Step 6: Extract background values from ORIGINAL image
    background_values = image[background_mask]

    # Step 7: Show final result
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='jet')
    plt.title('Original Phase Image')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(final_sample_mask, cmap='gray')
    plt.title('Sample Mask (White = Sample)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(background_mask, cmap='gray')
    plt.title('Background Mask (White = Background)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Print statistics
    total_pixels = image.size
    background_pixels = np.sum(background_mask)
    sample_pixels = np.sum(final_sample_mask)

    print(f"\nMask Statistics:")
    print(f"Total pixels: {total_pixels}")
    print(f"Background pixels: {background_pixels} ({100 * background_pixels / total_pixels:.1f}%)")
    print(f"Sample pixels: {sample_pixels} ({100 * sample_pixels / total_pixels:.1f}%)")
    print(f"Background values extracted: {len(background_values)}")

    return background_mask, background_values, threshold_value
