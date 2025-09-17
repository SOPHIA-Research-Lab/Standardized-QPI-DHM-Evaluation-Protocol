

# import libraries
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt
from skimage import measure
from skimage.segmentation import watershed
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import morphology
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


# Function to show two images side by side
def show_side_by_side(img1, img2, title1="Image 1", title2="Image 2", cmap="gray"):
    """
    Showing images using plt
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img1, cmap=cmap)
    axes[0].set_title(title1)
    axes[0].axis("off")

    axes[1].imshow(img2, cmap=cmap)
    axes[1].set_title(title2)
    axes[1].axis("off")

    plt.tight_layout()
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


def create_background_mask(
    image: np.ndarray,
    method: str = 'otsu',
    manual_threshold: float = 0.5,
    min_area: int = 100,
    hole_area: int = 50
):
    """
    Build a background mask (True where background) from a phase image.
    """
    img_u8 = ((image - np.min(image)) / (np.max(image) - np.min(image) + 1e-12) * 255).astype(np.uint8)

    if method == 'otsu':
        thr_u8, binary = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_value = float(thr_u8) / 255.0
    elif method == 'manual':
        thr_u8 = int(np.clip(manual_threshold, 0, 1) * 255)
        _, binary = cv2.threshold(img_u8, thr_u8, 255, cv2.THRESH_BINARY)
        threshold_value = float(thr_u8) / 255.0
    else:
        raise ValueError(f"Unknown thresholding method: {method}")

    # Preview: original + raw binary (white regions = pixels above threshold)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=False)

    im = ax1.imshow(image, cmap='jet')
    ax1.set_title('Original Phase Image')
    ax1.axis('off')
    ax1.set_aspect('equal')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    ax2.imshow(binary > 0, cmap='gray')
    ax2.set_title('Binary preview (White = pixels > threshold)')
    ax2.axis('off')
    ax2.set_aspect('equal')

    plt.subplots_adjust(wspace=0.15)
    plt.show()

    # Ask polarity relative to the preview just shown
    answer = simpledialog.askstring(
        "Sample Polarity",
        "In the binary preview, does the SAMPLE appear as:\n\nWhite regions (w) or Black regions (b)?\n\nType: w or b"
    )
    sample_is_white = (answer or 'w').strip().lower() == 'w'

    # Build sample mask explicitly from the chosen polarity
    preview_mask = (binary > 0)
    if sample_is_white:
        sample_mask = preview_mask.copy()
    else:
        sample_mask = ~preview_mask

    # Optional separation if available
    try:
        if np.any(sample_mask):
            sample_mask = separate_touching_samples(sample_mask)
    except NameError:
        pass

    # Clean mask
    if sample_is_white:
        cleaned = morphology.remove_small_objects(sample_mask, min_size=min_area)
    else:
        cleaned = ~morphology.remove_small_objects(~sample_mask, min_size=min_area)

    final_sample_mask = morphology.remove_small_holes(cleaned, area_threshold=hole_area)

    # Background is the complement
    background_mask = ~final_sample_mask
    background_values = image[background_mask]

    # Final visualization
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=False)

    imA = axA.imshow(image, cmap='jet')
    axA.set_title('Original Phase Image')
    axA.axis('off')
    axA.set_aspect('equal')
    divA = make_axes_locatable(axA)
    caxA = divA.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(imA, cax=caxA)

    axB.imshow(final_sample_mask, cmap='gray')
    axB.set_title('Sample Mask (White = Sample)')
    axB.axis('off')
    axB.set_aspect('equal')

    axC.imshow(background_mask, cmap='gray')
    axC.set_title('Background Mask (White = Background)')
    axC.axis('off')
    axC.set_aspect('equal')

    plt.subplots_adjust(wspace=0.2)
    plt.show()

    # Stats
    total_pixels = image.size
    background_pixels = int(np.sum(background_mask))
    sample_pixels = int(np.sum(final_sample_mask))

    print("\nMask Statistics:")
    print(f"Total pixels: {total_pixels}")
    print(f"Background pixels: {background_pixels} ({100 * background_pixels / total_pixels:.1f}%)")
    print(f"Sample pixels: {sample_pixels} ({100 * sample_pixels / total_pixels:.1f}%)")
    print(f"Background values extracted: {len(background_values)}")

    return background_mask, background_values, threshold_value

