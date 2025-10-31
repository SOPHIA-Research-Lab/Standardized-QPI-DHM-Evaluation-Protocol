from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt
from skimage import measure, morphology
from skimage.segmentation import watershed
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import wx


class SamplePolarityDialog(wx.Dialog):
    """Custom dialog to ask sample polarity."""
    
    def __init__(self, parent=None):
        wx.Dialog.__init__(self, parent, title="Sample Polarity", 
                          style=wx.DEFAULT_DIALOG_STYLE | wx.STAY_ON_TOP)
        self.polarity = None
        panel = wx.Panel(self)
        
        question = wx.StaticText(panel, label="In the binary preview, does the SAMPLE appear as:")
        description = wx.StaticText(panel, label="Select one option below:")

        white_btn = wx.Button(panel, label="White regions")
        black_btn = wx.Button(panel, label="Black regions")

        white_btn.Bind(wx.EVT_BUTTON, lambda e: self.select_polarity('w'))
        black_btn.Bind(wx.EVT_BUTTON, lambda e: self.select_polarity('b'))

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(question, 0, wx.ALL | wx.CENTER, 10)
        sizer.Add(description, 0, wx.ALL | wx.CENTER, 5)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddStretchSpacer(1)
        btn_sizer.Add(white_btn, 0, wx.ALL, 10)
        btn_sizer.Add(black_btn, 0, wx.ALL, 10)
        btn_sizer.AddStretchSpacer(1)

        sizer.Add(btn_sizer, 0, wx.EXPAND | wx.CENTER, 20)
        panel.SetSizer(sizer)

        self.Fit()
        self.SetSizeHints(400, 150)
        self.SetSize((420, 150))
        self.Center()
    
    def select_polarity(self, polarity):
        """Set polarity and close dialog."""
        self.polarity = polarity
        self.EndModal(wx.ID_OK)


def ask_sample_polarity(parent=None):
    """Show polarity dialog and return user selection."""
    dialog = SamplePolarityDialog(parent)
    if dialog.ShowModal() == wx.ID_OK:
        result = dialog.polarity
        dialog.Destroy()
        return result
    else:
        dialog.Destroy()
        return 'w'


def imageRead(namefile):
    """Read image from file."""
    imagen = Image.open(namefile)
    loadImage = ImageOps.grayscale(imagen)
    return loadImage


def imageShow(image, title):
    """Display image using matplotlib."""
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()


def show_side_by_side(img1, img2, title1="Image 1", title2="Image 2", cmap="gray"):
    """Display two images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img1, cmap=cmap)
    axes[0].set_title(title1)
    axes[0].axis("off")

    axes[1].imshow(img2, cmap=cmap)
    axes[1].set_title(title2)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


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
        print(f"Applying watershed separation: found {np.max(markers)} potential sample centers")
        labels = watershed(-distance, markers, mask=binary_mask)
        separated_mask = labels > 0

        original_components = len(measure.regionprops(measure.label(binary_mask)))
        new_components = len(measure.regionprops(measure.label(separated_mask)))
        print(f"Watershed result: {original_components} → {new_components} components")

        return separated_mask
    else:
        print("No watershed separation needed (single component or no clear centers)")
        return binary_mask


def create_background_mask(image: np.ndarray, method: str = 'otsu',
                          manual_threshold: float = 0.5, min_area: int = 100,
                          hole_area: int = 50, parent=None):
    """Build background mask (True=background) from phase image."""
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

    answer = ask_sample_polarity(parent=parent)
    if answer is None:
        print("Operation cancelled by user")
        return None, None, None
        
    sample_is_white = answer.strip().lower() == 'w'

    preview_mask = (binary > 0)
    sample_mask = preview_mask.copy() if sample_is_white else ~preview_mask

    try:
        if np.any(sample_mask):
            sample_mask = separate_touching_samples(sample_mask)
    except NameError:
        pass

    if sample_is_white:
        cleaned = morphology.remove_small_objects(sample_mask, min_size=min_area)
    else:
        cleaned = ~morphology.remove_small_objects(~sample_mask, min_size=min_area)

    final_sample_mask = morphology.remove_small_holes(cleaned, area_threshold=hole_area)
    background_mask = ~final_sample_mask
    background_values = image[background_mask]

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

    total_pixels = image.size
    background_pixels = int(np.sum(background_mask))
    sample_pixels = int(np.sum(final_sample_mask))

    print("\nMask Statistics:")
    print(f"Total pixels: {total_pixels}")
    print(f"Background pixels: {background_pixels} ({100 * background_pixels / total_pixels:.1f}%)")
    print(f"Sample pixels: {sample_pixels} ({100 * sample_pixels / total_pixels:.1f}%)")
    print(f"Background values extracted: {len(background_values)}")

    return background_mask, background_values, threshold_value