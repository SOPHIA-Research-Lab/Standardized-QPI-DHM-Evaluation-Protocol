import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from PIL import Image
from analysis.module1 import utilitiesRBPV as utRBPV
from analysis.module1 import residual_background as rb

def load_ground_truth(path):
    """ Load ground-truth image from file."""
    try:
        # if is a file type .mat
        if path.endswith('.mat'):
            import scipy.io as sio
            mat_data = sio.loadmat(path)
            # adjust according to the structure of your .mat
            if 'phase' in mat_data:
                return mat_data['phase']
            else:
                # Take the first field it finds
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        return mat_data[key]
        else:
            # If it's a normal image
            img = Image.open(path)
            grayscale = img.convert("L")
            img_array = np.array(grayscale, dtype=float)
            return utRBPV.grayscaleToPhase(img_array)
    except Exception as e:
        raise ValueError(f"Error loading ground-truth: {str(e)}")


def calculate_ssim(sample, ground_truth, use_unwrap=False):
    """ Calculate Structural Similarity Index (SSIM). """
    # Convert sample to numpy if it's PIL
    if isinstance(sample, Image.Image):
        grayscale = sample.convert("L")
        sample_array = np.array(grayscale, dtype=float)
        sample_phase = utRBPV.grayscaleToPhase(sample_array)
    else:
        sample_phase = sample

    # Unwrap if requested
    if use_unwrap:
        sample_phase = rb.unwrap_with_scikit(sample_phase)
        ground_truth = rb.unwrap_with_scikit(ground_truth)

    # Normalize both images to the same range
    sample_norm = (sample_phase - sample_phase.min()) / (sample_phase.max() - sample_phase.min())
    gt_norm = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min())

    # Calculate SSIM
    ssim_value = ssim(sample_norm, gt_norm, data_range=1.0)
    
    return ssim_value


def calculate_mse(sample, ground_truth, use_unwrap=False):
    """ Calculate Mean Squared Error (MSE)."""
    # Convert sample to numpy if it's PIL
    if isinstance(sample, Image.Image):
        grayscale = sample.convert("L")
        sample_array = np.array(grayscale, dtype=float)
        sample_phase = utRBPV.grayscaleToPhase(sample_array)
    else:
        sample_phase = sample

    # Unwrap if requested
    if use_unwrap:
        sample_phase = rb.unwrap_with_scikit(sample_phase)
        ground_truth = rb.unwrap_with_scikit(ground_truth)

    # Calculate MSE
    mse_value = mean_squared_error(ground_truth, sample_phase)
    
    return mse_value


def calculate_psnr(sample, ground_truth, use_unwrap=False):
    """ Calculate Peak Signal-to-Noise Ratio (PSNR)."""
    # Convert sample to numpy if it's PIL
    if isinstance(sample, Image.Image):
        grayscale = sample.convert("L")
        sample_array = np.array(grayscale, dtype=float)
        sample_phase = utRBPV.grayscaleToPhase(sample_array)
    else:
        sample_phase = sample

    # Unwrap if requested
    if use_unwrap:
        sample_phase = rb.unwrap_with_scikit(sample_phase)
        ground_truth = rb.unwrap_with_scikit(ground_truth)

    # Calculate data range
    data_range = max(ground_truth.max() - ground_truth.min(),
                     sample_phase.max() - sample_phase.min())

    # Calculate PSNR
    psnr_value = peak_signal_noise_ratio(ground_truth, sample_phase, data_range=data_range)
    
    return psnr_value