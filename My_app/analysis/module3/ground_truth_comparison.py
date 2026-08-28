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


def _prepare_sample_and_gt(sample, ground_truth, use_unwrap=False):
    
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

    sample_phase = np.asarray(sample_phase, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    if sample_phase.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch between sample {sample_phase.shape} and "
            f"ground-truth {ground_truth.shape}. Both images must have "
            f"the same dimensions to be compared."
        )

    return sample_phase, ground_truth


def calculate_ssim(sample, ground_truth, use_unwrap=False):
    """ Calculate Structural Similarity Index (SSIM). """
    sample_phase, ground_truth = _prepare_sample_and_gt(sample, ground_truth, use_unwrap)

    gt_min = ground_truth.min()
    gt_max = ground_truth.max()
    gt_range = gt_max - gt_min

    sample_norm = (sample_phase - gt_min) / gt_range
    gt_norm = (ground_truth - gt_min) / gt_range

    # Calculate SSIM
    ssim_value = ssim(sample_norm, gt_norm, data_range=1.0)

    return ssim_value


def calculate_mse(sample, ground_truth, use_unwrap=False):
    """ Calculate Mean Squared Error (MSE)."""
    sample_phase, ground_truth = _prepare_sample_and_gt(sample, ground_truth, use_unwrap)

    mse_value = mean_squared_error(ground_truth, sample_phase)

    return mse_value


def calculate_psnr(sample, ground_truth, use_unwrap=False):
    """ Calculate Peak Signal-to-Noise Ratio (PSNR)."""
    sample_phase, ground_truth = _prepare_sample_and_gt(sample, ground_truth, use_unwrap)

    data_range = ground_truth.max() - ground_truth.min()

    # Calculate PSNR
    psnr_value = peak_signal_noise_ratio(ground_truth, sample_phase, data_range=data_range)

    return psnr_value