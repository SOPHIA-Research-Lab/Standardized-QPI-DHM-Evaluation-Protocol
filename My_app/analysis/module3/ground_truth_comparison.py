import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from PIL import Image
from analysis.module1 import utilitiesRBPV as utRBPV


def _as_phase(variable):
    """A .mat variable as phase: real means radians already, complex needs the
    argument taken. What reaches a metric is always the phase, never a field."""
    array = np.asarray(variable)
    return np.angle(array) if np.iscomplexobj(array) else array


def load_ground_truth(path):
    """Load a ground-truth phase map from a .mat file or an image."""
    try:
        if path.endswith('.mat'):
            import scipy.io as sio
            matData = sio.loadmat(path)
            if 'phase' in matData:
                return _as_phase(matData['phase'])
            else:
                # No "phase" field: take the first one that is not metadata.
                for fieldName in matData.keys():
                    if not fieldName.startswith('__'):
                        return _as_phase(matData[fieldName])
        else:
            grayscale = Image.open(path).convert("L")
            grayLevels = np.array(grayscale, dtype=float)
            return utRBPV.gray_to_phase(grayLevels)
    except Exception as error:
        raise ValueError(f"Error loading ground-truth: {str(error)}")


def _prepare_sample_and_gt(sample, ground_truth):
    """Bring sample and ground truth to a comparable form, neither unwrapped."""
    # Unwrapping an 8-bit reference invents 2*pi jumps it never had.
    if isinstance(sample, Image.Image):
        grayscale = sample.convert("L")
        grayLevels = np.array(grayscale, dtype=float)
        samplePhase = utRBPV.gray_to_phase(grayLevels)
    else:
        samplePhase = sample

    samplePhase = np.asarray(samplePhase, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    if samplePhase.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch between sample {samplePhase.shape} and "
            f"ground-truth {ground_truth.shape}. Both images must have "
            f"the same dimensions to be compared."
        )

    return samplePhase, ground_truth


def calculate_ssim(sample, ground_truth):
    """Structural Similarity Index: 1 is identical, 0 is unrelated."""
    samplePhase, ground_truth = _prepare_sample_and_gt(sample, ground_truth)

    # Both normalized with the REFERENCE range, so amplitude differences stay.
    referenceMin = ground_truth.min()
    referenceRange = ground_truth.max() - referenceMin

    sampleNormalized = (samplePhase - referenceMin) / referenceRange
    referenceNormalized = (ground_truth - referenceMin) / referenceRange

    return ssim(sampleNormalized, referenceNormalized, data_range=1.0)


def calculate_mse(sample, ground_truth):
    """Mean Squared Error: 0 is identical, higher means larger differences."""
    samplePhase, ground_truth = _prepare_sample_and_gt(sample, ground_truth)
    return mean_squared_error(ground_truth, samplePhase)


def calculate_psnr(sample, ground_truth):
    """Peak Signal-to-Noise Ratio in dB: higher is better, infinite when MSE is 0."""
    samplePhase, ground_truth = _prepare_sample_and_gt(sample, ground_truth)
    referenceRange = ground_truth.max() - ground_truth.min()
    return peak_signal_noise_ratio(ground_truth, samplePhase,
                                   data_range=referenceRange)