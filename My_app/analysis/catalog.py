"""Metric id -> (function, table column label, needs unwrapped phase)."""
from analysis.module1 import residual_background as rb
from analysis.module2 import global_phase as gp
from analysis.module3 import ground_truth_comparison as gtc


METRIC_MAP = {

    # ===== MODULE 1: Residual Background Phase Variance =====
    'std_unwrapped': (rb.std_background, 'STD_U_B [rad]', True),
    'std_background': (rb.std_background, 'STD_B [rad]', False),
    'std_zones': (rb.std_background, 'STD_Zones [rad]', False),

    # MAD metrics
    'mad_unwrapped': (rb.mean_absolute_deviation_background, 'MAD_U_B [rad]', True),
    'mad_background': (rb.mean_absolute_deviation_background, 'MAD_B [rad]', False),
    'mad_zones': (rb.mean_absolute_deviation_background, 'MAD_Zones [rad]', False),

    # Mean (piston): RMS^2 = mean^2 + STD^2, so this is what RMS adds over STD.
    'mean_unwrapped': (rb.mean_background, 'Mean_U_B [rad]', True),
    'mean_background': (rb.mean_background, 'Mean_B [rad]', False),
    'mean_zones': (rb.mean_background, 'Mean_Zones [rad]', False),

    # RMS metrics
    'rms_unwrapped': (rb.rms_background, 'RMS_U_B [rad]', True),
    'rms_background': (rb.rms_background, 'RMS_B [rad]', False),
    'rms_zones': (rb.rms_background, 'RMS_Zones [rad]', False),

    # PV metrics
    'pv_unwrapped': (rb.pv_background, 'PV_U_B [rad]', True),
    'pv_background': (rb.pv_background, 'PV_B [rad]', False),
    'pv_zones': (rb.pv_background, 'PV_Zones [rad]', False),

    # FWHM metrics
    'fwhm_unwrapped': (rb.fwhm_background, 'FWHM_U_B [rad]', True),
    'fwhm_background': (rb.fwhm_background, 'FWHM_B [rad]', False),
    'fwhm_zones': (rb.fwhm_background, 'FWHM_Zones [rad]', False),

    # Entropy metrics
    'entropy_unwrapped': (rb.entropy_background, 'Entropy_U_B [bits]', True),
    'entropy_background': (rb.entropy_background, 'Entropy_B [bits]', False),
    'entropy_zones': (rb.entropy_background, 'Entropy_Zones [bits]', False),

    # Phase jumps: how much of the region still crosses a 2*pi cliff. Zero on
    # a compensated background; read beside Legendre, which smooths across them.
    'jumps_unwrapped': (rb.wrap_jump_density, 'Jumps_U_B [%]', True),
    'jumps_background': (rb.wrap_jump_density, 'Jumps_B [%]', False),
    'jumps_zones': (rb.wrap_jump_density, 'Jumps_Zones [%]', False),

    # Legendre metrics 
    'legendre_background': (rb.legendre_background, 'Legendre-B [rad]', False),
    'legendre_unwrapped': (rb.legendre_background, 'Legendre-U_B [rad]', True),
    'legendre_zones': (rb.legendre_background, 'Legendre_Zones [rad]', False),


    # ===== MODULE 2: Global Phase Distortion Metrics =====

    #  PV metrics
    'mmm_global': (gp.maximum_minus_minimum, 'PV_Global [rad]', False),
    'mmm_unwrapped': (gp.maximum_minus_minimum, 'PV_Global_U [rad]', True),

    # STD: the same dispersion as PV, read from every pixel instead of two.
    'std_global': (gp.std_global, 'STD_Global [rad]', False),
    'std_global_unwrapped': (gp.std_global, 'STD_Global_U [rad]', True),

    # Phase Gradient Metric: the local gradient, everywhere.
    'gradient_global': (gp.phase_gradient_prewitt, 'PGM_Global [rad/px]', False),
    'gradient_unwrapped': (gp.phase_gradient_prewitt, 'PGM_Unwrapped [rad/px]', True),

    # Global Phase Gradient: the tilt alone, from a least-squares plane.
    'tilt_global': (gp.phase_tilt_plane, 'GPG_Global [rad/px]', False),
    'tilt_unwrapped': (gp.phase_tilt_plane, 'GPG_Unwrapped [rad/px]', True),

    # TSM
    'tsm_global': (gp.tsm_global, 'TSM_Global [%]', False),
    'tsm_unwrapped': (gp.tsm_global, 'TSM_Unwrapped [%]', True),

    # Laplacian Energy
    'laplacian_global': (gp.laplacian_energy, 'LE_Global [rad^2/px^4]', False),
    'laplacian_unwrapped': (gp.laplacian_energy, 'LE_Unwrapped [rad^2/px^4]', True),

    # Spatial Frequency
    # Dimensionless: the phase is normalised to [0,1] before the differences.
    'spatial_freq_global': (gp.spatial_frequency_global, 'SF_Global', False),
    'spatial_freq_unwrapped': (gp.spatial_frequency_global, 'SF_Unwrapped', True),

    # Global Entropy
    'global_entropy': (gp.global_entropy_global, 'Entropy_Global [bits]', False),
    'global_entropy_unwrapped': (gp.global_entropy_global, 'Entropy_Unwrapped [bits]', True),

    # GSM
    'sharpness_global': (gp.sharpness_global, 'GSM_Global', False),
    'sharpness_unwrapped': (gp.sharpness_global, 'GSM_Unwrapped', True),

    # FWHM/Resolution
    'fwhm_global': (gp.fwhm_global, 'FWHM_Global [rad]', False),
    'fwhm_global_unwrapped': (gp.fwhm_global, 'FWHM_Global_U [rad]', True),


    # ===== MODULE 3: Ground-Truth Comparisons =====

    # SSIM
    'ssim': (gtc.calculate_ssim, 'SSIM', False),

    # MSE
    'mse': (gtc.calculate_mse, 'MSE', False),

    # PSNR
    'psnr': (gtc.calculate_psnr, 'PSNR', False),
}


#: Which protocol module each metric function lives in.
# The module is read from where the function is defined, not guessed from the
# id: the GUI used to match id prefixes, 'tilt_global' matched none of them, and
# the two GPG metrics were therefore handled as background metrics -- they asked
# for a segmentation they do not use, and were then called as f(phase, mask).
_MODULE_BY_PACKAGE = {
    'analysis.module1.residual_background': 1,
    'analysis.module2.global_phase': 2,
    'analysis.module3.ground_truth_comparison': 3,
}


def module_of(metric_id):
    """The module number a metric belongs to, or None if it is not catalogued."""
    entry = METRIC_MAP.get(metric_id)
    if entry is None:
        return None
    return _MODULE_BY_PACKAGE.get(getattr(entry[0], '__module__', None))
