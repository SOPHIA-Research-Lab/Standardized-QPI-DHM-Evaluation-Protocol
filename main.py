

import utilities as ut
import residual_background as rb

# Read image
#name = r"C:\Users\raul\PycharmProjects\standardized-QPI-DHM-Evaluation-Protocol\Samples\NredBlood_40x_632_4.65_30mm_PCA_105.76_fc_.bmp"
#name = r"C:\Users\raul\PycharmProjects\standardized-QPI-DHM-Evaluation-Protocol\Samples\NredBlood_40x_632_4.65_30mm_VortexLegPis_.bmp"
#name = r"C:\Users\raul\PycharmProjects\standardized-QPI-DHM-Evaluation-Protocol\Samples\NStar_20x_532_5.86_-4cm_PCA_118.27_fc_.bmp"
#name = r"C:\Users\raul\PycharmProjects\standardized-QPI-DHM-Evaluation-Protocol\Samples\Tglibocut_0x_632_4.75_SHCP_.bmp"
#name = r"C:\Users\raul\PycharmProjects\standardized-QPI-DHM-Evaluation-Protocol\Samples\Tglibocut_0x_632_4.75_VortexLegPis_.bmp"

#EAFIT - computer
#name = r"C:\Users\racastaneq\PycharmProjects\SEP\samples\NredBlood_40x_632_4.65_30mm_PCA_105.76_fc_.bmp"
name = r"C:\Users\racastaneq\PycharmProjects\SEP\samples\NredBlood_40x_632_4.65_30mm_VortexLegPis_.bmp"
#name = r"C:\Users\racastaneq\PycharmProjects\SEP\samples\NStar_20x_532_5.86_-4cm_PCA_118.27_fc_.bmp"
#name = r"C:\Users\racastaneq\PycharmProjects\SEP\samples\Tglibocut_0x_632_4.75_SHCP_.bmp"
#name = r"C:\Users\racastaneq\PycharmProjects\SEP\samples\Tglibocut_0x_632_4.75_VortexLegPis_.bmp"

# Example metrics
sample = ut.imageRead(name)
sample = ut.grayscaleToPhase(sample)
unwrapped = rb.unwrap_with_scikit(sample)

ut.show_side_by_side(sample, unwrapped, 'Sample', 'Unwrapped', cmap='gray')
background_mask, background_values, threshold = ut.create_background_mask(unwrapped, method='otsu')
# std = rb.std_background(sample, background_mask, manual=False, num_zones=2)
# mad = rb.mean_absolute_deviation_background(sample, background_mask, manual=True, num_zones=4)
# rms = rb.rms_background(sample, background_mask, manual=True, num_zones=4)
# fwhm = rb.fwhm_background(sample, background_mask, manual=False, num_zones=2)
# entropy = rb.entropy_background(sample, background_mask, manual=True, num_zones=1, bins=256, base=2.0)
sf = rb.spatial_frequency(sample, background_mask,  normalize=True, return_components=True)
