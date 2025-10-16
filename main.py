

import utilities as ut
import residual_background as rb
import unwrapping as uw
import numpy as np
import scipy.io as sio

# Read image
#name = r"C:\Users\raul\PycharmProjects\standardized-QPI-DHM-Evaluation-Protocol\Samples\NredBlood_40x_632_4.65_30mm_PCA_105.76_fc_.bmp"
#name = r"C:\Users\raul\PycharmProjects\standardized-QPI-DHM-Evaluation-Protocol\Samples\NredBlood_40x_632_4.65_30mm_VortexLegPis_.bmp"
#name = r"C:\Users\raul\PycharmProjects\standardized-QPI-DHM-Evaluation-Protocol\Samples\NStar_20x_532_5.86_-4cm_PCA_118.27_fc_.bmp"
#name = r"C:\Users\raul\PycharmProjects\standardized-QPI-DHM-Evaluation-Protocol\Samples\Tglibocut_0x_632_4.75_SHCP_.bmp"
#name = r"C:\Users\raul\PycharmProjects\standardized-QPI-DHM-Evaluation-Protocol\Samples\Tglibocut_0x_632_4.75_VortexLegPis_.bmp"

#EAFIT - computer
#name = r"C:\Users\racastaneq\PycharmProjects\SEP\samples\NredBlood_40x_632_4.65_30mm_PCA_105.76_fc_.bmp"
#name = r"C:\Users\racastaneq\PycharmProjects\SEP\samples\NredBlood_40x_632_4.65_30mm_VortexLegPis_.bmp"
#name = r"C:\Users\racastaneq\PycharmProjects\SEP\samples\comp_lev18.bmp"
#name = r"C:\Users\racastaneq\PycharmProjects\SEP\samples\Phase_micro.png"
#name = r"C:\Users\racastaneq\PycharmProjects\SEP\samples\NStar_20x_532_5.86_-4cm_PCA_118.27_fc_.bmp"
#name = r"C:\Users\racastaneq\PycharmProjects\SEP\samples\Tglibocut_0x_632_4.75_SHCP_.bmp"
#name = r"C:\Users\racastaneq\PycharmProjects\SEP\samples\Tglibocut_0x_632_4.75_VortexLegPis_.bmp"

# load .mat file
name1 = r"C:\Users\racastaneq\PycharmProjects\SEP\samples\PruebaStarAberraciones.mat"
data = sio.loadmat(name1)
complex_field = data['star']


'''load .npy'''
# load .npy file
#name1 = r"C:\Users\racastaneq\PycharmProjects\SEP\samples\matriz_compleja1.npy"
#complex_field = np.load(name1)


# Verificar el tipo de datos
phase = np.angle(complex_field)
abs = np.abs(complex_field)
ut.show_side_by_side(abs, phase, 'Sample', 'Unwrapped', cmap='gray')


# Example metrics
#sample = ut.imageRead(name)
#sample = ut.grayscaleToPhase(sample)
#unwrapped = uw.phase_unwrap(sample)

#ut.show_side_by_side(sample, unwrapped, 'Sample', 'Unwrapped', cmap='gray')
#background_mask, background_values, threshold = ut.create_background_mask(unwrapped, method='otsu')
#std = rb.std_background(sample, background_mask, manual=True, num_zones=3)
#mad = rb.mean_absolute_deviation_background(sample, background_mask, manual=True, num_zones=4)
#rms = rb.rms_background(sample, background_mask, manual=True, num_zones=4)
#fwhm = rb.fwhm_background(sample, background_mask, manual=False, num_zones=2)
#entropy = rb.entropy_background(sample, background_mask, manual=True, num_zones=1, bins=256, base=2.0)
#sf = rb.spatial_frequency(sample, background_mask,  normalize=True, return_components=True)
legendre = rb.legendre(complex_field, 250, False, True)