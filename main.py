
import utilities as ut
import residual_background as rb
import matplotlib.pyplot as plt

# Read image
#name = r"C:\Users\raul\PycharmProjects\standardized-QPI-DHM-Evaluation-Protocol\Samples\NredBlood_40x_632_4.65_30mm_PCA_105.76_fc_.bmp"
#name = r"C:\Users\raul\PycharmProjects\standardized-QPI-DHM-Evaluation-Protocol\Samples\NredBlood_40x_632_4.65_30mm_VortexLegPis_.bmp"
name = r"C:\Users\raul\PycharmProjects\standardized-QPI-DHM-Evaluation-Protocol\Samples\NStar_20x_532_5.86_-4cm_PCA_118.27_fc_.bmp"
#name = r"C:\Users\raul\PycharmProjects\standardized-QPI-DHM-Evaluation-Protocol\Samples\Tglibocut_0x_632_4.75_SHCP_.bmp"
#name = r"C:\Users\raul\PycharmProjects\standardized-QPI-DHM-Evaluation-Protocol\Samples\Tglibocut_0x_632_4.75_VortexLegPis_.bmp"

# Example metrics using Legendre to obtain background
sample = ut.imageRead(name)
sample = ut.grayscaleToPhase(sample)

background_mask, background_values, threshold = ut.create_background_mask(sample, method='otsu')
std1 = rb.std_background(sample,background_mask )
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(sample, cmap="gray")
plt.colorbar()
plt.subplot(1,2,2)
plt.title("Background")
plt.imshow(background_mask, cmap="gray")
plt.colorbar()
plt.show()


corrected, background = rb.legendre_background_correction(sample, 3)
std1 = rb.std_background(background)
entropy1 = rb.calculate_phase_entropy_background(sample, n_bins=256, method='shannon')
print(entropy1)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(sample, cmap="gray")
plt.colorbar()
plt.subplot(1,2,2)
plt.title("Background")
plt.imshow(background, cmap="gray")
plt.colorbar()
plt.show()

corrected, background = rb.pca_background_separation(sample,  n_components=6, patch_size=(8, 8), keep_components=2)
std2 = rb.std_background(background)

plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(sample, cmap="gray")
plt.colorbar()
plt.subplot(1,2,2)
plt.title("Background")
plt.imshow(corrected, cmap="gray")
plt.colorbar()
plt.show()

background, coeffs = rb.legendre_background_extraction(sample, limit=100, max_order=4, return_coefficients=True)
std3 = rb.std_background(background)

plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(sample, cmap="gray")
plt.colorbar()
plt.subplot(1,2,2)
plt.title("Background")
plt.imshow(background, cmap="gray")
plt.colorbar()
plt.show()

print(std1, std2, std3)

#ut.imageShow(sample, 'Phase 1')
#inp = np.array(sample)
#ut.create_binary_mask(inp, 'otsu')