
import utilities as ut
import numpy as np

# Read image
name = r"C:\Users\racastaneq\PycharmProjects\SEP\samples\NredBlood_40x_632_4.65_30mm_PCA_105.76_fc_.bmp"

sample = ut.imageRead(name)
ut.imageShow(sample, 'Phase 1')
inp = np.array(sample)
ut.create_binary_mask(inp, 'otsu')