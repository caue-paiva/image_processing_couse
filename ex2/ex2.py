import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import convolve

img = imageio.imread('hahaha_bw.png').astype(np.float64)

wSx = np.array([[-1, -2, -1],
                 [ 0,  0,  0],
                 [ 1,  2,  1]])

wSy = np.array([[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]])

gx = convolve(img, wSx)
gy = convolve(img, wSy)

# Remove 1px border
gx = gx[1:-1, 1:-1]
gy = gy[1:-1, 1:-1]

mag = np.sqrt(gx**2 + gy**2)
M = mag / mag.sum()

idx = np.unravel_index(np.argmax(M), M.shape)
print(f"{idx[0]}, {idx[1]}")
