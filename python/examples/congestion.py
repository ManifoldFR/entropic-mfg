import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

import build.python.pyentropicmfg as mfg

plt.rcParams['figure.dpi'] = 90


nx = 51
xar = np.linspace(0, 1, nx)
xg, yg = np.meshgrid(xar, xar)
extent = [xg.min(), xg.max(), yg.min(), yg.max()]

epsilon = 0.001
kernel = mfg.kernels.EuclideanKernel(
    nx, nx, 0, 1, 0, 1, epsilon)


# Define domain mask
mask = np.zeros((nx, nx), dtype=int)
mask[:2, :] = mask[-2:, :] = 1
mask[:, -2:] = mask[:, :2] = 1
mask[25:30, :21] = 1

def mask_to_img(mask: np.ndarray):
    mask_img = np.zeros(mask.shape + (4,))
    mask_img[mask.astype(bool), 3] = 1.
    return mask_img

mask_img = mask_to_img(mask)

# Initial distribution
rho_0 = np.zeros((nx, nx))
rho_0[10:15, 5:15] = 1.
rho_0 /= rho_0.sum()

fig = plt.figure()
plt.subplot(1,2,1)
plt.imshow(mask_img, zorder=2)
plt.imshow(rho_0, cmap=plt.cm.Blues, zorder=0, origin='lower')

plt.subplot(1,2,2)
plt.imshow(mask_img, zorder=2)
plt.imshow(kernel(rho_0), cmap=plt.cm.Blues, zorder=0, origin='lower')


## Problem setup

congest_max = 1.01 * rho_0.max()

### Define potential
import skfmm
exit_mask = np.zeros((nx, nx), dtype=int)
exit_mask[32:40, 5:15] = 1

exit_img = mask_to_img(exit_mask)
exit_img[exit_mask.astype(bool), 0] = .8

plt.figure()
plt.imshow(mask_img, zorder=2)
plt.imshow(exit_img, zorder=1, origin='lower')
plt.imshow(rho_0, cmap=plt.cm.Blues, zorder=0, origin='lower')
plt.show()

boundary_ = np.ma.MaskedArray(1. - exit_mask, mask=mask)
potential = skfmm.travel_time(boundary_, np.ones_like(boundary_))

plt.figure()
plt.imshow(mask_img, zorder=2, origin='lower', extent=extent)
plt.imshow(exit_img, zorder=1, origin='lower', extent=extent)
ct = plt.contourf(potential, zorder=1, levels=40, extent=extent)
plt.title("Potential function $\\Psi$")
plt.show()

prox = mfg.prox.CongestionObstacleProx(mask, congest_max, potential)




