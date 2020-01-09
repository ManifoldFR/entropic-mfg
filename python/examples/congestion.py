import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd() + 'build/python/')

import build.python.pyentropicmfg as mfg

plt.rcParams['figure.dpi'] = 90
plt.rcParams['savefig.dpi'] = 120


nx = 81
xar = np.linspace(0, 1, nx)
xg, yg = np.meshgrid(xar, xar)
extent = [xg.min(), xg.max(), yg.min(), yg.max()]


# Define domain mask
mask = (xg <= .02) | (xg >= .98)
mask |= (yg <= .02) | (yg >= .98)
mask |= (xg <= 0.5) & (np.abs(yg - 0.5) <= .08)
mask |= ((xg <= 0.5) & (xg >= 0.4)) & (yg <= .45) & (yg >= .15)

def mask_to_img(mask: np.ndarray):
    mask_img = np.zeros(mask.shape + (4,))
    mask_img[mask.astype(bool), 3] = 1.
    return mask_img

mask_img = mask_to_img(mask)

# Initial distribution
rho_0 = (np.abs(xg-0.2) < 0.1) & (np.abs(yg-0.2) < 0.1)
rho_0 = rho_0.astype(float)
rho_0 /= rho_0.sum()

fig = plt.figure(figsize=(8,4))
plt.subplot(1, 2, 1)
plt.imshow(mask_img, zorder=2, origin='lower')
plt.imshow(rho_0, cmap=plt.cm.Blues, zorder=0, origin='lower')

## Problem setup
kappa = 1.00
congest_max = kappa * rho_0.max()

### Define potential
import skfmm
exit_mask = (np.abs(xg - 0.2) < 0.08) & (np.abs(yg - 0.8) < 0.08)
exit_img = mask_to_img(exit_mask)
exit_img[exit_mask.astype(bool), 0] = .8

boundary_ = np.ma.MaskedArray(1. - exit_mask, mask=mask)
potential = skfmm.travel_time(boundary_, np.ones_like(boundary_))
potential = .1 * potential  # lower potential

plt.subplot(1, 2, 2)
plt.imshow(mask_img, zorder=2, origin='lower', extent=extent)
plt.imshow(exit_img, zorder=1, origin='lower', extent=extent)
ct = plt.contourf(potential.data * (1 - mask), zorder=1, levels=40, extent=extent)
plt.colorbar()
plt.title("Potential function $\\Psi$")
plt.show()


terminal_prox = mfg.prox.CongestionObstacleProx(mask, congest_max, potential)
running_prox = mfg.prox.CongestionObstacleProx(mask, congest_max, np.zeros_like(potential))

N_t = 31
dt = 1./ (N_t - 1)
epsilon = 0.1
kernel = mfg.kernels.EuclideanKernel(
    nx, nx, 0, 1, 0, 1, epsilon * dt)


sinkhorn = mfg.sinkhorn.MultiSinkhorn(
    running_prox, terminal_prox,
    kernel, rho_0)


a_s = [
    np.ones_like(rho_0, order='F') for _ in range(N_t)
]

print("Running sinkhorn...")
import time
t_a = time.time()
num_iters = 80
sinkhorn.run(a_s, num_iters)
sink_time = time.time() - t_a
print("Elapsed time:", sink_time)

print("Computing marginals...")
t_a = time.time()
marginals = sinkhorn.get_marginals(a_s)
marg_time = time.time() - t_a
print("Elapsed time:", marg_time)
sink_time += marg_time


num_to_plot = 6  # number of intermediary steps
skip = N_t // num_to_plot
steps_to_plot = [0] + [k*skip for k in range(1, num_to_plot + 1)] + [N_t - 1]

ncols = 4
nrows = len(steps_to_plot) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(9, 5))
axes = axes.ravel()

for i, t in enumerate(steps_to_plot):
    m = marginals[t]
    if i < len(axes):
        ax = axes[i]
        ax.imshow(mask_img, zorder=2, origin='lower', extent=extent)
        ax.imshow(m, zorder=1, origin='lower', extent=extent, cmap=plt.cm.Blues)
        ax.set_title("Time step $t=%d$" % t)
plt.tight_layout()
plt.suptitle("MFG evolution: CPU time %.1fs." % sink_time)
fig.savefig("python/examples/euclidean_simple.png")
plt.show()
