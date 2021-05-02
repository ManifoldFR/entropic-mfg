import sys
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

import build.python.pyentropicmfg as mfg
import skfmm
import termcolor as tcl


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
# mask |= ((xg <= 0.5) & (xg >= 0.4)) & (yg <= .45) & (yg >= .15)

def mask_to_img(mask: np.ndarray):
    mask_img = np.zeros(mask.shape + (4,))
    mask_img[mask.astype(bool), 3] = 1.
    return mask_img

mask_img = mask_to_img(mask)

# Initial distribution
rho_0 = (np.abs(xg-0.2) < 0.1) & (np.abs(yg-0.2) < 0.1)
rho_0 = rho_0.astype(float)
rho_0 /= rho_0.sum()

print("Plotting initial distribution & mask")
fig = plt.figure(figsize=(8,4))
plt.subplot(1, 2, 1)
plt.imshow(mask_img, zorder=2, origin='lower')
plt.imshow(rho_0, cmap=plt.cm.Blues, zorder=0, origin='lower')
plt.title(r"Initial distribution $\rho_0$, and boundaries")

## Problem setup
kappa = 1.00
congest_max = kappa * rho_0.max()

### Define potential
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
# plt.colorbar()
plt.title("Potential function $\\Psi$")
plt.show()



print("Defining dynamical problem...")

terminal_prox = mfg.prox.CongestionObstacleProx(mask, congest_max, potential)
running_prox = mfg.prox.CongestionObstacleProx(mask, congest_max, np.zeros_like(potential))

nsteps = 51
Tf = 10.
dt = Tf / (nsteps - 1)
variance = 0.1
kernel = mfg.kernels.EuclideanKernel(
    nx, nx, 0, 1, 0, 1, variance * dt ** .5)


sinkhorn = mfg.sinkhorn.MultiSinkhorn(
    running_prox, terminal_prox,
    kernel, rho_0)
sinkhorn.threshold = 1e-8

dual_potentials = [
    np.ones_like(rho_0, order='F') for _ in range(nsteps)
]


import time

t_a = time.time()
num_iters = 50

converged = sinkhorn.solve(num_iters, dual_potentials)
if converged:
    tcl.cprint("Converged.", "cyan")

sink_time = time.time() - t_a
print("Elapsed time:", sink_time)

print("Computing marginals...")
t_a = time.time()
marginals = sinkhorn.get_marginals()
marg_time = time.time() - t_a
print("Elapsed time:", marg_time)
sink_time += marg_time


def plot_convergence(solver):
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot(111)
    metric_vals = solver.conv_metrics_
    ax.plot(metric_vals, ls='--', lw=1.)
    ax.hlines(solver.threshold, 0, len(metric_vals),
              ls='dashdot', lw=1., color='gray', label='Threshold')
    ax.legend()
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"Hilbert metric $\mathcal{H}$")
    ax.set_yscale('log')
    plt.show()


def plot_fun():
    num_to_plot = 6  # number of intermediary steps
    skip = nsteps // num_to_plot
    steps_to_plot = [0] + [k*skip for k in range(1, num_to_plot + 1)] + [nsteps - 1]

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
    plt.show()


plot_fun()
plot_convergence(sinkhorn)
