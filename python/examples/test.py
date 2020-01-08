import sys
import os
import numpy as np
import matplotlib.pyplot as plt

print(os.getcwd())
sys.path.append(os.getcwd())

import build.python.pyentropicmfg as mfg

plt.rcParams['figure.dpi'] = 90


nx = 41
xar = np.linspace(0, 1, nx)
xg, yg = np.meshgrid(xar, xar)

# Define domain mask
mask = np.zeros((nx, nx), dtype=int)
mask[:2, :] = mask[-2:, :] = 1
mask[:, -2:] = mask[:, :2] = 1
mask[15:24, :21] = 1

# Initial distribution
rho_0 = np.zeros((nx, nx))
rho_0[10:20, 10:20] = 1.
rho_0 /= rho_0.sum()

congest_max = 1.01 * rho_0.max()
potential = np.zeros((nx, nx))
prox = mfg.prox.CongestionObstacleProx(mask, congest_max, potential)

x = np.random.randn(nx, nx)
y = prox(x)

plt.imshow(mask, cmap=plt.cm.binary)
plt.show()
