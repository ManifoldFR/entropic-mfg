import sys

import numpy as np

from build.python import pyentropicmfg
from build.python.pyentropicmfg import ObstacleProx

nx = 51
mask = np.zeros((nx, nx), dtype=int)

obst_prox = ObstacleProx(mask)
