import numpy as np
import build.python.pyentropicmfg as mfg
from build.python.pyentropicmfg.prox import BaseProximalOperator


class MyProx(BaseProximalOperator):
    def __init__(self):
        pass
        
    def __call__(self, x):
        return x

def test_allclose():
    my_prox = MyProx()
    nx = 41
    x = np.random.rand(nx, nx)

    assert np.allclose(x, my_prox(x))

if __name__ == "__main__":
    test_allclose()

