"""
Tests for OTlib.wasser: closed-form distances, and derivatives against
finite differences, including source and target of different lengths.

Run from the repository root with either
    python tests/test_otlib_wasser.py
or  python -m pytest tests
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from libs import OTlib as OT  # noqa: E402


def smooth_density(n, x0, x1, rng):
    """A positive two-bump density on n equally spaced points in [x0, x1]."""
    x = np.linspace(x0, x1, n)
    w = x1 - x0
    p = (np.exp(-0.5*((x - rng.uniform(x0, x1))/(0.15*w))**2)
         + 0.3*np.exp(-0.5*((x - rng.uniform(x0, x1))/(0.1*w))**2)
         + 1e-3*rng.random(n))
    return p, x


def rel_err(a, b):
    return np.max(np.abs(a - b))/np.max(np.abs(b))


def test_closed_form_shift():
    """Identical densities shifted by d: W1 = |d|, W2 (squared) = d^2."""
    rng = np.random.default_rng(0)
    p, x = smooth_density(200, -1.0, 1.0, rng)
    for d in [-0.3, 0.05, 0.7]:
        W1, W2 = OT.wasser(OT.OTpdf((p, x + d)), OT.OTpdf((p, x)), distfunc='W12')
        assert abs(W1 - abs(d)) < 1e-12 and abs(W2 - d*d) < 1e-12, (d, W1, W2)


def check_derivatives(nf, ng, reflect, rng, eps=1e-7, tol=1e-6):
    """dW1, dW2 (w.r.t. the source's unnormalised amplitudes) and dW/dx0
    (translation of the source) against central finite differences."""
    p, x = smooth_density(nf, -1.0, 1.3, rng)
    q, y = smooth_density(ng, -1.2, 1.1, rng)
    target = OT.OTpdf((q, y))
    W1, dW1, dW1x, W2, dW2, dW2x = OT.wasser(OT.OTpdf((p, x)), target, distfunc='W12',
                                             derivatives=True, reflect=reflect)
    assert len(dW1) == nf and len(dW2) == nf
    fd1, fd2 = np.zeros(nf), np.zeros(nf)
    for i in range(nf):
        pp, pm = p.copy(), p.copy()
        pp[i] += eps
        pm[i] -= eps
        a = OT.wasser(OT.OTpdf((pp, x)), target, distfunc='W12', reflect=reflect)
        b = OT.wasser(OT.OTpdf((pm, x)), target, distfunc='W12', reflect=reflect)
        fd1[i], fd2[i] = (a[0] - b[0])/(2*eps), (a[1] - b[1])/(2*eps)
    a = OT.wasser(OT.OTpdf((p, x + eps)), target, distfunc='W12', reflect=reflect)
    b = OT.wasser(OT.OTpdf((p, x - eps)), target, distfunc='W12', reflect=reflect)
    assert rel_err(dW1, fd1) < tol, (nf, ng, reflect, rel_err(dW1, fd1))
    assert rel_err(dW2, fd2) < tol, (nf, ng, reflect, rel_err(dW2, fd2))
    assert abs(dW1x - (a[0] - b[0])/(2*eps)) < 1e-4*abs(dW1x), (nf, ng, reflect)
    assert abs(dW2x - (a[1] - b[1])/(2*eps)) < 1e-4*abs(dW2x), (nf, ng, reflect)


def test_derivatives_equal_and_unequal_lengths():
    """Equal lengths, and different lengths in both orders (failed with a
    broadcast error before v1.1.0), with and without reflect."""
    rng = np.random.default_rng(7)
    for nf, ng in [(40, 40), (40, 55), (55, 40), (17, 90), (90, 17)]:
        for reflect in [False, True]:
            check_derivatives(nf, ng, reflect, rng)


def test_plan_derivative_unequal_lengths():
    """Derivative of the transport plan H (nf x ng) for nf != ng."""
    rng = np.random.default_rng(3)
    p, x = smooth_density(25, -1.0, 1.2, rng)
    q, y = smooth_density(37, -1.1, 1.0, rng)
    target = OT.OTpdf((q, y))
    W2, dW2, dW2x, H, dH = OT.wasser(OT.OTpdf((p, x)), target, distfunc='W2',
                                     derivatives=True, returnplan=True)
    assert H.shape == (25, 37) and dH.shape == (25, 25, 37)
    eps = 1e-7
    for i in range(len(p)):
        pp, pm = p.copy(), p.copy()
        pp[i] += eps
        pm[i] -= eps
        Hp = OT.wasser(OT.OTpdf((pp, x)), target, distfunc='W2', returnplan=True)[1]
        Hm = OT.wasser(OT.OTpdf((pm, x)), target, distfunc='W2', returnplan=True)[1]
        assert np.max(np.abs(dH[i] - (Hp - Hm)/(2*eps))) < 1e-6


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_') and callable(f):
            f()
            print(f"{name}: passed")
    print(f"OTlib {OT.__version__}: all tests passed")
