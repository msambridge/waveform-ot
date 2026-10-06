"""
Tests for OTlib.wasser_linear (wasser(..., interp='linear')): the 1D
Wasserstein distance between PDFs treated as continuous (each node's mass
spread uniformly over its cell), and its derivatives, including the cases
the point-mass formula can't differentiate (common CDF values).

Run from the repository root with either
    python tests/test_otlib_wasser_linear.py
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


def W(p, x, target, distfunc='W12'):
    return OT.wasser(OT.OTpdf((p, x)), target, distfunc=distfunc, interp='linear')


def rel_err(a, b):
    return np.max(np.abs(a - b))/max(np.max(np.abs(b)), 1e-300)


def fd_amplitudes(p, x, target, eps=1e-7, forward=()):
    """Central finite differences of (W1, W2) in each source amplitude
    (forward differences for the indices in `forward`, e.g. zero masses)."""
    fd = np.zeros((2, len(p)))
    for k in range(len(p)):
        pp, pm = p.copy(), p.copy()
        pp[k] += eps
        if k in forward:
            fd[:, k] = (np.array(W(pp, x, target)) - np.array(W(p, x, target)))/eps
        else:
            pm[k] -= eps
            fd[:, k] = (np.array(W(pp, x, target)) - np.array(W(pm, x, target)))/(2*eps)
    return fd


def test_closed_form_shift():
    """Identical densities shifted by d: W1 = |d|, W2 = d^2 (cells shift too)."""
    rng = np.random.default_rng(0)
    p, x = smooth_density(200, -1.0, 1.0, rng)
    for d in [-0.3, 0.05, 0.7]:
        W1, W2 = W(p, x + d, OT.OTpdf((p, x)))
        assert abs(W1 - abs(d)) < 1e-12 and abs(W2 - d*d) < 1e-12, (d, W1, W2)


def test_uniform_cells_closed_form():
    """One cell each: two uniform densities of widths a, b with centres c1, c2
    have W2 = (c1-c2)^2 + (a-b)^2/12."""
    a, b, c1, c2 = 0.4, 1.0, 0.3, -0.2
    W2 = OT.wasser_linear(OT.OTpdf((np.ones(2), np.array([c1-a/4, c1+a/4]))),
                          OT.OTpdf((np.ones(2), np.array([c2-b/4, c2+b/4]))), distfunc='W2')[0]
    assert abs(W2 - ((c1-c2)**2 + (a-b)**2/12)) < 1e-12, W2


def test_derivatives_equal_and_unequal_lengths():
    """dW1, dW2 (unnormalised source amplitudes) and dW/dx0 against finite differences."""
    rng = np.random.default_rng(7)
    for nf, ng in [(40, 40), (40, 55), (55, 40), (17, 90), (90, 17)]:
        p, x = smooth_density(nf, -1.0, 1.3, rng)
        q, y = smooth_density(ng, -1.2, 1.1, rng)
        target = OT.OTpdf((q, y))
        W1, dW1, dW1x, W2, dW2, dW2x = OT.wasser(OT.OTpdf((p, x)), target, derivatives=True, interp='linear')
        fd = fd_amplitudes(p, x, target)
        assert rel_err(dW1, fd[0]) < 1e-6 and rel_err(dW2, fd[1]) < 1e-6, (nf, ng, rel_err(dW1, fd[0]), rel_err(dW2, fd[1]))
        eps = 1e-7
        a, b = W(p, x + eps, target), W(p, x - eps, target)
        assert abs(dW1x - (a[0]-b[0])/(2*eps)) < 1e-5*max(abs(dW1x), 1) and abs(dW2x - (a[1]-b[1])/(2*eps)) < 1e-5*max(abs(dW2x), 1)


def test_common_cdf_values():
    """A source CDF value equal to a target CDF value: the point-mass formula's
    one-sided derivatives differ there (its kink), the linear ones agree."""
    x, y = np.array([-1.0, -0.3, 0.2, 0.9]), np.array([-0.5, 0.6])
    p, q = np.ones(4), np.ones(2)                  # both CDFs pass through 0.5
    target = OT.OTpdf((q, y))
    eps = 1e-6
    for interp in ['point', 'linear']:
        left, right = [], []
        for k in range(4):
            pp, pm = p.copy(), p.copy()
            pp[k] += eps
            pm[k] -= eps
            f = lambda v: OT.wasser(OT.OTpdf((v, x)), target, distfunc='W2', interp=interp)[0]
            right.append((f(pp) - f(p))/eps)
            left.append((f(p) - f(pm))/eps)
        gap = np.max(np.abs(np.array(right) - np.array(left)))
        if interp == 'point':
            assert gap > 1e-2, gap                 # a genuine kink for point masses
        else:
            assert gap < 1e-5, gap                 # continuous derivative
            dW2 = OT.wasser(OT.OTpdf((p, x)), target, distfunc='W2', derivatives=True, interp='linear')[1]
            assert rel_err(dW2, 0.5*(np.array(left) + np.array(right))) < 1e-5


def test_identical_inputs():
    """Source = target: W = 0 and gradient 0 (the point-mass formula refuses this case)."""
    rng = np.random.default_rng(3)
    p, x = smooth_density(60, -1.0, 1.0, rng)
    out = OT.wasser(OT.OTpdf((p, x)), OT.OTpdf((p, x)), derivatives=True, interp='linear')
    W1, dW1, dW1x, W2, dW2, dW2x = out
    assert abs(W1) < 1e-15 and abs(W2) < 1e-15
    assert np.max(np.abs(dW2)) < 1e-12 and abs(dW2x) < 1e-12
    fd = fd_amplitudes(p, x, OT.OTpdf((p, x)), eps=1e-6)
    assert np.max(np.abs(dW2 - fd[1])) < 1e-6


def test_zero_mass_cells():
    """Zero-mass source cells (gaps): one-sided derivatives (forward differences) match."""
    rng = np.random.default_rng(11)
    p, x = smooth_density(30, -1.0, 1.0, rng)
    zero = [5, 6, 7, 20]
    p[zero] = 0.
    q, y = smooth_density(45, -1.1, 0.9, rng)
    target = OT.OTpdf((q, y))
    W1, dW1, _, W2, dW2, _ = OT.wasser(OT.OTpdf((p, x)), target, derivatives=True, interp='linear')
    fd = fd_amplitudes(p, x, target, eps=1e-7, forward=zero)
    assert rel_err(dW2, fd[1]) < 1e-5 and rel_err(dW1, fd[0]) < 1e-5, (rel_err(dW2, fd[1]), rel_err(dW1, fd[0]))


def test_saturated_tails():
    """Gaussian-like PDFs whose tails are ~1e-16 of the peak (CDFs round to 0
    or 1 there): finite results, derivatives match away from the far tails."""
    x = np.linspace(-1., 1., 120)
    y = np.linspace(-1.2, 0.9, 140)
    p = np.exp(-0.5*((x-0.1)/0.08)**2)
    q = np.exp(-0.5*((y+0.05)/0.1)**2)
    target = OT.OTpdf((q, y))
    W1, dW1, _, W2, dW2, _ = OT.wasser(OT.OTpdf((p, x)), target, derivatives=True, interp='linear')
    assert np.all(np.isfinite(dW1)) and np.all(np.isfinite(dW2))
    core = np.where(p > 1e-6*p.max())[0]
    eps = 1e-9
    fd = np.zeros(len(core))
    for n, k in enumerate(core):
        pp, pm = p.copy(), p.copy()
        pp[k] += eps
        pm[k] -= eps
        fd[n] = (W(pp, x, target, 'W2')[0] - W(pm, x, target, 'W2')[0])/(2*eps)
    assert rel_err(dW2[core], fd) < 1e-4, rel_err(dW2[core], fd)


def test_gradient_continuous_along_a_crossing_path():
    """Along a path on which source CDF steps pass target CDF steps, the
    linear gradient varies continuously (no jumps beyond the smooth change)."""
    rng = np.random.default_rng(5)
    p0, x = smooth_density(25, -1.0, 1.0, rng)
    v = rng.standard_normal(25)
    q, y = smooth_density(31, -1.0, 1.0, rng)
    target = OT.OTpdf((q, y))
    ts = np.linspace(0., 0.2, 401)
    g = np.array([OT.wasser(OT.OTpdf((p0*np.exp(t*v), x)), target, distfunc='W2', derivatives=True,
                            interp='linear')[1] @ (p0*np.exp(t*v)*v) for t in ts])   # dW2/dt
    jumps = np.abs(np.diff(g))
    assert jumps.max() < 5*np.median(jumps) + 1e-12, (jumps.max(), np.median(jumps))


def test_unsupported_options():
    p, x = np.ones(5), np.linspace(0, 1, 5)
    s, t = OT.OTpdf((p, x)), OT.OTpdf((p, x + 0.1))
    for kw in [dict(returnplan=True), dict(reflect=True)]:
        try:
            OT.wasser(s, t, interp='linear', **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(kw)
    assert OT.wasser(s, t, interp='point') == OT.wasser(s, t)   # default unchanged


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_') and callable(f):
            f()
            print(f"{name}: passed")
    print(f"OTlib {OT.__version__}: all wasser_linear tests passed")
