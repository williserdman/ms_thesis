"""Framework-agnostic unit tests for the path + spectral math.

These run WITHOUT torch (numpy only), so they are the real local smoke test on a
CPU/torch-less box. Run from the repo root:

    python3 tests/test_math.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from mode_connectivity import paths, spectral


def approx(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def test_linear_interp():
    a, b = np.array([0.0, 2.0]), np.array([4.0, 6.0])
    assert np.allclose(paths.linear_interp(a, b, 0.0), a)
    assert np.allclose(paths.linear_interp(a, b, 1.0), b)
    assert np.allclose(paths.linear_interp(a, b, 0.5), [2.0, 4.0])


def test_bezier_zero_bend_equals_linear():
    a, b = np.array([1.0, -1.0]), np.array([3.0, 5.0])
    ctrl = paths.bezier_midpoint_init(a, b)
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        assert np.allclose(paths.bezier_interp(a, b, ctrl, t),
                           paths.linear_interp(a, b, t)), f"bend!=linear at t={t}"


def test_bezier_endpoints():
    a, b, c = np.array([0.0]), np.array([1.0]), np.array([5.0])
    assert np.allclose(paths.bezier_interp(a, b, c, 0.0), a)
    assert np.allclose(paths.bezier_interp(a, b, c, 1.0), b)


def test_barrier_flat_is_zero():
    ts = [0.0, 0.5, 1.0]
    # losses on the straight line between endpoints -> no barrier
    losses = [1.0, 1.5, 2.0]
    assert approx(paths.barrier_from_losses(losses, ts), 0.0)


def test_barrier_bump():
    ts = [0.0, 0.5, 1.0]
    losses = [1.0, 2.0, 1.0]  # baseline at t=.5 is 1.0 -> excess 1.0
    assert approx(paths.barrier_from_losses(losses, ts), 1.0)
    assert approx(paths.argmax_barrier_t(losses, ts), 0.5)


def test_barrier_dip_clamped():
    ts = [0.0, 0.5, 1.0]
    losses = [1.0, 0.2, 1.0]  # below baseline -> barrier clamped at 0
    assert approx(paths.barrier_from_losses(losses, ts), 0.0)


def test_linspace():
    xs = paths.linspace(5)
    assert len(xs) == 5 and approx(xs[0], 0.0) and approx(xs[-1], 1.0)


def test_basis_monomial():
    x = np.array([-1.0, 0.0, 0.5, 1.0])
    B = spectral.basis_matrix(x, K=3, basis="mono", domain="adj")
    assert B.shape == (4, 4)
    assert np.allclose(B[:, 0], 1.0)
    assert np.allclose(B[:, 1], x)
    assert np.allclose(B[:, 2], x ** 2)
    assert np.allclose(B[:, 3], x ** 3)


def test_basis_chebyshev_recurrence():
    x = np.array([-1.0, -0.3, 0.0, 0.7, 1.0])
    B = spectral.basis_matrix(x, K=3, basis="cheb", domain="adj")
    assert np.allclose(B[:, 0], 1.0)
    assert np.allclose(B[:, 1], x)
    assert np.allclose(B[:, 2], 2 * x ** 2 - 1)          # T2
    assert np.allclose(B[:, 3], 4 * x ** 3 - 3 * x)      # T3


def test_filter_response_linear_in_gamma():
    x = np.linspace(-1, 1, 17)
    gamma = np.array([0.0, 1.0, 0.0])  # picks out the degree-1 term
    g_mono = spectral.filter_response(gamma, x, basis="mono", domain="adj")
    assert np.allclose(g_mono, x)
    # linearity: response(a+b) == response(a) + response(b)
    ga = np.array([0.3, -0.2, 0.5])
    gb = np.array([-0.1, 0.4, 0.2])
    ra = spectral.filter_response(ga, x, "cheb", "adj")
    rb = spectral.filter_response(gb, x, "cheb", "adj")
    rab = spectral.filter_response(ga + gb, x, "cheb", "adj")
    assert np.allclose(ra + rb, rab)


def test_chebyshev_better_conditioned_than_monomial():
    # core idea-09/idea-16 premise: Chebyshev basis is far better conditioned
    eigs = np.linspace(-1, 1, 200)
    K = 12
    k_cheb = spectral.condition_number(eigs, K, basis="cheb", domain="adj")
    k_mono = spectral.condition_number(eigs, K, basis="mono", domain="adj")
    assert k_cheb < k_mono, (k_cheb, k_mono)


def test_response_distance_zero_for_equal():
    eigs = np.linspace(-1, 1, 50)
    g = np.array([0.1, 0.2, -0.3, 0.05])
    assert approx(spectral.response_distance(g, g, eigs), 0.0)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} math tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
