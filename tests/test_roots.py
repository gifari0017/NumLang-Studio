"""Tests for roots.py. Run with:  uv run pytest -v"""

import math

import pytest

from numlang.core.roots import bisection, false_position, newton

# classic textbook function, root near 2.0945514815
f = lambda x: x**3 - 2 * x - 5
df = lambda x: 3 * x**2 - 2
TRUE_ROOT = 2.0945514815423265


def test_bisection_finds_root():
    r = bisection(f, 2, 3, tol=1e-10)
    assert r.converged
    assert abs(r.root - TRUE_ROOT) < 1e-9


def test_false_position_finds_root():
    r = false_position(f, 2, 3, tol=1e-10)
    assert r.converged
    assert abs(r.root - TRUE_ROOT) < 1e-9


def test_newton_finds_root():
    r = newton(f, df, 2.0, tol=1e-12)
    assert r.converged
    assert abs(r.root - TRUE_ROOT) < 1e-11


def test_iteration_order_newton_fastest():
    """Newton < false position < bisection on this function."""
    n = newton(f, df, 2.0, tol=1e-10).iterations
    fp = false_position(f, 2, 3, tol=1e-10).iterations
    bi = bisection(f, 2, 3, tol=1e-10).iterations
    assert n < fp < bi


def test_bisection_iterations_match_theory():
    """Error after n steps is (b-a)/2**n, so n ~ log2((b-a)/tol)."""
    r = bisection(f, 2, 3, tol=1e-10)
    expected = math.ceil(math.log2(1 / 1e-10))
    assert abs(r.iterations - expected) <= 1


def test_bisection_rejects_bad_bracket():
    with pytest.raises(ValueError):
        bisection(f, 3, 4)  # no sign change


def test_false_position_rejects_bad_bracket():
    with pytest.raises(ValueError):
        false_position(f, 3, 4)


def test_newton_diverges_on_arctan_bad_start():
    """FAILURE CASE: atan(x) has root 0, but x0 = 2 makes Newton blow up."""
    r = newton(math.atan, lambda x: 1 / (1 + x * x), 2.0, max_iter=50)
    assert not r.converged


def test_newton_converges_on_arctan_good_start():
    r = newton(math.atan, lambda x: 1 / (1 + x * x), 0.5)
    assert r.converged
    assert abs(r.root) < 1e-9


def test_newton_zero_derivative():
    """FAILURE CASE: x^2 - 1 starting at x = 0 has f'(0) = 0."""
    r = newton(lambda x: x * x - 1, lambda x: 2 * x, 0.0)
    assert not r.converged
    assert "zero derivative" in r.message


def test_false_position_one_sided_stall():
    """FAILURE CASE: f = x^10 - 1 on [0, 1.3]. One end stays fixed and
    false position needs far more steps than bisection."""
    g = lambda x: x**10 - 1
    fp = false_position(g, 0, 1.3, tol=1e-8, max_iter=500)
    bi = bisection(g, 0, 1.3, tol=1e-8)
    assert fp.iterations > bi.iterations


def test_history_recorded():
    r = bisection(f, 2, 3)
    assert len(r.history) == r.iterations
    assert r.history[0]["i"] == 1
    assert "error" in r.history[0]
