"""Independent analytic solutions check all three ODE solvers."""
import math

import pytest

from numlang.core.ode import euler, modified_euler, rk4

METHODS = [euler, modified_euler, rk4]


@pytest.mark.parametrize("solver,expected", [(euler, 1.2), (modified_euler, 1.22), (rk4, 1.2214)])
def test_one_step(solver, expected):
    # Independently evaluated first update for y'=y, y(0)=1, h=0.2.
    r = solver(lambda x, y: y, 0, 1, 0.2, 0.2)
    # RK4's polynomial is 1 + h + h^2/2 + h^3/6 + h^4/24.
    assert r.y == pytest.approx(expected)
    assert r.steps == 1


@pytest.mark.parametrize("solver,tol", [(euler, .14), (modified_euler, .005), (rk4, 3e-6)])
def test_exponential_exact(solver, tol):
    r = solver(lambda x, y: y, 0, 1, .1, 1, exact=math.exp)
    assert r.x == 1 and r.steps == 10
    assert r.y == pytest.approx(math.e, abs=tol)
    assert r.error == pytest.approx(abs(r.y - math.e))
    assert r.history[0]["error"] == 0


@pytest.mark.parametrize("solver,tol", [(euler, .15), (modified_euler, .006), (rk4, 4e-6)])
def test_non_autonomous_exact(solver, tol):
    # y'=x+y, y(0)=1 has the analytic solution 2e^x-x-1.
    r = solver(lambda x, y: x+y, 0, 1, .05, 1)
    assert r.y == pytest.approx(2*math.e-2, abs=tol)
    assert r.error is None


@pytest.mark.parametrize("solver,order", [(euler, 1), (modified_euler, 2), (rk4, 4)])
def test_convergence_order(solver, order):
    coarse = solver(lambda x, y: y, 0, 1, .05, 1, exact=math.exp)
    fine = solver(lambda x, y: y, 0, 1, .025, 1, exact=math.exp)
    assert coarse.error / fine.error == pytest.approx(2**order, rel=.06)


@pytest.mark.parametrize("solver,tol", [(euler, .06), (modified_euler, .002), (rk4, 2e-7)])
def test_coupled_harmonic_oscillator(solver, tol):
    # x'=v, v'=-x has x=cos(t), v=-sin(t) for x(0)=1, v(0)=0.
    initial = [1, 0]
    exact = lambda t: (math.cos(t), -math.sin(t))
    r = solver(lambda t, y: (y[1], -y[0]), 0, initial, .05, 1, exact=exact)
    assert r.y == pytest.approx(exact(1), abs=tol)
    assert r.error == pytest.approx(max(abs(a-b) for a,b in zip(r.y, exact(1))))
    assert initial == [1, 0]


@pytest.mark.parametrize("solver", METHODS)
def test_short_final_step_and_history(solver):
    r = solver(lambda x, y: 2, 0, 3, .3, 1, exact=lambda x: 3+2*x)
    assert r.steps == 4
    assert r.x == 1 and r.y == pytest.approx(5)
    assert r.history[-1]["h"] == pytest.approx(.1)
    assert r.history[0]["y"] == 3
    assert len(r.history) == 5
    assert "numerical y" in r.table()


@pytest.mark.parametrize("solver", METHODS)
def test_backward_and_zero_length(solver):
    r = solver(lambda x, y: 2, 1, 5, .3, 0)
    assert r.x == 0 and r.y == pytest.approx(3)
    assert r.history[-1]["h"] < 0
    r = solver(lambda x, y: pytest.fail("must not evaluate RHS"), 1, 5, .3, 1)
    assert r.steps == 0 and r.y == 5


@pytest.mark.parametrize("solver", METHODS)
def test_invalid_step(solver):
    for h in [0, -1, math.inf, math.nan]:
        with pytest.raises(ValueError, match="positive"):
            solver(lambda x,y: y, 0, 1, h, 1)


@pytest.mark.parametrize("solver", METHODS)
def test_bad_dimension_and_nonfinite(solver):
    with pytest.raises(ValueError, match="dimensions"):
        solver(lambda x, y: [1], 0, [1, 2], .1, 1)
    with pytest.raises(ValueError, match="non-finite"):
        solver(lambda x, y: math.inf, 0, 1, .1, 1)


@pytest.mark.parametrize("solver", METHODS)
def test_max_steps(solver):
    with pytest.raises(ValueError, match="max_steps"):
        solver(lambda x, y: y, 0, 1, .1, 1, max_steps=2)


@pytest.mark.parametrize("solver", METHODS)
def test_no_roundoff_only_final_step(solver):
    r = solver(lambda x, y: 1, .3, 0, .2, .9)
    assert r.steps == 3
    assert r.x == .9 and r.y == pytest.approx(.6)
