import numpy as np
import pytest
from numlang.core.linear import gauss_solve, SingularMatrixError
from numlang.core.linear_fast import gauss_solve_array


@pytest.mark.parametrize('n', [2, 7, 220])
def test_array_matches_numpy_and_list(n):
    rng = np.random.default_rng(41)
    A = rng.normal(size=(n,n)) + n*np.eye(n)
    A[[0,1]] = A[[1,0]]  # force a row swap
    b = rng.normal(size=n)
    Ac, bc = A.copy(), b.copy()
    x = gauss_solve_array(A,b)
    assert x == pytest.approx(np.linalg.solve(A,b), abs=1e-11)
    assert x == pytest.approx(gauss_solve(A.tolist(),b.tolist()), abs=1e-11)
    assert np.array_equal(A,Ac) and np.array_equal(b,bc)


def test_exact_integer_inputs():
    assert gauss_solve_array(np.array([[0,2],[1,3]]),np.array([4,8])) == pytest.approx([2,2])


def test_singular():
    with pytest.raises(SingularMatrixError):
        gauss_solve_array(np.array([[1.,2.],[2.,4.]]),np.array([1.,2.]))


def test_bad_input():
    with pytest.raises(ValueError):
        gauss_solve_array(np.ones((2,3)),np.ones(2))
    with pytest.raises(ValueError):
        gauss_solve_array(np.eye(2),np.array([1.,np.nan]))
