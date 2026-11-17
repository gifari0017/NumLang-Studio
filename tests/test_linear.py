import numpy as np
import pytest

from numlang.core.linear import SingularMatrixError, gauss_solve


def test_simple_3x3():
    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]
    assert gauss_solve(A, b) == pytest.approx([2, 3, -1])


def test_needs_pivoting():
    # A[0][0] is zero: without a row swap this divides by zero.
    A = [[0, 1], [1, 0]]
    assert gauss_solve(A, [3, 5]) == pytest.approx([5, 3])


def test_singular_raises():
    A = [[1, 2], [2, 4]]
    with pytest.raises(SingularMatrixError):
        gauss_solve(A, [1, 2])


def test_inputs_not_changed():
    A = [[2, 1], [1, 3]]
    b = [3, 5]
    gauss_solve(A, b)
    assert A == [[2, 1], [1, 3]] and b == [3, 5]


def test_matches_numpy_random():
    rng = np.random.default_rng(0)
    for n in (3, 5, 10):
        A = rng.normal(size=(n, n))
        b = rng.normal(size=n)
        assert gauss_solve(A.tolist(), b.tolist()) == pytest.approx(
            np.linalg.solve(A, b), rel=1e-8
        )


def test_bad_shape():
    with pytest.raises(ValueError):
        gauss_solve([[1, 2, 3], [4, 5, 6]], [1, 2])
