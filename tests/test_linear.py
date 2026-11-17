import numpy as np
import pytest

from numlang.core.linear import (
    SingularMatrixError,
    determinant,
    gauss_jordan_solve,
    gauss_solve,
    inverse,
)


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


# Step 1b: Gauss-Jordan, determinant, inverse, and numerical stability.
def test_jordan_exact():
    assert gauss_jordan_solve(
        [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], [8, -11, -3]
    ) == pytest.approx([2, 3, -1])


def test_jordan_pivot():
    assert gauss_jordan_solve([[0, 1], [1, 0]], [3, 5]) == pytest.approx([5, 3])


@pytest.mark.parametrize("solver", [gauss_solve, gauss_jordan_solve])
def test_solvers_singular_and_shape(solver):
    with pytest.raises(SingularMatrixError):
        solver([[1, 2], [2, 4]], [1, 2])
    with pytest.raises(ValueError):
        solver([[1, 2]], [1])
    with pytest.raises(ValueError):
        solver([[1]], [1, 2])


def test_jordan_matches_numpy():
    rng = np.random.default_rng(42)
    for n in (1, 3, 5, 10):
        A = rng.normal(size=(n, n)) + n * np.eye(n)
        b = rng.normal(size=n)
        assert gauss_jordan_solve(A.tolist(), b.tolist()) == pytest.approx(
            np.linalg.solve(A, b), rel=1e-10, abs=1e-12
        )


def test_determinant_exact_and_swap_sign():
    assert determinant([[4, 7], [2, 6]]) == pytest.approx(10)
    assert determinant([[0, 1], [1, 0]]) == pytest.approx(-1)
    assert determinant([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) == pytest.approx(1)
    assert determinant([[7]]) == pytest.approx(7)


def test_determinant_singular():
    assert determinant([[1, 2], [2, 4]]) == 0.0


def test_determinant_matches_numpy():
    rng = np.random.default_rng(7)
    for n in (3, 5, 8):
        A = rng.normal(size=(n, n))
        assert determinant(A.tolist()) == pytest.approx(np.linalg.det(A), rel=1e-10)


def test_inverse_exact():
    actual = np.array(inverse([[4, 7], [2, 6]]))
    np.testing.assert_allclose(actual, [[0.6, -0.7], [-0.2, 0.4]])
    assert inverse([[2]]) == [[0.5]]


def test_inverse_matches_numpy_and_identity():
    rng = np.random.default_rng(19)
    for n in (3, 5, 8):
        A = rng.normal(size=(n, n)) + n * np.eye(n)
        actual = np.array(inverse(A.tolist()))
        np.testing.assert_allclose(actual, np.linalg.inv(A), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(A @ actual, np.eye(n), atol=1e-12)


def test_inverse_singular():
    with pytest.raises(SingularMatrixError):
        inverse([[1, 2], [2, 4]])


def test_new_functions_do_not_mutate_inputs():
    A, b = [[0, 1], [2, 3]], [1, 5]
    gauss_jordan_solve(A, b)
    determinant(A)
    inverse(A)
    assert A == [[0, 1], [2, 3]] and b == [1, 5]


@pytest.mark.parametrize("operation", [determinant, inverse])
def test_matrix_operations_validate_shape(operation):
    for A in ([], [[1, 2]], [[1, 2], [3]]):
        with pytest.raises(ValueError):
            operation(A)


def test_pivoting_avoids_catastrophic_cancellation():
    A, b = [[1e-20, 1], [1, 1]], [1, 2]
    expected = np.linalg.solve(A, b)
    good = gauss_solve(A, b, tol=1e-30)
    bad = gauss_solve(A, b, tol=1e-30, pivoting=False)
    assert good == pytest.approx(expected)
    assert np.linalg.norm(np.array(bad) - expected, np.inf) > 0.9
    with pytest.raises(SingularMatrixError):
        gauss_solve([[0, 1], [1, 0]], [3, 5], pivoting=False)


def test_condition_number_amplifies_small_rhs_perturbation():
    eps = 1e-8
    A = [[1, 1], [1, 1 + eps]]
    b, altered = [2, 2 + eps], [2, 2 + 2 * eps]
    x = np.array(gauss_solve(A, b))
    changed = np.array(gauss_solve(A, altered))
    assert x == pytest.approx([1, 1], abs=1e-7)
    assert changed == pytest.approx([0, 2], abs=1e-7)
    assert np.linalg.cond(A, np.inf) > 1e8
    relative_b = np.linalg.norm(np.subtract(altered, b), np.inf) / np.linalg.norm(b, np.inf)
    relative_x = np.linalg.norm(changed - x, np.inf) / np.linalg.norm(x, np.inf)
    assert relative_b < 1e-8 and relative_x > 0.9


def test_verbose_trace(capsys):
    gauss_jordan_solve([[0, 1], [2, 3]], [1, 5], verbose=True)
    trace = capsys.readouterr().out
    assert "swap row" in trace and "/=" in trace and "-=" in trace
