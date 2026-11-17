"""Hand-written Gaussian elimination and Gauss-Jordan operations.

All public functions copy their inputs. No numerical library is required.
The pivot tolerance is absolute (not a scale-invariant singularity test).
"""

from __future__ import annotations

import math


class SingularMatrixError(Exception):
    """Raised when no usable pivot exists at the requested tolerance."""


def _square_copy(A, tol):
    """Validate a nonempty square matrix and make a floating-point copy."""
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        raise ValueError("A must be a nonempty n x n matrix")
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    M = [[float(v) for v in row] for row in A]
    if any(not math.isfinite(v) for row in M for v in row):
        raise ValueError("A must contain finite numbers")
    return M


def _augmented(A, b, tol):
    M = _square_copy(A, tol)
    if len(b) != len(M):
        raise ValueError("b must have length n")
    for row, value in zip(M, b):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("b must contain finite numbers")
        row.append(value)
    return M


def _pivot(M, col, tol, verbose, pivoting=True):
    """Select and swap a pivot row; return whether a swap happened."""
    n = len(M)
    p = max(range(col, n), key=lambda r: abs(M[r][col])) if pivoting else col
    if abs(M[p][col]) < tol:
        raise SingularMatrixError(
            f"pivot in column {col} is {M[p][col]:.3e}: no usable pivot"
        )
    if p != col:
        M[col], M[p] = M[p], M[col]
        if verbose:
            print(f"swap row {col} <-> row {p}")
        return True
    return False


def _forward_eliminate(M, tol, verbose, pivoting=True):
    """Make the first n columns upper triangular, counting row swaps."""
    n, width = len(M), len(M[0])
    swaps = 0
    for col in range(n):
        swaps += _pivot(M, col, tol, verbose, pivoting)
        for r in range(col + 1, n):
            factor = M[r][col] / M[col][col]
            if factor == 0.0:
                continue
            M[r][col] = 0.0
            for c in range(col + 1, width):
                M[r][c] -= factor * M[col][c]
            if verbose:
                print(f"row {r} -= {factor:.6g} * row {col}")
    return swaps


def gauss_solve(A, b, *, tol=1e-12, verbose=False, pivoting=True):
    """Solve A x = b by forward elimination and back substitution.

    Partial pivoting is enabled by default. Disable it only to demonstrate
    the numerical danger of a tiny or zero pivot.
    """
    M = _augmented(A, b, tol)
    n = len(M)
    _forward_eliminate(M, tol, verbose, pivoting)

    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n] - sum(M[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / M[r][r]
    return x


def _gauss_jordan(M, tol, verbose):
    """Reduce [A | RHS] to [I | solution] using elementary row operations."""
    n, width = len(M), len(M[0])
    for col in range(n):
        _pivot(M, col, tol, verbose)
        pivot = M[col][col]

        # Divide the pivot row so its leading entry becomes one.
        for c in range(col, width):
            M[col][c] /= pivot
        if verbose:
            print(f"row {col} /= {pivot:.6g}")

        # Eliminate in EVERY other row, not just below the pivot.
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor == 0.0:
                continue
            M[r][col] = 0.0
            for c in range(col + 1, width):
                M[r][c] -= factor * M[col][c]
            if verbose:
                print(f"row {r} -= {factor:.6g} * row {col}")


def gauss_jordan_solve(A, b, *, tol=1e-12, verbose=False):
    """Solve A x = b by reducing [A | b] to [I | x]."""
    M = _augmented(A, b, tol)
    _gauss_jordan(M, tol, verbose)
    return [row[-1] for row in M]


def determinant(A, *, tol=1e-12, verbose=False):
    """Compute det(A) by elimination, without normalizing pivot rows.

    Returns 0.0 if a pivot is below the absolute tolerance. Choose a
    smaller tolerance for small-scale, nonsingular matrices.
    """
    M = _square_copy(A, tol)
    try:
        swaps = _forward_eliminate(M, tol, verbose)
    except SingularMatrixError:
        return 0.0
    return (-1.0 if swaps % 2 else 1.0) * math.prod(
        M[i][i] for i in range(len(M))
    )


def inverse(A, *, tol=1e-12, verbose=False):
    """Compute A inverse by reducing the augmented matrix [A | I]."""
    M = _square_copy(A, tol)
    n = len(M)
    for i in range(n):
        M[i].extend(1.0 if i == j else 0.0 for j in range(n))
    _gauss_jordan(M, tol, verbose)
    return [row[n:] for row in M]
