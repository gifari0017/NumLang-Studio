"""Linear systems A x = b, solved by hand (no numpy.linalg).

Step 1a: Gaussian elimination with partial pivoting.
"""

from __future__ import annotations


class SingularMatrixError(Exception):
    """Raised when no usable pivot exists: the matrix is singular (or too close)."""


def gauss_solve(A, b, *, tol=1e-12, verbose=False):
    """Solve A x = b by Gaussian elimination with partial pivoting.

    A: n x n list of lists. b: list of n numbers.
    Inputs are copied; the caller's data is never changed.
    Returns x as a list of n floats.
    """
    n = len(A)
    if n == 0 or any(len(row) != n for row in A) or len(b) != n:
        raise ValueError("A must be n x n and b must have length n")

    # Augmented matrix [A | b], copied.
    M = [[float(v) for v in A[i]] + [float(b[i])] for i in range(n)]

    # Forward elimination: make everything below the diagonal zero.
    for col in range(n):
        # Partial pivoting: pick the row with the largest |value| in this column.
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot_row][col]) < tol:
            raise SingularMatrixError(
                f"pivot in column {col} is {M[pivot_row][col]:.3e}: matrix is singular"
            )
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]
            if verbose:
                print(f"swap row {col} <-> row {pivot_row}")

        for r in range(col + 1, n):
            factor = M[r][col] / M[col][col]
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
            if verbose:
                print(f"row {r} -= {factor:.6g} * row {col}")

    # Back substitution: solve from the last row upward.
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n] - sum(M[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / M[r][r]
    return x
