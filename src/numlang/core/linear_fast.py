"""Numba implementation of our own partial-pivot Gaussian elimination."""
import numpy as np
from numba import njit
from .linear import SingularMatrixError


@njit(cache=True)
def gauss_solve_array(A, b, tol=1e-12):
    """Solve a dense float64 array system without altering A or b."""
    n = A.shape[0]
    if n == 0 or A.shape[1] != n or b.shape[0] != n:
        raise ValueError('expected square A and matching b')
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError('positive finite tolerance required')
    U = A.astype(np.float64)
    rhs = b.astype(np.float64)
    for i in range(n):
        if not np.isfinite(rhs[i]):
            raise ValueError('nonfinite RHS')
        for j in range(n):
            if not np.isfinite(U[i, j]):
                raise ValueError('nonfinite matrix')
    for c in range(n):
        p = c
        for r in range(c + 1, n):
            if abs(U[r, c]) > abs(U[p, c]):
                p = r
        if abs(U[p, c]) < tol:
            raise SingularMatrixError('no usable pivot')
        if p != c:
            for j in range(c, n):
                U[c, j], U[p, j] = U[p, j], U[c, j]
            rhs[c], rhs[p] = rhs[p], rhs[c]
        for r in range(c + 1, n):
            factor = U[r, c] / U[c, c]
            if factor == 0:
                continue
            U[r, c] = 0
            for j in range(c + 1, n):
                U[r, j] -= factor * U[c, j]
            rhs[r] -= factor * rhs[c]
    x = np.empty(n, dtype=np.float64)
    for r in range(n - 1, -1, -1):
        s = rhs[r]
        for c in range(r + 1, n):
            s -= U[r, c] * x[c]
        x[r] = s / U[r, r]
    return x
