"""Demonstrate linear solvers, pivoting and sensitivity without NumPy."""

from numlang.core.linear import determinant, gauss_jordan_solve, gauss_solve, inverse


def vector_norm_inf(v):
    return max(abs(value) for value in v)


def matrix_norm_inf(A):
    return max(sum(abs(value) for value in row) for row in A)


def relative_change(original, changed):
    return vector_norm_inf([b - a for a, b in zip(original, changed)]) / vector_norm_inf(original)


def show_vector(v):
    return "[" + ", ".join(f"{value:.8g}" for value in v) + "]"


def main():
    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]
    print("LINEAR SYSTEM: A x = b")
    print("Exact solution:    [2, 3, -1]")
    print("Gauss:            ", show_vector(gauss_solve(A, b)))
    print("Gauss-Jordan:     ", show_vector(gauss_jordan_solve(A, b)))
    print(f"Determinant:       {determinant(A):.8g}")
    print("Inverse:")
    for row in inverse(A):
        print("  " + show_vector(row))

    print("\nWHY PARTIAL PIVOTING MATTERS")
    A, b = [[1e-20, 1], [1, 1]], [1, 2]
    # Deliberately lower the tolerance so the unsafe method attempts the tiny pivot.
    print("With pivoting:    ", show_vector(gauss_solve(A, b, tol=1e-30)))
    print("Without pivoting: ", show_vector(gauss_solve(A, b, tol=1e-30, pivoting=False)))

    print("\nILL-CONDITIONED SYSTEM")
    eps = 1e-8
    A = [[1, 1], [1, 1 + eps]]
    b, changed = [2, 2 + eps], [2, 2 + 2 * eps]
    x, new_x = gauss_solve(A, b), gauss_solve(A, changed)
    cond = matrix_norm_inf(A) * matrix_norm_inf(inverse(A))
    print(f"Condition number (infinity norm): {cond:.3e}")
    print("Original solution:  ", show_vector(x))
    print("Perturbed solution: ", show_vector(new_x))
    print(f"Relative RHS change:      {relative_change(b, changed):.3e}")
    print(f"Relative solution change: {relative_change(x, new_x):.3e}")


if __name__ == "__main__":
    main()
