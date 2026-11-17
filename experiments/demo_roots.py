"""Demo: run all three root finders on one function and compare.

    uv run python experiments/demo_roots.py
"""

import math

from numlang.core.roots import bisection, false_position, newton

f = lambda x: x**3 - 2 * x - 5
df = lambda x: 3 * x**2 - 2

results = [
    bisection(f, 2, 3, tol=1e-10),
    false_position(f, 2, 3, tol=1e-10),
    newton(f, df, 2.0, tol=1e-10),
]

for r in results:
    print(r.table())
    print()

print("SUMMARY  f(x) = x^3 - 2x - 5")
print(f"{'method':<16}{'iterations':>11}{'root':>20}")
for r in results:
    print(f"{r.method:<16}{r.iterations:>11}{r.root:>20.12f}")

print("\nFAILURE: Newton on atan(x), start x0 = 2")
bad = newton(math.atan, lambda x: 1 / (1 + x * x), 2.0, max_iter=8)
print(bad.table())
