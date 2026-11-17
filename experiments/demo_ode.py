"""Analytical-vs-numerical comparison and convergence demonstration.

Run: uv run python experiments/demo_ode.py
Add --trace to print each time step of the three solvers.
"""
import math
import sys

from numlang.core.ode import euler, modified_euler, rk4


def main():
    solvers = [euler, modified_euler, rk4]
    f = lambda x, y: y
    print("ANALYTICAL VS NUMERICAL: y' = y, y(0) = 1, xend = 1")
    print(f"Exact solution: y(x) = exp(x), y(1) = {math.e:.12f}")
    print(f"{'Method':<18}{'h':>7}{'Numerical':>17}{'Exact':>17}{'Abs error':>15}")
    for solver in solvers:
        r = solver(f, 0, 1, .1, 1, exact=math.exp)
        print(f"{r.method:<18}{.1:7.3f}{r.y:17.10f}{math.e:17.10f}{r.error:15.6e}")
        if "--trace" in sys.argv:
            print(r.table())
    print("\nCONVERGENCE: HALVE h FROM 0.05 TO 0.025")
    print(f"{'Method':<18}{'Coarse error':>15}{'Fine error':>15}{'Ratio':>10}{'Order':>10}")
    for solver in solvers:
        a = solver(f, 0, 1, .05, 1, exact=math.exp)
        b = solver(f, 0, 1, .025, 1, exact=math.exp)
        ratio = a.error / b.error
        print(f"{a.method:<18}{a.error:15.6e}{b.error:15.6e}{ratio:10.3f}{math.log2(ratio):10.3f}")
    print("\nCOUPLED SYSTEM: position'=velocity, velocity'=-position")
    print("Initial position=1, velocity=0; t=1, h=0.05")
    exact = lambda t: (math.cos(t), -math.sin(t))
    print(f"Exact: position={exact(1)[0]:.9f}, velocity={exact(1)[1]:.9f}")
    for solver in solvers:
        r = solver(lambda t, y: (y[1], -y[0]), 0, [1, 0], .05, 1, exact=exact)
        print(f"{r.method:<18} position={r.y[0]:.9f}, velocity={r.y[1]:.9f}, error={r.error:.3e}")


if __name__ == "__main__":
    main()
