"""Root finding methods, written from scratch.

Three methods, one shared result type:

    bisection(f, a, b)        guaranteed, slow (linear convergence)
    false_position(f, a, b)   bracketing, usually faster, can stall on one side
    newton(f, df, x0)         fastest (quadratic), can diverge from a bad start

Every method records an iteration history so we can print tables,
plot convergence, and later drive the "trace mode" of NumLang.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

Func = Callable[[float], float]


@dataclass
class RootResult:
    """What every root finder returns."""

    method: str
    root: float
    iterations: int
    converged: bool
    message: str
    # one dict per iteration; keys differ per method but always include "i" and "error"
    history: list[dict] = field(default_factory=list)

    def table(self) -> str:
        """Plain-text iteration table (used in the REPL and the report)."""
        if not self.history:
            return f"{self.method}: no iterations"
        keys = list(self.history[0].keys())
        widths = {k: max(len(k), 14) for k in keys}
        widths["i"] = 4
        header = "  ".join(k.rjust(widths[k]) for k in keys)
        lines = [f"{self.method}", header, "-" * len(header)]
        for row in self.history:
            cells = []
            for k in keys:
                v = row[k]
                text = str(v) if k == "i" else f"{v:.10g}"
                cells.append(text.rjust(widths[k]))
            lines.append("  ".join(cells))
        lines.append(f"-> root = {self.root:.12g}, iterations = {self.iterations}, {self.message}")
        return "\n".join(lines)


def bisection(f: Func, a: float, b: float, tol: float = 1e-10, max_iter: int = 200) -> RootResult:
    """Bisection: halve the bracket [a, b] until it is smaller than tol.

    Needs f(a) and f(b) to have opposite signs. Then a root is guaranteed
    inside. Error after n steps is (b - a) / 2**n, so convergence is linear.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("bisection needs f(a) and f(b) with opposite signs")

    history: list[dict] = []
    c = a
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        error = (b - a) / 2
        history.append({"i": i, "a": a, "b": b, "c": c, "f(c)": fc, "error": error})
        if fc == 0 or error < tol:
            return RootResult("bisection", c, i, True, "converged", history)
        # keep the half that still contains the sign change
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return RootResult("bisection", c, max_iter, False, "max iterations reached", history)


def false_position(f: Func, a: float, b: float, tol: float = 1e-10, max_iter: int = 200) -> RootResult:
    """False position (regula falsi): cut the bracket where the secant line crosses zero.

    Same bracket rule as bisection, but the cut point uses the function values:
        c = (a*f(b) - b*f(a)) / (f(b) - f(a))
    Often faster than bisection. Weakness: one endpoint can stay fixed forever,
    so the bracket width never shrinks. That is why the stop test uses |c - c_prev|
    and not the bracket width.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("false position needs f(a) and f(b) with opposite signs")

    history: list[dict] = []
    c_prev = a
    c = a
    for i in range(1, max_iter + 1):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        error = abs(c - c_prev) if i > 1 else abs(b - a)
        history.append({"i": i, "a": a, "b": b, "c": c, "f(c)": fc, "error": error})
        if fc == 0 or (i > 1 and error < tol):
            return RootResult("false position", c, i, True, "converged", history)
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        c_prev = c
    return RootResult("false position", c, max_iter, False, "max iterations reached", history)


def newton(f: Func, df: Func, x0: float, tol: float = 1e-10, max_iter: int = 100) -> RootResult:
    """Newton-Raphson: follow the tangent line down to the x-axis.

        x_new = x - f(x) / f'(x)

    Quadratic convergence near a simple root (correct digits roughly double each step).
    Needs the derivative and a decent starting point. It can diverge (try atan, x0 = 2)
    or hit a zero derivative. We report those cases as converged = False.
    """
    history: list[dict] = []
    x = x0
    for i in range(1, max_iter + 1):
        fx, dfx = f(x), df(x)
        if not (math.isfinite(x) and math.isfinite(fx) and math.isfinite(dfx)):
            return RootResult("newton", x, i - 1, False, "diverged (non-finite value)", history)
        if dfx == 0:
            return RootResult("newton", x, i - 1, False, "zero derivative, cannot continue", history)
        x_new = x - fx / dfx
        error = abs(x_new - x)
        history.append({"i": i, "x": x, "f(x)": fx, "f'(x)": dfx, "x_new": x_new, "error": error})
        x = x_new
        if error < tol:
            return RootResult("newton", x, i, True, "converged", history)
    return RootResult("newton", x, max_iter, False, "max iterations reached", history)
