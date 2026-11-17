"""Hand-written fixed-step ODE solvers for y' = f(x, y).

A scalar initial state uses f(x, float) -> float. A system uses
f(x, tuple) -> a sequence of derivatives in the same order as the state.
The step size h is a positive magnitude; the interval sets the direction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

State = float | tuple[float, ...]
RHS = Callable[[float, State], float | Sequence[float]]
Exact = Callable[[float], float | Sequence[float]]


@dataclass
class ODEResult:
    """Final state and a trace including the initial condition (step zero)."""

    method: str
    x: float
    y: State
    history: list[dict]

    @property
    def steps(self) -> int:
        return len(self.history) - 1

    @property
    def error(self) -> float | None:
        return self.history[-1]["error"]

    def table(self) -> str:
        def fmt(value):
            if value is None:
                return "-"
            if isinstance(value, tuple):
                return "(" + ", ".join(f"{v:.10g}" for v in value) + ")"
            return f"{value:.10g}"

        lines = [self.method, "step | x | h | numerical y | exact y | abs error"]
        for row in self.history:
            lines.append(" | ".join([str(row["i"])] + [fmt(row[k]) for k in ("x", "h", "y", "exact", "error")]))
        return "\n".join(lines)


def _vector(value, scalar: bool, size: int | None = None) -> tuple[float, ...]:
    """Validate and copy every state into an immutable internal vector."""
    try:
        result = (float(value),) if scalar else tuple(float(v) for v in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("expected a scalar or a matching vector of real numbers") from exc
    if not result or (size is not None and len(result) != size):
        raise ValueError("state and derivative dimensions must match and be nonempty")
    if not all(math.isfinite(v) for v in result):
        raise ValueError("ODE state or derivative became non-finite")
    return result


def _add(y, h, k):
    return tuple(a + h * b for a, b in zip(y, k))


def _euler(f, x, y, h):
    k1 = f(x, y)
    return _add(y, h, k1)


def _heun(f, x, y, h):
    k1 = f(x, y)
    prediction = _add(y, h, k1)
    k2 = f(x + h, prediction)
    return tuple(a + h / 2 * (b + c) for a, b, c in zip(y, k1, k2))


def _rk4(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, _add(y, h / 2, k1))
    k3 = f(x + h / 2, _add(y, h / 2, k2))
    k4 = f(x + h, _add(y, h, k3))
    return tuple(a + h / 6 * (b + 2*c + 2*d + e)
                 for a, b, c, d, e in zip(y, k1, k2, k3, k4))


def _solve(method, step, f, x0, y0, h, xend, exact, max_steps):
    x0, h, xend = float(x0), float(h), float(xend)
    if not all(math.isfinite(v) for v in (x0, h, xend)) or h <= 0:
        raise ValueError("x0 and xend must be finite; h must be finite and positive")
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    scalar = isinstance(y0, (int, float))
    y = _vector(y0, scalar)
    size = len(y)

    def public(state):
        return state[0] if scalar else state

    def rhs(x, state):
        state = _vector(state, False, size)
        return _vector(f(x, public(state)), scalar, size)

    history = []

    def record(i, x, state, step_h):
        reference = _vector(exact(x), scalar, size) if exact is not None else None
        # For a system, use the largest component-wise absolute error.
        error = max(abs(a-b) for a, b in zip(state, reference)) if reference is not None else None
        history.append({"i": i, "x": x, "h": step_h, "y": public(state),
                        "exact": public(reference) if reference is not None else None,
                        "error": error})

    x = x0
    record(0, x, y, 0.0)
    direction = 1 if xend >= x0 else -1
    for i in range(1, max_steps + 1):
        if x == xend:
            break
        # Use the original grid, not repeated x += h, to avoid accumulated drift.
        target = x0 + direction * i * h
        next_x = min(target, xend) if direction > 0 else max(target, xend)
        # Snap away a tiny endpoint roundoff, not a real short final step.
        endpoint_tol = min(h * 1e-12, 2 * max(math.ulp(x0), math.ulp(xend)))
        if abs(next_x - xend) <= endpoint_tol:
            next_x = xend
        actual_h = next_x - x
        if actual_h == 0:
            raise ValueError("h is too small to advance x in floating-point arithmetic")
        y = _vector(step(rhs, x, y, actual_h), False, size)
        x = next_x
        record(i, x, y, actual_h)
    if x != xend:
        raise ValueError("max_steps exceeded before reaching xend")
    return ODEResult(method, x, public(y), history)


def euler(f: RHS, x0: float, y0: State | Sequence[float], h: float,
          xend: float, *, exact: Exact | None = None, max_steps: int = 1_000_000) -> ODEResult:
    """First order: use one slope at the start of the step."""
    return _solve("euler", _euler, f, x0, y0, h, xend, exact, max_steps)


def modified_euler(f: RHS, x0: float, y0: State | Sequence[float], h: float,
                   xend: float, *, exact: Exact | None = None, max_steps: int = 1_000_000) -> ODEResult:
    """Second order (Heun): predict, then average the start/end slopes."""
    return _solve("modified euler", _heun, f, x0, y0, h, xend, exact, max_steps)


def rk4(f: RHS, x0: float, y0: State | Sequence[float], h: float,
        xend: float, *, exact: Exact | None = None, max_steps: int = 1_000_000) -> ODEResult:
    """Fourth order: weighted average of four slopes."""
    return _solve("rk4", _rk4, f, x0, y0, h, xend, exact, max_steps)
