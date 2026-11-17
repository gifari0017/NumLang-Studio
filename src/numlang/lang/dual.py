"""Forward-mode automatic differentiation over real numbers."""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Dual:
    value: float
    derivative: float = 0.0

    @staticmethod
    def lift(other):
        return other if isinstance(other, Dual) else Dual(float(other))

    def __add__(self, other):
        other = self.lift(other)
        return Dual(self.value + other.value, self.derivative + other.derivative)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.value, -self.derivative)

    def __sub__(self, other):
        return self + (-self.lift(other))

    def __rsub__(self, other):
        return self.lift(other) - self

    def __mul__(self, other):
        other = self.lift(other)
        u, v = self.value, other.value
        return Dual(u * v, self.derivative * v + u * other.derivative)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.lift(other)
        u, v = self.value, other.value
        q = u / v
        return Dual(q, (self.derivative - q * other.derivative) / v)

    def __rtruediv__(self, other):
        return self.lift(other) / self

    def __pow__(self, other):
        other = self.lift(other)
        u, v = self.value, other.value
        result = math.pow(u, v)
        if other.derivative == 0:
            # Constant exponent rule also works for negative bases and integers.
            d = 0.0 if v == 0 or self.derivative == 0 else v * math.pow(u, v - 1) * self.derivative
        else:
            if u <= 0:
                raise ValueError("a varying exponent requires a positive base")
            d = result * (other.derivative * math.log(u) + v * self.derivative / u)
        return Dual(result, d)

    def __rpow__(self, other):
        return self.lift(other) ** self


FUNCTIONS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "ln": math.log, "log": math.log,
    "sqrt": math.sqrt, "abs": abs, "atan": math.atan,
}


def apply_function(name: str, arg: float | Dual) -> float | Dual:
    """Apply the function and its chain rule if the argument is dual."""
    if not isinstance(arg, Dual):
        return FUNCTIONS[name](arg)
    u, d = arg.value, arg.derivative
    value = FUNCTIONS[name](u)
    # Reject singular elementary derivatives conservatively, even when
    # the incoming tangent vanishes (e.g. sqrt(x^2) at x=0).
    if name in ("sqrt", "abs") and u == 0:
        raise ValueError(f"{name} derivative is undefined at zero")
    if d == 0:
        return Dual(value, 0.0)
    if name == "sin": factor = math.cos(u)
    elif name == "cos": factor = -math.sin(u)
    elif name == "tan": factor = 1 / math.cos(u)**2
    elif name == "exp": factor = value
    elif name in ("ln", "log"): factor = 1 / u
    elif name == "sqrt": factor = 1 / (2 * value)
    elif name == "abs": factor = 1 if u > 0 else -1
    elif name == "atan": factor = 1 / (1 + u*u)
    else: raise ValueError(f"unknown function {name}")
    return Dual(value, factor * d)
