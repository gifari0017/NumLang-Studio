"""Immutable expression tree, numeric evaluation and symbolic differentiation."""
from __future__ import annotations
from dataclasses import dataclass
import math
import operator
from typing import Mapping

from .dual import Dual, apply_function
from .lexer import LanguageError


class Expr:
    def evaluate(self, **variables: float) -> float:
        return float(_evaluate(self, variables))

    def autodiff(self, variable: str = "x", **variables: float) -> Dual:
        if variable not in ("x", "y"):
            raise LanguageError(f"unknown differentiation variable {variable!r}")
        env = {name: Dual(float(v), float(name == variable)) for name, v in variables.items()}
        return Dual.lift(_evaluate(self, env))

    def differentiate(self, variable: str = "x") -> Expr:
        if variable not in ("x", "y"):
            raise LanguageError(f"unknown differentiation variable {variable!r}")
        return simplify(_diff(self, variable))

    def __str__(self) -> str:
        return _format(self)


@dataclass(frozen=True)
class Number(Expr):
    value: float


@dataclass(frozen=True)
class Variable(Expr):
    name: str


@dataclass(frozen=True)
class Constant(Expr):
    name: str


@dataclass(frozen=True)
class Unary(Expr):
    operand: Expr


@dataclass(frozen=True)
class Binary(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Call(Expr):
    name: str
    arg: Expr


_OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


def _evaluate(node: Expr, env: Mapping[str, float | Dual]):
    if isinstance(node, Number): return node.value
    if isinstance(node, Constant): return math.pi if node.name == "pi" else math.e
    if isinstance(node, Variable):
        if node.name not in env:
            raise LanguageError(f"missing value for {node.name}")
        return env[node.name]
    if isinstance(node, Unary): return -_evaluate(node.operand, env)
    if isinstance(node, Call): return apply_function(node.name, _evaluate(node.arg, env))
    if isinstance(node, Binary):
        a, b = _evaluate(node.left, env), _evaluate(node.right, env)
        if node.op == "^":
            return a ** b if isinstance(a, Dual) or isinstance(b, Dual) else math.pow(a, b)
        return _OPS[node.op](a, b)
    raise TypeError("invalid expression node")


def _b(op, a, b): return Binary(op, a, b)


def _n(value): return Number(float(value))


def _is(node, value): return isinstance(node, Number) and node.value == value


def simplify(node: Expr) -> Expr:
    """Small algebraic simplifier; identities hold on the original domain."""
    if isinstance(node, Unary):
        a = simplify(node.operand)
        if isinstance(a, Number): return _n(-a.value)
        if isinstance(a, Unary): return a.operand
        return Unary(a)
    if isinstance(node, Call): return Call(node.name, simplify(node.arg))
    if not isinstance(node, Binary): return node
    a, b, op = simplify(node.left), simplify(node.right), node.op
    if isinstance(a, Number) and isinstance(b, Number):
        try:
            value = _evaluate(_b(op, a, b), {})
            if math.isfinite(value): return _n(value)
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
    if op == "+":
        if _is(a, 0): return b
        if _is(b, 0): return a
    elif op == "-":
        if _is(b, 0): return a
        if _is(a, 0): return simplify(Unary(b))
    elif op == "*":
        if _is(a, 0) or _is(b, 0): return _n(0)
        if _is(a, 1): return b
        if _is(b, 1): return a
    elif op == "/":
        if _is(a, 0): return _n(0)
        if _is(b, 1): return a
    elif op == "^":
        if _is(b, 0): return _n(1)
        if _is(b, 1): return a
    return _b(op, a, b)


def _diff(node: Expr, var: str) -> Expr:
    if isinstance(node, (Number, Constant)): return _n(0)
    if isinstance(node, Variable): return _n(node.name == var)
    if isinstance(node, Unary): return Unary(_diff(node.operand, var))
    if isinstance(node, Binary):
        u, v, op = node.left, node.right, node.op
        du, dv = simplify(_diff(u, var)), simplify(_diff(v, var))
        if op in ("+", "-"): return _b(op, du, dv)
        if op == "*": return _b("+", _b("*", du, v), _b("*", u, dv))
        if op == "/": return _b("/", _b("-", _b("*", du, v), _b("*", u, dv)), _b("^", v, _n(2)))
        if op == "^":
            if _is(dv, 0):
                if _is(du, 0) or _is(v, 0): return _n(0)
                return _b("*", _b("*", v, _b("^", u, _b("-", v, _n(1)))), du)
            # General power rule: (u^v)' = u^v * (v' ln(u) + v u'/u).
            return _b("*", node, _b("+", _b("*", dv, Call("ln", u)), _b("/", _b("*", v, du), u)))
    if isinstance(node, Call):
        u, name = node.arg, node.name
        du = simplify(_diff(u, var))
        if _is(du, 0): return _n(0)
        if name == "sin": factor = Call("cos", u)
        elif name == "cos": factor = Unary(Call("sin", u))
        elif name == "tan": factor = _b("/", _n(1), _b("^", Call("cos", u), _n(2)))
        elif name == "exp": factor = node
        elif name in ("ln", "log"): factor = _b("/", _n(1), u)
        elif name == "sqrt": factor = _b("/", _n(1), _b("*", _n(2), node))
        elif name == "abs": factor = _b("/", u, node)
        elif name == "atan": factor = _b("/", _n(1), _b("+", _n(1), _b("^", u, _n(2))))
        else: raise LanguageError(f"unknown function {name}")
        return _b("*", factor, du)
    raise TypeError("invalid expression node")


def _precedence(node):
    if isinstance(node, Binary): return {"+": 1, "-": 1, "*": 2, "/": 2, "^": 4}[node.op]
    if isinstance(node, Unary) or isinstance(node, Number) and node.value < 0: return 3
    return 5


def _format(node, minimum=0):
    """Print a tree as NumLang source while preserving operator grouping."""
    if isinstance(node, Number): text = repr(float(node.value)).removesuffix(".0")
    elif isinstance(node, (Variable, Constant)): text = node.name
    elif isinstance(node, Call): text = f"{node.name}({_format(node.arg)})"
    elif isinstance(node, Unary): text = "-" + _format(node.operand, 3)
    elif isinstance(node, Binary):
        p = _precedence(node)
        if node.op == "^":
            text = _format(node.left, 5) + "^" + _format(node.right, 3)
        else:
            sep = f" {node.op} " if node.op in ("+", "-") else node.op
            text = _format(node.left, p) + sep + _format(node.right, p + 1)
    else: raise TypeError("invalid expression node")
    return f"({text})" if _precedence(node) < minimum else text
