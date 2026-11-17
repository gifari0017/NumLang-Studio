"""Recursive-descent expression parser; no eval() or external parser."""
from __future__ import annotations
import math
from .ast import Binary, Call, Constant, Expr, Number, Unary, Variable
from .dual import FUNCTIONS
from .lexer import LanguageError, tokenize


class Parser:
    def __init__(self, source: str, variables=("x", "y")):
        self.tokens = tokenize(source)
        self.pos = 0
        self.variables = set(variables)

    @property
    def current(self): return self.tokens[self.pos]

    def accept(self, text):
        if self.current.kind == "op" and self.current.text == text:
            self.pos += 1
            return True
        return False

    def fail(self, message):
        raise LanguageError(f"column {self.current.column}: {message}")

    def need(self, text):
        if not self.accept(text): self.fail(f"expected {text!r}")

    def parse(self) -> Expr:
        result = self.expression()
        if self.current.kind != "end": self.fail(f"unexpected token {self.current.text!r}")
        return result

    def expression(self):
        left = self.term()
        while self.current.text in ("+", "-"):
            op = self.current.text
            self.pos += 1
            left = Binary(op, left, self.term())
        return left

    def term(self):
        left = self.unary()
        while True:
            if self.accept("*"): op = "*"
            elif self.accept("/"): op = "/"
            elif self.current.kind == "name" or self.current.text == "(": op = "*"
            else: break
            left = Binary(op, left, self.unary())
        return left

    def unary(self):
        if self.accept("-"): return Unary(self.unary())
        if self.accept("+"): return self.unary()
        return self.power()

    def power(self):
        base = self.atom()
        if self.accept("^"): return Binary("^", base, self.unary())
        return base

    def atom(self):
        token = self.current
        if token.kind == "number":
            value = float(token.text)
            if not math.isfinite(value): self.fail("number is not finite")
            self.pos += 1
            return Number(value)
        if token.kind == "name":
            if token.text not in self.variables and token.text not in ("pi", "e") and token.text not in FUNCTIONS:
                self.fail(f"unknown name {token.text!r}")
            self.pos += 1
            if token.text in self.variables: return Variable(token.text)
            if token.text in ("pi", "e"): return Constant(token.text)
            self.need("(")
            arg = self.expression()
            self.need(")")
            return Call(token.text, arg)
        if self.accept("("):
            result = self.expression()
            self.need(")")
            return result
        self.fail("expression ends too early" if token.kind == "end" else f"unexpected token {token.text!r}")


def parse_expression(source: str, variables=("x", "y")) -> Expr:
    """Compile source into a reusable expression tree."""
    try:
        return Parser(source, variables).parse()
    except RecursionError as exc:
        raise LanguageError("expression is nested too deeply") from exc
