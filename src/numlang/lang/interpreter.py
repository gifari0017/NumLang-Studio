"""Execute NumLang commands by delegating to the hand-written numerical core.

The command grammar is line-oriented; mathematical expressions still use the
recursive-descent parser. No Python eval(), exec() or external solver is used.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass, field, replace
from io import StringIO
import math
import re

from numlang.core.linear import SingularMatrixError, gauss_solve, gauss_jordan_solve
from numlang.core.ode import euler, modified_euler, rk4
from numlang.core.roots import RootResult, bisection, false_position, newton
from .lexer import LanguageError
from .parser import parse_expression


@dataclass(frozen=True)
class SceneConfig:
    name: str = 'curtain'
    cols: int = 13
    rows: int = 9
    spacing: float = 0.05
    stiffness: float = 150.0
    wind: float = 0.0
    integrator: str = 'rk4'
    h: float = 0.004  # seconds, not milliseconds


@dataclass
class CommandResult:
    kind: str
    text: str
    data: dict = field(default_factory=dict)


ROOTS = ('bisection', 'falsepos', 'newton')
LINEAR = {'gauss': gauss_solve, 'jordan': gauss_jordan_solve}
ODES = {'euler': euler, 'modeuler': modified_euler, 'rk4': rk4}
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
ROOT_PATTERN = re.compile(rf'root\s+(.+?)\s+in\s+({NUMBER})\s+({NUMBER})(?:\s+using\s+(\w+))?', re.I)
SOLVE_PATTERN = re.compile(r'solve\s+\[([^\[\]]+)\]\s*=\s*\[([^\[\]]+)\](?:\s+using\s+(\w+))?', re.I)
ODE_PATTERN = re.compile(
    rf"ode\s+y'\s*=\s*(.+?),\s*y\s*\(\s*({NUMBER})\s*\)\s*=\s*({NUMBER})\s*,\s*h\s*=\s*({NUMBER})\s*,\s*to\s+({NUMBER})(?:\s+using\s+(\w+))?(?:\s+exact\s+(?:=\s*)?(.+))?", re.I)

HELP = """Commands (one per line; # starts a comment):
  root x^3 - 2x - 5 in 2 3 [using bisection|falsepos|newton|all]
  solve [2 1 -1; -3 -1 2; -2 1 2] = [8 -11 -3] [using gauss|jordan|all]
  ode y' = y, y(0) = 1, h = 0.1, to 1 [using euler|modeuler|rk4|all] [exact exp(x)]
  compare root ... | compare solve ... | compare ode ...
  trace <command> | trace on | trace off
  scene curtain ... end (multiline configuration)
  help | quit (REPL only)
Defaults: root all, solve gauss, ode rk4. Newton starts at the first endpoint.
"""


def _number(s):
    if re.fullmatch(NUMBER, s) is None:
        raise LanguageError(f'expected a number, got {s!r}')
    value = float(s)
    if not math.isfinite(value):
        raise LanguageError('number must be finite')
    return value


def _fmt(v):
    return '-' if v is None else f'{v:.12g}'


def _methods(selected, choices, default, compare):
    chosen = (selected or ('all' if compare else default)).lower()
    if chosen != 'all' and chosen not in choices:
        raise LanguageError(f'unknown method {chosen!r}; choose {", ".join(choices)} or all')
    if compare and chosen != 'all':
        raise LanguageError('compare runs all methods; omit using or write using all')
    return list(choices) if chosen == 'all' else [chosen]


def _root(match, compare, trace):
    source, a, b, method = match.groups()
    a, b = _number(a), _number(b)
    methods = _methods(method, ROOTS, 'all', compare)
    if a >= b:
        raise LanguageError('root interval must satisfy a < b')
    expr = parse_expression(source, variables=('x',))
    derivative = str(expr.differentiate())

    def f(x):
        value = expr.evaluate(x=x)
        if not math.isfinite(value):
            raise ValueError('function value is not finite')
        return value

    def df(x):
        return expr.autodiff(x=x).derivative

    lines = [f'ROOT: f(x) = {expr}', f"f'(x) = {derivative}",
             'Method           Root             Iterations  Status / residual']
    records = []
    traces = []
    for key in methods:
        label = 'false position' if key == 'falsepos' else key
        try:
            # Handle exact endpoints before core bracketing algorithms.
            fa = f(a)
            fb = f(b) if key != 'newton' else None
            if fa == 0 or fb == 0:
                r = RootResult(label, a if fa == 0 else b, 0, True, 'exact endpoint')
            elif key == 'newton':
                r = newton(f, df, a)
            else:
                if (fa > 0) == (fb > 0):
                    raise ValueError('f(a) and f(b) need opposite signs')
                r = (bisection if key == 'bisection' else false_position)(f, a, b)
            try:
                residual = abs(f(r.root))
            except (ValueError, ArithmeticError):
                residual = None
            status = 'converged' if r.converged else r.message
            lines.append(f'{label:<17}{_fmt(r.root):<17}{r.iterations:<12}{status}; |f|={_fmt(residual)}')
            records.append({'method': key, 'result': r, 'residual': residual})
            if trace:
                traces.append(r.table())
        except (ValueError, ArithmeticError, RecursionError) as exc:
            if len(methods) == 1:
                raise
            lines.append(f'{label:<17}{"-":<17}{"-":<12}failed: {exc}')
            records.append({'method': key, 'result': None, 'error': str(exc)})
    return CommandResult('root', '\n'.join(lines + traces),
                         {'expression': source, 'derivative': derivative, 'a': a, 'b': b, 'results': records})


def _row(source):
    # Matrix values are whitespace-separated numbers, not expressions.
    values = source.split()
    if not values:
        raise LanguageError('matrix rows must not be empty')
    return [_number(s) for s in values]


def _solve(match, compare, trace):
    matrix, rhs, method = match.groups()
    A = [_row(row) for row in matrix.split(';')]
    b = _row(rhs)
    methods = _methods(method, LINEAR, 'gauss', compare)
    lines = ['LINEAR SYSTEM: A x = b', 'Method           Solution                     Max residual']
    records = []
    traces = []
    for key in methods:
        # Capture the existing teaching solver's printed elementary row operations.
        buff = StringIO()
        with redirect_stdout(buff):
            x = LINEAR[key](A, b, verbose=trace)
        residual = max(abs(sum(a*v for a, v in zip(row, x))-bi) for row, bi in zip(A, b))
        if not all(math.isfinite(v) for v in (*x, residual)):
            raise LanguageError('linear solver produced a non-finite result')
        solution = '[' + ', '.join(_fmt(v) for v in x) + ']'
        lines.append(f'{key:<17}{solution:<29}{_fmt(residual)}')
        records.append({'method': key, 'solution': x, 'residual': residual})
        if trace:
            traces.append(f'{key} row operations:\n{buff.getvalue().rstrip()}')
    return CommandResult('solve', '\n'.join(lines + traces), {'results': records})


def _ode(match, compare, trace):
    source, x0, y0, h, xend, method, exact_source = match.groups()
    x0, y0, h, xend = map(_number, (x0, y0, h, xend))
    methods = _methods(method, ODES, 'rk4', compare)
    expr = parse_expression(source)
    reference = parse_expression(exact_source, variables=('x',)) if exact_source else None
    f = lambda x, y: expr.evaluate(x=x, y=y)
    exact = (lambda x: reference.evaluate(x=x)) if reference is not None else None
    lines = [f"ODE: y' = {expr}, y({x0:g}) = {y0:g}, h = {h:g}, to {xend:g}"]
    if reference is not None:
        lines.append(f'Exact solution supplied: y = {reference}')
        if not math.isclose(exact(x0), y0, rel_tol=1e-9, abs_tol=1e-9):
            raise LanguageError('exact solution does not match the initial condition')
    lines.append('Method           Steps    Numerical y       Exact y          Abs error')
    records = []
    traces = []
    for key in methods:
        r = ODES[key](f, x0, y0, h, xend, exact=exact, max_steps=100_000)
        lines.append(f'{key:<17}{r.steps:<9}{_fmt(r.y):<18}{_fmt(r.history[-1]["exact"]):<17}{_fmt(r.error)}')
        records.append({'method': key, 'result': r})
        if trace:
            traces.append(r.table())
    return CommandResult('ode', '\n'.join(lines + traces), {'results': records})


class Interpreter:
    """Stateful command interpreter; scene blocks commit only at 'end'."""
    def __init__(self):
        self.scene = SceneConfig()
        self.trace = False
        self._pending = None
        self._scene_line = 0

    @property
    def in_scene(self):
        return self._pending is not None

    def abort_scene(self):
        self._pending = None

    def _scene_command(self, line):
        if line.lower() == 'end':
            self.scene = self._pending
            self.abort_scene()
            c = self.scene
            return CommandResult('scene', f'SCENE {c.name}: cloth {c.cols} x {c.rows}, spacing {c.spacing:g}, '
                                 f'k={c.stiffness:g}, wind={c.wind:g}, {c.integrator}, h={c.h*1000:g} ms',
                                 {'config': c})
        m = re.fullmatch(rf'cloth\s+(\d+)\s*x\s*(\d+)\s+spacing\s+({NUMBER})', line, re.I)
        if m:
            cols, rows = map(int, m.groups()[:2])
            spacing = _number(m[3])
            if not (2 <= cols <= 32 and 2 <= rows <= 32 and spacing > 0):
                raise LanguageError('cloth dimensions must be 2..32 and spacing positive')
            self._pending = replace(self._pending, cols=cols, rows=rows, spacing=spacing)
            return None
        m = re.fullmatch(r'(stiffness|wind|integrator)\s+(\S+)', line, re.I)
        if m:
            key, raw = m[1].lower(), m[2]
            if key == 'integrator':
                value = raw.lower()
                if value not in (*ODES, 'implicit'):
                    raise LanguageError('unknown cloth integrator; choose euler, modeuler, rk4, implicit')
            else:
                value = _number(raw)
                lo, hi = (40, 600) if key == 'stiffness' else (0, 12)
                if not lo <= value <= hi:
                    raise LanguageError(f'{key} must be in {lo}..{hi}')
            self._pending = replace(self._pending, **{key: value})
            return None
        m = re.fullmatch(rf'h\s*=\s*({NUMBER})\s*(ms|s)?', line, re.I)
        if m:
            value = _number(m[1]) / (1 if (m[2] or 'ms').lower() == 's' else 1000)
            if not 0.001 <= value <= 0.033:
                raise LanguageError('scene h must be between 1 and 33 ms')
            self._pending = replace(self._pending, h=value)
            return None
        raise LanguageError('inside a scene use cloth, stiffness, wind, integrator, h or end')

    def execute_line(self, raw: str, line_number: int = 1) -> CommandResult | None:
        """Execute one line; all expected user errors have a source line number."""
        line = raw.split('#', 1)[0].strip()
        if not line:
            return None
        try:
            if self.in_scene:
                return self._scene_command(line)
            m = re.fullmatch(r'scene\s+([A-Za-z_][A-Za-z_0-9]*)', line, re.I)
            if m:
                self._pending = SceneConfig(name=m[1])
                self._scene_line = line_number
                return None
            if line.lower() == 'help':
                return CommandResult('help', HELP)
            m = re.fullmatch(r'trace\s+(on|off)', line, re.I)
            if m:
                self.trace = m[1].lower() == 'on'
                return CommandResult('trace', f'Trace {m[1].lower()}')
            trace, compare = self.trace, False
            # Prefixes can be combined in either order.
            for _ in range(2):
                m = re.match(r'(trace|compare)\s+', line, re.I)
                if not m:
                    break
                if m[1].lower() == 'trace':
                    trace = True
                else:
                    compare = True
                line = line[m.end():]
            for pattern, handler in ((ROOT_PATTERN, _root), (SOLVE_PATTERN, _solve), (ODE_PATTERN, _ode)):
                match = pattern.fullmatch(line)
                if match:
                    return handler(match, compare, trace)
            command = line.split()[0].lower()
            usage = {'root': 'root <expression> in <a> <b> [using method]',
                     'solve': 'solve [matrix rows; ...] = [rhs] [using method]',
                     'ode': "ode y' = <expression>, y(<x0>) = <y0>, h = <h>, to <xend> [using method] [exact expression]"}
            if command in usage:
                raise LanguageError('write ' + usage[command])
            raise LanguageError(f'unknown command {command!r}; type help')
        except (ValueError, ArithmeticError, SingularMatrixError, RecursionError) as exc:
            self.abort_scene()
            raise LanguageError(f'line {line_number}: {exc}') from exc

    def finish(self):
        if self.in_scene:
            self.abort_scene()
            raise LanguageError(f"line {self._scene_line}: scene is missing 'end'")

    def run_script(self, source: str) -> list[CommandResult]:
        results = []
        for i, line in enumerate(source.splitlines(), 1):
            r = self.execute_line(line, i)
            if r is not None:
                results.append(r)
        self.finish()
        return results
