import math
import pytest
import sympy as sp
from numlang.lang import parse_expression
from numlang.core.roots import newton

x, y = sp.symbols('x y', real=True)


@pytest.mark.parametrize('source,reference,point', [
    ('x^3 - 2x - 5', x**3-2*x-5, 2.0),
    ('sin(x^2)+cos(3x)', sp.sin(x**2)+sp.cos(3*x), .7),
    ('tan(x)/sqrt(x+2)', sp.tan(x)/sp.sqrt(x+2), .3),
    ('exp(-x)*ln(x)+log(x)', sp.exp(-x)*sp.log(x)+sp.log(x), 1.3),
    ('abs(x)+atan(x)', sp.Abs(x)+sp.atan(x), -1.5),
    ('x^x', x**x, 2.4), ('2^x', 2**x, .7),
    ('(x+1)^(x-1)', (x+1)**(x-1), .6),
    ('x^-2', x**-2, -2.0), ('x^0', sp.Integer(1), 0.0),
    ('x^2', x**2, 0.0), ('pi+e', sp.pi+sp.E, .4),
    ('(x^2+1)/(x-2)', (x**2+1)/(x-2), .5),
    ('x^2*y + sin(y)', x**2*y+sp.sin(y), 1.7),
])
def test_derivatives_against_sympy(source, reference, point):
    expr = parse_expression(source)
    env = {'x': point, 'y': 1.2}
    subs = {x: point, y: 1.2}
    expected_value = float(reference.evalf(subs=subs))
    expected_derivative = float(sp.diff(reference, x).evalf(subs=subs))
    dual = expr.autodiff(**env)
    symbolic = expr.differentiate()
    assert dual.value == pytest.approx(expected_value)
    assert dual.derivative == pytest.approx(expected_derivative, abs=1e-12)
    assert symbolic.evaluate(**env) == pytest.approx(expected_derivative, abs=1e-12)
    assert parse_expression(str(symbolic)).evaluate(**env) == pytest.approx(expected_derivative, abs=1e-12)


def test_polynomial_derivative_text():
    assert str(parse_expression('x^3 - 2x - 5').differentiate()) == '3*x^2 - 2'


def test_partial_derivative_y():
    expr = parse_expression('x^2*y + sin(y)')
    expected = 4 + math.cos(1.2)
    assert expr.autodiff('y', x=2, y=1.2).derivative == pytest.approx(expected)
    assert expr.differentiate('y').evaluate(x=2, y=1.2) == pytest.approx(expected)


def test_newton_integration():
    expr = parse_expression('x^3 - 2x - 5')
    result = newton(lambda x: expr.evaluate(x=x), lambda x: expr.autodiff(x=x).derivative, 2)
    assert result.converged
    assert result.root == pytest.approx(2.0945514815423265, abs=1e-10)
    assert result.iterations == 5


def test_singular_composition_and_constant():
    assert parse_expression('sqrt(0)+abs(0)').autodiff(x=0).derivative == 0
    with pytest.raises(ValueError, match='undefined'):
        parse_expression('sqrt(x^2)').autodiff(x=0)
