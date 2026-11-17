import math
import pytest
from numlang.lang import LanguageError, parse_expression
from numlang.lang.ast import Binary, Number, Unary, Variable


@pytest.mark.parametrize('source,expected', [
    ('2+3*4', 14), ('(2+3)*4', 20), ('2^3^2', 512),
    ('-2^2', -4), ('(-2)^2', 4), ('2^-3', .125),
    ('8/4/2', 1), ('2x + 3(x+1)', 13), ('(x+1)(x-1)', 3),
    ('2sin(pi/2)', 2), ('2e', 2*math.e), ('1.5e-2 + .5E+1', 5.015),
    ('x+y', 5), ('+--x', 2), ('LOG(e)', 1), ('sqrt(0)', 0),
    ('1/2x', 1), ('abs(-x)', 2)])
def test_expressions(source, expected):
    expr = parse_expression(source)
    assert expr.evaluate(x=2, y=3) == pytest.approx(expected)
    assert parse_expression(str(expr)).evaluate(x=2, y=3) == pytest.approx(expected)


def test_ast_structure():
    assert parse_expression('-x^2 + 3') == Binary('+', Unary(Binary('^', Variable('x'), Number(2))), Number(3))


@pytest.mark.parametrize('source', ['', '2+', 'sin x', 'cos()', '3 4', 'x)', '(x+1', 'foo(x)', '1e999', 'x^^2', 'xy', 'x**2'])
def test_malformed_input(source):
    with pytest.raises(LanguageError, match='column'):
        parse_expression(source)


def test_variable_scope():
    with pytest.raises(LanguageError, match='unknown name'):
        parse_expression('x+y', variables=('x',))
    with pytest.raises(LanguageError, match='missing value for y'):
        parse_expression('x+y').evaluate(x=1)


def test_real_domain():
    with pytest.raises(ValueError): parse_expression('sqrt(-1)').evaluate()
    with pytest.raises(ValueError): parse_expression('(-1)^.5').evaluate()
    with pytest.raises(ZeroDivisionError): parse_expression('1/0').evaluate()
