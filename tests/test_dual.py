import math
import pytest
from numlang.lang.dual import Dual, apply_function


def test_dual_arithmetic():
    x = Dual(2, 1)
    f = (x*x + 3*x - 1)/(x + 1)
    assert (f.value, f.derivative) == pytest.approx((3, 4/3))
    assert (5-x).derivative == -1
    assert (1/x).derivative == pytest.approx(-0.25)
    assert (-x).derivative == -1


def test_dual_power():
    x = Dual(2, 1)
    assert (x**x).derivative == pytest.approx(4*(1+math.log(2)))
    assert (2**x).derivative == pytest.approx(4*math.log(2))
    assert (Dual(-2, 1)**3).derivative == 12
    assert (Dual(0, 1)**0).derivative == 0
    assert (Dual(0, 1)**2).derivative == 0


@pytest.mark.parametrize('name,value,derivative', [
    ('sin', math.sin(2), math.cos(2)), ('cos', math.cos(2), -math.sin(2)),
    ('tan', math.tan(2), 1/math.cos(2)**2), ('exp', math.exp(2), math.exp(2)),
    ('ln', math.log(2), .5), ('log', math.log(2), .5),
    ('sqrt', math.sqrt(2), 1/(2*math.sqrt(2))), ('abs', 2, 1),
    ('atan', math.atan(2), .2)])
def test_function_chain_rule(name, value, derivative):
    result = apply_function(name, Dual(2, 3))
    assert (result.value, result.derivative) == pytest.approx((value, 3*derivative))


@pytest.mark.parametrize('name', ['sqrt', 'abs'])
def test_undefined_derivative_at_zero(name):
    with pytest.raises(ValueError, match='undefined'):
        apply_function(name, Dual(0, 1))
