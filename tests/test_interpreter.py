"""Independent numerical and command-grammar tests."""
import math
import pytest

from numlang.lang.interpreter import Interpreter, SceneConfig
from numlang.lang.lexer import LanguageError


@pytest.mark.parametrize('method', ['bisection', 'falsepos', 'newton'])
def test_root_methods(method):
    result = Interpreter().execute_line(f'root x^2 - 2 in 1 2 using {method}')
    record, = result.data['results']
    assert record['method'] == method
    assert record['result'].converged
    assert record['result'].root == pytest.approx(math.sqrt(2), abs=1e-9)
    assert record['residual'] < 1e-8
    assert result.data['derivative'] == '2*x'


def test_root_comparison_matches_reference():
    r = Interpreter().execute_line('compare root x^3 - 2x - 5 in 2 3')
    assert [v['result'].iterations for v in r.data['results']] == [34, 22, 5]
    assert all(v['result'].root == pytest.approx(2.0945514815423265, abs=1e-9) for v in r.data['results'])
    assert '3*x^2 - 2' in r.text


@pytest.mark.parametrize('method', ['bisection', 'falsepos', 'newton'])
def test_root_exact_endpoint(method):
    r = Interpreter().execute_line(f'root (x-1)^2 in 1 2 using {method}').data['results'][0]['result']
    assert r.root == 1 and r.converged and r.iterations == 0


def test_root_compare_reports_failure_without_hiding_newton():
    r = Interpreter().execute_line('root x^2 - 4 in 3 4')
    assert r.data['results'][0]['result'] is None
    assert r.data['results'][1]['result'] is None
    assert r.data['results'][2]['result'].root == pytest.approx(2)
    assert 'opposite signs' in r.text


@pytest.mark.parametrize('method', ['gauss', 'jordan', 'all'])
def test_solve_methods(method):
    r = Interpreter().execute_line(f'solve [2 1 -1; -3 -1 2; -2 1 2] = [8 -11 -3] using {method}')
    assert len(r.data['results']) == (2 if method == 'all' else 1)
    for v in r.data['results']:
        assert v['solution'] == pytest.approx([2, 3, -1])
        assert v['residual'] < 1e-12


@pytest.mark.parametrize('method,error_limit', [('euler', .15), ('modeuler', .005), ('rk4', 3e-6)])
def test_ode_methods(method, error_limit):
    r = Interpreter().execute_line(f"ode y' = y, y(0) = 1, h = .1, to 1 using {method} exact exp(x)")
    v = r.data['results'][0]['result']
    assert v.steps == 10
    assert v.y == pytest.approx(math.e, abs=error_limit)
    assert v.error == pytest.approx(abs(v.y-math.e))
    assert v.history[-1]['exact'] == pytest.approx(math.e)


def test_ode_comparison_and_non_autonomous_exact():
    r = Interpreter().execute_line("compare ode y' = x + y, y(0)=1, h=.1, to 1 exact 2exp(x)-x-1")
    values = [v['result'] for v in r.data['results']]
    assert [v.history[-1]['exact'] for v in values] == pytest.approx([2*math.e-2]*3)
    assert values[2].error < values[1].error < values[0].error


def test_ode_without_exact_and_backwards():
    r = Interpreter().execute_line("ode y' = 2x, y(1)=1, h=.3, to 0")
    v = r.data['results'][0]['result']
    assert v.y == pytest.approx(0, abs=1e-12)
    assert v.steps == 4 and v.error is None


def test_scene_configuration_and_units():
    interp = Interpreter()
    r, = interp.run_script('scene demo\n cloth 12 x 9 spacing 0.05\n stiffness 200\n wind 2\n integrator implicit\n h = 12 ms\nend')
    assert interp.scene == SceneConfig('demo', 12, 9, .05, 200, 2, 'implicit', .012)
    assert r.data['config'] == interp.scene
    interp.run_script('scene next\nh = .004 s\nend')
    assert interp.scene.h == .004


def test_trace_mode_and_comments():
    i = Interpreter()
    assert i.execute_line('# ignored') is None
    assert 'Trace on' == i.execute_line('trace on').text
    assert 'row operations' in i.execute_line('solve [0 1; 1 0] = [1 2]').text
    assert 'swap row' in i.execute_line('solve [0 1; 1 0] = [1 2]').text
    i.execute_line('trace off')
    assert 'row operations' not in i.execute_line('solve [0 1; 1 0] = [1 2]').text
    assert 'x_new' in i.execute_line('trace root x^2 - 2 in 1 2 using newton # comment').text
    assert 'step | x | h' in i.execute_line("trace compare ode y'=y, y(0)=1,h=.1,to .2").text


@pytest.mark.parametrize('command, message', [
    ('root x^2 + 1 in 0 1 using bisection', 'opposite signs'),
    ('root x in 2 1', 'a < b'),
    ('root y in -1 1', 'unknown name'),
    ('root x in 0 1 using secant', 'unknown method'),
    ('root x in 2', 'write root'),
    ('compare root x in -1 1 using newton', 'compare runs all'),
    ('solve [1 2; 3] = [1 2]', 'n x n'),
    ('solve [1 2; 2 4] = [1 2]', 'pivot'),
    ('solve [1oops] = [2]', 'expected a number'),
    ('solve [1e999] = [2]', 'finite'),
    ("ode y'=y, y(0)=1,h=0,to 1", 'positive'),
    ("ode y'=y, y(0)=1,h=.1,to 1 exact 2exp(x)", 'initial condition'),
    ("ode y'=y, y(0)=1,h=.1,to 1 using gauss", 'unknown method'),
    ('nonsense', 'unknown command'),
    ('end', 'unknown command'),
])
def test_line_numbered_errors(command, message):
    with pytest.raises(LanguageError) as err:
        Interpreter().execute_line(command, 7)
    assert str(err.value).startswith('line 7:')
    assert message in str(err.value)


@pytest.mark.parametrize('bad', ['cloth 0 x 9 spacing .05', 'h = 0 ms', 'stiffness 900', 'wind -1', 'integrator gauss', 'root x in -1 1'])
def test_invalid_scene_is_not_committed(bad):
    i = Interpreter()
    with pytest.raises(LanguageError, match='line 3:'):
        i.run_script(f'scene invalid\n stiffness 200\n {bad}\nend')
    assert not i.in_scene
    assert i.scene == SceneConfig()


def test_unclosed_scene():
    i = Interpreter()
    with pytest.raises(LanguageError, match="line 2: scene is missing 'end'"):
        i.run_script('# comment\nscene unfinished\nh=4 ms')
    assert not i.in_scene
