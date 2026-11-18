"""Independent checks of the browser-to-Python bridge."""
import json
from threading import Thread
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pytest
from numlang.app.server import Studio, make_server


def test_initial_state_and_root():
    d = Studio().action({'action': 'init'})
    s = d['state']
    assert (s['cols'], s['rows'], s['free_dofs']) == (13, 9, 220)
    assert len(s['positions']) == 117 and s['active_springs'] == 404
    assert [r['iterations'] for r in d['root']['results']] == [34, 22, 5]
    assert d['root']['results'][-1]['root'] == pytest.approx(2.0945514815423)
    assert d['root']['derivative'] == '3*x^2 - 2'


def test_fixed_steps_and_pause():
    s = Studio()
    s.advance(.003)
    assert s.cloth.t == 0
    s.advance(.007)
    assert s.cloth.t == pytest.approx(.008)
    s.action({'action': 'pause'})
    s.advance(.05)
    assert s.cloth.t == pytest.approx(.008)
    s.action({'action': 'pause'})
    s.advance(.004)
    assert s.cloth.t == pytest.approx(.012)


def test_live_switch_and_reset():
    s = Studio()
    s.advance(.02)
    s.configure('method', 'implicit')
    assert s.cloth.t == pytest.approx(.02)
    s.advance(.004)
    assert s.cloth.t == pytest.approx(.024)
    s.action({'action': 'reset'})
    assert s.cloth.t == 0 and s.cloth.method == 'implicit'


def test_drag_release_cut_reset():
    s = Studio()
    d = s.action({'action': 'drag', 'index': 14, 'x': .1, 'y': .2})['state']
    assert d['positions'][14] == [.1, .2] and d['free_dofs'] == 218
    assert s.action({'action': 'release'})['state']['free_dofs'] == 220
    d = s.action({'action': 'cut', 'index': d['springs'][0][0]})['state']
    assert d['active_springs'] == 403 and len(d['springs']) == 211
    assert s.action({'action': 'reset'})['state']['active_springs'] == 404


def test_script_real_solvers():
    d = Studio().action({'action': 'script', 'source': "root x^2 - 2 in 1 2 using newton\n"
                        "compare solve [2 1; 1 3] = [4 7]\n"
                        "compare ode y'=y, y(0)=1, h=.1, to 1 exact exp(x)"})
    assert d['root']['results'][0]['root'] == pytest.approx(2**.5)
    assert '[1, 2]' in d['output'] and '2.71827974414' in d['output']


def test_atomic_script_and_new_scene():
    s = Studio()
    old = s.cloth
    with pytest.raises(ValueError, match='line 4'):
        s.run_script('scene small\ncloth 3 x 3 spacing .05\nend\nnotacommand')
    assert s.cloth is old
    s.run_script('scene small\ncloth 3 x 3 spacing .05\nend')
    assert len(s.cloth.x) == 9


@pytest.mark.parametrize('key,value', [('h',0),('wind',13),('stiffness',float('nan')),('method','bad')])
def test_invalid_controls(key,value):
    with pytest.raises(ValueError):
        Studio().configure(key,value)


def test_diverged_state_serializable_and_recoverable():
    s = Studio()
    s.configure('method','euler')
    s.configure('h',.014)
    for _ in range(100):
        s.advance(.05)
        if s.cloth.exploded:
            break
    assert s.snapshot()['exploded']
    json.dumps(s.snapshot(), allow_nan=False)
    s.configure('method','rk4')
    assert not s.cloth.exploded


def test_http_routes_and_errors():
    server = make_server(0)
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(url,timeout=5) as r:
            html = r.read().decode()
            assert 'PYTHON ENGINE' in html and "fetch('/api'" in html
            assert 'function stepRK4' not in html and 'function newton(' not in html
        def post(action,origin=None):
            headers={'Content-Type':'application/json'}
            if origin:
                headers['Origin']=origin
            return urlopen(Request(url+'/api',data=json.dumps(action).encode(),headers=headers),timeout=5)
        with post({'action':'init'}) as r:
            assert json.load(r)['state']['cols']==13
        with pytest.raises(HTTPError) as e:
            post({'action':'script','source':'invalid'})
        assert e.value.code==400 and 'line 1' in json.load(e.value)['error']
        with pytest.raises(HTTPError) as e:
            post({'action':'reset'},'https://example.com')
        assert e.value.code==403
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=5)
