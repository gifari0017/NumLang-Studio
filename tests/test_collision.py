import json
import pytest
from numlang.physics.collision import floor_impact
from numlang.physics.cloth import Cloth, METHODS
from numlang.app.server import Studio


def test_crossing_and_rebound_exact_linear_path():
    hit = floor_impact(.3, .5, 2., 2., .1, .4)
    assert hit.time == pytest.approx(.05, abs=1e-10)
    assert hit.vy == pytest.approx(-.4)
    assert hit.y == pytest.approx(.38, abs=1e-10)
    assert hit.bisection_iterations >= 1 and hit.newton_iterations >= 1


def test_no_contact_and_endpoint():
    assert floor_impact(.2, .3, 1, 1, .1, .4) is None
    hit = floor_impact(.3, .4, 1, 1, .1, .4)
    assert (hit.time, hit.y, hit.vy) == pytest.approx((.1,.4,-.2))
    assert floor_impact(.4, .4, 0, 0, .1, .4) is None


def test_existing_penetration_and_away_velocity():
    hit = floor_impact(.42,.43,-1,-1,.01,.4)
    assert (hit.time,hit.vy,hit.y) == pytest.approx((0,-1,.39))


@pytest.mark.parametrize('h,e', [(0,.2),(.1,1.1),(.1,float('nan'))])
def test_invalid_contact_parameters(h,e):
    with pytest.raises(ValueError):
        floor_impact(.3,.5,2,2,h,.4,e)


@pytest.mark.parametrize('method', METHODS)
def test_floor_in_all_integrators(method):
    c = Cloth(2,2,method=method,h=.004)
    c.alive[:] = False
    c.set_floor(True)
    pins = c.x[c.pinned].copy()
    c.x[2,1] = c.floor_y-.001
    c.v[2,1] = 1.
    for _ in range(100):
        assert c.step()
        assert c.x[:,1].max() <= c.floor_y+1e-12
    assert c.contacts > 0
    assert c.x[c.pinned] == pytest.approx(pins)
    assert c.last_impact.newton_iterations >= 1


def test_floor_activation_drag_reset_and_disable():
    c = Cloth(2,2)
    c.drag_node(2,0,1)
    assert c.x[2,1] == 1
    c.set_floor(True)
    assert c.x[2,1] == c.floor_y
    c.drag_node(2,0,1)
    assert c.x[2,1] == c.floor_y
    c.reset()
    assert c.floor_enabled and c.contacts == 0 and c.last_impact is None
    c.set_floor(False)
    c.drag_node(2,0,1)
    assert c.x[2,1] == 1


def test_backend_floor_state_and_live_contact():
    s = Studio()
    assert not s.snapshot()['floor_enabled']
    s.action({'action':'floor'})
    for _ in range(20):
        result = s.action({'action':'frame','dt':.05})
    assert result['state']['contacts'] == 0
    # Use the same drag/release requests as the browser. Pull the bottom
    # centre sideways and down; neighbouring free nodes can hit the floor.
    i = (s.cloth.rows-1)*s.cloth.cols + s.cloth.cols//2
    s.action({'action':'drag','index':i,'x':.4,'y':s.cloth.floor_y})
    for _ in range(10):
        s.action({'action':'frame','dt':.05})
    s.action({'action':'release'})
    for _ in range(20):
        state = s.action({'action':'frame','dt':.05})['state']
    assert not state['exploded']
    assert state['floor_enabled'] and state['contacts'] > 0
    assert state['last_impact']['newton_iterations'] > 0
    json.dumps(state,allow_nan=False)
    assert not s.action({'action':'floor'})['state']['floor_enabled']


@pytest.mark.parametrize('method,h', [('modeuler',.003),('rk4',.004),('implicit',.033)])
def test_idle_default_cloth_does_not_hit_floor(method,h):
    c = Cloth(method=method,h=h)
    c.set_floor(True)
    assert c.floor_y - (c.rows-1)*c.spacing == pytest.approx(.10)
    while c.t < 5.:
        assert c.step()
    assert c.contacts == 0 and c.last_impact is None
    assert c.x[:,1].max() < c.floor_y


@pytest.mark.parametrize('method', METHODS)
def test_resting_nodes_do_not_generate_hits(method):
    c = Cloth(2,2,method=method,h=.004)
    c.alive[:] = False
    c.x[~c.pinned,1] = c.floor_y
    c.set_floor(True)
    for _ in range(500):
        assert c.step()
    assert c.contacts == 0 and c.last_impact is None
    assert c.x[~c.pinned,1] == pytest.approx(c.floor_y)
    assert c.v[~c.pinned,1] == pytest.approx(0.)


def test_real_impact_recorded_then_rest_then_reimpact():
    c = Cloth(2,2,method='rk4',h=.004)
    c.alive[:] = False
    c.x[~c.pinned,1] = c.floor_y
    c.set_floor(True)
    i = 2
    c.drag_node(i,0,c.floor_y-.002)
    c.release()
    c.v[i,1] = .6
    assert c.step()
    assert c.contacts == 1
    assert c.last_impact.bisection_iterations > 0
    first = c.last_impact
    for _ in range(500):
        assert c.step()
    # A real rebound can cause another impact, but resting cannot.
    settled_count, settled_hit = c.contacts,c.last_impact
    assert settled_hit is not None and first.time > 0
    for _ in range(500):
        assert c.step()
    assert c.contacts == settled_count and c.last_impact is settled_hit
    c.drag_node(i,0,c.floor_y-.002)
    c.release()
    c.v[i,1] = .6
    assert c.step()
    assert c.contacts == settled_count+1
    assert c.last_impact is not settled_hit


def test_drag_to_floor_is_not_an_impact():
    c = Cloth(2,2,method='rk4')
    c.alive[:] = False
    c.x[~c.pinned,1] = c.floor_y
    c.set_floor(True)
    c.drag_node(2,0,c.floor_y+.05)
    assert c.x[2,1] == c.floor_y
    c.release()
    for _ in range(100):
        assert c.step()
    assert c.contacts == 0
