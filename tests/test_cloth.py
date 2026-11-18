import math
import numpy as np
import pytest
from numlang.core.ode import euler, modified_euler, rk4
from numlang.lang.interpreter import Interpreter
from numlang.physics.cloth import Cloth, MASS, GRAVITY, GAMMA, DAMPING, METHODS, implicit_system, mechanical_energy, measure_stability


def test_topology_and_scene():
    i = Interpreter()
    c = Cloth.from_scene(i.scene)
    assert (c.cols,c.rows,len(c.x),len(c.edges),c.free_dofs) == (13,9,117,404,220)
    assert list(np.flatnonzero(c.pinned)) == [0,2,4,6,8,10,12]
    assert sum(c.weights==1) == 212 and sum(c.weights==.5) == 192
    assert len({tuple(sorted(e)) for e in c.edges}) == 404
    assert c.x[c.edges[:,0]].shape == (404,2)
    assert np.linalg.norm(c.x[c.edges[:,1]]-c.x[c.edges[:,0]],axis=1) == pytest.approx(c.rest)
    assert c.x[0] == pytest.approx([-.3,0])
    assert c.x[-1] == pytest.approx([.3,.4])
    assert i.run_script('scene other\n cloth 4 x 3 spacing .1\nend')[0].data['config'].cols == 4
    assert Cloth.from_scene(i.scene).x.shape == (12,2)


def test_initial_acceleration():
    c = Cloth()
    a = c.accelerations()
    assert a[c.pinned] == pytest.approx(np.zeros((7,2)))
    assert a[~c.pinned,0] == pytest.approx(0,abs=1e-10)
    assert a[~c.pinned,1] == pytest.approx(GRAVITY)


def test_hooke_force_and_damping():
    c=Cloth(2,2)
    c.alive[:]=False
    s=next(s for s,e in enumerate(c.edges) if tuple(e)==(2,3))
    c.alive[s]=True
    c.x[3,0]+=.01
    c.v[3,0]=.2
    force=c.stiffness*.01+DAMPING*.2
    a=c.accelerations()
    assert a[2,0] == pytest.approx(force/MASS)
    assert a[3,0] == pytest.approx(-force/MASS-GAMMA*.2)
    E=mechanical_energy(c.x,c.v,c.edges,c.rest,c.weights,c.alive,c.stiffness)
    assert E == pytest.approx(.5*MASS*.2**2+.5*c.stiffness*.01**2)


@pytest.mark.parametrize('method,solver', [('euler',euler),('modeuler',modified_euler),('rk4',rk4)])
def test_explicit_step_matches_independent_ode(method,solver):
    c=Cloth(2,2,method=method,h=.004)
    c.alive[:]=False
    # Exact model is a point mass under gravity and linear drag.
    f=lambda t,y:(y[1],GRAVITY-GAMMA*y[1])
    expected=solver(f,0,(.05,0.),.004,.004).y
    c.step()
    assert (c.x[2,1],c.v[2,1]) == pytest.approx(expected)
    c=Cloth(2,2,method=method,h=.001)
    c.alive[:]=False
    for _ in range(100): c.step()
    exact_v=GRAVITY/GAMMA*(1-math.exp(-GAMMA*.1))
    exact_y=.05+GRAVITY/GAMMA*(.1-(1-math.exp(-GAMMA*.1))/GAMMA)
    assert (c.x[2,1],c.v[2,1]) == pytest.approx((exact_y,exact_v),abs=.001)


def test_implicit_system_matches_numpy_reference():
    c=Cloth(3,2,method='implicit',h=.01)
    c.x[-1]+=[.002,.003]
    c.v[-1]=[.1,.2]
    A,b,idx=implicit_system(*c._args(c.fixed),c.h)
    assert A == pytest.approx(A.T,abs=1e-13)
    assert min(np.linalg.eigvalsh(A)) >= MASS
    oldx,oldv=c.x.copy(),c.v.copy()
    dv=np.linalg.solve(A,b)
    assert c.step()
    for i,p in enumerate(idx):
        if p>=0:
            expected=oldv[i]+dv[p:p+2]
            assert c.v[i] == pytest.approx(expected,abs=1e-12)
            assert c.x[i] == pytest.approx(oldx[i]+c.h*expected,abs=1e-12)
        else:
            assert c.x[i] == pytest.approx(oldx[i])


def test_implicit_free_fall_exact_discrete_step():
    c=Cloth(2,2,method='implicit',h=.033)
    c.alive[:]=False
    c.step()
    v=c.h*GRAVITY/(1+c.h*GAMMA)
    assert c.v[2,1] == pytest.approx(v)
    assert c.x[2,1] == pytest.approx(.05+c.h*v)


@pytest.mark.parametrize('method',METHODS)
def test_pins_drag_cut_reset(method):
    c=Cloth(4,3,method=method,h=.001)
    pins=c.x[c.pinned].copy()
    c.drag_node(5,.1,.2)
    c.cut_spring(2)
    for _ in range(5): assert c.step()
    assert c.x[5] == pytest.approx([.1,.2])
    assert c.v[5] == pytest.approx([0,0])
    assert c.x[c.pinned] == pytest.approx(pins)
    assert c.free_dofs==18 and not c.alive[2]
    c.release()
    assert c.free_dofs==20
    c.reset()
    assert c.t==0 and c.drag is None and c.alive.all() and not c.exploded


@pytest.mark.parametrize('method,h,diverged', [('euler',.001,True),('modeuler',.003,False),('modeuler',.004,True),('rk4',.010,False),('rk4',.014,True),('implicit',.033,False)])
def test_measured_stability(method,h,diverged):
    r=measure_stability(method,h,3.)
    assert r.diverged == diverged
    if diverged:
        assert r.duration < 3 and r.peak_energy>1e8
    else:
        assert r.duration == pytest.approx(3)
        assert r.peak_energy < 100


def test_nonfinite_and_reset():
    c=Cloth(method='euler')
    c.v[9,0]=1e9
    assert not c.step() and c.exploded
    t=c.t
    assert not c.step() and c.t==t
    c.reset()
    assert c.step()


def test_nan_detection():
    c=Cloth(method='rk4')
    c.v[9,0]=np.nan
    assert not c.step() and c.exploded


def test_implicit_assembly_against_finite_difference():
    # At rest, the clamped stiffness agrees with the true spring derivative.
    c=Cloth(3,3,method='implicit',h=.01)
    A,b,idx=implicit_system(*c._args(c.fixed),c.h)
    free=np.flatnonzero(~c.fixed)
    n=c.free_dofs
    Jx=np.zeros((n,n)); Jv=np.zeros((n,n)); eps=1e-6
    for j,i in enumerate(free):
        for d in range(2):
            c.x[i,d]+=eps
            plus=c.accelerations()[free].ravel()
            c.x[i,d]-=2*eps
            minus=c.accelerations()[free].ravel()
            c.x[i,d]+=eps
            Jx[:,2*j+d]=(plus-minus)/(2*eps)
            c.v[i,d]+=eps
            plus=c.accelerations()[free].ravel()
            c.v[i,d]-=2*eps
            minus=c.accelerations()[free].ravel()
            c.v[i,d]+=eps
            Jv[:,2*j+d]=(plus-minus)/(2*eps)
    expected=MASS*(np.eye(n)-c.h*Jv-c.h**2*Jx)
    assert A == pytest.approx(expected,abs=1e-9)
    assert b == pytest.approx(c.h*MASS*c.accelerations()[free].ravel(),abs=1e-10)
