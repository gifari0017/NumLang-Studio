import math
import pytest
from numlang.app.plots import root_data, ode_data, order_data, energy_data, generate


def test_root_plot_data():
    data = root_data()
    assert [len(v) for v in data.values()] == [34,22,5]
    assert all(series[-1][1] < 1e-8 for series in data.values())


def test_ode_plot_data():
    data = ode_data()
    assert data['Euler'][-1]['y'] == pytest.approx(1.1**10)
    assert data['RK4'][-1]['exact'] == pytest.approx(math.e)
    hs,errors = order_data()
    for p,values in enumerate(errors.values()):
        expected_order = (1,2,4)[p]
        assert math.log2(values[-2]/values[-1]) == pytest.approx(expected_order,abs=.1)


def test_energy_plot_data():
    data = energy_data()
    assert [diverged for _,diverged in data.values()] == [True,True,True,False]
    series,_ = data['implicit']
    assert series[-1][0] == pytest.approx(3.)
    assert series[-1][1] < 100


def test_generate_pngs(tmp_path):
    paths = generate(tmp_path)
    assert len(paths) == 4
    for path in paths:
        assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert path.stat().st_size > 10000
