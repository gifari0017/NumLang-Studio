"""Generate reproducible PNG figures using the project solvers.

Run: uv run python -m numlang.app.plots
No external solver is used; Matplotlib only draws the measured data.
"""
import argparse
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numlang.core.roots import bisection, false_position, newton
from numlang.core.ode import euler, modified_euler, rk4
from numlang.physics.cloth import Cloth, METHODS, ENERGY_LIMIT

SOLVERS = (euler, modified_euler, rk4)
LABELS = ('Euler', 'Modified Euler', 'RK4')


def root_data():
    f = lambda x: x**3-2*x-5
    results = (bisection(f,2,3), false_position(f,2,3), newton(f,lambda x:3*x*x-2,2))
    # A common residual metric is comparable across the three stopping rules.
    return {r.method: [(row['i'],abs(f(row['x_new'] if r.method=='newton' else row['c'])))
                       for row in r.history] for r in results}


def ode_data():
    return {label: solver(lambda x,y:y,0,1,.1,1,exact=math.exp).history
            for label,solver in zip(LABELS,SOLVERS)}


def order_data():
    hs = [.2,.1,.05,.025,.0125]
    return hs, {label: [solver(lambda x,y:y,0,1,h,1,exact=math.exp).error for h in hs]
                for label,solver in zip(LABELS,SOLVERS)}


def energy_data(duration=3., h=.014):
    data = {}
    for method in METHODS:
        cloth = Cloth(method=method,h=h)
        series = [(0.,0.)]
        while cloth.t < duration-1e-12:
            ok = cloth.step(min(h,duration-cloth.t))
            if math.isfinite(cloth.energy):
                series.append((cloth.t,cloth.energy))
            if not ok:
                break
        data[method] = (series,cloth.exploded)
    return data


def generate(directory):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    paths = []
    def save(fig,name):
        path = directory / name
        fig.tight_layout()
        fig.savefig(path,dpi=240,bbox_inches='tight')
        plt.close(fig)
        paths.append(path)

    fig,ax = plt.subplots(figsize=(7,4.3))
    for name,series in root_data().items():
        # Zero residual cannot appear on a logarithmic axis; clip only for drawing.
        ax.semilogy([i for i,_ in series],[max(err,1e-16) for _,err in series],'.-',label=name.title())
    ax.set(xlabel='Iteration',ylabel='Absolute residual |f(x)|',title='Root-finding convergence: x³ − 2x − 5')
    ax.grid(True,which='both',alpha=.25); ax.legend()
    save(fig,'root_convergence.png')

    fig,ax = plt.subplots(figsize=(7,4.3))
    xs = [i/100 for i in range(101)]
    ax.plot(xs,[math.exp(x) for x in xs],'-',linewidth=2.5,label='Analytical exp(x)')
    for label,history in ode_data().items():
        ax.plot([r['x'] for r in history],[r['y'] for r in history],'o--',markersize=3,label=label)
    ax.set(xlabel='x',ylabel='y(x)',title="Analytical and numerical solutions: y′ = y, h = 0.1")
    ax.grid(alpha=.25); ax.legend()
    save(fig,'ode_comparison.png')

    fig,ax = plt.subplots(figsize=(7,4.3))
    hs,errors = order_data()
    for label,values in errors.items():
        ax.loglog(hs,values,'o-',label=label)
    ax.set(xlabel='Step size h',ylabel='Absolute error at x = 1',title='Global error and convergence order')
    ax.grid(True,which='both',alpha=.25); ax.legend()
    save(fig,'ode_error_order.png')

    fig,ax = plt.subplots(figsize=(7,4.3))
    for label,(series,diverged) in energy_data().items():
        ts,es = zip(*series)
        es = [max(e,1e-12) for e in es]  # log-axis display floor only
        line, = ax.semilogy(ts,es,label=label+(' (diverged)' if diverged else ' (survived)'))
        if diverged:
            ax.scatter(ts[-1],es[-1],marker='x',color=line.get_color())
    ax.axhline(ENERGY_LIMIT,linestyle='--',color='gray',label='Divergence threshold')
    ax.set(xlabel='Simulation time (s)',ylabel='Kinetic + elastic energy (J)',title='13 × 9 cloth stability: h = 14 ms, k = 150 N/m')
    ax.grid(True,which='both',alpha=.25); ax.legend(fontsize=8)
    save(fig,'cloth_energy.png')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='docs/figures')
    args = parser.parse_args()
    for path in generate(args.output):
        print(f'Saved {path}')


if __name__ == '__main__':
    main()
