"""Two-dimensional spring cloth. Positive y points down, as in the LCD.

NumPy provides array storage and arithmetic; all forces, integration stages,
Jacobian assembly and Gaussian elimination are implemented here by hand.
"""
from dataclasses import dataclass
import math
import numpy as np
from numba import njit
from numlang.core.linear import SingularMatrixError
from numlang.core.linear_fast import gauss_solve_array

MASS = 0.015
GRAVITY = 9.81
DAMPING = 0.03
GAMMA = 0.6
ENERGY_LIMIT = 1e8
METHODS = ('euler', 'modeuler', 'rk4', 'implicit')


@njit(cache=True)
def acceleration(x, v, fixed, edges, rest, weights, alive, stiffness, wind, t):
    a = np.zeros_like(x)
    for s in range(len(edges)):
        if not alive[s]:
            continue
        i, j = edges[s]
        dx = x[j, 0] - x[i, 0]
        dy = x[j, 1] - x[i, 1]
        L = math.hypot(dx, dy)
        if L < 1e-9:
            continue
        nx, ny = dx / L, dy / L
        dvx = v[j, 0] - v[i, 0]
        dvy = v[j, 1] - v[i, 1]
        force = weights[s] * (stiffness * (L - rest[s]) + DAMPING * (dvx*nx + dvy*ny))
        a[i, 0] += force * nx
        a[i, 1] += force * ny
        a[j, 0] -= force * nx
        a[j, 1] -= force * ny
    for i in range(len(x)):
        if fixed[i]:
            a[i, 0] = a[i, 1] = 0.0
        else:
            a[i, 0] = a[i, 0] / MASS + wind * (1 + .4*math.sin(2.2*t+3*x[i, 1])) - GAMMA*v[i, 0]
            a[i, 1] = a[i, 1] / MASS + GRAVITY - GAMMA*v[i, 1]
    return a


@njit(cache=True)
def explicit_step(x, v, fixed, edges, rest, weights, alive, stiffness, wind, t, h, method):
    a1 = acceleration(x, v, fixed, edges, rest, weights, alive, stiffness, wind, t)
    if method == 0:  # Euler: both slopes use the old state.
        return x + h*v, v + h*a1
    if method == 1:  # Heun predictor, then average slopes.
        xp, vp = x + h*v, v + h*a1
        a2 = acceleration(xp, vp, fixed, edges, rest, weights, alive, stiffness, wind, t+h)
        return x + h/2*(v+vp), v + h/2*(a1+a2)
    # RK4: position slopes are velocities; velocity slopes are accelerations.
    v2 = v + h/2*a1
    a2 = acceleration(x+h/2*v, v2, fixed, edges, rest, weights, alive, stiffness, wind, t+h/2)
    v3 = v + h/2*a2
    a3 = acceleration(x+h/2*v2, v3, fixed, edges, rest, weights, alive, stiffness, wind, t+h/2)
    v4 = v + h*a3
    a4 = acceleration(x+h*v3, v4, fixed, edges, rest, weights, alive, stiffness, wind, t+h)
    return x+h/6*(v+2*v2+2*v3+v4), v+h/6*(a1+2*a2+2*a3+a4)


@njit(cache=True)
def implicit_system(x, v, fixed, edges, rest, weights, alive, stiffness, wind, t, h):
    """Assemble (M+h*C+h²*K) dv = h*f0-h²*K*v.

    K is a positive stiffness approximation: compression's negative
    transverse stiffness is clamped, as in the original prototype.
    Damping directions and wind are frozen at the beginning of the step.
    """
    idx = np.full(len(x), -1, dtype=np.int64)
    n = 0
    for i in range(len(x)):
        if not fixed[i]:
            idx[i] = n
            n += 2
    A = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    acc = acceleration(x, v, fixed, edges, rest, weights, alive, stiffness, wind, t)
    for i in range(len(x)):
        p = idx[i]
        if p >= 0:
            for d in range(2):
                A[p+d, p+d] = MASS*(1+h*GAMMA)
                b[p+d] = h*MASS*acc[i, d]
    for s in range(len(edges)):
        if not alive[s]:
            continue
        i, j = edges[s]
        p, q = idx[i], idx[j]
        if p < 0 and q < 0:
            continue
        dx = x[j, 0] - x[i, 0]
        dy = x[j, 1] - x[i, 1]
        L = math.hypot(dx, dy)
        if L < 1e-9:
            continue
        nx, ny = dx/L, dy/L
        k, c = stiffness*weights[s], DAMPING*weights[s]
        tr = k*max(0., L-rest[s])/L
        # 2x2 local stiffness B and damping Q contributions.
        B = np.empty((2, 2)); Q = np.empty((2, 2))
        B[0, 0] = k*nx*nx + tr*(1-nx*nx)
        B[0, 1] = B[1, 0] = (k-tr)*nx*ny
        B[1, 1] = k*ny*ny + tr*(1-ny*ny)
        Q[0, 0] = h*c*nx*nx+h*h*B[0, 0]
        Q[0, 1] = Q[1, 0] = h*c*nx*ny+h*h*B[0, 1]
        Q[1, 1] = h*c*ny*ny+h*h*B[1, 1]
        for d in range(2):
            g = 0.0
            for e in range(2):
                g += h*h*B[d, e]*(v[j, e]-v[i, e])
                if p >= 0:
                    A[p+d, p+e] += Q[d, e]
                if q >= 0:
                    A[q+d, q+e] += Q[d, e]
                if p >= 0 and q >= 0:
                    A[p+d, q+e] -= Q[d, e]
                    A[q+d, p+e] -= Q[d, e]
            if p >= 0:
                b[p+d] += g
            if q >= 0:
                b[q+d] -= g
    return A, b, idx


@njit(cache=True)
def implicit_step(x, v, fixed, edges, rest, weights, alive, stiffness, wind, t, h):
    A, b, idx = implicit_system(x, v, fixed, edges, rest, weights, alive, stiffness, wind, t, h)
    xn, vn = x.copy(), v.copy()
    if len(b) == 0:
        return xn, vn
    dv = gauss_solve_array(A, b)
    for i in range(len(x)):
        p = idx[i]
        if p >= 0:
            for d in range(2):
                vn[i, d] += dv[p+d]
                xn[i, d] += h*vn[i, d]
    return xn, vn


@njit(cache=True)
def mechanical_energy(x, v, edges, rest, weights, alive, stiffness):
    """Kinetic plus elastic energy (prototype's nonnegative blow-up metric)."""
    E = 0.0
    for i in range(len(v)):
        E += .5*MASS*(v[i, 0]**2+v[i, 1]**2)
    for s in range(len(edges)):
        if alive[s]:
            i, j = edges[s]
            dx = x[j, 0]-x[i, 0]
            dy = x[j, 1]-x[i, 1]
            E += .5*stiffness*weights[s]*(math.hypot(dx, dy)-rest[s])**2
    return E


class Cloth:
    """Owns the state and topology. Reset restores the initial flat grid."""
    def __init__(self, cols=13, rows=9, spacing=.05, stiffness=150., wind=0., method='rk4', h=.004):
        if not isinstance(cols, int) or not isinstance(rows, int) or not (2 <= cols <= 32 and 2 <= rows <= 32):
            raise ValueError('cloth dimensions must be integers in 2..32')
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError('spacing must be positive and finite')
        self.cols, self.rows, self.spacing = cols, rows, spacing
        self.stiffness, self.wind, self.method, self.h = stiffness, wind, method, h
        self._validate()
        edges, rest, weights = [], [], []
        def add(i, j, diagonal=False):
            edges.append((i, j))
            rest.append(spacing*(math.sqrt(2) if diagonal else 1))
            weights.append(.5 if diagonal else 1.)
        for r in range(rows):
            for c in range(cols):
                i = r*cols+c
                if c+1 < cols:
                    add(i, i+1)
                if r+1 < rows:
                    add(i, i+cols)
                if r+1 < rows and c+1 < cols:
                    add(i, i+cols+1, True)
                    add(i+1, i+cols, True)
        self.edges = np.array(edges, dtype=np.int64)
        self.rest = np.array(rest, dtype=np.float64)
        self.weights = np.array(weights, dtype=np.float64)
        self.pinned = np.zeros(cols*rows, dtype=np.bool_)
        self.pinned[:cols:2] = True
        self.reset()

    @classmethod
    def from_scene(cls, scene):
        return cls(scene.cols, scene.rows, scene.spacing, scene.stiffness, scene.wind, scene.integrator, scene.h)

    def _validate(self):
        if self.method not in METHODS:
            raise ValueError('unknown cloth integrator')
        if not math.isfinite(self.h) or self.h <= 0:
            raise ValueError('step must be positive and finite')
        if not math.isfinite(self.stiffness) or self.stiffness <= 0 or not math.isfinite(self.wind):
            raise ValueError('invalid stiffness or wind')

    def reset(self):
        self.x = np.array([((c-(self.cols-1)/2)*self.spacing, r*self.spacing)
                           for r in range(self.rows) for c in range(self.cols)], dtype=np.float64)
        self.v = np.zeros_like(self.x)
        self.alive = np.ones(len(self.edges), dtype=np.bool_)
        self.drag = None
        self.t = 0.0
        self.exploded = False
        self.energy = 0.0
        self.reason = ''

    @property
    def fixed(self):
        fixed = self.pinned.copy()
        if self.drag is not None:
            fixed[self.drag] = True
        return fixed

    @property
    def free_dofs(self):
        return 2*(len(self.x)-int(self.fixed.sum()))

    def drag_node(self, index, x, y):
        if not 0 <= index < len(self.x) or self.pinned[index]:
            raise ValueError('choose a free node')
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('drag position must be finite')
        self.drag = index
        self.x[index] = x, y
        self.v[index] = 0.

    def release(self):
        self.drag = None

    def cut_spring(self, index):
        if not 0 <= index < len(self.edges):
            raise ValueError('invalid spring index')
        self.alive[index] = False

    def _args(self, fixed):
        return (self.x, self.v, fixed, self.edges, self.rest, self.weights,
                self.alive, float(self.stiffness), float(self.wind), self.t)

    def accelerations(self):
        return acceleration(*self._args(self.fixed))

    def step(self, h=None):
        """Advance one step; retain last finite display state after divergence."""
        if self.exploded:
            return False
        self._validate()
        h = self.h if h is None else float(h)
        if not math.isfinite(h) or h <= 0:
            raise ValueError('step must be positive and finite')
        fixed = self.fixed
        self.v[fixed] = 0.
        try:
            if self.method == 'implicit':
                xn, vn = implicit_step(*self._args(fixed), h)
            else:
                xn, vn = explicit_step(*self._args(fixed), h, METHODS.index(self.method))
        except (SingularMatrixError, OverflowError, FloatingPointError, ValueError) as exc:
            self.exploded = True
            self.reason = str(exc)
            return False
        self.t += h
        xn[fixed] = self.x[fixed]
        vn[fixed] = 0.
        E = mechanical_energy(xn, vn, self.edges, self.rest, self.weights, self.alive, self.stiffness)
        self.energy = float(E)
        if not math.isfinite(E) or E > ENERGY_LIMIT or not np.isfinite(xn).all() or not np.isfinite(vn).all():
            self.exploded = True
            self.reason = 'energy ran away'
            return False
        self.x, self.v = xn, vn
        return True


@dataclass(frozen=True)
class StabilityResult:
    method: str
    h: float
    duration: float
    diverged: bool
    peak_energy: float
    final_energy: float


def measure_stability(method, h, duration=3., **kwargs):
    """No threshold is preassigned to a method; run and measure it."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('duration must be positive and finite')
    c = Cloth(method=method, h=h, **kwargs)
    peak = 0.
    while c.t < duration - 1e-12:
        ok = c.step(min(h, duration-c.t))
        peak = max(peak, c.energy) if math.isfinite(c.energy) else math.inf
        if not ok:
            break
    return StabilityResult(method, h, c.t, c.exploded, peak, c.energy)
