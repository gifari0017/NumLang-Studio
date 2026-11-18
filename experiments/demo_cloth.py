"""Measure stability rather than hard-code the prototype's claims."""
import argparse
from time import perf_counter
from numlang.physics.cloth import Cloth, METHODS, measure_stability


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration',type=float,default=3.)
    args=parser.parse_args()
    c=Cloth()
    print(f'CLOTH: {c.cols} x {c.rows} | nodes={len(c.x)} | pins={c.pinned.sum()} | springs={len(c.edges)} | free DOFs={c.free_dofs}')
    print(f'k={c.stiffness:g} N/m | wind=0 | duration={args.duration:g} s | divergence: kinetic + elastic energy > 1e8 J or nonfinite')
    print('Compiling Numba kernels (first run may take longer)...',flush=True)
    for m in METHODS:
        Cloth(method=m).step()
    print(f'{"h (ms)":<9}' + ''.join(f'{m:>18}' for m in METHODS))
    for ms in (1,3,4,10,14,33):
        cells=[]
        for m in METHODS:
            r=measure_stability(m,ms/1000,args.duration)
            cells.append(f'DIVERGED {r.duration:.2f}s' if r.diverged else 'SURVIVED')
        print(f'{ms:<9}' + ''.join(f'{v:>18}' for v in cells),flush=True)
    print('\nBENCHMARK (after compilation; machine dependent)')
    for m in METHODS:
        c=Cloth(method=m)
        start=perf_counter()
        for _ in range(100):
            c.step()
            if c.exploded: c.reset()
        print(f'{m:<10}: {(perf_counter()-start)*10:.3f} ms/step')


if __name__=='__main__':
    main()
