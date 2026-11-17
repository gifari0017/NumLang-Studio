"""Demonstrate the expression pipeline and automatic Newton derivative."""
from numlang.lang import parse_expression, tokenize
from numlang.core.roots import newton


def main():
    source = 'x^3 - 2x - 5'
    expr = parse_expression(source, variables=('x',))
    dual = expr.autodiff(x=2)
    print('SOURCE:        ', source)
    print('TOKENS:        ', ' | '.join(t.text for t in tokenize(source) if t.kind != 'end'))
    print('PARSED:        ', expr)
    print('SYMBOLIC f\'(x):', expr.differentiate())
    print(f'AT x=2: f(x)={dual.value:g}, f\'(x)={dual.derivative:g}')
    result = newton(lambda x: expr.evaluate(x=x), lambda x: expr.autodiff(x=x).derivative, 2)
    print(f'NEWTON: root={result.root:.10f}, iterations={result.iterations}, converged={result.converged}')
    print('\nNewton iteration trace:')
    print(result.table())
    ode_expr = parse_expression('x + y')
    print(f'\nODE expression x + y at x=2, y=3: {ode_expr.evaluate(x=2, y=3):g}')


if __name__ == '__main__':
    main()
