"""Local Instrument bridge. The browser never integrates or solves equations."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import math
from pathlib import Path
import webbrowser
from threading import Lock

from numlang.lang.interpreter import Interpreter
from numlang.physics.cloth import Cloth, METHODS

DEFAULT_SCRIPT = '''scene curtain
  cloth 13 x 9 spacing 0.05
  stiffness 150
  wind 0
  integrator rk4
  h = 4 ms
end

root x^3 - 2x - 5 in 2 3
'''
HTML = Path(__file__).with_name('instrument.html')
MAX_BODY = 65536


def number(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'expected a finite number between {low} and {high}')
    return float(value)


def finite(value):
    return float(value) if math.isfinite(value) else None


class Studio:
    """The sole owner of simulation state; requests are processed sequentially."""
    def __init__(self):
        self.cloth = Cloth()
        self.paused = False
        self.remainder = 0.0
        self.root = None
        self.output = ''
        self.run_script(DEFAULT_SCRIPT)

    def run_script(self, source):
        if not isinstance(source, str) or len(source) > 16000:
            raise ValueError('script must be text of at most 16000 characters')
        # Run against a fresh interpreter: a failing script leaves the live scene unchanged.
        results = Interpreter().run_script(source)
        next_cloth = None
        next_root = None
        for result in results:
            if result.kind == 'scene':
                config = result.data['config']
                if not .0001 <= config.spacing <= 1:
                    raise ValueError('Instrument scene spacing must be 0.0001..1 metre')
                next_cloth = Cloth.from_scene(config)
            elif result.kind == 'root':
                data = result.data
                records = []
                for row in data['results']:
                    r = row['result']
                    records.append({'method': row['method'], 'ok': bool(r and r.converged),
                                    'root': finite(r.root) if r else None,
                                    'iterations': r.iterations if r else 0,
                                    'message': r.message if r else row.get('error', 'failed')})
                next_root = {'expression': data['expression'], 'derivative': data['derivative'],
                             'a': data['a'], 'b': data['b'], 'results': records}
        if next_cloth is not None:
            self.cloth = next_cloth
            self.remainder = 0.0
        if next_root is not None:
            self.root = next_root
        self.output = '\n\n'.join(r.text for r in results) or 'No commands to execute.'
        # Prevent very long traces from overwhelming the browser.
        if len(self.output) > 120000:
            self.output = self.output[:120000] + '\n[Display truncated; use the CLI for the full trace.]'

    def configure(self, key, value):
        c = self.cloth
        if key == 'method':
            if value not in METHODS:
                raise ValueError('unknown integrator')
        elif key == 'h':
            value = number(value, .001, .033)
        elif key == 'stiffness':
            value = number(value, 40, 600)
        elif key == 'wind':
            value = number(value, 0, 12)
        else:
            raise ValueError('unknown control')
        setattr(c, key, value)
        if c.exploded:
            c.reset()
        self.remainder = 0.0

    def advance(self, dt):
        c = self.cloth
        dt = number(dt, 0, .1)
        if self.paused or c.exploded:
            self.remainder = 0.
            return
        self.remainder = min(self.remainder + dt, 40*c.h)
        # Never change the selected numerical step to match the display frame rate.
        while self.remainder + 1e-12 >= c.h:
            self.remainder = max(0., self.remainder - c.h)
            if not c.step():
                self.remainder = 0.
                break

    def snapshot(self):
        c = self.cloth
        # Only structural springs are rendered, exactly as in the prototype.
        springs = []
        for s, (i, j) in enumerate(c.edges):
            if c.alive[s] and c.weights[s] == 1:
                dx, dy = c.x[j] - c.x[i]
                strain = (math.hypot(dx, dy)-c.rest[s])/c.rest[s]
                springs.append([s, int(i), int(j), finite(strain)])
        return {'positions': c.x.tolist(), 'pinned': c.pinned.tolist(), 'springs': springs,
                'cols': c.cols, 'rows': c.rows, 'spacing': c.spacing, 'method': c.method,
                'h': c.h, 'stiffness': c.stiffness, 'wind': c.wind, 't': c.t,
                'energy': finite(c.energy), 'exploded': c.exploded, 'reason': c.reason,
                'drag': c.drag, 'paused': self.paused, 'free_dofs': c.free_dofs,
                'active_springs': int(c.alive.sum())}

    def action(self, data):
        if not isinstance(data, dict):
            raise ValueError('expected a JSON object')
        kind = data.get('action')
        c = self.cloth
        if kind == 'frame':
            self.advance(data.get('dt', 0))
        elif kind == 'configure':
            self.configure(data.get('key'), data.get('value'))
        elif kind == 'reset':
            c.reset()
            self.remainder = 0.
        elif kind == 'pause':
            self.paused = not self.paused
            self.remainder = 0.
        elif kind == 'drag':
            index = data.get('index')
            if type(index) is not int or not 0 <= index < len(c.x):
                raise ValueError('invalid node index')
            if not c.exploded:
                c.drag_node(index, number(data.get('x'), -10, 10), number(data.get('y'), -10, 10))
        elif kind == 'release':
            c.release()
        elif kind == 'cut':
            index = data.get('index')
            if type(index) is not int or not 0 <= index < len(c.edges):
                raise ValueError('invalid spring index')
            if not c.exploded:
                c.cut_spring(index)
        elif kind == 'script':
            self.run_script(data.get('source'))
        elif kind != 'init':
            raise ValueError('unknown action')
        result = {'state': self.snapshot()}
        if kind in ('init', 'script'):
            result.update(root=self.root, output=self.output)
        if kind == 'init':
            result['source'] = DEFAULT_SCRIPT
        return result


def make_server(port=8765, studio=None):
    studio = studio if studio is not None else Studio()
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Frame requests would otherwise flood CMD.

        def send_bytes(self, status, body, mime):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_json(self, status, data):
            self.send_bytes(status, json.dumps(data, allow_nan=False).encode('utf-8'),
                            'application/json; charset=utf-8')

        def local_host(self):
            return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'

        def do_GET(self):
            if not self.local_host():
                self.send_json(403, {'error': 'Open the local URL printed in CMD.'})
            elif self.path == '/':
                self.send_bytes(200, HTML.read_bytes(), 'text/html; charset=utf-8')
            else:
                self.send_json(404, {'error': 'not found'})

        def do_POST(self):
            origin = self.headers.get('Origin')
            if not self.local_host() or origin not in (None, f'http://127.0.0.1:{self.server.server_port}'):
                self.send_json(403, {'error': 'local requests only'})
                return
            if self.path != '/api':
                self.send_json(404, {'error': 'not found'})
                return
            if self.headers.get_content_type() != 'application/json':
                self.send_json(415, {'error': 'expected JSON'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= MAX_BODY:
                    raise ValueError('invalid request size')
                data = json.loads(self.rfile.read(size))
                with lock:
                    result = studio.action(data)
                self.send_json(200, result)
            except (ValueError, TypeError, ArithmeticError, RecursionError) as exc:
                with lock:
                    state = studio.snapshot()
                self.send_json(400, {'error': str(exc), 'state': state})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description='Launch the NumLang Studio Instrument')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    print('Preparing Python / Numba numerical engine...', flush=True)
    for method in METHODS:
        Cloth(method=method).step()
    try:
        server = make_server(args.port)
    except OSError as exc:
        parser.exit(1, f'Cannot start local server: {exc}\nTry --port 8766.\n')
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'NumLang Studio | Python engine ready\nOpen {url}\nPress Ctrl+C to stop.', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nNumLang Studio stopped.')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
