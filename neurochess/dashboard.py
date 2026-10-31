"""Read-only, dependency-free training dashboard. Bind localhost; use Colab/SSH proxy."""
import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
from urllib.parse import parse_qs, urlparse

WEB = Path(__file__).with_name('web')


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path is outside run root')
    return path


def read_json(path):
    try:
        return json.loads(path.read_text()[:65536])
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_rows(path, limit=5000):
    """Bound reads and ignore a concurrent incomplete final JSONL line."""
    try:
        with path.open('rb') as f:
            f.seek(0, 2)
            size = f.tell(); start = max(0, size - 4 * 1024 * 1024)
            f.seek(start)
            if start:
                f.readline()
            lines = f.read().splitlines()
    except FileNotFoundError:
        return []
    rows = []
    for line in lines:
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        except (ValueError, UnicodeDecodeError):
            continue
    return rows[-limit:]


def pipeline_state(root, path):
    """Expose the enclosing curriculum without changing per-stage metric axes."""
    root=Path(root).resolve();base=Path(path).resolve()
    while base.is_relative_to(root):
        if (base/'curriculum-plan.json').is_file() and (base/'pipeline-status.json').is_file():break
        if base==root:return None
        base=base.parent
    else:return None
    state=read_json(base/'pipeline-status.json');plan=read_json(base/'curriculum-plan.json')
    stage=state.get('stage','setup');active=None
    candidates=[(base/'stage-a',int(plan.get('stage_a',4000)))]
    candidates.extend((base/f'hard-wave-{i}',1000) for i in range(1,4));candidates.append((base,1000))
    completed=sum(max(0,min(goal,int(read_json(folder/'status.json').get('step',0)))) for folder,goal in candidates)
    if stage=='stage-a':active=base/'stage-a'
    elif stage in [f'hard-wave-{i}' for i in range(1,5)]:
        wave=int(stage.rsplit('-',1)[1]);active=base/f'hard-wave-{wave}' if wave<4 else base
    error=read_json(base/'pipeline-error.json')
    if error.get('time',0)<state.get('updated_at',0):error={}
    return dict(stage=stage,updated_at=state.get('updated_at'),completed_steps=completed,
        total_steps=plan.get('additional_optimizer_steps',8000),error=error,
        active_run=str(active.relative_to(root)) if active and (active/'run.json').is_file() else None)


def snapshot(root, run):
    path = safe_path(root, run)
    if not (path / 'run.json').is_file():
        raise FileNotFoundError('Run not found')
    def latest_per_step(rows):
        return sorted({r['step']: r for r in rows if isinstance(r.get('step'), int)}.values(), key=lambda r: r['step'])
    return dict(run=run, pipeline=pipeline_state(root,path), config=read_json(path / 'run.json'), status=read_json(path / 'status.json'),
        dataset=read_json(path / 'dataset.json'), replay_dataset=read_json(path / 'replay-dataset.json'),
        runtime=read_json(path / 'runtime.json'), persistence=read_json(path / 'persistence.json'),
        validation_dataset=read_json(path / 'validation-dataset.json'), lineage=read_json(path / 'lineage.json'),
        checkpoints={name: dict(bytes=(path/name).stat().st_size, modified_at=(path/name).stat().st_mtime)
                     for name in ['last.pt','best.pt'] if (path/name).is_file()},
        metrics=latest_per_step(read_rows(path / 'metrics.jsonl')),
        validation=latest_per_step(read_rows(path / 'validation.jsonl')),
        arena=read_rows(path / 'arena.jsonl', 1000), server_time=time.time(),
        scope='Last 5000 metric records (up to 4 MiB per log); same-step resumed records use latest value')


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, root, **kwargs):
        self.root = Path(root).resolve()
        super().__init__(*args, **kwargs)

    def log_message(self, *args):
        pass

    def send(self, body, content_type='application/json', status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == '/favicon.ico':
                return self.send(b'', 'image/x-icon', 204)
            if url.path in ['/', '/index.html']:
                return self.send((WEB / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            if url.path in ['/app.js', '/style.css']:
                return self.send((WEB / url.path[1:]).read_bytes(),
                    'text/javascript; charset=utf-8' if url.path.endswith('.js') else 'text/css; charset=utf-8')
            if url.path == '/api/info':
                return self.send(json.dumps(dict(service='neurochess', root=str(self.root))).encode())
            if url.path == '/api/runs':
                paths = []
                if self.root.exists():
                    for path in self.root.rglob('run.json'):
                        if path.resolve().is_relative_to(self.root):
                            paths.append(str(path.parent.relative_to(self.root)))
                        if len(paths) >= 200:
                            break
                return self.send(json.dumps(dict(runs=sorted(paths))).encode())
            run = query.get('run', ['.'])[0]
            if url.path == '/api/run':
                return self.send(json.dumps(snapshot(self.root, run)).encode())
            if url.path == '/api/pgn':
                run_path = safe_path(self.root, run)
                relative = query.get('file', [''])[0]
                path = safe_path(run_path, relative)
                if not path.is_relative_to(run_path / 'arena') or path.name != 'games.pgn':
                    raise ValueError('Not an arena PGN')
                return self.send(path.read_bytes(), 'application/x-chess-pgn; charset=utf-8')
            return self.send(b'{"error":"Not found"}', status=404)
        except (ValueError, FileNotFoundError):
            return self.send(b'{"error":"Run or file not found"}', status=404)


def serve(root, host='127.0.0.1', port=8765):
    return ThreadingHTTPServer((host, port), partial(Handler, root=root))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='runs')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    a = p.parse_args()
    server = serve(a.root, a.host, a.port)
    print(f'NeuroChess monitor: http://{a.host}:{server.server_port}  root={Path(a.root).resolve()}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
