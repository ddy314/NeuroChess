"""Upload source/data archives, then run via colab exec. NC_BACKGROUND=1 returns promptly.

NC_JOB: JSON list of neurochess.train arguments. NC_DASHBOARD defaults to 1.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import urllib.request

root = Path('/content/neurochess')
root.mkdir(exist_ok=True)
for name in ['neurochess-source.tar.gz', 'neurochess-data.tar.gz']:
    archive_path = Path('/content') / name
    if archive_path.exists():
        with tarfile.open(archive_path) as archive:
            archive.extractall(root, filter='data')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(root)], check=True)
os.chdir(root)
args = json.loads(os.environ['NC_JOB'])
if not isinstance(args, list) or not all(isinstance(s, str) for s in args):
    raise ValueError('NC_JOB must be a JSON list of training arguments')
if '--out' not in args:
    raise ValueError('NC_JOB must include --out followed by a persistent output directory')
out = Path(args[args.index('--out') + 1]).resolve()
if str(out).startswith('/content/drive/') and not os.path.ismount('/content/drive'):
    raise RuntimeError('Mount Google Drive before starting; refusing to put a persistent checkpoint on ephemeral storage')
out.mkdir(parents=True, exist_ok=True)
if os.environ.get('NC_DASHBOARD', '1') == '1':
    port = int(os.environ.get('NC_PORT', '8765'))
    monitor_root = out.parent
    def probe():
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/info', timeout=2) as response:
                return json.load(response)
        except OSError:
            return None
    info = probe()
    if info is not None and info != dict(service='neurochess', root=str(monitor_root)):
        raise RuntimeError('Monitor port belongs to another root/service; choose NC_PORT')
    if info is None:
        with (root / 'dashboard.log').open('a') as log:
            server = subprocess.Popen([sys.executable, '-u', '-m', 'neurochess.dashboard', '--root', str(monitor_root),
                '--port', str(port)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(40):
            if probe():
                break
            if server.poll() is not None:
                raise RuntimeError('Dashboard failed; inspect /content/neurochess/dashboard.log')
            time.sleep(.25)
        else:
            raise RuntimeError('Dashboard did not become ready')
    print(f'Dashboard ready on localhost:{port}. In the connected Colab notebook run:', flush=True)
    print(f'from google.colab import output; output.serve_kernel_port_as_iframe({port}, height=900)', flush=True)
import uuid
spec = dict(args=args, arena=os.environ.get('NC_ARENA', '1') == '1',
            prepare_only=os.environ.get('NC_PREPARE_ONLY', '0') == '1',
            stockfish=os.environ.get('NC_STOCKFISH'), data_profile=os.environ.get('NC_DATA_PROFILE', 'configs/data/elite-recent.json'),
            data_cache=os.environ.get('NC_DATA_CACHE'),
            require_profile=os.environ.get('NC_REQUIRE_PROFILE', '1') == '1', pgn_url=os.environ.get('NC_PGN_URL'), min_elo=int(os.environ.get('NC_MIN_ELO', '2000')),
            max_positions=int(os.environ.get('NC_MAX_POSITIONS', '1000000')))
spec_dir = root / '.jobs'; spec_dir.mkdir(exist_ok=True)
spec_path = spec_dir / f'{uuid.uuid4().hex}.json'
spec_path.write_text(json.dumps(spec))
command = [sys.executable, '-u', '-m', 'neurochess.job', '--spec', str(spec_path)]
if os.environ.get('NC_BACKGROUND', '0') == '1':
    with (out / 'console.log').open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(2)
    if process.poll() is not None and process.returncode != 0:
        raise RuntimeError(f'Training startup failed; inspect {out / "console.log"}')
    (out / 'process.json').write_text(json.dumps(dict(pid=process.pid, started_at=time.time(), command=command)))
    print(f'Pipeline launched: PID={process.pid}, console={out / "console.log"}', flush=True)
    print('This launcher returning does not mean training finished. Check status.json / dashboard.', flush=True)
else:
    subprocess.run(command, check=True)
