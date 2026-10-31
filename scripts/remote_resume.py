"""Resume this Colab run with local writes and retryable Drive replication."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time

root = Path('/content/neurochess')
root.mkdir(exist_ok=True)
with tarfile.open('/content/neurochess-source.tar.gz') as archive:
    archive.extractall(root, filter='data')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(root)], check=True)
os.chdir(root)
sys.path.insert(0, str(root))
from neurochess.persistence import RunMirror

mount = Path('/content/drive')
persistent = mount / 'MyDrive/neurochess/human-recent'
local = Path('/content/neurochess-runs/human-recent')
if not os.path.ismount(mount):
    raise RuntimeError('Mount Google Drive before resuming')
if local.exists():
    raise RuntimeError('Local recovery directory already exists; inspect before rerunning')
print('Copying saved run and cached dataset to VM-local storage', flush=True)
shutil.copytree(persistent, local)
config = json.loads((local / 'run.json').read_text())
dataset = root / config['data']
shutil.copytree(mount / 'MyDrive/neurochess-data/elite-recent-2026q3', dataset)
if not Path(config['arena_stockfish']).is_file():
    subprocess.run(['apt-get', 'update', '-qq'], check=True)
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'stockfish'], check=True)
config.update(out=str(local), resume=str(local / 'last.pt'), stop_after=None, init=None)
args = []
for key, value in config.items():
    if value is None or value is False:
        continue
    args.append('--' + key.replace('_', '-'))
    if value is not True:
        args.append(str(value))

with (root / 'dashboard.log').open('a') as log:
    dashboard = subprocess.Popen([sys.executable, '-u', '-m', 'neurochess.dashboard',
        '--root', str(local.parent), '--port', '8765'], stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True)
mirror = RunMirror(local, persistent, mount)
# The initial copies already match; avoid rewriting the 1.7 GB checkpoint.
for path in local.rglob('*'):
    if path.is_file():
        stat = path.stat()
        mirror.copied[path.relative_to(local)] = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
stop = threading.Event()
def replicate():
    while not stop.wait(30):
        try:
            mirror.sync()
        except OSError as error:
            print('Drive sync delayed; training continues locally:', error, flush=True)
worker = threading.Thread(target=replicate, daemon=True)
worker.start()
print('Starting foreground training; dashboard on port 8765', flush=True)
with (local / 'console.log').open('a') as log:
    process = subprocess.Popen([sys.executable, '-u', '-m', 'neurochess.train', *args],
                               stdout=log, stderr=subprocess.STDOUT)
    code = process.wait()
stop.set()
worker.join()
# Keep the kernel executing until final artifacts have reached Drive.
while True:
    try:
        mirror.sync()
        break
    except OSError as error:
        print('Waiting to persist final artifacts:', error, flush=True)
        time.sleep(30)
print('Training exited; final artifacts persisted, return code:', code, flush=True)
if code:
    raise RuntimeError(f'Training failed with exit code {code}; inspect console.log')
