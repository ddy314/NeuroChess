"""Run after uploading the prepared bundle and mounting Drive; safe to resume."""
import hashlib
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
mount = Path('/content/drive')
if not os.path.ismount(mount):
    raise RuntimeError('Mount Google Drive in this runtime first; no training was started')
root.mkdir(exist_ok=True)
with tarfile.open('/content/neurochess-phase2.tar.gz') as archive:
    archive.extractall(root, filter='data')
os.chdir(root)
sys.path.insert(0, str(root))
# Verify local payload before installing or starting GPU work.
checksums = json.loads((root / 'phase2-checksums.json').read_text())
for relative, expected in checksums.items():
    with (root / relative).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise RuntimeError(f'Bundle integrity check failed: {relative}')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(root)], check=True)
from neurochess.persistence import RunMirror

local = Path('/content/neurochess-runs/engine-v1')
persistent = mount / 'MyDrive/neurochess/engine-v1'
replay = root / 'data/human-replay'
source = mount / 'MyDrive/neurochess-data/elite-recent-2026q3'
if not (replay / 'manifest.json').is_file():
    if not (source / 'manifest.json').is_file():
        raise RuntimeError(f'Human replay cache missing: {source}')
    if replay.exists():
        shutil.rmtree(replay)  # Only an incomplete VM-local cache.
    shutil.copytree(source, replay)
if not local.exists():
    if (persistent / 'last.pt').is_file():
        shutil.copytree(persistent, local)
    else:
        local.mkdir(parents=True)
persistent.mkdir(parents=True, exist_ok=True)
if not Path('/usr/games/stockfish').is_file():
    subprocess.run(['apt-get', 'update', '-qq'], check=True)
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'stockfish'], check=True)
import torch
profile = os.environ.get('NC_TRAIN_PROFILE') or ('configs/phase2-l4.json' if 'L4' in torch.cuda.get_device_name() else 'configs/phase2.json')
config = json.loads((root / profile).read_text())
print('GPU:', torch.cuda.get_device_name(), 'training profile:', profile, flush=True)
config.update(data=str(root / 'data/evals-v1'), out=str(local), replay_data=str(replay))
config['resume' if (local / 'last.pt').is_file() else 'init'] = str(
    local / 'last.pt' if (local / 'last.pt').is_file() else root / 'phase2-init.pt')
args = []
for key, value in config.items():
    args.append('--' + key.replace('_', '-'))
    if value is not True:
        args.append(str(value))
with (local / 'dashboard.log').open('a') as log:
    subprocess.Popen([sys.executable, '-m', 'neurochess.dashboard', '--root', str(local.parent),
                      '--port', '8765'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
mirror = RunMirror(local, persistent, mount)
stop = threading.Event()
def replicate():
    while not stop.wait(30):
        try:
            mirror.sync()
            (local / 'persistence.json').write_text(json.dumps(dict(last_sync=time.time(), destination=str(persistent))))
        except OSError as error:
            print('Drive sync delayed; local training continues:', error, flush=True)
worker = threading.Thread(target=replicate, daemon=True)
worker.start()
print('Training phase 2; dashboard port 8765; log:', local / 'console.log', flush=True)
try:
    with (local / 'console.log').open('a') as log:
        code = subprocess.call([sys.executable, '-u', '-m', 'neurochess.train', *args],
                               stdout=log, stderr=subprocess.STDOUT)
finally:
    stop.set()
    worker.join()
# Do not report completion until the final checkpoint reaches persistent storage.
while True:
    try:
        mirror.sync()
        (local / 'persistence.json').write_text(json.dumps(dict(last_sync=time.time(), destination=str(persistent))))
        mirror.sync()
        break
    except OSError as error:
        print('Final persistence pending:', error, flush=True)
        time.sleep(30)
print('Final artifacts persisted; training exit code:', code, flush=True)
if code:
    raise RuntimeError(f'Training exited with {code}; see console.log')
