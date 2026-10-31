"""Stage closed NPY shards to an existing runtime while preparation continues."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--socket', required=True)
    p.add_argument('--destination', required=True)
    p.add_argument('--report', required=True)
    a = p.parse_args()
    root, base = Path(a.data), Path(a.base)
    report = Path(a.report)
    sent = json.loads(report.read_text()) if report.exists() else {}
    old = {p.name for p in base.glob('*.npy')}
    while True:
        complete = (root/'manifest.json').exists()
        ready = []
        for file in sorted(root.glob('*.npy')):
            if file.name in old or file.name in sent or time.time()-file.stat().st_mtime < 2:
                continue
            try:
                rows = np.load(file, mmap_mode='r', allow_pickle=False)
                if file.stat().st_size != rows.offset + rows.nbytes:
                    continue
                ready.append(file)
            except (OSError, ValueError):
                continue
        if ready and (len(ready) >= 8 or complete):
            files = ready[:8]
            command = ['ssh','-S',a.socket,'-o','BatchMode=yes','root@colab',
                       'tar -xzf - -C '+shlex.quote(a.destination)]
            with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
                with tarfile.open(fileobj=process.stdin, mode='w|gz', compresslevel=1) as tar:
                    for file in files: tar.add(file, arcname=file.name)
                process.stdin.close()
                if process.wait(): raise RuntimeError('Shard upload failed; rerun to retry batch')
            for file in files:
                with file.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
                sent[file.name] = dict(bytes=file.stat().st_size, sha256=digest, uploaded_at=time.time())
            report.write_text(json.dumps(sent,indent=2))
            print(json.dumps(dict(uploaded_shards=len(sent),last=files[-1].name)),flush=True)
            continue
        if complete and all(f.name in old or f.name in sent for f in root.glob('*.npy')):
            break
        time.sleep(3)


if __name__ == '__main__':
    main()
