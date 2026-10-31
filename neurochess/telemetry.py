import json
from pathlib import Path
import time


def append_json(path, value):
    with Path(path).open('a') as output:
        output.write(json.dumps(value, allow_nan=False) + '\n')


def write_status(root, phase, **extra):
    path = Path(root) / 'status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(phase=phase, updated_at=time.time(), **extra)))
    tmp.replace(path)
