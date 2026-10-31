import json
from pathlib import Path

import pytest

from neurochess.persistence import RunMirror


def test_unmounted_drive_does_not_create_run(tmp_path, monkeypatch):
    source = tmp_path / 'local'
    source.mkdir()
    (source / 'metrics.jsonl').write_text('new log\n')
    destination = tmp_path / 'drive' / 'run'
    monkeypatch.setattr('neurochess.persistence.os.path.ismount', lambda p: False)
    with pytest.raises(OSError):
        RunMirror(source, destination, tmp_path / 'drive').sync()
    assert not destination.exists()


def test_sync_recovers_and_publishes_checkpoint_before_complete(tmp_path, monkeypatch):
    source = tmp_path / 'local'
    destination = tmp_path / 'drive' / 'run'
    source.mkdir()
    destination.mkdir(parents=True)
    (source / 'last.pt').write_bytes(b'new checkpoint')
    (destination / 'last.pt').write_bytes(b'old checkpoint')
    (source / 'status.json').write_text(json.dumps({'phase': 'complete'}))
    mounted = [False]
    monkeypatch.setattr('neurochess.persistence.os.path.ismount', lambda p: mounted[0])
    mirror = RunMirror(source, destination, destination.parent)
    with pytest.raises(OSError):
        mirror.sync()
    assert (destination / 'last.pt').read_bytes() == b'old checkpoint'
    original = Path.replace
    def replace(path, target):
        if target.name == 'status.json':
            assert (destination / 'last.pt').read_bytes() == b'new checkpoint'
        return original(path, target)
    monkeypatch.setattr(Path, 'replace', replace)
    mounted[0] = True
    mirror.sync()
    assert json.loads((destination / 'status.json').read_text())['phase'] == 'complete'
    (source / 'last.pt').write_bytes(b'next checkpoint')
    mirror.sync()
    assert (destination / 'last.pt').read_bytes() == b'next checkpoint'


def test_interrupted_copy_keeps_previous_checkpoint(tmp_path, monkeypatch):
    source = tmp_path / 'local'
    destination = tmp_path / 'drive' / 'run'
    source.mkdir()
    destination.mkdir(parents=True)
    (source / 'last.pt').write_bytes(b'new checkpoint')
    (destination / 'last.pt').write_bytes(b'old checkpoint')
    (source / 'status.json').write_text('{"phase":"complete"}')
    monkeypatch.setattr('neurochess.persistence.os.path.ismount', lambda p: True)
    def disconnect(src, dst, size):
        dst.write(b'partial')
        raise OSError('Drive disconnected')
    monkeypatch.setattr('neurochess.persistence.shutil.copyfileobj', disconnect)
    with pytest.raises(OSError):
        RunMirror(source, destination, destination.parent).sync()
    assert (destination / 'last.pt').read_bytes() == b'old checkpoint'
    assert not (destination / 'status.json').exists()
