import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('stream_evals', Path('scripts/stream_evals.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Response:
    status = 206
    headers = {'ETag': '"v1"', 'Content-Range': 'bytes 0-3/10'}
    def __init__(self, data): self.data = data
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return self.data


def test_short_http_range_retries_without_accepting_truncated_data(monkeypatch):
    responses = iter([Response(b'ab'), Response(b'abcd')])
    monkeypatch.setattr(module, 'urlopen', lambda *a, **k: next(responses))
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    assert module.fetch_range('https://example.invalid/data', '"v1"', 0, 3) == b'abcd'


def test_changed_source_version_is_rejected(monkeypatch):
    response = Response(b'abcd'); response.headers = {**response.headers, 'ETag': '"v2"'}
    monkeypatch.setattr(module, 'urlopen', lambda *a, **k: response)
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='version changed'):
        module.fetch_range('https://example.invalid/data', '"v1"', 0, 3)
