import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import urlopen
import pytest
from neurochess.dashboard import serve, snapshot, safe_path
from neurochess.telemetry import append_json, write_status


def test_live_api_and_safety(tmp_path):
    run = tmp_path / 'human'; run.mkdir()
    (run / 'run.json').write_text(json.dumps({'steps': 10}))
    append_json(run / 'metrics.jsonl', {'step': 1, 'loss': [3, 2, 0, 2]})
    write_status(run, 'training', step=1)
    server = serve(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    def get(path):
        with urlopen(base + path) as response:
            return json.load(response)
    try:
        assert get('/api/runs')['runs'] == ['human']
        assert get('/api/run?run=human')['metrics'][0]['step'] == 1
        append_json(run / 'metrics.jsonl', {'step': 2, 'loss': [2, 1, 0, 2]})
        append_json(run / 'metrics.jsonl', {'step': 2, 'loss': [1, .5, 0, 1]})
        with (run / 'metrics.jsonl').open('a') as output:
            output.write('{"step":3')
        rows = get('/api/run?run=human')['metrics']
        assert len(rows) == 2 and rows[-1]['loss'][0] == 1
        for path in ['/api/run?run=../outside', '/api/pgn?run=human&file=run.json', '/api/pgn?run=human&file=../run.json']:
            with pytest.raises(HTTPError) as e:
                get(path)
            assert e.value.code == 404
        arena = run / 'arena/step-1'; arena.mkdir(parents=True)
        (arena / 'games.pgn').write_text('[Result "*"]\n\n*\n')
        with urlopen(base + '/api/pgn?run=human&file=arena/step-1/games.pgn') as response:
            assert b'[Result' in response.read()
        for asset in ['/', '/app.js', '/style.css']:
            with urlopen(base + asset) as response:
                assert response.status == 200 and len(response.read()) > 100
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_symlink_escape(tmp_path):
    root = tmp_path / 'root'; root.mkdir()
    (root / 'escape').symlink_to(tmp_path)
    with pytest.raises(ValueError):
        safe_path(root, 'escape/secret')


def test_curriculum_transition_exposes_next_stage_and_total_progress(tmp_path):
    root=tmp_path/'curriculum-v3';first=root/'stage-a';first.mkdir(parents=True)
    (root/'curriculum-plan.json').write_text(json.dumps(dict(stage_a=4000,additional_optimizer_steps=8000)))
    (root/'pipeline-status.json').write_text(json.dumps(dict(stage='filtering-old-hard-labels-1',updated_at=10)))
    (first/'run.json').write_text('{"steps":4000}');write_status(first,'complete',step=4000)
    result=snapshot(tmp_path,'curriculum-v3/stage-a')
    assert result['status']['phase']=='complete'
    assert result['pipeline']['completed_steps']==4000 and result['pipeline']['total_steps']==8000
    assert result['pipeline']['active_run'] is None
    second=root/'hard-wave-1';second.mkdir();(second/'run.json').write_text('{"steps":1000}')
    write_status(second,'training',step=25)
    (root/'pipeline-status.json').write_text(json.dumps(dict(stage='hard-wave-1',updated_at=20)))
    result=snapshot(tmp_path,'curriculum-v3/stage-a')
    assert result['pipeline']['active_run']=='curriculum-v3/hard-wave-1'
    assert result['pipeline']['completed_steps']==4025
    (root/'pipeline-error.json').write_text(json.dumps(dict(time=21,error='failed')))
    assert snapshot(tmp_path,'curriculum-v3/stage-a')['pipeline']['error']['error']=='failed'
    (root/'pipeline-status.json').write_text(json.dumps(dict(stage='hard-wave-1',updated_at=22)))
    assert not snapshot(tmp_path,'curriculum-v3/stage-a')['pipeline']['error']
    for i in range(1,4):
        folder=root/f'hard-wave-{i}';folder.mkdir(exist_ok=True);write_status(folder,'complete',step=1000)
    (root/'run.json').write_text('{"steps":1000}');write_status(root,'complete',step=1000)
    (root/'pipeline-status.json').write_text(json.dumps(dict(stage='sf1000-evaluation',updated_at=30)))
    result=snapshot(tmp_path,'curriculum-v3')
    assert result['pipeline']['completed_steps']==8000 and result['pipeline']['active_run'] is None
