import json
from pathlib import Path
import sys

import pytest
import torch

from neurochess import autostop


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    source=tmp_path/'local';source.mkdir()
    mount=tmp_path/'drive';destination=mount/'run';destination.mkdir(parents=True)
    monkeypatch.setattr('os.path.ismount',lambda path:Path(path)==mount)
    checkpoint=dict(step=2,origin_step=1500,data_hash='expected',optimizer={'state':{}},model={})
    torch.save(checkpoint,source/'last.pt')
    (source/'best.pt').write_bytes(b'best model')
    for name in ['metrics.jsonl','validation.jsonl']:(source/name).write_text('{"step":2}\n')
    for name in ['dataset.json','validation-dataset.json','replay-dataset.json','lineage.json']:
        (source/name).write_text('{}')
    (source/'run.json').write_text(json.dumps(dict(steps=2)))
    (source/'status.json').write_text(json.dumps(dict(phase='complete',step=2,total_steps=2)))
    (source/'training-code.tar.gz').write_bytes(b'code archive')
    cache=mount/'data.tar.gz';cache.write_bytes(b'training data archive')
    (source/'data-cache.json').write_text(json.dumps(dict(path=str(cache),sha256=autostop.digest(cache))))
    return source,destination,mount


def test_release_only_after_drive_checkpoint_and_data_readback(completed_run,monkeypatch):
    source,destination,mount=completed_run;calls=[]
    def release(address):
        proof=json.loads((destination/'autostop-verification.json').read_text())
        assert proof['step']==2 and proof['cumulative_step']==1502
        assert proof['verified_files']['last.pt']==autostop.digest(destination/'last.pt')
        assert json.loads((destination/'autostop.json').read_text())['phase']=='shutdown_requested'
        calls.append(address)
    monkeypatch.setattr(autostop,'request_unassign',release)
    result=autostop.finalize_and_stop(source,destination,mount,2,'expected','127.0.0.1:8011')
    assert result['step']==2 and calls==['127.0.0.1:8011']


@pytest.mark.parametrize('failure',['paused','wrong_step','wrong_hash','no_optimizer','missing_file','bad_cache'])
def test_failure_never_releases_runtime(completed_run,monkeypatch,failure):
    source,destination,mount=completed_run;calls=[]
    monkeypatch.setattr(autostop,'request_unassign',lambda *args:calls.append(args))
    if failure=='paused':(source/'status.json').write_text('{"phase":"paused","step":2,"total_steps":2}')
    elif failure=='missing_file':(source/'best.pt').unlink()
    elif failure=='bad_cache':(mount/'data.tar.gz').write_bytes(b'corrupted')
    else:
        checkpoint=torch.load(source/'last.pt',weights_only=False)
        if failure=='wrong_step':checkpoint['step']=1
        elif failure=='wrong_hash':checkpoint['data_hash']='other'
        else:del checkpoint['optimizer']
        torch.save(checkpoint,source/'last.pt')
    with pytest.raises(RuntimeError):
        autostop.finalize_and_stop(source,destination,mount,2,'expected','127.0.0.1:8011')
    assert not calls


def test_unmounted_drive_prevents_release_and_writes(completed_run,monkeypatch):
    source,destination,mount=completed_run;calls=[]
    monkeypatch.setattr('os.path.ismount',lambda _:False)
    monkeypatch.setattr(autostop,'request_unassign',lambda *a:calls.append(a))
    with pytest.raises(OSError):autostop.finalize_and_stop(source,destination,mount,2,'expected','127.0.0.1:8011')
    assert not list(destination.iterdir()) and not calls


def test_persistent_checkpoint_corruption_is_detected(completed_run):
    source,destination,mount=completed_run
    autostop.RunMirror(source,destination,mount).sync()
    (destination/'last.pt').write_bytes(b'corrupt persistent checkpoint')
    with pytest.raises(RuntimeError,match='Drive readback differs'):
        autostop.verify_completed_run(source,destination,mount,2,'expected')


def test_unassign_uses_runtime_local_post(monkeypatch):
    events=[]
    class Connection:
        def __init__(self,host,port,timeout):events.append((host,port,timeout))
        def request(self,method,path,body):events.append((method,path,body))
        def getresponse(self):return self
        def read(self):return b''
        status=200
        def close(self):events.append('closed')
    monkeypatch.setattr(autostop.http.client,'HTTPConnection',Connection)
    autostop.request_unassign('172.28.0.1:8011')
    assert ('POST','/unassign',b'') in events and events[-1]=='closed'
    with pytest.raises(ValueError):autostop.request_unassign('8.8.8.8:80')


@pytest.mark.parametrize('success',[False,True])
def test_guard_waits_for_launcher_and_requires_final_success_marker(completed_run,monkeypatch,success):
    source,destination,mount=completed_run;events=[]
    log=source/'launcher.log';log.write_text(autostop.SUCCESS_LINE+'\n' if success else 'RuntimeError: final sync failed\n')
    identities=iter(['start','start',None])
    monkeypatch.setattr(autostop,'process_identity',lambda _:next(identities))
    monkeypatch.setattr(autostop,'find_runtime_address',lambda:'127.0.0.1:8011')
    monkeypatch.setattr(autostop.time,'sleep',lambda _:events.append('waited'))
    original=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda p:b'python\0scripts/expanded_colab.py\0' if str(p)=='/proc/123/cmdline' else original(p))
    def stop(*args):
        events.append('stopped');return {'step':2}
    monkeypatch.setattr(autostop,'finalize_and_stop',stop)
    monkeypatch.setattr(sys,'argv',['autostop','--source',str(source),'--destination',str(destination),
        '--mount',str(mount),'--launcher-pid','123','--launcher-log',str(log),'--expected-steps','2','--data-hash','expected'])
    if success:
        autostop.main();assert events==['waited','stopped']
    else:
        with pytest.raises(RuntimeError,match='successful final-persistence marker'):autostop.main()
        assert events==['waited']
        assert json.loads((destination/'autostop.json').read_text())['phase']=='blocked'


def test_required_evaluation_failure_prevents_release(completed_run,monkeypatch):
    source,destination,mount=completed_run;calls=[]
    (source/'evaluation-required.json').write_text(json.dumps(dict(directory='evaluations/final',games_per_mode=64)))
    (source/'selected.pt').write_bytes(b'selected model')
    def reject(*args):raise ValueError('Evaluation is incomplete')
    monkeypatch.setattr('neurochess.evaluation_check.verify_arena',reject)
    monkeypatch.setattr(autostop,'request_unassign',lambda *a:calls.append(a))
    with pytest.raises(ValueError,match='incomplete'):
        autostop.finalize_and_stop(source,destination,mount,2,'expected','127.0.0.1:8011')
    assert not calls
