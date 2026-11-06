"""Single-branch guarded cloud continuation of the locally validated value MLP."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import chess
import numpy as np
import torch
from neurochess.policy_value_pair import checkpoint, from_original_states, policy_digest
from neurochess.multihead import MultiheadPlayer
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.encoding import encode, legal_ids
from neurochess.persistence import RunMirror
from neurochess.autostop import find_runtime_address, request_unassign
from scripts.local_value_repair import write, metrics
from scripts.local_value_sustained import make_tasks, train_step
from scripts.diagnose_value_training import load_cache
from scripts.search_repair_run_cloud import gate_decision


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(json.dumps(list(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def gpu_metrics(model, records):
    ce, values = [], []
    with torch.inference_mode():
        for start in range(0, len(records), 32):
            r = records[start:start+32]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model.value_only(torch.as_tensor(r['tokens'].copy(), device='cuda'),
                                          torch.as_tensor(r['meta'].copy(), device='cuda')).float()
            target = torch.as_tensor(r['value'].copy(), device='cuda')
            ce.extend((-(target*logits.log_softmax(-1)).sum(-1)).cpu().tolist())
            p = logits.softmax(-1); values.extend((p[:, 0]-p[:, 2]).cpu().tolist())
    truth = records['value'][:, 0]-records['value'][:, 2]; error = abs(np.asarray(values)-truth)
    by = {}
    for name, mask in [('winning', truth>.25), ('neutral', abs(truth)<=.25), ('losing', truth<-.25)]:
        by[name] = dict(n=int(mask.sum()), mae=float(error[mask].mean()))
    return dict(n=len(records), ce=float(np.mean(ce)), mae=float(error.mean()), by_stratum=by)


def numeric_checks(reference, candidate):
    """Original numerical gate limits, additionally anchored to old value before repair."""
    return dict(general_ce=candidate['general']['ce']<=reference['general']['ce']*1.03,
                general_mae=candidate['general']['mae']<=reference['general']['mae']+.02,
                balanced_mae=candidate['balanced']['mae']<=reference['balanced']['mae']+.005,
                per_value_stratum=all(candidate['balanced']['by_stratum'][k]['mae']<=reference['balanced']['by_stratum'][k]['mae']+.03
                                     for k in ['winning', 'neutral', 'losing']))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True); a = p.parse_args()
    plan = json.loads(Path(a.plan).read_text()); mount = Path(plan['mount'])
    if not os.path.ismount(mount):
        raise RuntimeError('CLI Drive mount required')
    lock = open('/tmp/value-sustained-job.lock', 'w'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    work = Path(plan['work']); save = Path(plan['save']); work.mkdir(exist_ok=True); save.mkdir(parents=True, exist_ok=True)
    mirror = RunMirror(work, save, mount); sync_lock = threading.Lock(); done = threading.Event()
    phase = ['checking-inputs']; model = None; reference = None; last_pass_head = None; step = 0
    def sync():
        with sync_lock:
            mirror.sync()
    def heartbeat():
        while not done.wait(30):
            try:
                write(work/'pipeline-status.json', dict(phase=phase[0], additional_steps=step, absolute_step=1000+step, updated_at=time.time()))
                sync()
            except Exception as e:
                print('Persistence retry:', repr(e), flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    try:
        torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
        if 'L4' not in torch.cuda.get_device_name():
            raise RuntimeError('Expected the authorized single L4')
        inputs = Path(plan['inputs']); manifest = json.loads((inputs/'manifest.json').read_text())
        for rel, expected in manifest['input_sha256'].items():
            if sha(inputs/rel) != expected:
                raise ValueError('Input SHA mismatch: '+rel)
        parents = []
        for name in ['policy', 'value']:
            entry = plan['parents'][name]
            if sha(entry['path']) != entry['sha256']:
                raise ValueError('Persistent parent SHA mismatch: '+name)
            parents.append(torch.load(entry['path'], map_location='cpu', mmap=True, weights_only=False))
        model = from_original_states(*parents)
        old_head = copy.deepcopy(model.value_model.value).cpu().eval()
        selected = torch.load(inputs/'head-1000.pt', map_location='cpu', weights_only=False)
        if selected['step'] != 1000 or selected['accepted_updates'] != 1000:
            raise ValueError('Wrong locally qualified checkpoint')
        model.value_model.value.load_state_dict(selected['value_head'])
        if state_digest(model.state_dict()) != manifest['qualified_model_state_sha256']:
            raise ValueError('Reconstruction differs from actual local1000-step checkpoint tensors')
        lineage = dict(parent_local_file_sha256=manifest['qualified_local_file_sha256'],
                       parent_model_state_sha256=manifest['qualified_model_state_sha256'],
                       exact_original_trunks=True, training_branches=1, optimizer_reset=True,
                       anchor_model='original old value head, unchanged from local validation')
        policy_sha = policy_digest(model); last_pass_head = copy.deepcopy(selected['value_head'])
        data = load_cache(inputs/'cache'); tasks = make_tasks(old_head, data)
        initial_numeric = {name: metrics(old_head, *data[name]) for name in ['balanced', 'general']}
        write(work/'original-value-numeric-reference.json', initial_numeric)
        write(work/'run.json', plan); write(work/'input-verification.json', manifest)
        # Small locally qualified head plus persistent original parents are exact recovery inputs.
        torch.save(selected, work/'head-initial-1000.pt'); sync()
        torch.save(checkpoint(model,1000,lineage,gate_passed=True,local_gate_passed=True),work/'initial.pt'); sync()
        model = model.to('cuda').eval(); del parents
        records = {name: np.concatenate([np.load(f) for f in sorted((inputs/'data'/folder).glob('val-*.npy'))])
                   for name, folder in [('balanced','val'), ('general','general-val')]}
        roots = json.loads((inputs/'data/gate-roots.json').read_text())
        config = json.loads((ROOT/'configs/search-baseline-lock.json').read_text())['config']
        network = MultiheadPlayer(model, 'cuda')
        b = chess.Board(); t, m = encode(b); _, ids = legal_ids(b)
        moves = np.full((64,256), -1, np.int16); moves[:, :len(ids)] = ids
        for size in [1,64]:
            network.raw_outputs(np.repeat(t[None],size,axis=0), np.repeat(m[None],size,axis=0), moves[:size])
        torch.cuda.synchronize()
        def gate(label):
            phase[0] = 'gate-'+label
            report = {name:gpu_metrics(model, r) for name,r in records.items()}; opponent = []
            for group in ['diagnostics', 'holdout']:
                result = dict(correct=0,total=0,rows=[]); report[group] = result
                for job in roots[group]:
                    b = chess.Board(job['initial_fen'])
                    for u in job['history']: b.push_uci(u)
                    player = NativeMCTSPlayer(network, **config)
                    try:
                        move = player.choose(b); stats = player.search_stats
                    finally:
                        player.close()
                    b.push(move); correct = move.uci()==job['solution'][0] or b.is_checkmate()
                    result['correct'] += int(correct); result['total'] += 1
                    result['rows'].append(dict(key=job['key'],move=move.uci(),expected=job['solution'][0],correct=correct,stats=stats))
                    if group=='diagnostics':
                        b = chess.Board(job['initial_fen'])
                        for u in job['history']+[job['solution'][0]]: b.push_uci(u)
                        t,m = encode(b)
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                            prob = model.value_only(torch.as_tensor(t[None],device='cuda'),torch.as_tensor(m[None],device='cuda')).float().softmax(-1)
                        opponent.append(float((prob[0,0]-prob[0,2]).cpu()))
                    write(work/(label+'-progress.json'), report)
            report['opponent_mean_value'] = float(np.mean(opponent))
            report['policy_q_digest'] = policy_digest(model)
            if report['policy_q_digest'] != policy_sha: raise ValueError('Frozen policy/Q changed')
            write(work/(label+'-gate.json'),report); sync(); return report
        reference = gate('initial-1000')
        write(work/'baseline-protocol.json',dict(search_reference='Initial qualified1000-step model on this L4, fixed for entire continuation',
            additional_numeric_reference='Original old value numeric metrics; original tolerances unchanged',search_config=config,
            full_games=0,baseline_remeasured_per_stage=False,scope='Fixed-position gate, not Elo'))
        head = copy.deepcopy(model.value_model.value).cpu().eval()
        optimizer = torch.optim.AdamW(head.parameters(), lr=plan['learning_rate'], weight_decay=.01)
        torch.manual_seed(plan['seed']); accepted = 0; final_decision = None
        for step in range(1,plan['additional_steps']+1):
            phase[0] = 'training'; update = train_step(head, tasks, optimizer, plan['batch_per_training_stratum'])
            accepted += int(update['actual_norm']>1e-9)
            if step%50==0:
                val = {name:metrics(head,*data[name]) for name in ['balanced','general']}
                row = dict(step=step,absolute_step=1000+step,accepted_updates=accepted,loss=[update['objective'],0.,0.,update['objective']],
                           validation=val,actual_update_norm=update['actual_norm'],projection=update['projection'],timestamp=time.time())
                with (work/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                write(work/'status.json',dict(phase='training',step=step,total_steps=plan['additional_steps'],absolute_step=1000+step,updated_at=time.time()))
                print(json.dumps(row),flush=True)
            if step%plan['gate_every']==0:
                small = dict(value_head=head.state_dict(),step=1000+step,additional_steps=step,optimizer=optimizer.state_dict(),
                             rng_state=torch.get_rng_state(),plan=plan,lineage=lineage,accepted_updates=accepted)
                torch.save(small,work/f'head-{1000+step}.pt'); sync()
                model.value_model.value.load_state_dict(head.state_dict())
                report = gate(str(1000+step)); decision = gate_decision(reference,report)
                decision['original_numeric_checks'] = numeric_checks(initial_numeric,report)
                decision['passed'] &= all(decision['original_numeric_checks'].values()) and accepted==step
                write(work/f'{1000+step}-decision.json',decision); sync(); final_decision=decision
                if not decision['passed']:
                    phase[0]='gate-failed-stopped'; break
                last_pass_head = copy.deepcopy(head.state_dict())
        else:
            phase[0]='completed'
        assert policy_digest(model)==policy_sha
        torch.save(checkpoint(model,1000+step,lineage,gate_passed=final_decision['passed'],optimizer=optimizer.state_dict()),work/'last.pt')
        model.value_model.value.load_state_dict(last_pass_head)
        last_pass_step = 1000+step if final_decision['passed'] else 1000+step-plan['gate_every']
        torch.save(checkpoint(model,last_pass_step,lineage,gate_passed=True),work/'selected.pt')
        write(work/'completion.json',dict(phase=phase[0],additional_steps=step,absolute_step=1000+step,
            selected_absolute_step=last_pass_step,decision=final_decision,default_model_promoted=False,training_branches=1))
        write(work/'status.json',dict(phase='complete' if phase[0]=='completed' else 'stopped',
            step=step,total_steps=plan['additional_steps'],absolute_step=1000+step,reason=phase[0],updated_at=time.time()))
    except BaseException:
        phase[0]='failed'; write(work/'failure.json',dict(traceback=traceback.format_exc(),additional_steps=step,phase=phase[0]))
        if 'head' in locals():
            torch.save(dict(value_head=head.state_dict(),additional_steps=step,optimizer=optimizer.state_dict(),
                rng_state=torch.get_rng_state(),unvalidated_failure_state=True,lineage=lineage),work/'failure-head.pt')
        write(work/'status.json',dict(phase='failed',step=step,total_steps=plan['additional_steps'],updated_at=time.time()))
        print(traceback.format_exc(),flush=True)
    finally:
        done.set(); thread.join(timeout=35)
        console = Path('/content/value-sustained-launcher.log')
        if console.exists():shutil.copyfile(console,work/'console-snapshot.log')
        sync()
        hashes = {str(f.relative_to(work)):sha(f) for f in work.rglob('*') if f.is_file() and f.suffix!='.tmp'}
        for rel,expected in hashes.items():
            if sha(save/rel)!=expected:raise IOError('Persistent readback mismatch: '+rel)
        write(work/'persistence-verification.json',dict(phase=phase[0],sha256=hashes,verified_at=time.time()))
        sync()
        if sha(work/'persistence-verification.json')!=sha(save/'persistence-verification.json'):
            raise IOError('Final verification marker not persistent')
        print('Persisted and verified; releasing runtime. Phase:',phase[0],flush=True)
        request_unassign(find_runtime_address())


if __name__=='__main__':
    main()
