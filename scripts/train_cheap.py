"""Distill engine-searched WDL values locally; exported evaluation runs on CPU."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if importlib.util.find_spec('chess') is None:
    for site in (ROOT/'.venv/lib').glob('python*/site-packages'):
        if (site/'chess').is_dir():
            sys.path.append(str(site)); break
import numpy as np
import torch
from torch.nn import functional as F
from neurochess.cheap import CheapConfig, CheapModel, save_cheap
from neurochess.data import DTYPE


def batches(root, split, size, rng, limit=None):
    files = sorted(Path(root).glob(f'{split}-*.npy'))
    if not files:
        raise ValueError(f'No {split} shards: {root}')
    if split == 'train':
        rng.shuffle(files)
    count = 0
    for file in files:
        rows = np.load(file, mmap_mode='r', allow_pickle=False)
        if rows.dtype != DTYPE:
            raise ValueError(f'Unexpected schema: {file}')
        order = rng.permutation(len(rows)) if split == 'train' else np.arange(len(rows))
        for start in range(0, len(rows), size):
            r = rows[order[start:start+size]]
            good = r['value_weight'] > 0
            r = r[good]
            if not len(r):
                continue
            if limit is not None:
                r = r[:limit-count]
            if not len(r):
                return
            yield (torch.from_numpy(np.array(r['tokens'], copy=True)),
                   torch.from_numpy(np.array(r['meta'], copy=True)),
                   torch.from_numpy(np.array(r['value'][:, 0]-r['value'][:, 2], copy=True)))
            count += len(r)
            if limit is not None and count >= limit:
                return


@torch.inference_mode()
def validate(model, root, size, limit=20000):
    errors, baseline_errors = [], []
    device = next(model.parameters()).device
    for tokens, meta, target in batches(root, 'val', size, np.random.default_rng(0), limit):
        tokens, meta, target = tokens.to(device), meta.to(device), target.to(device)
        prediction = model(tokens, meta)
        own, _ = model.features(tokens)
        baseline = (own@model.material).tanh()
        errors.extend((prediction-target).cpu().numpy().tolist())
        baseline_errors.extend((baseline-target).cpu().numpy().tolist())
    if not errors:
        raise ValueError('No validation labels')
    e = np.asarray(errors); b = np.asarray(baseline_errors)
    manifest=Path(root)/'manifest.json'
    split=json.loads(manifest.read_text()).get('provenance',{}).get('split','position-hash holdout') if manifest.exists() else 'unspecified'
    return dict(positions=len(e), mse=float((e*e).mean()), mae=float(np.abs(e).mean()),
                absolute_error_p90=float(np.quantile(np.abs(e), .9)),
                material_mse=float((b*b).mean()), material_mae=float(np.abs(b).mean()),
                split=split,note='Dataset split is recorded separately; this is not a calibrated error bound or Elo measurement.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--init')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--cosine', action='store_true', help='Decay learning rate to 10% over this run')
    p.add_argument('--hard-data', help='Additional searched-frontier labels; separate RNG and validation')
    p.add_argument('--hard-weight', type=float, default=.25)
    p.add_argument('--seed', type=int, default=31)
    p.add_argument('--validate-every', type=int, default=250)
    a = p.parse_args()
    if min(a.steps, a.batch_size, a.threads, a.validate_every, a.lr) <= 0 or a.hard_weight<0:
        p.error('Positive resource limits required')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable in this Python environment; use a CUDA-enabled existing environment')
    rng = np.random.default_rng(a.seed)
    hard_rng = np.random.default_rng(a.seed+991903)
    hard_iterator = None
    if a.init:
        from neurochess.cheap import load_cheap
        model, _ = load_cheap(a.init)
    else:
        model = CheapModel(CheapConfig(a.width, a.hidden))
    model.to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.0001, fused=a.device == 'cuda')
    manifest = Path(a.data)/'manifest.json'
    provenance = dict(args=vars(a), config=asdict(model.config), device=a.device,
        parameters=sum(p.numel() for p in model.parameters()),
        torch=str(torch.__version__), gpu=torch.cuda.get_device_name() if a.device == 'cuda' else None,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        dataset=json.loads(manifest.read_text()), target='side-to-move W-L of engine-searched score proxy')
    if a.init:
        with open(a.init, 'rb') as f:
            provenance['initial_checkpoint_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
    if a.hard_data:
        hard_manifest=Path(a.hard_data)/'manifest.json'
        provenance['hard_manifest_sha256']=hashlib.sha256(hard_manifest.read_bytes()).hexdigest()
        provenance['hard_dataset']=json.loads(hard_manifest.read_text())
    (out/'run.json').write_text(json.dumps(provenance, indent=2))
    start = time.perf_counter(); step = seen = 0; best = float('inf')
    hard_seen = 0
    initial = validate(model, a.data, a.batch_size)
    (out/'initial.json').write_text(json.dumps(initial, indent=2))
    with (out/'metrics.jsonl').open('w') as log:
        while step < a.steps:
            epoch_seen = 0
            for tokens, meta, target in batches(a.data, 'train', a.batch_size, rng):
                tokens, meta, target = tokens.to(a.device), meta.to(a.device), target.to(a.device)
                if a.cosine:
                    lr = a.lr*(.1+.9*.5*(1+math.cos(math.pi*step/a.steps)))
                    for group in opt.param_groups:
                        group['lr'] = lr
                model.train(); opt.zero_grad(set_to_none=True)
                logits = model.logits(tokens, meta)
                # A weak logit term prevents completely flat gradients at saturation.
                loss = F.mse_loss(logits.tanh(), target) + .02*F.smooth_l1_loss(
                    logits, torch.atanh(target.clamp(-.995, .995)))
                if a.hard_data:
                    if hard_iterator is None:
                        hard_iterator=iter(batches(a.hard_data,'train',min(512,a.batch_size),hard_rng))
                    try: ht,hm,hy=next(hard_iterator)
                    except StopIteration:
                        hard_iterator=iter(batches(a.hard_data,'train',min(512,a.batch_size),hard_rng))
                        ht,hm,hy=next(hard_iterator)
                    ht,hm,hy=ht.to(a.device),hm.to(a.device),hy.to(a.device)
                    hl=model.logits(ht,hm)
                    loss=loss+a.hard_weight*(F.mse_loss(hl.tanh(),hy)+.02*F.smooth_l1_loss(
                        hl,torch.atanh(hy.clamp(-.995,.995))))
                    hard_seen+=len(hy)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
                step += 1; seen += len(target); epoch_seen += len(target)
                if step % a.validate_every == 0 or step == a.steps:
                    model.eval(); val = validate(model, a.data, a.batch_size)
                    row = dict(step=step, seen=seen, seconds=time.perf_counter()-start,
                               train_loss=float(loss.detach()), lr=opt.param_groups[0]['lr'], validation=val,
                               phase='complete' if step == a.steps else 'training')
                    if a.hard_data:
                        row['hard_seen']=hard_seen
                        row['hard_validation']=validate(model,a.hard_data,a.batch_size)
                    print(json.dumps(row), flush=True); log.write(json.dumps(row)+'\n'); log.flush()
                    save_cheap(model, out/'last.pt', step=step, seen=seen, validation=val, provenance=provenance)
                    model.export(out/'last.bin')
                    if val['mse'] < best:
                        best = val['mse']
                        save_cheap(model, out/'best.pt', step=step, seen=seen, validation=val, provenance=provenance)
                        model.export(out/'best.bin')
                    (out/'status.json').write_text(json.dumps(row, indent=2))
                if step == a.steps:
                    break
            if not epoch_seen:
                raise ValueError('No training labels')


if __name__ == '__main__':
    main()
