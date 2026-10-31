"""Compare local CPU/CUDA student training, including host-to-device copies."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if importlib.util.find_spec('chess') is None:
    for site in (ROOT/'.venv/lib').glob('python*/site-packages'):
        if (site/'chess').is_dir():
            sys.path.append(str(site)); break
import numpy as np
import torch
from neurochess.cheap import CheapModel
from torch.nn import functional as F


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=200); p.add_argument('--threads', type=int, default=4)
    a = p.parse_args(); torch.set_num_threads(a.threads)
    data = np.load(next(Path(a.data).glob('train-*.npy')), mmap_mode='r')
    rows = []
    for batch in [1024, 4096]:
        # Hold identical, real data in host RAM; every step includes input copies.
        r = data[:batch]
        host = [torch.from_numpy(np.array(x, copy=True)) for x in [
            r['tokens'], r['meta'], r['value'][:,0]-r['value'][:,2]]]
        for device in ['cpu', 'cuda']:
            if device == 'cuda' and not torch.cuda.is_available():
                continue
            torch.manual_seed(31); model = CheapModel().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=.001, fused=device == 'cuda')
            def step():
                tokens, meta, target = [x.to(device) for x in host]
                opt.zero_grad(set_to_none=True)
                logits = model.logits(tokens, meta)
                loss = F.mse_loss(logits.tanh(), target)+.02*F.smooth_l1_loss(
                    logits, torch.atanh(target.clamp(-.995,.995)))
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
            for _ in range(20): step()
            if device == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(a.steps): step()
            if device == 'cuda': torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            row = dict(device=device, batch_size=batch, steps=a.steps, seconds=seconds,
                       positions_per_second=a.steps*batch/seconds)
            rows.append(row); print(json.dumps(row),flush=True)
            del opt, model
            if device == 'cuda': torch.cuda.empty_cache()
    report = dict(torch=torch.__version__, threads=a.threads,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        note='Warmup excluded; input transfer included; one host-resident shard; other local jobs may contend.', results=rows)
    Path(a.out).write_text(json.dumps(report,indent=2))


if __name__ == '__main__': main()
