"""Sustained local value-head training with class-specific update constraints.

Only training samples determine gradients. Steps500 and1000 are declared before
the run; no early checkpoint selection, cloud calls or search changes occur here.
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from scripts.diagnose_value_training import load_cache, load_head, strata
from scripts.local_value_repair import metrics, write


def project_update(proposal, gradients, max_sweeps=2048):
    """Project the actual AdamW displacement onto g_i dot displacement <= 0.

    Projecting raw gradients before Adam is insufficient: its per-coordinate
    scaling and momentum change the update direction. Solve the small dual
    coordinate problem after forming the real optimizer displacement instead.
    """
    norms = gradients.norm(dim=1)
    g = gradients[norms > 1e-12] / norms[norms > 1e-12, None]
    if not len(g):
        return proposal, dict(sweeps=0, max_violation=0.)
    gram = (g @ g.T).double().numpy()
    dot = (g @ proposal).double().numpy()
    dual = np.zeros(len(g))
    tolerance = max(1e-12, float(np.max(np.abs(dot))) * 1e-7)
    for sweep in range(max_sweeps):
        for j in range(len(g)):
            dual[j] = max(0., dual[j] + (dot[j] - gram[j] @ dual) / gram[j, j])
        if np.max(dot - gram @ dual) <= tolerance:
            break
    projected = proposal - torch.from_numpy(dual).to(proposal.dtype) @ g
    violation = (g @ projected).max().item()
    if violation > max(2e-9, 1e-5 * proposal.norm().item()):
        raise RuntimeError(f'Update projection failed to converge: {violation}')
    return projected, dict(sweeps=sweep+1, max_violation=violation)


def parameter_vector(head):
    return torch.nn.utils.parameters_to_vector(head.parameters()).detach().clone()


def assign_vector(head, vector):
    with torch.no_grad():
        offset = 0
        for p in head.parameters():
            p.copy_(vector[offset:offset+p.numel()].view_as(p))
            offset += p.numel()


def make_tasks(initial, data):
    tasks = {}
    for name in ['repair', 'anchor']:
        x, y = data[name]
        for label, mask in strata(y).items():
            xx, yy = x[mask], y[mask]
            with torch.no_grad():
                base = torch.cat([initial(b).softmax(-1) for b in xx.split(512)])
            tasks[name+'/'+label] = (xx, yy, base)
    return tasks


def train_step(head, tasks, optimizer, batch=96, constraint_projection=True):
    objectives = []; protected = []; losses = {}
    for name, (x, y, base) in tasks.items():
        ix = torch.randint(len(x), (batch,)); logits = head(x[ix]); logprob = logits.log_softmax(-1)
        probability = logprob.exp(); value = probability[:, 0] - probability[:, 2]
        target_value = y[ix, 0] - y[ix, 2]
        ce = -(y[ix]*logprob).sum(-1).mean()
        mse = (value-target_value).square().mean(); mae = (value-target_value).abs().mean()
        kl = F.kl_div(logprob, base[ix], reduction='batchmean') if name.startswith('anchor/') else 0.
        objectives.append(ce+.5*mse+kl)
        if constraint_projection:
            for loss in [ce, mae]:
                protected.append(torch.cat([g.flatten() for g in torch.autograd.grad(loss, head.parameters(), retain_graph=True)]))
        losses[name] = dict(ce=ce.item(), mae=mae.item())
    optimizer.zero_grad(set_to_none=True)
    objective = torch.stack(objectives).mean(); objective.backward()
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite value-head gradient')
    before = parameter_vector(head); optimizer.step(); proposal = parameter_vector(head)-before
    actual, projection = project_update(proposal, torch.stack(protected)) if constraint_projection else (proposal, dict(disabled=True))
    assign_vector(head, before+actual)
    return dict(objective=objective.item(), training_losses=losses, projection=projection,
                actual_norm=actual.norm().item(), proposal_norm=proposal.norm().item())


def train(cache, old, out, steps=1000, batch=96, lr=3e-6, seed=20261105, constraint_projection=True):
    torch.set_num_threads(4)
    head = load_head(old); initial = copy.deepcopy(head)
    data = load_cache(cache); torch.manual_seed(seed)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    plan = dict(steps=steps, mandatory_checkpoints=[500, 1000], batch_per_training_stratum=batch,
                learning_rate=lr, seed=seed, trainable='old-value-MLP only, original encoders and fresh policy/Q frozen',
                objective='mean of six stratum-specific CE+0.5*value_MSE+anchor_KL losses',
                constraints='Project actual AdamW displacement against separate CE and value_MAE gradients for all six training strata' if constraint_projection else 'Disabled for numerical diagnosis only; not a replacement candidate',
                validation_used_for_updates=False, early_checkpoint_selection=False, cloud_allowed=False,
                old_sha256=hashlib.file_digest(open(old, 'rb'), 'sha256').hexdigest(),
                cache_sha256={name: hashlib.file_digest(open(Path(cache)/(name+'-features.npz'), 'rb'), 'sha256').hexdigest() for name in data})
    write(out/'plan.json', plan)
    reference = {name: metrics(initial, *data[name]) for name in ['balanced', 'general']}
    write(out/'initial-validation.json', reference)
    tasks = make_tasks(initial, data)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=.01)
    accepted = 0; displacement_sum = 0.; projection_fractions = []
    initial_vector = parameter_vector(head)
    for step in range(1, steps+1):
        update = train_step(head, tasks, optimizer, batch, constraint_projection)
        actual_norm = update['actual_norm']; accepted += int(actual_norm > 1e-9)
        displacement_sum += actual_norm
        projection_fractions.append(actual_norm/max(update['proposal_norm'], 1e-20))
        if step % 50 == 0:
            validation = {name: metrics(head, *data[name]) for name in ['balanced', 'general']}
            row = dict(step=step, objective=update['objective'], validation=validation, training_losses=update['training_losses'],
                       accepted_updates=accepted, cumulative_update_norm=displacement_sum,
                       parameter_change_norm=(parameter_vector(head)-initial_vector).norm().item(),
                       mean_retained_update_fraction=float(np.mean(projection_fractions[-50:])), projection=update['projection'])
            with (out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            write(out/'status.json', row); print(json.dumps(row), flush=True)
        if step in [500, 1000]:
            torch.save(dict(value_head=head.state_dict(), step=step, plan=plan, validation=validation,
                            accepted_updates=accepted, early_checkpoint_selection=False), out/f'head-{step}.pt')
    if accepted != steps:
        raise RuntimeError(f'Only {accepted} nonzero updates of {steps}; no-op training is not a repair')
    write(out/'result.json', dict(complete=True, reference=reference, final_validation=validation,
                                 accepted_updates=accepted, cumulative_update_norm=displacement_sum,
                                 parameter_change_norm=(parameter_vector(head)-initial_vector).norm().item(),
                                 chess_gate_run=False, early_checkpoint_selection=False))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--cache', required=True); p.add_argument('--old', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--diagnostic-no-projection', action='store_true', help='Numerical ablation only; never selects the gate candidate.')
    a = p.parse_args()
    train(a.cache, a.old, a.out, constraint_projection=not a.diagnostic_no_projection)


if __name__ == '__main__':
    main()
