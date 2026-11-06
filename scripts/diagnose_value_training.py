"""Audit cached training features and conflicting value-head gradients locally."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn


def load_head(path):
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    config = state['model_config']
    head = nn.Sequential(nn.Linear(config['width'], config['head_width']), nn.GELU(),
                         nn.Linear(config['head_width'], 3))
    head.load_state_dict({k[6:]: v for k, v in state['model'].items() if k.startswith('value.')})
    return head.eval()


def load_cache(folder):
    data = {}
    for name in ['repair', 'anchor', 'balanced', 'general']:
        with np.load(Path(folder) / (name + '-features.npz')) as cache:
            x = torch.from_numpy(cache['x'].astype(np.float32))
            y = torch.from_numpy(cache['value'].copy())
            assert torch.isfinite(x).all() and torch.isfinite(y).all()
            assert torch.allclose(y.sum(-1), torch.ones(len(y)), atol=2e-5)
            data[name] = (x, y)
    return data


def strata(y):
    value = y[:, 0] - y[:, 2]
    return {'winning': value > .25, 'neutral': value.abs() <= .25, 'losing': value < -.25}


def audit(head, data):
    report = {'distributions': {}, 'training_gradients': {}, 'gradient_cosines': {}}
    gradients = {}
    for name, (x, y) in data.items():
        report['distributions'][name] = {}
        for label, mask in strata(y).items():
            xx, yy = x[mask], y[mask]
            with torch.no_grad():
                probabilities = torch.cat([head(b).softmax(-1) for b in xx.split(512)])
                prediction = probabilities[:, 0] - probabilities[:, 2]
                truth = yy[:, 0] - yy[:, 2]
            report['distributions'][name][label] = dict(
                count=len(xx), target_mean=truth.mean().item(), prediction_mean=prediction.mean().item(),
                signed_error=(prediction-truth).mean().item(), mae=(prediction-truth).abs().mean().item(),
                mean_probability_residual=(probabilities-yy).mean(0).tolist())
            # Only TRAIN subsets participate in gradient-conflict diagnosis.
            if name in ['repair', 'anchor']:
                index = torch.linspace(0, len(xx)-1, min(2048, len(xx))).long()
                logits = head(xx[index]); loss = -(yy[index]*logits.log_softmax(-1)).sum(-1).mean()
                grad = torch.cat([g.flatten() for g in torch.autograd.grad(loss, head.parameters())])
                key = name + '/' + label
                gradients[key] = grad
                report['training_gradients'][key] = dict(norm=grad.norm().item(), ce=loss.item())
    for a, ga in gradients.items():
        report['gradient_cosines'][a] = {b: torch.nn.functional.cosine_similarity(ga, gb, dim=0).item()
                                        for b, gb in gradients.items()}
    return report


def main():
    p = argparse.ArgumentParser(); p.add_argument('--cache', required=True)
    p.add_argument('--old', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); torch.set_num_threads(4)
    result = audit(load_head(a.old), load_cache(a.cache))
    Path(a.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
