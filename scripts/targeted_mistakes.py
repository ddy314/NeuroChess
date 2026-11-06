"""Targeted teacher checks on student and teacher continuations. No training side effects."""
import math
import numpy as np
from neurochess.targets import score_targets


def classify(pvs, move, value, turn, regret_cp=100, value_error=.25):
    scores={p['line'].split()[0]:score_targets(p,turn) for p in pvs}
    if move not in scores: raise ValueError('Student choice absent from coherent teacher candidate set')
    best=max(scores,key=lambda m:scores[m][0]); chosen=scores[move]; target=scores[best]
    if not math.isfinite(value): raise ValueError('Nonfinite student value')
    # Mate length differences between already winning moves are not centipawn blunders.
    bp=next(p for p in pvs if p['line'].split()[0]==best)
    cp=next(p for p in pvs if p['line'].split()[0]==move)
    if 'mate' not in bp and 'mate' not in cp: regret=max(0.,target[0]-chosen[0]); bad=regret>=regret_cp
    else:
        regret=None; bad=(target[1]>.99 and chosen[1]<.99) or (target[1]>-.99 and chosen[1]<-.99)
    return dict(best=best,student=move,regret_cp=regret,policy_error=bool(bad),
                value_error=abs(float(value)-target[1]),value_mistake=abs(float(value)-target[1])>=value_error,
                target_wdl=target[2].tolist(),target_value=target[1])


def stable_labels(first, second, turn, tolerance_cp=30):
    a={p['line'].split()[0]:p for p in first};b={p['line'].split()[0]:p for p in second}
    if set(a)!=set(b):return False
    for move,p in a.items():
        q=b[move]
        if ('mate' in p)!=('mate' in q):return False
        if 'mate' in p:
            if np.sign(score_targets(p,turn)[1])!=np.sign(score_targets(q,turn)[1]):return False
        elif abs(p['cp']-q['cp'])>tolerance_cp:return False
    return True


def needs_repair(decision):
    return decision['policy_error'] or decision['value_mistake']
