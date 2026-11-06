"""Qualify the actual P/Q/WDL targets, without inventing missing supervision."""
import numpy as np
from .targets import score_targets
from scripts.targeted_mistakes import classify


def qualify(previous,current,turn,cp_tolerance=30,wdl_tolerance=.02):
    a={p['line'].split()[0]:p for p in previous};b={p['line'].split()[0]:p for p in current}
    if not a or set(a)!=set(b):raise ValueError('Stability requires identical coherent candidate sets')
    cp_stable=[];q_stable=[]
    def target_close(x,y):return bool(np.max(abs(x[2]-y[2]))<=wdl_tolerance+1e-7 and abs(x[1]-y[1])<=wdl_tolerance+1e-7)
    scores=[{m:score_targets(p,turn) for m,p in group.items()} for group in [a,b]]
    for m in a:
        if 'bound' in a[m] or 'bound' in b[m]:continue
        if target_close(scores[0][m],scores[1][m]):q_stable.append(m)
        same_kind=('mate' in a[m])==('mate' in b[m])
        if same_kind and (np.sign(a[m]['mate'])==np.sign(b[m]['mate']) and abs(scores[0][m][0]-scores[1][m][0])<=cp_tolerance if 'mate' in a[m] else abs(a[m]['cp']-b[m]['cp'])<=cp_tolerance):cp_stable.append(m)
    best=[max(s,key=lambda m:s[m][0]) for s in scores]
    value=all('bound' not in g[m] for g,m in zip([a,b],best)) and target_close(scores[0][best[0]],scores[1][best[1]])
    return dict(policy=len(cp_stable)==len(a),value=bool(value),q_moves=sorted(q_stable),cp_stable_moves=sorted(cp_stable),
                candidates=len(a),wdl_tolerance=wdl_tolerance,cp_tolerance=cp_tolerance,
                acceptance='all_cp_stable' if len(cp_stable)==len(a) else 'target_stable_masked')


def qualify_guarded(previous,current,turn,spec,value_audit=None):
    """Candidate-restricted agreement cannot certify the global board value.

    Keep the established numeric targets, but mask V unless independent,
    unrestricted PV1 searches agree with both coherent candidate searches.
    A globally discovered move absent from the candidate union also masks P.
    Conditional action Q targets retain their own per-action qualification.
    """
    result=qualify(previous,current,turn,spec['stability_cp'],spec['wdl_stability'])
    if not spec.get('unrestricted_value_guard'):return result
    audit=value_audit or [];verified=len(audit)==2
    if verified:
        verified=all(s.get('independent') and s.get('unrestricted') and s.get('candidates') is None and len(s['teacher'])==1 for s in audit)
        verified=verified and [s['budget'] for s in audit]==spec['value_guard_nodes']
    global_stable=False;agreement=False;missing=[]
    if verified:
        global_targets=[score_targets(s['teacher'][0],turn) for s in audit]
        coherent=[max((score_targets(p,turn) for p in ps),key=lambda t:t[0]) for ps in [previous,current]]
        tol=spec['wdl_stability']+1e-7
        global_stable=all('bound' not in s['teacher'][0] for s in audit) and np.max(abs(global_targets[0][2]-global_targets[1][2]))<=tol and abs(global_targets[0][1]-global_targets[1][1])<=tol
        agreement=all(np.max(abs(a[2]-b[2]))<=tol and abs(a[1]-b[1])<=tol for a,b in zip(coherent,global_targets))
        candidates={p['line'].split()[0] for p in current}
        missing=sorted({s['teacher'][0]['line'].split()[0] for s in audit}-candidates)
    result['value']=bool(result['value'] and verified and global_stable and agreement)
    result['policy']=bool(result['policy'] and verified and not missing)
    result['unrestricted_guard']=dict(verified=bool(verified),global_wdl_stable=bool(global_stable),coherent_global_agreement=bool(agreement),missing_global_moves=missing)
    return result


def qualified_decision(previous,current,pred,turn,supervision):
    d=[classify(p,pred['move'],pred['value'],turn) for p in [previous,current]]
    value=supervision['value'] and all(x['value_mistake'] for x in d)
    policy=supervision['policy'] and all(x['policy_error'] for x in d)
    known={p['line'].split()[0]:score_targets(p,turn)[1] for p in current}
    best=d[-1]['best'];move=pred['move']
    q=all(x['policy_error'] for x in d) and {best,move}.issubset(supervision['q_moves']) and known[best]-known[move]>=.05
    first={p['line'].split()[0]:score_targets(p,turn)[1] for p in previous}
    q_errors={m:min(abs(float(v)-known[m]),abs(float(v)-first[m])) for m,v in pred.get('support_q',{}).items() if m in supervision['q_moves']}
    direct_q=any(error>=.25 for error in q_errors.values())
    return dict(d[-1],usable_error=bool(value or policy or q or direct_q),q_mistake=bool(direct_q),qualified_q_errors=q_errors,
                repair_heads=[name for name,yes in [('policy',policy),('Q',q or direct_q),('value',value)] if yes])
