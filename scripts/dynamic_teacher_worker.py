"""CPU-only adaptive teacher checks for roots and independent successors."""
import hashlib,threading,time
import chess,chess.engine
from neurochess.teacher_uci import FastTeacher
from neurochess.dynamic_curriculum import board_for,state_id,digest
from scripts.targeted_mistakes import classify,needs_repair,stable_labels
from neurochess.teacher_stability import qualify,qualify_guarded,qualified_decision
from neurochess.targets import score_targets

ENGINE=None


def initialize(binary,hash_mb=64):
    global ENGINE
    ENGINE=FastTeacher(binary,hash_mb=hash_mb);threading._register_atexit(ENGINE.quit)


def label(task):
    j=task['job'];pred=task['prediction'];spec=task['spec'];b=board_for(j);attempts=[];value_audit=[];started=time.monotonic()
    if b.is_game_over(claim_draw=True):raise ValueError('Terminal successor must use explicit terminal target')
    def analyze(budget,moves=None,global_value=False):
        infos=ENGINE.analyse(b,chess.engine.Limit(nodes=budget),multipv=1 if global_value else min(4,b.legal_moves.count()) if moves is None else len(moves),root_moves=None if moves is None else [b.parse_uci(m) for m in moves])
        pvs=[]
        for info in infos:
            s=info['score'].white();pv=dict(line=' '.join(m.uci() for m in info['pv']),depth=info['depth'],nodes=info['nodes'])
            pv['mate' if s.is_mate() else 'cp']=s.mate() if s.is_mate() else s.score()
            if info.get('lowerbound') or info.get('upperbound'):pv['bound']='lower' if info.get('lowerbound') else 'upper'
            pvs.append(pv)
        if moves is not None and {p['line'].split()[0] for p in pvs}!=set(moves):raise ValueError('Teacher omitted a coherent candidate')
        (value_audit if global_value else attempts).append(dict(budget=budget,independent=True,candidates=moves,teacher=pvs,**({'unrestricted':True} if global_value else {})))
        return pvs
    coarse=analyze(spec['seed_nodes']);moves=list(dict.fromkeys([p['line'].split()[0] for p in coarse]+pred['top_moves']+[pred['move']]))
    first=analyze(spec['screen_nodes'],moves);decision=classify(first,pred['move'],pred['value'],b.turn)
    candidate_values={p['line'].split()[0]:score_targets(p,b.turn)[1] for p in first}
    q_mistake=any(abs(v-candidate_values[m])>=.25 for m,v in pred.get('support_q',{}).items() if m in candidate_values)
    audit=int(state_id(j)[:12],16)%10000<int(spec['easy_audit_fraction']*10000)
    must_confirm=task.get('force_confirm',False) or needs_repair(decision) or q_mistake or audit or j['split']!='train'
    teacher=first;stable=None;supervision=None
    if must_confirm:
        previous=None
        for budget in spec['confirm_nodes']:
            teacher=analyze(budget,moves)
            if previous is not None and budget>=spec['minimum_confirm_nodes']:
                supervision=qualify(previous,teacher,b.turn,spec['stability_cp'],spec['wdl_stability'])
                stable=supervision['value'] or supervision['policy'] or bool(supervision['q_moves'])
                decision=qualified_decision(previous,teacher,pred,b.turn,supervision)
                fast=spec.get('early_stop_when_any_target_is_stable',False) and j['split']=='train'
                if (fast and stable) or supervision['value'] or decision['usable_error']:break
            previous=teacher
        if supervision is None:raise ValueError('At least two independent confirmation budgets required')
        if spec.get('unrestricted_value_guard'):
            if supervision['value'] or supervision['policy'] or task.get('needs_best_reply'):
                cached=task.get('cached_value_audit')
                if cached:value_audit=cached
                else:
                    for budget in spec['value_guard_nodes']:analyze(budget,global_value=True)
            supervision=qualify_guarded(attempts[-2]['teacher'],teacher,b.turn,spec,value_audit)
            stable=supervision['value'] or supervision['policy'] or bool(supervision['q_moves'])
            decision=qualified_decision(attempts[-2]['teacher'],teacher,pred,b.turn,supervision)
    pending=[] if supervision is None else ([h for h in ['policy','value'] if not supervision[h]]+(['Q_actions'] if len(supervision['q_moves'])<len(moves) else []))
    return dict(job=j,prediction=pred,teacher=teacher,stable=stable,decision=decision,attempts=attempts,pending_heads=pending,
                repair=bool(stable and decision.get('usable_error',False)),training_eligible=stable is True,supervision=supervision,
                teacher_spec=spec,value_audit=value_audit,coarse_audit=audit,seconds=time.monotonic()-started,cache_hit=False)
