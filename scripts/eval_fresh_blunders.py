"""Coherent finite-budget teacher regret on frozen shared puzzle roots.

Candidate union contains every compared model's direct/search choice, original
solution, and a fresh teacher proposal. Mates are events, never artificial CP.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json,multiprocessing,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess,chess.engine
import numpy as np
from eval_fresh_puzzles import board_for,paired_groups


def inspect_chunk(item):
    tasks,engine,nodes=item;results=[]
    with chess.engine.SimpleEngine.popen_uci(engine,timeout=120) as sf:
        sf.configure({'Threads':1,'Hash':64,'Skill Level':20,'UCI_LimitStrength':False})
        for job,choices in tasks:
            b=board_for(job)
            coarse=sf.analyse(b,chess.engine.Limit(nodes=min(nodes,100000)),multipv=min(4,b.legal_moves.count()),game=object())
            candidates=list(dict.fromkeys([x['pv'][0] for x in coarse]+[b.parse_uci(m) for m in choices.values()]+[b.parse_uci(job['solution'][0])]))
            infos=sf.analyse(b,chess.engine.Limit(nodes=nodes),root_moves=candidates,multipv=len(candidates),game=object())
            scores={}
            for info in infos:
                score=info['score'].pov(b.turn);scores[info['pv'][0].uci()]={'mate':score.mate()} if score.is_mate() else {'cp':score.score()}
            positive_mate=any(s.get('mate',0)>0 for s in scores.values())
            safe=any('cp' in s or s.get('mate',0)>0 for s in scores.values())
            finite=[s['cp'] for s in scores.values() if 'cp' in s]
            stats={}
            for method,move in choices.items():
                chosen=scores[move];regret=None
                if not positive_mate and 'cp' in chosen and finite:regret=max(finite)-chosen['cp']
                stats[method]=dict(move=move,cp_regret=regret,blunder100=regret is not None and regret>=100,
                    missed_winning_mate=positive_mate and chosen.get('mate',0)<=0,
                    avoidable_reported_mate_loss=safe and chosen.get('mate',0)<0)
            results.append(dict(key=job['key'],group=job['group'],category=job['category'],teacher_scores=scores,methods=stats))
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--puzzle-result',required=True);p.add_argument('--jobs',required=True)
    p.add_argument('--stockfish',required=True);p.add_argument('--out',required=True);p.add_argument('--nodes',type=int,default=500000);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();report=json.loads(Path(a.puzzle_result).read_text());keys=set(report['search_keys'])
    jobs={j['key']:j for j in (json.loads(l) for l in Path(a.jobs).read_text().splitlines()) if j['key'] in keys}
    choices={key:{} for key in keys};methods=[]
    for i,model in enumerate(report['results']):
        for mode in ['direct','search']:
            method=f'{i}-{mode}';methods.append(method)
            for prediction in model[mode]['predictions']:
                if prediction['key'] in keys:choices[prediction['key']][method]=prediction['predicted']
    if len(jobs)!=256 or any(len(v)!=len(methods) for v in choices.values()):raise ValueError('Incomplete frozen shared predictions')
    tasks=[(jobs[key],choices[key]) for key in sorted(keys)];rows=[]
    chunks=[(tasks[i:i+16],a.stockfish,a.nodes) for i in range(0,len(tasks),16)]
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for results in pool.map(inspect_chunk,chunks):rows.extend(results)
    summary={}
    for method in methods:
        stats=[r['methods'][method] for r in rows];cp=[s['cp_regret'] for s in stats if s['cp_regret'] is not None]
        summary[method]=dict(cp_positions=len(cp),mean_cp_regret=float(np.mean(cp)) if cp else None,
            blunders100=sum(s['blunder100'] for s in stats),missed_winning_mates=sum(s['missed_winning_mate'] for s in stats),
            avoidable_reported_mate_losses=sum(s['avoidable_reported_mate_loss'] for s in stats))
    paired={}
    job_order=[jobs[r['key']] for r in rows]
    for mode in ['direct','search']:
        before=f'1-{mode}';after=f'2-{mode}'
        paired[mode]=paired_groups([r['methods'][before]['blunder100'] for r in rows],[r['methods'][after]['blunder100'] for r in rows],job_order)
    out=dict(teacher_nodes=a.nodes,positions=len(rows),summary=summary,paired_blunder_rate_change=paired,rows=rows,
        models=[dict(index=i,checkpoint=m['checkpoint'],sha256=m['sha256']) for i,m in enumerate(report['results'])],
        scope='Finite-node teacher candidate-union regret, not exact chess truth. Mate events separate from centipawns.')
    Path(a.out).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
