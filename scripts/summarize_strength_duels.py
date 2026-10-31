"""Paired, budget-aware summaries for the frozen causal search matches."""
import argparse
import json
from pathlib import Path
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_search_causes import paired_ci


def pair_scores(rows):
    grouped={}
    for row in rows:
        if row['score'] is not None:grouped.setdefault(row['opening'],[]).append(row['score'])
    return {k:float(np.mean(v)) for k,v in grouped.items() if len(v)==2}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);summary={};results={}
    for folder in ['candidate-duel','baseline-duel','reuse-duel','puct-reuse-control','coverage-only-duel']:
        path=root/folder/'result.json'
        if not path.exists():continue
        result=json.loads(path.read_text());results[folder]=result
        for opponent,rows in result['games'].items():
            key=folder+'/'+opponent;players={}
            for name in rows[0]['choices']:
                choices=sum(r['choices'][name] for r in rows);positions=sum(r['large_model_positions'][name] for r in rows)
                reuses=sum(r['stats'][name].get('tree_reuses',0) for r in rows)
                retained=sum(r['stats'][name].get('retained_nodes',0) for r in rows)
                players[name]=dict(choices=choices,large_positions=positions,positions_per_choice=positions/max(1,choices),
                    seconds_per_choice=sum(r['seconds'][name] for r in rows)/max(1,choices),tree_reuses=reuses,
                    reuse_fraction=reuses/max(1,choices),retained_nodes_per_reuse=retained/max(1,reuses),
                    correction_requests_per_choice=sum(r['stats'][name].get('correction_positions',0) for r in rows)/max(1,choices) if name!=opponent else None,
                    cache_hits_per_choice=sum(r['stats'][name].get('cache_hits',0) for r in rows)/max(1,choices) if name!=opponent else None)
            summary[key]=dict(**result['results'][opponent],players=players,audit=json.loads((root/folder/'audit.json').read_text())['passed'])
    contrasts={}
    for variant in ['candidate-duel','reuse-duel','coverage-only-duel']:
        if variant not in results or 'baseline-duel' not in results:continue
        for opponent,rows in results[variant]['games'].items():
            if opponent not in results['baseline-duel']['games']:continue
            v=pair_scores(rows);b=pair_scores(results['baseline-duel']['games'][opponent]);common=sorted(v.keys()&b.keys())
            if common:
                delta=[v[k]-b[k] for k in common]
                contrasts[variant+'/'+opponent]=dict(common_opening_pairs=len(common),mean_score_change=float(np.mean(delta)),
                    bootstrap95=paired_ci(delta),variant_score=float(np.mean([v[k] for k in common])),baseline_score=float(np.mean([b[k] for k in common])))
    if 'puct-reuse-control' in results and 'baseline-duel' in results:
        v=pair_scores(results['puct-reuse-control']['games']['puct-31-no-reuse']);b=pair_scores(results['baseline-duel']['games']['puct-31']);common=sorted(v.keys()&b.keys());delta=[v[k]-b[k] for k in common]
        contrasts['PUCT-reuse-off-minus-on']=dict(common_opening_pairs=len(common),mean_contender_score_change=float(np.mean(delta)),bootstrap95=paired_ci(delta))
    report=dict(summary=summary,paired_contrasts=contrasts,total_completed_games=sum(x['completed'] for x in summary.values()),
        note='Score differences use only common opening pairs. Bootstrap is approximate and wide for eight pairs. Elo is conditional on these models, budgets and openings; unfinished games excluded. These are fixed inference caps, not fixed-time or equal actual-forward tests.')
    (root/'duel-analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
