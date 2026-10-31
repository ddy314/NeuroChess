"""Summarize single-game latency, simulation throughput and finite teacher quality."""
import argparse,json
from pathlib import Path
import numpy as np


def interval(values):
    a=np.asarray(values,float);rng=np.random.default_rng(131331)
    return np.quantile(a[rng.integers(len(a),size=(20000,len(a)))].mean(1),[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=Path(a.root)
    d=json.loads((r/'validation/result.json').read_text());summary={};by={}
    for v in d['suite']:
        rows=[next(x for x in p['methods'] if x['method']==v['name']) for p in d['positions']];by[v['name']]=rows
        times=[x['seconds'] for x in rows];sims=[x['stats']['completed_simulations'] for x in rows]
        cp=[x['cp_regret'] for x in rows if x['cp_regret'] is not None]
        summary[v['name']]=dict(samples=len(rows),median_seconds=float(np.median(times)),p95_seconds=float(np.quantile(times,.95)),max_seconds=max(times),over_2s=sum(t>2 for t in times),
            simulations_min=min(sims),simulations_median=float(np.median(sims)),simulations_max=max(sims),mean_inputs=float(np.mean([x['large_positions'] for x in rows])),
            mean_native_seconds=float(np.mean([x['stats']['native_seconds'] for x in rows])),mean_inference_seconds=float(np.mean([x['stats']['inference_seconds'] for x in rows])),
            time_stops=sum(x['stats'].get('time_stops',0) for x in rows),mean_cp_regret=float(np.mean(cp)) if cp else None,cp_positions=len(cp),blunders_100=sum(x>=100 for x in cp))
    gains=[b['stats']['completed_simulations']/a['stats']['completed_simulations']-1 for a,b in zip(by['reference-2s'],by['optimized-2s']) if a['stats']['completed_simulations']>0 and a['stats'].get('time_stops') and b['stats'].get('time_stops')]
    speed=[1-b['seconds']/a['seconds'] for a,b in zip(by['fixed-2048'],by['cached-2048'])]
    cpdelta=[b['cp_regret']-a['cp_regret'] for a,b in zip(by['reference-2s'],by['optimized-2s']) if a['cp_regret'] is not None and b['cp_regret'] is not None]
    result=dict(summary=summary,paired=dict(time_budget_simulation_gain_mean=float(np.mean(gains)),time_budget_simulation_gain95=interval(gains),eligible_timed_roots=len(gains),
        fixed2048_time_reduction_mean=float(np.mean(speed)),fixed2048_time_reduction95=interval(speed),
        fixed2048_move_parity=sum(a['move']==b['move'] for a,b in zip(by['fixed-2048'],by['cached-2048'])),
        timed_cp_change=float(np.mean(cpdelta)),timed_cp_change95=interval(cpdelta)),
        note='Cold single-game roots. Paired root bootstrap is approximate. Simulation counts include terminal backups/cache reuse and are not independent neural observations. Fixed2048 parity controls finite numerical effects from changing GPU batch shape.')
    final=r/'final-validation/result.json'
    if final.exists():
        fd=json.loads(final.read_text());fs={};fr={}
        for variant in fd['suite']:
            name=variant['name'];rows=[next(x for x in p['methods'] if x['method']==name) for p in fd['positions']];fr[name]=rows
            times=[x['seconds'] for x in rows];sims=[x['stats']['completed_simulations'] for x in rows]
            fs[name]=dict(**fd['summary'][name],p95_seconds=float(np.quantile(times,.95)),max_seconds=max(times),over_2s=sum(t>2 for t in times),
                simulations_median=float(np.median(sims)),simulations_min=min(sims),simulations_max=max(sims))
        deltas=[b['cp_regret']-a['cp_regret'] for a,b in zip(fr['reference-2s'],fr['optimized-2s']) if a['cp_regret'] is not None and b['cp_regret'] is not None]
        result['final_validation']=dict(summary=fs,paired_cp_change=float(np.mean(deltas)),paired_cp_change95=interval(deltas),
            note='Separate new16 roots, frozen after discovering the guard on original32 roots. CP teacher is finite; single-root labels are not game outcomes.')
        case=json.loads((r/'guard-case/result.json').read_text())['positions'][0]
        result['diagnostic_case']=case
    arena=r/'arena/result.json'
    if arena.exists():
        duel=json.loads(arena.read_text());games=duel['games']['puct-512'];timings={}
        for name in ['optimized-2s','puct-512']:
            decisions=[x for g in games for x in g['decisions'][name]];ts=[x['seconds'] for x in decisions]
            timings[name]=dict(choices=len(ts),median_seconds=float(np.median(ts)),p95_seconds=float(np.quantile(ts,.95)),max_seconds=max(ts),over_2s=sum(t>2 for t in ts),mean_inputs=float(np.mean([x['large_positions'] for x in decisions])),median_simulations=float(np.median([x['simulations'] for x in decisions])))
        result['arena']=dict(result=duel['results']['puct-512'],timings=timings,note='Eight games/four opening pairs against lower-compute512sim, not an equal-time algorithm comparison or reliable Elo estimate.')
    (r/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
