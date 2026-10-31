"""Local paired games for the experimental CPU student / sparse-query search."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if importlib.util.find_spec('chess') is None:
    for site in (ROOT/'.venv/lib').glob('python*/site-packages'):
        if (site/'chess').is_dir():
            sys.path.append(str(site)); break
import torch
from neurochess.best_first import BestFirstPlayer
from neurochess.multihead import MultiheadPlayer
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.play import run_match
from neurochess.train import load_model


def digest(path):
    with open(path,'rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cheap',required=True); p.add_argument('--checkpoint')
    p.add_argument('--stockfish',required=True); p.add_argument('--openings',required=True)
    p.add_argument('--out',required=True); p.add_argument('--methods',default='best-first,mcts')
    p.add_argument('--resume',action='store_true',help='Continue completed method blocks with the identical plan')
    p.add_argument('--pairs',type=int,default=4); p.add_argument('--nodes',type=int,default=1000)
    p.add_argument('--expansions',type=int,default=256); p.add_argument('--neural-budget',type=int,default=32)
    p.add_argument('--quiescence-depth',type=int,default=0)
    p.add_argument('--simulations',type=int,default=512); p.add_argument('--device',default='cuda')
    p.add_argument('--small-simulations',type=int,default=31)
    p.add_argument('--small-leaf-batch',type=int,default=4)
    p.add_argument('--root-policy-weight',type=float,default=0.)
    p.add_argument('--align-root-objective',action='store_true')
    p.add_argument('--local-exploration',action='store_true')
    p.add_argument('--correction-clip',type=float,default=0.)
    p.add_argument('--root-value-clip',type=float,default=0.)
    p.add_argument('--cpu-correction-clip',type=float,default=0.)
    p.add_argument('--mate-nodes',type=int,default=0);p.add_argument('--mate-depth',type=int,default=7)
    p.add_argument('--tactical-nodes',type=int,default=0);p.add_argument('--tactical-depth',type=int,default=2)
    p.add_argument('--verify-final',action='store_true')
    p.add_argument('--verification-reserve',type=int,default=8)
    p.add_argument('--anchor-decay',type=float,default=0.)
    p.add_argument('--correction-scheduler',choices=['hierarchical','frontier'],default='hierarchical')
    p.add_argument('--max-plies',type=int,default=600); p.add_argument('--max-seconds',type=float,default=1200)
    a=p.parse_args(); methods=a.methods.split(',')
    if not methods or any(m not in ['cheap','best-first','mcts','mcts-small','direct'] for m in methods):
        p.error('Methods: cheap,best-first,mcts,mcts-small,direct')
    if any(m!='cheap' for m in methods) and not a.checkpoint:
        p.error('Large-model checkpoint required for these methods')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=a.resume)
    fens=[s.strip() for s in Path(a.openings).read_text().splitlines() if s.strip()][:a.pairs]
    if a.pairs<1 or len(fens)!=a.pairs:
        p.error('Not enough opening pairs')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    network=None
    if a.checkpoint:
        model,state=load_model(a.checkpoint,a.device); del state
        network=MultiheadPlayer(model,a.device)
    plan_args=vars(a).copy();plan_args.pop('resume',None)
    if not a.local_exploration:plan_args.pop('local_exploration',None)
    for key,default in dict(correction_clip=0.,root_value_clip=0.,cpu_correction_clip=0.,mate_nodes=0,mate_depth=7,tactical_nodes=0,tactical_depth=2).items():
        if plan_args[key]==default:plan_args.pop(key,None)
    plan=dict(args=plan_args,cheap_sha256=digest(a.cheap),
        checkpoint_sha256=digest(a.checkpoint) if a.checkpoint else None,
        stockfish_sha256=digest(a.stockfish),openings=fens,
        note='Development paired games, not absolute Elo or a final independent test.')
    plan_path=out/'plan.json'
    if a.resume and (not plan_path.exists() or json.loads(plan_path.read_text())!=plan):
        raise ValueError('Resume requires the identical arena plan and input fingerprints')
    plan_path.write_text(json.dumps(plan,indent=2))
    reports={}
    for name in methods:
        existing=out/name/'result.json'
        if a.resume and existing.exists():
            report=json.loads(existing.read_text())
            if 'search_config' not in report:
                raise ValueError(f'Incomplete method report: {existing}')
            reports[name]=report
            continue
        if name=='direct': player=network
        elif name=='mcts': player=NativeMCTSPlayer(network,simulations=a.simulations,leaf_batch_size=64)
        elif name=='mcts-small': player=NativeMCTSPlayer(network,simulations=a.small_simulations,leaf_batch_size=a.small_leaf_batch)
        else: player=BestFirstPlayer(a.cheap,network if name=='best-first' else None,
            expansions=a.expansions,neural_budget=a.neural_budget,quiescence_depth=a.quiescence_depth,
            root_policy_weight=a.root_policy_weight,anchor_decay=a.anchor_decay,
            correction_scheduler=a.correction_scheduler,align_root_objective=a.align_root_objective,
            verify_final=a.verify_final,verification_reserve=a.verification_reserve,
            local_exploration=a.local_exploration,correction_clip=a.correction_clip,
            mate_nodes=a.mate_nodes,mate_depth=a.mate_depth,tactical_nodes=a.tactical_nodes,tactical_depth=a.tactical_depth,
            root_value_clip=a.root_value_clip,cpu_correction_clip=a.cpu_correction_clip)
        before=network.search_stats['forward_positions'] if network else 0
        try:
            def progress(completed,total):
                state=dict(method=name,completed=completed,total=total,search_stats=dict(player.search_stats))
                (out/'status.json').write_text(json.dumps(state,indent=2)); print(json.dumps(state),flush=True)
            report=run_match(player,a.stockfish,fens,out/name,a.nodes,a.max_plies,a.max_seconds,progress)
            report['search_config']=player.config if hasattr(player,'config') else None
            report['search_stats']=dict(player.search_stats)
            report['large_model_positions']=(network.search_stats['forward_positions']-before) if network else 0
            (out/name/'result.json').write_text(json.dumps(report,indent=2))
            reports[name]=report; (out/'result.json').write_text(json.dumps(reports,indent=2))
        finally:
            if hasattr(player,'close'): player.close()


if __name__=='__main__': main()
