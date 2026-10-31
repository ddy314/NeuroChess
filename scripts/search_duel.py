"""Resumable direct, color-swapped search matches using one frozen GPU model."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if importlib.util.find_spec('chess') is None:sys.path.append(str(next((ROOT/'.venv/lib').glob('python*/site-packages'))))
import chess
import chess.pgn
import numpy as np
import torch
import neurochess.best_first as bf
import neurochess.native_mcts as mcts
from neurochess.multihead import MultiheadPlayer
from neurochess.train import load_model


def digest(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def rating(score):
    return 400*math.log10(score/(1-score)) if 0<score<1 else None


def setup_game(board,white,black):
    game=chess.pgn.Game();game.setup(board)
    # Game.setup normalizes away an EP square with no legal capture, but that
    # square is still part of the neural observation supplied to the match.
    raw_fen=board.fen(en_passant='fen')
    if raw_fen!=chess.STARTING_FEN:
        game.headers['FEN']=raw_fen;game.headers['SetUp']='1'
    game.headers['White']=white;game.headers['Black']=black
    return game


def summarize(rows,scheduled):
    completed=[r for r in rows if r['score'] is not None];by_pair={}
    for r in completed:by_pair.setdefault(r['opening'],[]).append(r['score'])
    pairs=np.array([np.mean(v) for v in by_pair.values() if len(v)==2])
    result=dict(scheduled=scheduled,attempted=len(rows),completed=len(completed),truncated=sum(r['score'] is None for r in rows),
        wins=sum(r['score']==1 for r in completed),draws=sum(r['score']==.5 for r in completed),losses=sum(r['score']==0 for r in completed),
        independent_completed_pairs=len(pairs))
    if len(pairs):
        rng=np.random.default_rng(713031);samples=pairs[rng.integers(len(pairs),size=(20000,len(pairs)))].mean(1)
        ci=np.quantile(samples,[.025,.975]).tolist();score=float(pairs.mean())
        # Distribution-free bound for independent opening-pair scores in [0,1].
        eps=math.sqrt(math.log(40)/(2*len(pairs)));conservative=[max(0.,score-eps),min(1.,score+eps)]
        result.update(paired_score=score,score_bootstrap95=ci,relative_elo=rating(score),
            relative_elo_bootstrap95=[rating(v) for v in ci],score_hoeffding95=conservative,
            relative_elo_hoeffding95=[rating(v) for v in conservative],
            pentanomial=[int(np.sum(np.isclose(pairs,v))) for v in [0,.25,.5,.75,1]],
            rating_note='Relative to this opponent, budgets and opening distribution only. Bootstrap pairs, not individual games. Null Elo endpoints mean unbounded; identical pair outcomes make bootstrap degenerate; consult conservative interval.')
    return result


def main():
    p=argparse.ArgumentParser()
    for key in ['suite','contender','opponents','openings','checkpoint','out']:p.add_argument('--'+key,required=True)
    p.add_argument('--pairs',type=int,default=32);p.add_argument('--max-plies',type=int,default=1000)
    p.add_argument('--resume',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=a.resume)
    suite={v['name']:v for v in json.loads(Path(a.suite).read_text())};opponents=a.opponents.split(',')
    fens=Path(a.openings).read_text().splitlines()[:a.pairs]
    if len(fens)!=a.pairs or a.pairs<1:raise ValueError('Insufficient openings')
    for fen in fens:
        b=chess.Board(fen)
        if not b.is_valid() or b.is_game_over(claim_draw=True):raise ValueError('Invalid opening')
    native_bf=bf.load_library();native_mcts=mcts.load_library();bf.load_library=lambda:native_bf;mcts.load_library=lambda:native_mcts
    for filename in ['libbestfirst.so','libmcts.so']:
        if not a.resume:shutil.copy2(ROOT/'neurochess/native'/filename,out/filename)
        elif digest(ROOT/'neurochess/native'/filename)!=digest(out/filename):raise ValueError('Resume requires identical native libraries')
    plan_args=vars(a).copy();plan_args.pop('resume')
    plan=dict(args=plan_args,variants={n:suite[n] for n in [a.contender]+opponents},openings=fens,
        checkpoint_sha256=digest(a.checkpoint),students={v['cheap']:digest(v['cheap']) for v in suite.values() if v.get('cheap')},
        native_sha256={n:digest(out/n) for n in ['libbestfirst.so','libmcts.so']},
        source_sha256={n:digest(n) for n in ['neurochess/native/best_first.cpp','neurochess/native/mcts.cpp','neurochess/best_first.py','neurochess/native_mcts.py','neurochess/multihead.py','neurochess/inference_cache.py',__file__]},
        rules='Direct play, swapped colors, fresh players per game; original PUCT subtree reuse retained. Full python-chess history and claimable draws. Unfinished games excluded, never counted as draws.')
    if a.resume and json.loads((out/'plan.json').read_text())!=plan:raise ValueError('Resume plan mismatch')
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high');model,state=load_model(a.checkpoint,'cuda');del state
    network=MultiheadPlayer(model,'cuda');start=time.perf_counter()
    def make(name):
        v=suite[name]
        return bf.BestFirstPlayer(v['cheap'],network,**v['options']) if v['kind']=='best-first' else mcts.NativeMCTSPlayer(network,**v['options'])
    all_rows={n:[] for n in opponents}
    for opening,fen in enumerate(fens):
        # Alternate opponent order by opening to distribute machine drift.
        order=opponents[opening%len(opponents):]+opponents[:opening%len(opponents)]
        for opponent in order:
            folder=out/opponent;folder.mkdir(exist_ok=True)
            for color in [chess.WHITE,chess.BLACK]:
                path=folder/f'{opening:03d}-{int(color)}.json'
                if path.exists():all_rows[opponent].append(json.loads(path.read_text()));continue
                players={a.contender:make(a.contender),opponent:make(opponent)};board=chess.Board(fen)
                game=setup_game(board,a.contender if color else opponent,opponent if color else a.contender)
                node=game;seconds={n:0. for n in players};positions={n:0 for n in players};choices={n:0 for n in players};decisions={n:[] for n in players}
                try:
                    for ply in range(a.max_plies):
                        if board.is_game_over(claim_draw=True):break
                        name=a.contender if board.turn==color else opponent;before=network.search_stats['forward_positions'];tic=time.perf_counter()
                        move=players[name].choose(board);elapsed=time.perf_counter()-tic;inputs=network.search_stats['forward_positions']-before
                        seconds[name]+=elapsed;positions[name]+=inputs;choices[name]+=1
                        last=getattr(players[name],'last_result',{})
                        decisions[name].append(dict(seconds=elapsed,large_positions=inputs,
                            simulations=last.get('new_simulations'),time_stopped=last.get('time_stopped'),proven_value=last.get('proven_value')))
                        if move not in board.legal_moves:raise RuntimeError('Illegal move')
                        board.push(move);node=node.add_variation(move)
                    outcome=board.outcome(claim_draw=True);score=None if outcome is None else .5 if outcome.winner is None else float(outcome.winner==color)
                    game.headers['Result']='*' if outcome is None else outcome.result()
                    if outcome:game.headers['Termination']=outcome.termination.name
                    row=dict(opening=opening,contender_color='white' if color else 'black',opponent=opponent,fen=fen,
                        score=score,result=game.headers['Result'],termination=outcome.termination.name if outcome else 'ply-limit',
                        plies=len(board.move_stack),seconds=seconds,large_model_positions=positions,choices=choices,
                        stats={n:dict(v.search_stats) for n,v in players.items()},decisions=decisions)
                    (folder/f'{opening:03d}-{int(color)}.pgn').write_text(str(game)+'\n\n')
                    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(row,indent=2));temporary.replace(path)
                    all_rows[opponent].append(row)
                finally:
                    for v in players.values():v.close()
                reports={n:summarize(r,2*a.pairs) for n,r in all_rows.items()}
                status=dict(elapsed_seconds=time.perf_counter()-start,current_opening=opening,results=reports)
                (out/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    reports={n:summarize(r,2*a.pairs) for n,r in all_rows.items()}
    (out/'result.json').write_text(json.dumps(dict(plan=plan,results=reports,games=all_rows,seconds=time.perf_counter()-start),indent=2))


if __name__=='__main__':main()
