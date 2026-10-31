"""Resumable multihead native PUCT versus full-strength fixed-node Stockfish."""
import importlib.util,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if importlib.util.find_spec('chess') is None:
    sys.path.append(str(next((ROOT/'.venv/lib').glob('python*/site-packages'))))
import argparse,hashlib,json,time,shutil,math
from concurrent.futures import ThreadPoolExecutor
import chess,chess.engine,chess.pgn
import numpy as np
import torch
from neurochess.multihead import MultiheadPlayer
from neurochess.native_mcts import NativeMCTSPlayer
import neurochess.native_mcts as native_module
from neurochess.train import load_model


def digest(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def elo(score):
    return 400*math.log10(score/(1-score)) if 0<score<1 else None


def summarize(rows,pairs):
    result={}
    for nodes in [500,1000,1500]:
        games=[r for r in rows if r['nodes']==nodes];done=[r for r in games if r['score'] is not None]
        grouped={}
        for r in done:grouped.setdefault(r['opening'],[]).append(r['score'])
        values=np.array([np.mean(scores) for scores in grouped.values() if len(scores)==2])
        item=dict(completed=len(done),scheduled=2*pairs,wins=sum(r['score']==1 for r in done),
            draws=sum(r['score']==.5 for r in done),losses=sum(r['score']==0 for r in done),
            truncated=sum(r['score'] is None for r in games),independent_pairs=len(values))
        if len(values):
            rng=np.random.default_rng(20261031);samples=values[rng.integers(len(values),size=(50000,len(values)))].mean(1)
            score=float(values.mean());ci=np.quantile(samples,[.025,.975]);eps=math.sqrt(math.log(40)/(2*len(values)))
            conservative=[max(0,score-eps),min(1,score+eps)]
            item.update(score=score,relative_elo=elo(score),paired_bootstrap_score95=ci.tolist(),
                paired_bootstrap_elo95=[elo(float(s)) for s in ci],hoeffding_score95=conservative,
                hoeffding_elo95=[elo(s) for s in conservative],
                pentanomial=[int(np.sum(np.isclose(values,v))) for v in [0,.25,.5,.75,1]])
        result[str(nodes)]=item
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--stockfish',required=True)
    p.add_argument('--openings',required=True);p.add_argument('--out',required=True);p.add_argument('--pairs',type=int,default=16)
    p.add_argument('--batch-games',type=int,default=16);p.add_argument('--simulations',type=int,default=512)
    p.add_argument('--resume',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=a.resume)
    if a.batch_games<2 or a.batch_games%2:raise ValueError('Batch must hold full opening pairs')
    fens=Path(a.openings).read_text().splitlines()[:a.pairs]
    if len(fens)!=a.pairs or len(set(fens))!=a.pairs:raise ValueError('Opening count differs')
    for fen in fens:
        b=chess.Board(fen)
        if not b.is_valid() or b.is_game_over(claim_draw=True):raise ValueError('Invalid opening')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    model,state=load_model(a.checkpoint,'cuda');assert model.config.history_free
    network=MultiheadPlayer(model,'cuda')
    library=native_module.load_library();native_module.load_library=lambda:library
    player=NativeMCTSPlayer(network,simulations=a.simulations,leaf_batch_size=64)
    engines=[chess.engine.SimpleEngine.popen_uci(a.stockfish,timeout=60) for _ in range(a.batch_games)]
    options={'Threads':1,'Hash':64,'Skill Level':20,'UCI_LimitStrength':False}
    for e in engines:e.configure(options)
    plan=dict(checkpoint_sha256=digest(a.checkpoint),checkpoint_step=state['step'],origin_step=state.get('origin_step'),
        stockfish=engines[0].id,stockfish_sha256=digest(a.stockfish),stockfish_options=options,
        opponent_nodes=[500,1000,1500],openings=fens,pairs=a.pairs,batch_games=a.batch_games,
        search=player.config,device=torch.cuda.get_device_name(),torch=torch.__version__,
        native_sha256=digest(ROOT/'neurochess/native/libmcts.so'),
        source_sha256={str(f):digest(ROOT/f) for f in ['neurochess/native/mcts.cpp','neurochess/native/chess_rules.hpp','neurochess/native_mcts.py','neurochess/multihead.py','scripts/fixed_node_arena.py']},
        protocol='Same roots and colors for every node budget. Fixed 512-simulation search; no teacher consulted by model. Hash cleared per game. Complete terminal outcomes only; unfinished games never count as draws.',
        elo_scope='Relative to each fixed-node Stockfish opponent under this protocol; no absolute/human Elo anchor. Pair bootstrap retains color correlation; unbounded interval endpoints are null.')
    del state
    if a.resume:
        if json.loads((out/'plan.json').read_text())!=plan:raise ValueError('Resume fingerprints differ')
    else:
        save(out/'plan.json',plan)
        shutil.copy2(ROOT/'neurochess/native/libmcts.so',out/'libmcts.frozen.so')
        for file in plan['source_sha256']:
            target=out/'source'/file;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/file,target)
    rows=[]
    for f in out.glob('games/*.json'):rows.append(json.loads(f.read_text()))
    completed={(r['nodes'],r['opening'],r['color']) for r in rows};(out/'games').mkdir(exist_ok=True)
    started=time.time();initial=len(rows);cached_size=-1;cached_summary=None
    def status(phase):
        nonlocal cached_size,cached_summary
        if cached_size!=len(rows):cached_summary=summarize(rows,a.pairs);cached_size=len(rows)
        info=dict(phase=phase,completed=len(rows),scheduled=6*a.pairs,results=cached_summary,
            updated_at=time.time(),elapsed_seconds=time.time()-started,resumed_games=initial,
            network_stats=network.search_stats,search_stats=player.search_stats)
        save(out/'status.json',info);return info
    try:
        with ThreadPoolExecutor(8) as pool:
            # Rotate opponent order between blocks to spread thermal drift.
            for block,offset in enumerate(range(0,len(fens),a.batch_games//2)):
                order=[500,1000,1500];order=order[block%3:]+order[:block%3]
                for nodes in order:
                    active=[]
                    for opening in range(offset,min(offset+a.batch_games//2,len(fens))):
                        for color in [True,False]:
                            label='white' if color else 'black'
                            if (nodes,opening,label) in completed:continue
                            board=chess.Board(fens[opening]);game=chess.pgn.Game();game.setup(board)
                            raw_fen=board.fen(en_passant='fen')
                            if raw_fen!=chess.STARTING_FEN:game.headers.update(FEN=raw_fen,SetUp='1')
                            game.headers.update(Event='NeuroChess fixed-node local evaluation',Round=str(opening+1),
                                White='NeuroChess-PUCT512' if color else engines[0].id['name'],
                                Black=engines[0].id['name'] if color else 'NeuroChess-PUCT512',
                                ModelMode='multihead-puct',SFNodes=str(nodes),SearchSimulations=str(a.simulations))
                            active.append(dict(board=board,game=game,node=game,opening=opening,color=color,slot=len(active),plies=0))
                    for g in active:engines[g['slot']].configure({'Clear Hash':None})
                    status('running')
                    while active:
                        alive=[]
                        for g in active:
                            outcome=g['board'].outcome(claim_draw=True)
                            if outcome is None and g['plies']<1000:alive.append(g);continue
                            score=None if outcome is None else .5 if outcome.winner is None else float(outcome.winner==g['color'])
                            row=dict(nodes=nodes,opening=g['opening'],color='white' if g['color'] else 'black',score=score,
                                result=outcome.result() if outcome else '*',termination=outcome.termination.name if outcome else 'max_plies',plies=g['plies'])
                            g['game'].headers.update(Result=row['result'],Termination=row['termination'])
                            stem=out/'games'/f"{nodes}-{g['opening']:03d}-{row['color']}"
                            stem.with_suffix('.pgn').write_text(str(g['game'])+'\n\n');save(stem.with_suffix('.json'),row)
                            rows.append(row);completed.add((nodes,g['opening'],row['color']));print(json.dumps(row),flush=True)
                        active=alive
                        if not active:break
                        def push(g,move):
                            if move not in g['board'].legal_moves:raise RuntimeError('Illegal move')
                            g['board'].push(move);g['node']=g['node'].add_variation(move);g['plies']+=1
                        pending=[g for g in active if g['board'].turn==g['color']]
                        if pending:
                            for g,move in zip(pending,player.choose_many([g['board'] for g in pending])):push(g,move)
                        pending=[g for g in active if g['board'].turn!=g['color'] and g['plies']<1000 and not g['board'].is_game_over(claim_draw=True)]
                        def sf(g):return engines[g['slot']].play(g['board'],chess.engine.Limit(nodes=nodes),game=g['game']).move
                        for g,move in zip(pending,pool.map(sf,pending)):push(g,move)
                        status('running')
        if len(rows)!=6*a.pairs or any(r['score'] is None for r in rows):raise RuntimeError('Incomplete games; inspect PGNs')
        result=status('complete');save(out/'result.json',result)
        with (out/'games.pgn').open('w') as pgn,(out/'games.jsonl').open('w') as log:
            for f in sorted((out/'games').glob('*.json')):
                pgn.write(f.with_suffix('.pgn').read_text());log.write(json.dumps(json.loads(f.read_text()))+'\n')
    finally:
        player.close()
        for engine in engines:engine.quit()

if __name__=='__main__':main()
