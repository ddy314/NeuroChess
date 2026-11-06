"""Paired evaluation with all three trained heads and fixed opponent budget."""
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if importlib.util.find_spec('chess') is None:
    for site in (ROOT/'.venv/lib').glob('python*/site-packages'):
        if (site/'chess').is_dir():sys.path.append(str(site));break
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time
import chess
import chess.engine
import chess.pgn
import torch
from neurochess.multihead import MultiheadPlayer
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.train import load_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--openings',required=True)
    p.add_argument('--stockfish',required=True);p.add_argument('--out',required=True)
    p.add_argument('--pairs',type=int,default=8);p.add_argument('--batch-games',type=int,default=16)
    p.add_argument('--simulations',type=int,default=512);p.add_argument('--nodes',type=int,default=1000)
    p.add_argument('--search-config',help='Native PUCT configuration JSON; timed searches run one board per player')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    fens=Path(a.openings).read_text().splitlines();assert len(fens)==a.pairs and len(set(fens))==a.pairs and a.batch_games>=2 and a.batch_games%2==0
    model,state=load_model(a.checkpoint,'cuda');assert model.config.history_free
    with open(a.checkpoint,'rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    search_options=json.loads(Path(a.search_config).read_text()) if a.search_config else dict(simulations=a.simulations,leaf_batch_size=64)
    if search_options.get('time_limit') is not None:a.batch_games=2
    direct=MultiheadPlayer(model,'cuda');search=NativeMCTSPlayer(direct,**search_options)
    slot_search=[search]+[NativeMCTSPlayer(direct,**search_options) for _ in range(a.batch_games-1)] if search_options.get('time_limit') is not None else [search]
    search.mode='multihead-puct'
    engines=[chess.engine.SimpleEngine.popen_uci(a.stockfish,timeout=30) for _ in range(a.batch_games)]
    for e in engines:e.configure({'Threads':1,'Hash':64,'Skill Level':20,'UCI_LimitStrength':False})
    plan=dict(checkpoint_sha256=sha,checkpoint_step=state['step'],origin_step=state.get('origin_step'),
        opponent=engines[0].id,stockfish_options=dict(Threads=1,Hash=64,SkillLevel=20,UCI_LimitStrength=False),
        opponent_nodes=a.nodes,search=search.config,head_fusion=dict(top_k=5,max_logit_gap=3,q_weight=2,
        wdl_confidence='1 - entropy(WDL)/log(3)',q_virtual_visits=2,q_initial='0.5*candidate_Q+0.5*parent_value'),
        no_search='One forward at root; no child positions, mate scan, or lookahead.',
        openings=fens,opening_sha256=hashlib.sha256(Path(a.openings).read_bytes()).hexdigest(),
        max_plies=1000,device=torch.cuda.get_device_name(),torch=torch.__version__,absolute_elo_claim=False)
    del state
    (out/'plan.json').write_text(json.dumps(plan,indent=2));rows=[];started=time.time()
    def status(phase):
        summary={}
        for mode in ['direct','search']:
            rs=[r for r in rows if r['mode']==mode]
            summary[mode]=dict(played=len(rs),wins=sum(r['score']==1 for r in rs),draws=sum(r['score']==.5 for r in rs),
                losses=sum(r['score']==0 for r in rs),truncated=sum(r['score'] is None for r in rs))
        record=dict(phase=phase,completed=len(rows),scheduled=len(fens)*4,results=summary,updated_at=time.time(),
            elapsed_seconds=time.time()-started,network_stats=direct.search_stats,search_stats=search.search_stats)
        tmp=out/'status.tmp';tmp.write_text(json.dumps(record,indent=2));tmp.replace(out/'status.json')
        return record
    try:
        with ThreadPoolExecutor(8) as pool,(out/'games.jsonl').open('w') as log,(out/'games.pgn').open('w') as pgn:
            for mode,player in [('direct',direct),('search',search)]:
                for offset in range(0,len(fens),a.batch_games//2):
                    active=[]
                    for i in range(offset,min(offset+a.batch_games//2,len(fens))):
                        fen=fens[i]
                        for color in [True,False]:
                            board=chess.Board(fen);game=chess.pgn.Game();game.setup(board)
                            if board.fen(en_passant='fen')!=chess.STARTING_FEN:
                                game.headers.update(FEN=board.fen(en_passant='fen'),SetUp='1')
                            game.headers.update(Event='Engine-supervised multihead evaluation',Round=str(i+1),
                                White='NeuroChess-'+mode if color else engines[0].id['name'],
                                Black=engines[0].id['name'] if color else 'NeuroChess-'+mode,
                                ModelMode=mode,SearchSimulations=str(search_options['simulations'] if mode=='search' else 0))
                            active.append(dict(board=board,game=game,node=game,color=color,slot=len(active),opening=i,plies=0))
                    status('running')
                    while active:
                        alive=[]
                        for g in active:
                            result=g['board'].outcome(claim_draw=True)
                            if result is None and g['plies']<1000:alive.append(g);continue
                            row=dict(mode=mode,opening=g['opening'],color='white' if g['color'] else 'black',plies=g['plies'],
                                score=None if result is None else .5 if result.winner is None else float(result.winner==g['color']),
                                result='*' if result is None else result.result(),termination='max_plies' if result is None else result.termination.name)
                            g['game'].headers['Result']=row['result'];rows.append(row)
                            log.write(json.dumps(row)+'\n');log.flush();print(g['game'],file=pgn,end='\n\n');pgn.flush()
                            print(json.dumps(row),flush=True);status('running')
                        active=alive
                        if not active:break
                        pending=[g for g in active if g['board'].turn==g['color']]
                        def push(g,move):
                            if move not in g['board'].legal_moves:raise RuntimeError('Illegal move')
                            g['board'].push(move);g['node']=g['node'].add_variation(move);g['plies']+=1
                        if pending:
                            if mode=='search' and search_options.get('time_limit') is not None:
                                for g in pending:push(g,slot_search[g['slot']].choose(g['board']))
                            else:
                                for g,move in zip(pending,player.choose_many([g['board'] for g in pending])):push(g,move)
                        pending=[g for g in active if g['board'].turn!=g['color'] and g['plies']<1000 and not g['board'].is_game_over(claim_draw=True)]
                        def sf(g):return engines[g['slot']].play(g['board'],chess.engine.Limit(nodes=a.nodes),game=g['game']).move
                        for g,move in zip(pending,pool.map(sf,pending)):push(g,move)
                        status('running')
            report=status('complete');(out/'result.json').write_text(json.dumps(report,indent=2))
    finally:
        for player in slot_search:player.close()
        for engine in engines:engine.quit()


if __name__=='__main__':main()
