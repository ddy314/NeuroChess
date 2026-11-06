"""Direct paired play against the immutable2s floor, on one serialized GPU."""
import json,hashlib,os,shutil,sys,time,traceback
from pathlib import Path
import chess,chess.pgn,chess.engine,numpy as np,torch
from neurochess.train import load_model
from neurochess.multihead import MultiheadPlayer
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.pawn_data import sha,write_json,contains,REVISION,SOURCE
from neurochess.encoding import position_key,encode,legal_ids
from scripts.search_duel import summarize


def main():
    work=Path('/content/pawn-arena');work.mkdir(exist_ok=True)
    save=Path('/content/drive/MyDrive/neurochess/pawn-nodes1024-20261103/arena-2s-20261104');save.mkdir(parents=True,exist_ok=True)
    def persist():
        for p in work.iterdir():
            if p.is_file() and p.suffix in ['.json','.pgn','.jsonl']:
                tmp=save/(p.name+'.tmp');shutil.copyfile(p,tmp);tmp.replace(save/p.name)
    try:
        torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
        lock=json.loads(Path('configs/search-baseline-lock.json').read_text());options=lock['config']
        # The same frozen source is used by both players. Rebuilt native binary
        # has a separately recorded platform hash, without changing its source.
        for p,want in lock['source_sha256'].items():
            if p.endswith('.so'):continue
            if sha(p)!=want:raise ValueError('Locked source mismatch: '+p)
        baseline=Path('/content/baseline.pt');assert sha(baseline)==lock['checkpoint_sha256']
        final=Path('/content/pawn-final.pt');shutil.copyfile('/content/drive/MyDrive/neurochess/pawn-nodes1024-20261103/training/last.pt',final)
        assert sha(final)=='fc6cace9e86c44d8b35b54f8bc76c93ada0994e3a975cd8e8f02c3ac671e8ebc'
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq
        import requests
        repo=requests.get(f'https://huggingface.co/api/datasets/{SOURCE}/revision/{REVISION}',timeout=90).json()
        files=sorted(x['rfilename'] for x in repo['siblings'] if x['rfilename'].startswith('test/nodes_1024/') and x['rfilename'].endswith('.parquet'))
        prior_path=work/'prior.npy';shutil.copyfile('/content/drive/MyDrive/neurochess-data/pawn-nodes1024-20m-20261103/known-prior-keys.npy',prior_path)
        prior=np.load(prior_path,mmap_mode='r');fs=HfFileSystem();openings=[]
        for path in files:
            with fs.open(f'datasets/{SOURCE}@{REVISION}/{path}','rb',block_size=1024*1024) as stream:
                pf=pq.ParquetFile(stream)
                for batch in pf.iter_batches(batch_size=32,columns=['uci','cp_evals','global_game_index']):
                    for game in batch.to_pylist():
                        if len(game['uci'])<24:continue
                        b=chess.Board()
                        for u in game['uci'][:12]:b.push_uci(u)
                        scores=game['cp_evals'][12]
                        if not scores or abs(max(e['score_cp'] for e in scores))>50:continue
                        key=np.array([position_key(b)],dtype='V16')
                        if contains(prior,key)[0] or any(o['key']==bytes(key[0]).hex() for o in openings):continue
                        openings.append(dict(game=int(game['global_game_index']),prefix=game['uci'][:12],fen=b.fen(),key=bytes(key[0]).hex()))
                        if len(openings)==8:break
                    if len(openings)==8:break
            if len(openings)==8:break
        if len(openings)!=8:raise ValueError('Not enough isolated balanced openings')
        plan=dict(baseline_id=lock['name'],baseline_sha256=sha(baseline),candidate_sha256=sha(final),options=options,
            gpu=torch.cuda.get_device_name(),native_sha256=sha('neurochess/native/libmcts.so'),openings=openings,
            selection='First8 separate official test games with ply12 teacher best score within50cp and no known prior root key',
            protocol='16 games, swapped colors, serial GPU execution, full prefix retained, claimable draws;600 searched plies cap; truncations excluded; no default promotion',
            source_revision=REVISION,source_sha256={p:sha(p) for p in lock['source_sha256'] if not p.endswith('.so')})
        write_json(work/'plan.json',plan);persist()
        nets={};warm={}
        for name,path in [('baseline',baseline),('pawn',final)]:
            start=time.time();model,state=load_model(path,'cuda');del state;model.eval();net=MultiheadPlayer(model,'cuda')
            b=chess.Board(openings[0]['fen']);tokens,meta=encode(b);ids=np.full((64,256),-1,np.int16);li=legal_ids(b)[1];ids[:,:len(li)]=li
            for n in range(1,65):net.evaluate_search_heads(np.tile(tokens,(n,1)),np.tile(meta,(n,1)),ids[:n])
            warm[name]=time.time()-start;nets[name]=net
        rows=[];started=time.time()
        # Frozen regression roots are evaluated before direct play, with fresh
        # search trees and one shared finite teacher search per candidate union.
        roots=Path('/content/frozen-roots.fen').read_text().splitlines()
        with chess.engine.SimpleEngine.popen_uci('/usr/games/stockfish') as sf:
            sf.configure({'Threads':1,'Hash':64})
            frozen=dict(teacher=sf.id,roots=[])
            for index,fen in enumerate(roots):
                b=chess.Board(fen);choices={}
                for name in nets:
                    player=NativeMCTSPlayer(nets[name],**options);tick=time.time()
                    try:move=player.choose(b)
                    finally:player.close()
                    choices[name]=dict(move=move.uci(),seconds=time.time()-tick)
                proposal=sf.analyse(b,chess.engine.Limit(nodes=100000),game=object())['pv'][0]
                moves=list(dict.fromkeys([proposal]+[b.parse_uci(v['move']) for v in choices.values()]))
                info=sf.analyse(b,chess.engine.Limit(nodes=1000000),root_moves=moves,multipv=len(moves),game=object())
                labels={x['pv'][0].uci():dict(cp=x['score'].pov(b.turn).score(mate_score=100000),mate=x['score'].pov(b.turn).mate()) for x in info}
                best=max(v['cp'] for v in labels.values())
                for choice in choices.values():choice['cp_regret']=best-labels[choice['move']]['cp']
                frozen['roots'].append(dict(fen=fen,choices=choices,teacher=labels))
                write_json(work/'frozen-regression.json',frozen);persist()
        for index,o in enumerate(openings):
            for color in ([True,False] if index%2==0 else [False,True]):
                players={name:NativeMCTSPlayer(net,**options) for name,net in nets.items()}
                board=chess.Board();game=chess.pgn.Game();node=game
                for uci in o['prefix']:move=board.parse_uci(uci);board.push(move);node=node.add_variation(move)
                game.headers['White']='pawn' if color else 'baseline';game.headers['Black']='baseline' if color else 'pawn'
                decisions={name:[] for name in nets}
                try:
                    for ply in range(600):
                        if board.is_game_over(claim_draw=True):break
                        name='pawn' if board.turn==color else 'baseline';t=time.perf_counter();move=players[name].choose(board);elapsed=time.perf_counter()-t
                        if move not in board.legal_moves:raise ValueError('Illegal move')
                        decisions[name].append(dict(seconds=elapsed,root=players[name].last_result));board.push(move);node=node.add_variation(move)
                        if ply%10==0:
                            write_json(work/'live.json',dict(opening=index,candidate_white=color,searched_plies=ply+1,completed_games=len(rows),elapsed_seconds=time.time()-started));persist()
                    outcome=board.outcome(claim_draw=True);score=None if outcome is None else .5 if outcome.winner is None else float(outcome.winner==color)
                    game.headers['Result']=outcome.result() if outcome else '*';game.headers['Termination']=outcome.termination.name if outcome else 'ply-limit'
                    stem=f'{index:03d}-{int(color)}';(work/(stem+'.pgn')).write_text(str(game)+'\n\n')
                    # Independently replay the complete PGN, including prefix.
                    import io
                    replay=chess.pgn.read_game(io.StringIO(str(game)));audit=replay.board()
                    if replay.errors:raise ValueError('PGN parse failed')
                    for move in replay.mainline_moves():
                        if move not in audit.legal_moves:raise ValueError('PGN replay illegal')
                        audit.push(move)
                    if audit.fen()!=board.fen() or audit.outcome(claim_draw=True)!=outcome:raise ValueError('PGN result/history mismatch')
                    row=dict(opening=index,contender_color='white' if color else 'black',score=score,result=game.headers['Result'],termination=game.headers['Termination'],plies=len(board.move_stack)-12,decisions=decisions,replay_verified=True)
                    write_json(work/(stem+'.json'),row);rows.append(row)
                finally:
                    for player in players.values():player.close()
                report=dict(plan=plan,summary=summarize(rows,16),games=rows,warmup_seconds=warm,elapsed_seconds=time.time()-started)
                write_json(work/'result.json',report);persist();print(json.dumps(report['summary']),flush=True)
        for name in nets:
            seconds=[d['seconds'] for r in rows for d in r['decisions'][name]]
            report.setdefault('latency',{})[name]=dict(median=float(np.median(seconds)),p95=float(np.quantile(seconds,.95)),maximum=max(seconds),moves=len(seconds))
        report['complete']=True;write_json(work/'result.json',report);persist()
    except BaseException:
        write_json(work/'failure.json',dict(traceback=traceback.format_exc()));persist();raise
    finally:
        from neurochess.autostop import request_unassign,find_runtime_address
        write_json(save/'shutdown-request.json',dict(requested_at=time.time()));request_unassign(find_runtime_address())


if __name__=='__main__':main()
