"""CPU-only color-swapped Stockfish node-budget ladder; no absolute Elo anchor."""
import argparse,hashlib,json,time
from pathlib import Path
import chess,chess.engine,chess.pgn


def digest(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--stockfish',required=True);p.add_argument('--openings',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);fens=Path(a.openings).read_text().splitlines()
    options={'Threads':1,'Hash':64,'Skill Level':20,'UCI_LimitStrength':False};rows=[];start=time.time()
    engines=[chess.engine.SimpleEngine.popen_uci(a.stockfish,timeout=60) for _ in range(2)]
    for e in engines:e.configure(options)
    plan=dict(stockfish=engines[0].id,stockfish_sha256=digest(a.stockfish),options=options,openings=fens,openings_sha256=digest(a.openings),pairs_per_matchup=len(fens),matchups=[[500,1000],[500,1500],[1000,1500]],anchor='SF1000=0, a relative scale only')
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    try:
        with (out/'games.jsonl').open('w') as log,(out/'games.pgn').open('w') as pgn:
            for opening,fen in enumerate(fens):
                for low,high in plan['matchups']:
                    for color in [True,False]:
                        board=chess.Board(fen);game=chess.pgn.Game();game.setup(board);node=game
                        raw=board.fen(en_passant='fen')
                        if raw!=chess.STARTING_FEN:game.headers.update(FEN=raw,SetUp='1')
                        game.headers.update(Event='SF18 node-budget calibration',Round=str(opening+1),White=f'SF18-{low if color else high}',Black=f'SF18-{high if color else low}')
                        for e in engines:e.configure({'Clear Hash':None})
                        for ply in range(1000):
                            if board.is_game_over(claim_draw=True):break
                            mine=board.turn==color;move=engines[0 if mine else 1].play(board,chess.engine.Limit(nodes=low if mine else high),game=game).move
                            if move not in board.legal_moves:raise RuntimeError('Illegal SF move')
                            board.push(move);node=node.add_variation(move)
                        outcome=board.outcome(claim_draw=True)
                        if outcome is None:raise RuntimeError('Unfinished calibration game')
                        score=.5 if outcome.winner is None else float(outcome.winner==color)
                        row=dict(opening=opening,low=low,high=high,low_color='white' if color else 'black',score=score,result=outcome.result(),termination=outcome.termination.name,plies=len(board.move_stack))
                        game.headers.update(Result=row['result'],Termination=row['termination']);rows.append(row)
                        log.write(json.dumps(row)+'\n');log.flush();pgn.write(str(game)+'\n\n');pgn.flush()
                        (out/'status.json').write_text(json.dumps(dict(completed=len(rows),scheduled=6*len(fens),elapsed_seconds=time.time()-start)))
            (out/'result.json').write_text(json.dumps(dict(plan=plan,games=rows,seconds=time.time()-start),indent=2))
            print(json.dumps(dict(completed=len(rows),seconds=time.time()-start)),flush=True)
    finally:
        for e in engines:e.quit()

if __name__=='__main__':main()
