"""Independently replay saved PGNs and check result/budget bookkeeping."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import chess.pgn
from scripts.search_duel import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--duel',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.duel);plan=json.loads((root/'plan.json').read_text());rows={};hashes={}
    for folder in sorted(x for x in root.iterdir() if x.is_dir()):
        games=[]
        for path in sorted(folder.glob('[0-9][0-9][0-9]-[01].json')):
            row=json.loads(path.read_text());pgn=path.with_suffix('.pgn')
            with pgn.open() as f:game=chess.pgn.read_game(f)
            assert game is not None and not game.errors
            board=game.board();assert board.fen(en_passant='fen')==row['fen']==plan['openings'][row['opening']]
            moves=list(game.mainline_moves())
            for move in moves:
                assert move in board.legal_moves;board.push(move)
            assert len(moves)==row['plies']==sum(row['choices'].values())
            outcome=board.outcome(claim_draw=True);color=row['contender_color']=='white'
            score=None if outcome is None else .5 if outcome.winner is None else float(outcome.winner==color)
            assert score==row['score']
            assert game.headers['Result']==row['result']==('*' if outcome is None else outcome.result())
            assert game.headers['White']==(plan['args']['contender'] if color else row['opponent'])
            for name,count in row['large_model_positions'].items():
                config=plan['variants'][name]
                cap=config['options'].get('neural_budget',32) if config['kind']=='best-first' else config['options']['simulations']+1
                assert 0<=count<=row['choices'][name]*cap
            hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[str(pgn.relative_to(root))]=hashlib.sha256(pgn.read_bytes()).hexdigest();games.append(row)
        if games:rows[folder.name]=games
    result=dict(passed=True,checked_games=sum(map(len,rows.values())),
        summary={n:summarize(r,2*plan['args']['pairs']) for n,r in rows.items()},sha256=hashes,
        note='All saved complete game records replayed legally; unfinished results remain excluded, not draws. Actual neural positions checked against aggregate per-game caps.')
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='sha256'},indent=2))


if __name__=='__main__':main()
