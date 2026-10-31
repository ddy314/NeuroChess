"""Restore exact recorded start FENs when PGN export omitted nonlegal EP.

python-chess Game.setup uses its normalized FEN by default. The actual match
starts from the raw FEN in the immutable plan/JSON. Restore that exact input
for reproducibility, preserving original files and every move/result.
"""
import argparse
import hashlib
import json
from pathlib import Path

import chess
import chess.pgn


def main():
    p=argparse.ArgumentParser();p.add_argument('--duel',required=True);a=p.parse_args();root=Path(a.duel);changes=[]
    for folder in sorted(f for f in root.iterdir() if f.is_dir() and f.name!='pgn-before-fen-normalization'):
        for row_path in sorted(folder.glob('[0-9][0-9][0-9]-[01].json')):
            row=json.loads(row_path.read_text());path=row_path.with_suffix('.pgn')
            with path.open() as f:game=chess.pgn.read_game(f)
            assert game is not None and not game.errors
            old=game.board();new=chess.Board(row['fen'])
            if old.fen(en_passant='fen')==row['fen']:continue
            old_parts=old.fen(en_passant='fen').split();new_parts=row['fen'].split()
            assert old_parts[:3]+old_parts[4:]==new_parts[:3]+new_parts[4:]
            assert sorted(m.uci() for m in old.legal_moves)==sorted(m.uci() for m in new.legal_moves)
            moves=list(game.mainline_moves())
            for move in moves:
                assert move in old.legal_moves and move in new.legal_moves;old.push(move);new.push(move)
            assert old.fen(en_passant='fen')==new.fen(en_passant='fen')
            backup=root/'pgn-before-fen-normalization'/folder.name/path.name;backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists():backup.write_bytes(path.read_bytes())
            before=hashlib.sha256(path.read_bytes()).hexdigest();game.headers['FEN']=row['fen'];game.headers['SetUp']='1'
            path.write_text(str(game)+'\n\n')
            changes.append(dict(file=str(path.relative_to(root)),before_sha256=before,after_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fen=row['fen']))
    previous=root/'fen-normalization.json'
    if changes or not previous.exists():previous.write_text(json.dumps(dict(changed=len(changes),changes=changes,
        note='Metadata only: exact raw start FEN restored from original match records; legal moves and final positions checked identical; original PGNs retained.'),indent=2))
    print(json.dumps(dict(changed_this_run=len(changes))))


if __name__=='__main__':main()
