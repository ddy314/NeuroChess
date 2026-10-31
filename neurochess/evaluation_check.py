"""Verify a completed paired evaluation before publishing or releasing a VM."""
import hashlib,json
from pathlib import Path
import chess,chess.pgn


def verify_arena(root,games_per_mode=64,checkpoint=None):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());report=json.loads((root/'result.json').read_text())
    if plan['opponent_nodes']!=1000 or plan['stockfish_options']['SkillLevel']!=20 or plan['stockfish_options']['UCI_LimitStrength']:
        raise ValueError('Evaluation opponent budget/strength differs from SF1000')
    if len(plan['openings'])*2!=games_per_mode:raise ValueError('Wrong opening count')
    if checkpoint:
        with open(checkpoint,'rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        if sha!=plan['checkpoint_sha256']:raise ValueError('Evaluated checkpoint differs from saved selection')
    rows=[json.loads(l) for l in (root/'games.jsonl').read_text().splitlines()]
    if len(rows)!=games_per_mode*2:raise ValueError('Evaluation is incomplete')
    index={(r['mode'],r['opening'],r['color']):r for r in rows}
    if len(index)!=len(rows):raise ValueError('Duplicate games')
    expected={(mode,i,color) for mode in ['direct','search'] for i in range(len(plan['openings'])) for color in ['white','black']}
    if set(index)!=expected:raise ValueError('Opening/color coverage differs')
    seen=set()
    with (root/'games.pgn').open() as f:
        while (game:=chess.pgn.read_game(f)) is not None:
            if game.errors:raise ValueError('Invalid PGN')
            board=game.board();plies=0
            for move in game.mainline_moves():
                if move not in board.legal_moves:raise ValueError('Illegal PGN move')
                board.push(move);plies+=1
            result=board.outcome(claim_draw=True)
            if result is None or result.result()!=game.headers['Result']:raise ValueError('Truncated or incorrect PGN result')
            color=game.headers['White'].startswith('NeuroChess');mode=game.headers['ModelMode'];opening=int(game.headers['Round'])-1
            key=(mode,opening,'white' if color else 'black')
            if opening<0 or opening>=len(plan['openings']) or game.board().fen()!=chess.Board(plan['openings'][opening]).fen():raise ValueError('PGN opening differs')
            if key in seen or key not in index:raise ValueError('Unexpected PGN game')
            row=index[key];seen.add(key)
            score=.5 if result.winner is None else float(result.winner==color)
            if row['score']!=score or row['plies']!=plies or row['termination']!=result.termination.name:raise ValueError('PGN and results disagree')
    if len(seen)!=len(rows):raise ValueError('Missing PGN games')
    for mode in ['direct','search']:
        group=[r for r in rows if r['mode']==mode]
        if len(group)!=games_per_mode or any(r['score'] is None for r in group):raise ValueError('Incomplete mode')
        stats=report['results'][mode]
        if stats['truncated'] or stats['wins']+stats['draws']+stats['losses']!=games_per_mode:raise ValueError('Incomplete reported mode')
        for name,value in [('wins',1.),('draws',.5),('losses',0.)]:
            if stats[name]!=sum(r['score']==value for r in group):raise ValueError('Incorrect aggregate score')
    return dict(games=len(rows),games_per_mode=games_per_mode,all_moves_legal=True,all_scores_verified=True,
                checkpoint_sha256=plan['checkpoint_sha256'],stockfish_nodes=1000,results=report['results'])
