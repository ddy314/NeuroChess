"""Replay every game and report protocol-relative Elo with paired uncertainty."""
import argparse,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess,chess.pgn
import numpy as np
from io import StringIO


def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def rating(score):
    return 400*math.log10(score/(1-score)) if 0<score<1 else None


def exact_pair_bootstrap(scores):
    """Exact distribution of the empirical opening-pair bootstrap mean."""
    units=np.rint(np.asarray(scores)*4).astype(int)
    probabilities=np.bincount(units,minlength=5)/len(units);mass=np.array([1.])
    for _ in units:mass=np.convolve(mass,probabilities)
    cdf=np.cumsum(mass);cdf[-1]=1.
    bounds=np.searchsorted(cdf,[.025,.975])/(4*len(units))
    return bounds.tolist()


def fit_ladder(wins):
    """Expected-score Bradley-Terry fit, anchoring participant 2 (SF1000) to zero."""
    wins=np.asarray(wins,float);games=wins+wins.T;x=np.zeros(4);free=[0,1,3]
    for _ in range(80):
        prob=1/(1+np.exp(np.clip(x[None,:]-x[:,None],-40,40)))
        gradient=(wins-games*prob).sum(axis=1)
        weight=games*prob*(1-prob);laplacian=np.diag(weight.sum(axis=1))-weight
        if np.max(np.abs(gradient[free]))<1e-8:break
        delta=np.linalg.solve(laplacian[np.ix_(free,free)]+np.eye(3)*1e-10,gradient[free])
        x[free]+=delta/max(1.,float(np.max(np.abs(delta)))/4)
        if np.max(np.abs(x))>35:break
    answer=x*(400/math.log(10));reach=wins>0;np.fill_diagonal(reach,True)
    for k in range(4):reach|=reach[:,k,None]&reach[k,None,:]
    for i in free:
        if reach[2,i] and not reach[i,2]:answer[i]=-np.inf
        elif reach[i,2] and not reach[2,i]:answer[i]=np.inf
    return answer


def calibration_and_ladder(root,model_games,plan):
    folder=root/'calibration';cal=json.loads((folder/'result.json').read_text());rows=cal['games']
    if cal['plan']['stockfish_sha256']!=plan['stockfish_sha256'] or cal['plan']['openings']!=plan['openings']:raise ValueError('Calibration protocol differs')
    if cal['plan']['options']!=plan['stockfish_options']:raise ValueError('Calibration engine options differ')
    expected={(i,lo,hi,c) for i in range(plan['pairs']) for lo,hi in [[500,1000],[500,1500],[1000,1500]] for c in ['white','black']}
    seen=set()
    with (folder/'games.pgn').open() as file:
        for row in rows:
            key=(row['opening'],row['low'],row['high'],row['low_color'])
            if key not in expected or key in seen:raise ValueError('Calibration game identity differs')
            seen.add(key);game=chess.pgn.read_game(file)
            if game is None or game.errors:raise ValueError('Invalid calibration PGN')
            board=game.board();plies=0
            if board.fen(en_passant='fen')!=chess.Board(plan['openings'][row['opening']]).fen(en_passant='fen'):raise ValueError('Calibration root differs')
            for move in game.mainline_moves():
                if move not in board.legal_moves:raise ValueError('Illegal calibration move')
                board.push(move);plies+=1
            result=board.outcome(claim_draw=True);color=row['low_color']=='white'
            if result is None:raise ValueError('Incomplete calibration game')
            score=.5 if result.winner is None else float(result.winner==color)
            if row['score']!=score or row['result']!=result.result() or game.headers['Result']!=row['result'] or plies!=row['plies']:raise ValueError('Calibration outcome differs')
            if game.headers['White']!=f"SF18-{row['low'] if color else row['high']}":raise ValueError('Calibration color differs')
            row['pgn']=str(game)+'\n\n'
        if chess.pgn.read_game(file) is not None:raise ValueError('Unexpected calibration PGN')
    if seen!=expected:raise ValueError('Calibration coverage differs')
    ids={500:1,1000:2,1500:3};blocks=np.zeros((plan['pairs'],4,4))
    for row in model_games:
        i,j=0,ids[row['nodes']];blocks[row['opening'],i,j]+=row['score'];blocks[row['opening'],j,i]+=1-row['score']
    for row in rows:
        i,j=ids[row['low']],ids[row['high']];blocks[row['opening'],i,j]+=row['score'];blocks[row['opening'],j,i]+=1-row['score']
    point=fit_ladder(blocks.sum(axis=0));rng=np.random.default_rng(20261031)
    samples=np.array([fit_ladder(blocks[indices].sum(axis=0)) for indices in rng.integers(plan['pairs'],size=(10000,plan['pairs']))])
    intervals=np.quantile(samples,[.025,.975],axis=0,method='inverted_cdf');names=['NeuroChess','SF500','SF1000','SF1500']
    finite=lambda x:float(x) if np.isfinite(x) else None
    ladder={name:dict(relative_elo=finite(point[i]),joint_opening_bootstrap95=[finite(x) for x in intervals[:,i]],separated_bootstrap_samples=int(np.isinf(samples[:,i]).sum())) for i,name in enumerate(names)}
    return dict(games=len(rows),all_moves_legal=True,all_scores_verified=True,plan=cal['plan'],results=rows),dict(anchor='SF1000 = 0; relative, not absolute Elo',ratings=ladder,
        method='Expected-score Bradley-Terry MLE. 10000 bootstrap resamples of complete opening blocks jointly across all six matchups, preserving color and cross-budget correlations.',
        games=len(rows)+len(model_games),independent_opening_blocks=plan['pairs'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);arena=root/'matches'
    plan=json.loads((arena/'plan.json').read_text());reported=json.loads((arena/'result.json').read_text());pairs=plan['pairs']
    checkpoint=root/'model/selected.pt'
    if sha(checkpoint)!=plan['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    if sha(arena/'libmcts.frozen.so')!=plan['native_sha256']:raise ValueError('Native library changed')
    for file,expected in plan['source_sha256'].items():
        if sha(arena/'source'/file)!=expected:raise ValueError('Frozen source differs: '+file)
    expected={(n,i,c) for n in [500,1000,1500] for i in range(pairs) for c in ['white','black']};seen=set();games=[]
    for path in sorted((arena/'games').glob('*.json')):
        row=json.loads(path.read_text());key=(row['nodes'],row['opening'],row['color'])
        if key not in expected or key in seen:raise ValueError('Unexpected or duplicate game')
        seen.add(key);pgn=path.with_suffix('.pgn').read_text();game=chess.pgn.read_game(StringIO(pgn))
        if game is None or game.errors:raise ValueError('Invalid PGN')
        board=game.board();initial=chess.Board(plan['openings'][row['opening']])
        if board.fen(en_passant='fen')!=initial.fen(en_passant='fen'):raise ValueError('Opening differs')
        if int(game.headers['SFNodes'])!=row['nodes'] or int(game.headers['SearchSimulations'])!=512:raise ValueError('Game budget differs')
        color=row['color']=='white'
        if game.headers['White'].startswith('NeuroChess')!=color:raise ValueError('Model color differs')
        moves=[]
        for move in game.mainline_moves():
            if move not in board.legal_moves:raise ValueError('Illegal move')
            moves.append(move.uci());board.push(move)
        outcome=board.outcome(claim_draw=True)
        if outcome is None:raise ValueError('Unfinished game')
        score=.5 if outcome.winner is None else float(outcome.winner==color)
        if score!=row['score'] or len(moves)!=row['plies'] or outcome.result()!=row['result'] or game.headers['Result']!=row['result'] or outcome.termination.name!=row['termination']:raise ValueError('Result does not match replay')
        games.append(dict(**row,pgn=pgn,uci_moves=moves))
    if seen!=expected:raise ValueError('Not all scheduled games completed')
    results={}
    for nodes in [500,1000,1500]:
        rows=[r for r in games if r['nodes']==nodes]
        scores=[np.mean([r['score'] for r in rows if r['opening']==i]) for i in range(pairs)]
        score=float(np.mean(scores));bounds=exact_pair_bootstrap(scores);eps=math.sqrt(math.log(40)/(2*pairs))
        conservative=[max(0.,score-eps),min(1.,score+eps)]
        item=dict(games=len(rows),independent_opening_pairs=pairs,wins=sum(r['score']==1 for r in rows),draws=sum(r['score']==.5 for r in rows),
            losses=sum(r['score']==0 for r in rows),score=score,relative_elo=rating(score),
            pair_bootstrap_score95=bounds,pair_bootstrap_elo95=[rating(x) for x in bounds],
            hoeffding_score95=conservative,hoeffding_elo95=[rating(x) for x in conservative],
            pentanomial=[sum(np.isclose(s,v) for s in scores).item() for v in [0,.25,.5,.75,1]])
        for field in ['wins','draws','losses']:
            if item[field]!=reported['results'][str(nodes)][field]:raise ValueError('Aggregate differs')
        if not np.isclose(score,reported['results'][str(nodes)]['score']):raise ValueError('Score differs')
        results[str(nodes)]=item
    calibration,ladder=calibration_and_ladder(root,games,plan)
    cloud=json.loads((root/'cloud/completion-summary.json').read_text())
    report=dict(training_and_cloud_evaluation=cloud,local_plan=plan,results=results,stockfish_calibration=calibration,relative_ladder=ladder,
        checkpoint=dict(path=str(checkpoint.resolve()),bytes=checkpoint.stat().st_size,sha256=sha(checkpoint)),
        verification=dict(games=len(games),all_moves_legal=True,all_terminal_results_verified=True,all_budget_ids_verified=True,all_opening_color_pairs_complete=True),
        interpretation=dict(elo='Delta Elo against each fixed-node SF18 configuration: 400*log10(score/(1-score)). No calibrated absolute/human rating.',
            uncertainty='Exact nonparametric bootstrap over whole two-game opening pairs; 95% marginal intervals per node budget. Hoeffding intervals are an additional conservative independent-pair bound. Null Elo endpoints mean negative or positive infinity at score 0 or 1, or a separated ladder fit.',
            scope='16 frozen opening pairs per opponent. Shared openings make different budget estimates correlated; these are not 96 independent openings. No pooling with cloud games.',
            source='https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html#the-elo-rating-of-stockfish'),games=games)
    target=root/'all-results.json';target.write_text(json.dumps(report,indent=2))
    def num(x):return f'{x:+.0f}' if x is not None else 'unbounded'
    lines=['# Local fixed-node Stockfish comparison','',f'Model SHA-256: `{plan["checkpoint_sha256"]}`. Multihead native PUCT, 512 simulations per move.','',
        '| SF nodes | W-D-L | Score | Relative Elo | Paired bootstrap 95% |','|---:|---:|---:|---:|---:|']
    for node,r in results.items():lines.append(f'| {node} | {r["wins"]}-{r["draws"]}-{r["losses"]} | {100*r["score"]:.1f}% | {num(r["relative_elo"])} | {num(r["pair_bootstrap_elo95"][0])} to {num(r["pair_bootstrap_elo95"][1])} |')
    lines+=['','Relative ladder (SF1000 anchored at zero):',*[f'- {name}: {num(r["relative_elo"])} Elo, joint opening bootstrap 95% {r["joint_opening_bootstrap95"]}' for name,r in ladder['ratings'].items()],'','All 96 PGNs replayed and terminal scores verified. Each budget uses the same 16 openings with colors swapped.','',
        'These are Elo differences versus the specified SF node budgets, not calibrated human or unrestricted-engine ratings. Pair bootstrap intervals are conditional on this small opening sample; conservative bounds and complete PGNs are in all-results.json.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(results=results,verification=report['verification'],report=str(target)),indent=2))

if __name__=='__main__':main()
