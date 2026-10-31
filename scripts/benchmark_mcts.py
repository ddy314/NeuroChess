"""Compare reference/native search and optional real checkpoint inference, no installs."""
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if importlib.util.find_spec('chess') is None:
    for site in (ROOT / '.venv/lib').glob('python*/site-packages'):
        if (site / 'chess/__init__.py').exists(): sys.path.append(str(site)); break
import chess
import numpy as np
import torch
from neurochess.encoding import encode, legal_ids
from neurochess.mcts import MCTSPlayer
from neurochess.native_mcts import NativeMCTSPlayer


class ZeroNetwork:
    device = 'cpu'
    def evaluate(self, boards):
        for b in boards: encode(b)
        legal = [legal_ids(b)[0] for b in boards]
        return legal, np.zeros((len(boards),256),np.float32), None, np.zeros(len(boards),np.float32)
    def evaluate_encoded(self, tokens, meta, ids):
        return np.zeros(ids.shape,np.float32), np.zeros(len(tokens),np.float32)


def measure(network, boards, simulations, repeats):
    rows = []
    for backend, batch in [('python',1), ('cpp',1), ('cpp',16)]:
        elapsed, samples, stats = [], [], []
        for _ in range(repeats):
            player = MCTSPlayer(network,simulations=simulations) if backend=='python' else NativeMCTSPlayer(network,simulations=simulations,leaf_batch_size=batch)
            start = time.perf_counter()
            # One game slot, no cross-game batch: measure single-game latency.
            moves = [player.choose(b).uci() for b in boards]
            elapsed.append(time.perf_counter()-start)
            samples.append(moves); stats.append(dict(player.search_stats))
            if hasattr(player,'close'): player.close()
        rows.append(dict(backend=backend,leaf_batch_size=batch,seconds=elapsed,
                         median_seconds=statistics.median(elapsed),moves=samples,stats=stats))
    base = rows[0]['median_seconds']
    for row in rows: row['speedup_vs_python'] = base/row['median_seconds']
    return rows


def main():
    p = argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--device',default='cuda')
    p.add_argument('--simulations',type=int,default=128);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(4)
    boards = [chess.Board(),chess.Board('rnbqkb1r/ppp2ppp/8/3pp3/2B1n3/5N2/PPPPQPPP/RNB1K2R w KQkq - 0 5'),
              chess.Board('rnbqkb1r/pp2pppp/5n2/3p4/3P4/2N5/PP2PPPP/R1BQKBNR w KQkq - 1 5')]
    report=dict(simulations_per_move=a.simulations,repeats=a.repeats,positions=[b.fen() for b in boards],
                notes=['Single game slot; cold trees for unrelated roots; model loading excluded.',
                       'Zero-network timings include chess rules, encoding, tree work and bridge, not neural inference.',
                       'Batched selection changes search ordering; equal simulation budgets do not imply identical moves.'],
                zero_network=measure(ZeroNetwork(),boards,a.simulations,a.repeats))
    print(json.dumps({'zero_network':report['zero_network']},indent=2),flush=True)
    if a.checkpoint:
        from neurochess.play import Player
        network=Player(a.checkpoint,a.device)
        for _ in range(3):
            network.evaluate([boards[0]]*16)
            encoded=[encode(boards[0]) for _ in range(16)]
            ids=np.full((16,256),-1,np.int16); ms=legal_ids(boards[0])[1];ids[:,:len(ms)]=ms
            network.evaluate_encoded(np.stack([x[0] for x in encoded]),np.stack([x[1] for x in encoded]),ids)
        report['checkpoint']=str(Path(a.checkpoint).resolve())
        report['gpu']=torch.cuda.get_device_name() if a.device.startswith('cuda') else None
        report['real_network']=measure(network,boards,a.simulations,a.repeats)
        # Exercise reuse on a legal game trajectory with real weights.
        player=NativeMCTSPlayer(network,simulations=a.simulations,leaf_batch_size=16)
        board=chess.Board(); trace=[]
        for ply in range(12):
            if board.is_game_over(claim_draw=True): break
            start=time.perf_counter();move=player.choose(board)
            trace.append(dict(ply=ply,fen=board.fen(),move=move.uci(),seconds=time.perf_counter()-start,stats=dict(player.search_stats)))
            board.push(move)
        report['real_network_reuse_trace']=trace
        player.close()
    report['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['neurochess/native/mcts.cpp','neurochess/native/vendor/chess.hpp','neurochess/native_mcts.py','neurochess/play.py','scripts/benchmark_mcts.py']}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n')
    print(a.out)

if __name__=='__main__':main()
