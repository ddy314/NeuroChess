"""Development-only tools for causal search/evaluator comparisons.

Stockfish here is an oracle intervention, never the deployed player's evaluator.
Its labels and runtimes must be reported separately from large-model queries.
"""
from collections import defaultdict

import chess
import chess.engine
import numpy as np

from .encoding import move_id


def decode_board(tokens, meta):
    """Invert the effective square-token input for diagnostic FEN evaluation.

    Repetition history is unavailable in an encoded observation. Exact game-rule
    handling stays in native search; diagnostic Stockfish labels are FEN-only.
    """
    turn = bool(meta[9]); flip = 0 if turn else 56
    board = chess.Board(None); board.turn = turn
    for square, token in enumerate(tokens):
        token = int(token)
        if token:
            board.set_piece_at(square^flip,chess.Piece((token-1)%6+1,turn if token<=6 else not turn))
    for color, offset in [(turn,0),(not turn,2)]:
        if meta[offset]: board.castling_rights |= chess.BB_H1 if color else chess.BB_H8
        if meta[offset+1]: board.castling_rights |= chess.BB_A1 if color else chess.BB_A8
    board.ep_square = (round(float(meta[4])*63)^flip) if meta[11] else None
    board.halfmove_clock = round(float(meta[5])*150)
    board.fullmove_number = max(1,round(float(meta[10])*300))
    if not board.is_valid():
        raise ValueError('Diagnostic decoder produced an invalid board')
    return board


def teacher_label(engine, board, nodes=50000):
    info=engine.analyse(board,chess.engine.Limit(nodes=nodes),game=object())
    score=info['score'].pov(board.turn); wdl=score.wdl(model='sf16',ply=32)
    return dict(cp=score.score(),mate=score.mate(),value=(wdl.wins-wdl.losses)/1000,
                depth=info.get('depth'),pv=[m.uci() for m in info.get('pv',[])[:12]])


class OracleCorrections:
    """Swap correction WDL, keeping policy/Q fixed. Not a centipawn-order oracle.

    The fixed-ply WDL proxy saturates: different losing centipawn scores can
    have identical targets. Encoded observations also omit full repetition history.
    """
    def __init__(self, network, engine, nodes=50000):
        self.network,self.engine,self.nodes=network,engine,nodes
        self.device=network.device;self.model=network.model;self.labels={};self.calls=0

    def evaluate_search_heads(self, tokens, meta, ids):
        policy,q,values=self.network.evaluate_search_heads(tokens,meta,ids)
        values=np.asarray(values).copy()
        for i in range(len(tokens)):
            b=decode_board(tokens[i],meta[i]);key=b.fen(en_passant='fen')
            if key not in self.labels:
                self.labels[key]=teacher_label(self.engine,b,self.nodes);self.calls+=1
            values[i]=self.labels[key]['value']
        return policy,q,values


def tree_children(trace):
    children=defaultdict(list)
    for node in trace['nodes']:children[node['parent']].append(node['id'])
    return children


def node_path(trace, index):
    nodes=trace['nodes'];moves=[]
    while nodes[index]['parent']>=0:
        moves.append(nodes[index]['move']);index=nodes[index]['parent']
    return list(reversed(moves))


def node_board(root, trace, index):
    board=root.copy()
    for move in node_path(trace,index):board.push_uci(move)
    return board


def fixed_tree_backup(trace, mode='pure'):
    """Same expanded topology, different leaf/internal-value rules.

    This isolates backup effects; it does not claim the new rule would generate
    this same tree. Node creation order is parent-before-child in native search.
    """
    nodes=trace['nodes'];children=tree_children(trace);values=[0.]*len(nodes)
    for n in reversed(nodes):
        i=n['id'];kids=children.get(i,[])
        if n['solved']!=2:value=float(n['solved'])
        elif kids:
            value=max(-values[c] for c in kids)
            if mode=='anchored':
                w=n.get('effective_anchor_weight',n['anchor_weight'])
                value=(1-w)*value+w*n['anchor']
        else:
            value=n['anchor'] if n['corrected'] else n['cheap']
            if mode=='anchored':value=n['value']
        values[i]=value
    rank=sorted(children[0],key=lambda i:values[i])
    return dict(move=nodes[rank[0]]['move'],leaders=[dict(move=nodes[i]['move'],value=-values[i]) for i in rank],values=values)
