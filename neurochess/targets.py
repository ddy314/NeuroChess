"""Versioned engine targets shared by cache import and live distillation."""
import chess.engine
import numpy as np

TARGET_SPEC = dict(schema=1, policy='softmax-side-to-move-centipawns', temperature_cp=100,
                   wdl_model='sf16', wdl_ply=32, q='W-L', q_scope='annotated-legal-moves-only',
                   value='best-annotated-action-WDL', history_free=True)


def score_targets(pv, turn):
    if ('cp' in pv) == ('mate' in pv):
        raise ValueError('exactly one of cp/mate required')
    value=pv.get('cp',pv.get('mate'))
    if not isinstance(value,int) or isinstance(value,bool):raise ValueError('score must be an integer')
    signed=value if turn else -value
    if 'mate' in pv:
        if not signed:raise ValueError('terminal mate score on nonterminal board')
        wdl=np.array([1.,0.,0.] if signed>0 else [0.,0.,1.],np.float32)
        rank=np.sign(signed)*(100000-min(abs(signed),10000))
    else:
        w=chess.engine.Cp(signed).wdl(model=TARGET_SPEC['wdl_model'],ply=TARGET_SPEC['wdl_ply'])
        wdl=np.array([w.wins,w.draws,w.losses],np.float32)/1000
        rank=np.clip(signed,-50000,50000)
    return float(rank),float(wdl[0]-wdl[2]),wdl


def assign_targets(row, board, legal, pvs, temperature_cp=100):
    indices=[];ranks=[];wdls=[]
    for pv in pvs:
        move=board.parse_uci(pv['line'].split()[0]);index=legal.index(move)
        if index in indices:raise ValueError('duplicate first move in MultiPV')
        rank,q,wdl=score_targets(pv,board.turn)
        indices.append(index);ranks.append(rank);wdls.append(wdl)
        row['q'][index]=q;row['qmask'][index]=True
    if not indices or temperature_cp<=0:raise ValueError('nonempty candidates and positive temperature required')
    values=np.asarray(ranks,np.float64)
    probabilities=np.exp(np.maximum(-80,(values-values.max())/temperature_cp))
    row['policy'][indices]=probabilities/probabilities.sum()
    row['value']=wdls[int(values.argmax())];row['value_weight']=1
    return len(indices)
