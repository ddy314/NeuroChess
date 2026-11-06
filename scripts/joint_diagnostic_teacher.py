"""Independent CPU-only paired teacher adjudication for post-training analysis."""
import threading
import chess,chess.engine
from neurochess.encoding import position_key
from scripts.targeted_mistakes import stable_labels

ENGINE=None


def initialize(binary):
    global ENGINE
    ENGINE=chess.engine.SimpleEngine.popen_uci(binary,timeout=180)
    ENGINE.configure({'Threads':1,'Hash':64,'Skill Level':20,'UCI_LimitStrength':False})
    threading._register_atexit(ENGINE.quit)


def adjudicate(task):
    j=task['job'];b=chess.Board(j['initial_fen'])
    for u in j['history']:b.push_uci(u)
    assert position_key(b).hex()==j['key']
    def analyze(nodes,moves=None):
        kwargs={} if moves is None else dict(root_moves=[b.parse_uci(m) for m in moves])
        info=ENGINE.analyse(b,chess.engine.Limit(nodes=nodes),multipv=min(4,b.legal_moves.count()) if moves is None else len(moves),game=object(),**kwargs)
        result=[]
        for i in info:
            s=i['score'].white();p=dict(line=' '.join(m.uci() for m in i['pv']),depth=i.get('depth'),nodes=i.get('nodes'))
            p['mate' if s.is_mate() else 'cp']=s.mate() if s.is_mate() else s.score();result.append(p)
        return result
    deep=analyze(1000000)
    moves=list(dict.fromkeys([p['line'].split()[0] for p in deep]+task.get('moves',[])))
    first=analyze(100000,moves);second=analyze(1000000,moves);stable=stable_labels(first,second,b.turn)
    refined=None
    if not stable:
        refined=analyze(3000000,moves);stable=stable_labels(second,refined,b.turn)
    return dict(job=j,unrestricted_deep=deep,first=first,second=second,teacher=refined or second,stable=stable,refined=refined is not None,protocol='Unrestricted1M top4; coherent candidate union100k/1M; unstable union independently3M;30cp stability')
