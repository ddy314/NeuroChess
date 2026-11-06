"""Bounded fixed-position parent/final check; no full games or promotion."""
import json
from pathlib import Path
import sys
import time
import chess.engine
import numpy as np
import torch
from neurochess.data import record
from neurochess.model import ModelConfig
from neurochess.policy_value_pair import PolicyValuePair
from neurochess.multihead import MultiheadPlayer, fuse_heads
from neurochess.native_mcts import NativeMCTSPlayer
from scripts.local_teacher_worker import board_for
from scripts.targeted_mistakes import classify, stable_labels
from scripts.local_targeted_annotation import write


def main(plan_path):
    plan=json.loads(Path(plan_path).read_text());paths=plan['paths'];work=Path(paths['work'])
    jobs=json.loads(Path('/content/joint-test-jobs.json').read_text())
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/search-baseline-lock.json').read_text())['config']
    torch.set_num_threads(3);models={}
    for name,path in [('parent',paths['parent']),('final',str(work/'last.pt'))]:
        state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        model=PolicyValuePair(ModelConfig(**state['model_config']));model.load_state_dict(state['model'])
        models[name]=model.cuda().eval();del state
    reports=[];started=time.time()
    with chess.engine.SimpleEngine.popen_uci('/content/neurochess/bin/stockfish',timeout=120) as engine:
        engine.configure({'Threads':1,'Hash':64,'Skill Level':20,'UCI_LimitStrength':False})
        def teacher(board,nodes,moves=None):
            kw={} if moves is None else dict(root_moves=[board.parse_uci(m) for m in moves])
            info=engine.analyse(board,chess.engine.Limit(nodes=nodes),multipv=min(4,board.legal_moves.count()) if moves is None else len(moves),game=object(),**kw)
            result=[]
            for i in info:
                score=i['score'].white();p=dict(line=' '.join(m.uci() for m in i['pv']))
                p['mate' if score.is_mate() else 'cp']=score.mate() if score.is_mate() else score.score();result.append(p)
            return result
        for repeat in range(3):
            for index,job in enumerate(jobs):
                board=board_for(job);row,_=record(board);choices=[]
                for name in (['parent','final'] if (repeat+index)%2==0 else ['final','parent']):
                    net=MultiheadPlayer(models[name],'cuda')
                    raw=net.raw_outputs(np.array([row['tokens']]),np.array([row['meta']]),np.array([row['moves']]))
                    fused,_,values,_=fuse_heads(raw[:,:256],raw[:,256:512],raw[:,512:],np.array([row['moves']>=0]))
                    legal=list(board.legal_moves);direct=legal[int(fused[0,:len(legal)].argmax())].uci()
                    player=NativeMCTSPlayer(net,**config);start=time.time()
                    try:move=player.choose(board);stats=player.search_stats
                    finally:player.close()
                    choices.append(dict(model=name,move=move.uci(),direct=direct,value=float(values[0]),seconds=time.time()-start,stats=stats))
                coarse=teacher(board,10000);moves=list(dict.fromkeys([p['line'].split()[0] for p in coarse]+[c[k] for c in choices for k in ['move','direct']]))
                first=teacher(board,100000,moves);second=teacher(board,300000,moves)
                stable=stable_labels(first,second,board.turn)
                for c in choices:
                    c['search_decision']=classify(second,c['move'],c['value'],board.turn)
                    c['direct_decision']=classify(second,c['direct'],c['value'],board.turn)
                reports.append(dict(repeat=repeat,key=job['key'],group=job['group'],ply=job.get('ply'),stable=stable,choices=choices,teacher=second))
                write(work/'fixed-position-evaluation.json',dict(protocol='16 distinct held-out games x3 counterbalanced repeats; locked2s search; coherent100k/300k teacher; no full games, Elo or automatic promotion',completed_pairs=len(reports),total_pairs=48,rows=reports,seconds=time.time()-started))
                write(work/'status.json',dict(phase='fixed-position-evaluation',step=plan['next_training']['steps'],total_steps=plan['next_training']['steps'],updated_at=time.time(),completed_pairs=len(reports),total_pairs=48))


if __name__=='__main__':main(sys.argv[1])
