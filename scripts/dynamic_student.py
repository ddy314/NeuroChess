"""Batched current-student predictions; never starts an optimizer or cloud VM."""
import hashlib,time
from pathlib import Path
import numpy as np
import chess
from neurochess.encoding import encode,legal_ids
from neurochess.multihead import MultiheadPlayer,fuse_heads
from neurochess.dynamic_curriculum import board_for


class Student:
    def __init__(self,path,expected_sha=None):
        import torch
        from scripts.analyze_joint_checkpoint import load
        self.path=Path(path);self.sha256=hashlib.file_digest(self.path.open('rb'),'sha256').hexdigest()
        if expected_sha and self.sha256!=expected_sha:raise ValueError('Student checkpoint SHA differs')
        torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
        model,self.step=load(self.path);self.net=MultiheadPlayer(model,'cuda')

    def boards(self,boards):
        encoded=[encode(b) for b in boards];moves=np.full((len(boards),256),-1,np.int16);legals=[]
        for i,b in enumerate(boards):
            ms,ids=legal_ids(b)
            if not ms:raise ValueError('Predict nonterminal positions only')
            moves[i,:len(ids)]=ids;legals.append(ms)
        raw=self.net.raw_outputs(np.stack([t[0] for t in encoded]),np.stack([t[1] for t in encoded]),moves)
        return self.predictions(raw,moves,legals)

    def predictions(self,raw,moves,legals):
        fused,_,values,support=fuse_heads(raw[:,:256],raw[:,256:512],raw[:,512:],moves>=0)
        result=[]
        for i,legal in enumerate(legals):
            choice=int(fused[i].argmax());top=np.argsort(-raw[i,:len(legal)])[:5];active=np.flatnonzero(support[i])
            candidates=list(dict.fromkeys([legal[choice].uci()]+[legal[k].uci() for k in active]))
            result.append(dict(model_sha256=self.sha256,model_step=self.step,move=legal[choice].uci(),top_moves=candidates,
                               policy_top5=[legal[k].uci() for k in top],value=float(values[i]),wdl=raw[i,512:].tolist(),
                               support_q={legal[k].uci():float(raw[i,256+k]) for k in active}))
        return result

    def jobs(self,jobs):
        boards=[]
        for j in jobs:
            boards.append(board_for(j))
        return self.boards(boards)

    def values(self,jobs):
        """Exact existing V branch for coarse prioritization, no P/Q forward."""
        import torch
        encoded=[]
        for j in jobs:
            encoded.append(encode(board_for(j)))
        return self.values_encoded(np.stack([t[0] for t in encoded]),np.stack([t[1] for t in encoded]))

    def values_encoded(self,tokens,meta):
        import torch
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=self.net.model.value_only(torch.as_tensor(tokens,device='cuda'),torch.as_tensor(meta,device='cuda'))
            wdl=logits.softmax(-1).cpu().numpy()
        return [dict(model_sha256=self.sha256,model_step=self.step,value=float(w[0]-w[2]),wdl=w.tolist(),value_only=True) for w in wdl]
