"""Read-only P/Q/V factorial interventions for five audited search regressions.

Outputs include each source's full original trunk. Every diagnostic cell runs
both networks; fixed simulations control work, not latency or playing strength.
The locked timed baseline and all checkpoint/source files remain immutable.
"""
import torch
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.append('/home/xia/code/llmlearn/.venv/lib/python3.13/site-packages')
import argparse,concurrent.futures,fcntl,hashlib,itertools,json,math,os,signal,time,traceback
import numpy as np
import chess,chess.engine
from scripts.lichess_search_audit import (verify_inputs,load_net,PARENT,CANDIDATE,ENGINE,FastTeacher,
    scored_infos,sha,write,read,teacher_wl)
from neurochess.encoding import encode,legal_ids,position_key
from neurochess.multihead import fuse_heads
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.search_diagnostics import decode_board

AUDIT=ROOT/'output/lichess-search-audit-512-20261105'
OUT=ROOT/'output/lichess-five-failure-mechanisms-20261105'
FAILURES=[147,296,302,386,420]

def pack(board):
    t,m=encode(board);moves,ids=legal_ids(board);a=np.full((1,256),-1,np.int16);a[0,:len(ids)]=ids
    return t[None],m[None],a,moves

def effective_key(tokens,meta,ids):
    meta=np.array(meta,copy=True);meta[[5,6,7,10]]=0
    return tokens.tobytes()+meta.tobytes()+ids.tobytes()

class Mixed:
    device='cuda'
    def __init__(self,nets,sources,route='normal',root=None,capture=False):
        self.nets=nets;self.sources=sources;self.route=route;self.model=nets[0].model
        self.features={};self.handles=[];self.samples={};self.capture=capture
        self.rootkey=None if root is None else effective_key(*[a[0] for a in pack(root)[:3]])
        for i,net in enumerate(nets):
            def hook(module,inputs,i=i):self.features[i]=inputs[0].detach()
            self.handles.append(net.model.value[0].register_forward_pre_hook(hook))

    def close(self):
        for h in self.handles:h.remove()
        self.handles=[]

    def evaluate_search_heads(self,tokens,meta,moves):
        raw=[n.raw_outputs(tokens,meta,moves) for n in self.nets]
        p,q,v=self.sources;wdl=raw[v][:,512:] if v<2 else None
        if v>=2:
            # v2: old Value MLP on new trunk; v3: new Value MLP on old trunk.
            head,trunk=(0,1) if v==2 else (1,0)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                wdl=self.nets[head].model.value(self.features[trunk]).float().softmax(-1).cpu().numpy()
        logits,Q,value,support=fuse_heads(raw[p][:,:256],raw[q][:,256:512],wdl,moves>=0)
        if self.route!='normal':
            candidate=raw[1];old=raw[0];newV=candidate[:,512]-candidate[:,514];oldV=old[:,512]-old[:,514]
            baseP,baseQ,baseV,baseSupport=fuse_heads(candidate[:,:256],candidate[:,256:512],candidate[:,512:],moves>=0)
            if self.route=='old_leaf_FPU':logits,Q,value=baseP,baseQ,oldV
            elif self.route=='old_confidence':
                logits,_,_,_=fuse_heads(candidate[:,:256],candidate[:,256:512],old[:,512:],moves>=0)
                Q,value=baseQ,baseV
            elif self.route=='old_value_Q_seed':
                logits,value=baseP,baseV
                Q=np.where(baseSupport,.5*candidate[:,256:512]+.5*oldV[:,None],np.nan)
            elif self.route=='old_V_root_input_only':
                oldP,oldQ,oldValue,_=fuse_heads(candidate[:,:256],candidate[:,256:512],old[:,512:],moves>=0)
                mask=np.array([effective_key(tokens[i],meta[i],moves[i])==self.rootkey for i in range(len(tokens))])
                logits,Q,value=baseP,baseQ,baseV
                logits[mask]=oldP[mask];Q[mask]=oldQ[mask];value[mask]=oldValue[mask]
            else:raise ValueError('Unknown routing intervention')
        if self.capture:
            drift=np.abs((raw[1][:,512]-raw[1][:,514])-(raw[0][:,512]-raw[0][:,514]))
            for i in range(len(tokens)):
                key=hashlib.sha256(effective_key(tokens[i],meta[i],moves[i])).hexdigest()
                if key in self.samples:continue
                record=dict(drift=float(drift[i]),tokens=tokens[i].copy(),meta=meta[i].copy(),ids=moves[i].copy(),old=raw[0][i].copy(),new=raw[1][i].copy())
                self.samples[key]=record
            if len(self.samples)>48:
                self.samples=dict(sorted(self.samples.items(),key=lambda kv:-kv[1]['drift'])[:32])
        return np.asarray(logits,np.float32),np.asarray(Q,np.float32),np.asarray(value,np.float32)

class ObserveLibrary:
    """Read root snapshots after completed backups; no traversal/state edits."""
    def __init__(self,lib,trace):self.lib=lib;self.trace=trace;self.last=-1
    def __getattr__(self,name):return getattr(self.lib,name)
    def nc_submit_heads(self,*args):
        rc=self.lib.nc_submit_heads(*args)
        if rc>=0:
            result=json.loads(self.lib.nc_result(args[0]));n=result['roots'][0]['new_simulations']
            if self.last<0 or n-self.last>=128:
                self.trace.append(result);self.last=n
        return rc

def root_tables(nets,job):
    board=chess.Board(job['fen']);t,m,ids,moves=pack(board)
    raws=[n.raw_outputs(t,m,ids)[0] for n in nets];result={}
    for name,r in zip(['parent','candidate'],raws):
        p=r[:256];q=r[256:512];wdl=r[512:];prob=np.exp(p-p.max());prob/=prob.sum()
        fused,prior,value,support=fuse_heads(p[None],q[None],wdl[None],ids>=0)
        fp=np.exp(fused[0]-fused[0].max());fp/=fp.sum()
        result[name]=dict(wdl=wdl.tolist(),value=float(value[0]),moves={move.uci():dict(policy_probability=float(prob[i]),Q=float(q[i]),fused_prior=float(fp[i]),
            Q_seed=None if not np.isfinite(prior[0,i]) else float(prior[0,i]),supported=bool(support[0,i])) for i,move in enumerate(moves)})
    return result

def prepare_jobs():
    jobs={i:read(AUDIT/'search'/f'{i:03d}.json') for i in range(512)}
    teachers={i:read(AUDIT/'teachers'/f'{i:03d}.json') for i in range(512)}
    controls=[]
    for i in FAILURES:
        j=jobs[i];b=chess.Board(j['fen']);options=[]
        for k,x in jobs.items():
            t=teachers[k]
            if k in FAILURES+controls or x['cohort']!=j['cohort'] or x['selection_stratum']!=j['selection_stratum']:continue
            if x['search']['parent']['move']!=x['search']['candidate']['move'] or not t['final']['all_cp'] or not t['final']['all_exact'] or t['final']['candidate_regret']>100:continue
            c=chess.Board(x['fen']);score=abs(len(c.piece_map())-len(b.piece_map()))+abs(c.legal_moves.count()-b.legal_moves.count())/10
            options.append((score,k))
        if not options:raise ValueError('No retained matched control')
        controls.append(min(options)[1])
    return [dict(jobs[i],diagnostic_group='failure' if i in FAILURES else 'retained_control') for i in FAILURES+controls]

def teacher_case(out,job,candidates):
    path=out/'teachers'/f"{job['index']:03d}.json"
    if path.exists():return read(path)
    engine=FastTeacher(ENGINE,threads=1,hash_mb=64,timeout=300);b=chess.Board(job['fen']);stages=[]
    try:
        discovery=scored_infos(engine,b,5000000);moves=list(dict.fromkeys(candidates+[discovery[0]['move']]))
        for nodes in ([10000000,20000000,40000000] if job['index'] in FAILURES else [5000000]):
            rows=scored_infos(engine,b,nodes,moves)
            stages.append(dict(requested_nodes=nodes,teacher=rows))
            if min(r['depth'] for r in rows)>=24 and not any(r['lowerbound'] or r['upperbound'] for r in rows):break
        result=dict(index=job['index'],discovery=discovery,stages=stages,final=stages[-1]['teacher'],
            depth24_exact=min(r['depth'] for r in rows)>=24 and not any(r['lowerbound'] or r['upperbound'] for r in rows),identity=engine.identity)
        write(path,result);return result
    finally:engine.quit()

def main():
    out=OUT;out.mkdir(parents=True,exist_ok=True)
    handle=(out/'run.lock').open('w');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('2-hour bounded diagnostic limit')));signal.alarm(7200)
    def status(phase,**kw):write(out/'status.json',dict(phase=phase,pid=os.getpid(),seconds=time.time()-started,updated_at=time.time(),training_resumed=False,**kw));print(json.dumps(dict(phase=phase,**kw)),flush=True)
    try:
        torch.set_num_threads(2);torch.set_float32_matmul_precision('high');lock,cert=verify_inputs()
        jobs=prepare_jobs();write(out/'jobs.json',jobs)
        plan=dict(failures=FAILURES,controls=[j['index'] for j in jobs if j['index'] not in FAILURES],checkpoint_sha256=cert['checkpoint_sha256'],
            root_config=lock['config'],diagnostic_budgets=[2048,4096],factorial='All2x2x2 P/Q/V outputs including each original trunk; both full networks evaluated in every cell.',
            extra_V_modes=['old_leaf_FPU','old_confidence','old_value_Q_seed','old_V_root_input_only','old_V_MLP_new_trunk','new_V_MLP_old_trunk'],
            scope='Selected5 failures and5 retained matched controls, not unbiased quality/strength proof. Fixed-budget diagnostics change only time/simulation stopping and use original engine; no speed claim. All original files immutable.',
            teacher='Five failures: unrestricted5M, common candidate10M/20M/40M until depth24 exact, bounded. Controls5M. No training labels.',
            script_sha256=sha(__file__),training_resumed=False)
        if (out/'plan.json').exists() and read(out/'plan.json')!=plan:raise ValueError('Diagnostic plan changed')
        write(out/'plan.json',plan);status('loading')
        nets=[load_net(p) for p in [PARENT,CANDIDATE]]
        b=chess.Board(jobs[0]['fen']);t,m,ids,_=pack(b)
        for n in nets:
            for size in [1,64]:n.raw_outputs(np.repeat(t,size,0),np.repeat(m,size,0),np.repeat(ids,size,0))
        root={str(j['index']):root_tables(nets,j) for j in jobs};write(out/'root-outputs.json',root)
        rows=[]
        def search(job,code,budget,route='normal',timed=False,capture=False):
            label=f'{job["index"]:03d}-{code}-{budget}-{route}'+('-timed' if timed else '')
            path=out/'search'/(label+'.json')
            if path.exists():r=read(path);rows.append(r);return r
            cfg=dict(lock['config'])
            if not timed:cfg.update(time_limit=None,simulations=budget)
            mixed=None
            if timed:net=nets[int(code[0])]
            else:net=mixed=Mixed(nets,tuple(map(int,code)),route,root=chess.Board(job['fen']),capture=capture)
            player=NativeMCTSPlayer(net,**cfg);trace=[]
            if capture:player.lib=ObserveLibrary(player.lib,trace)
            tick=time.time()
            try:move=player.choose(chess.Board(job['fen']));last=player.last_result;stats=player.search_stats
            finally:
                player.close()
                if mixed:mixed.close()
            r=dict(index=job['index'],group=job['diagnostic_group'],code=code,budget=budget,route=route,timed=timed,
                move=move.uci(),root=last,stats=stats,seconds=time.time()-tick,trace=trace)
            if capture and mixed.samples:
                chosen=sorted(mixed.samples.items(),key=lambda a:-a[1]['drift'])[:24]
                r['node_samples']=[dict(input_sha256=k,drift=x['drift'],fen=decode_board(x['tokens'],x['meta']).fen(),
                    parent_value=float(x['old'][512]-x['old'][514]),candidate_value=float(x['new'][512]-x['new'][514])) for k,x in chosen]
            write(path,r);rows.append(r);return r
        # Unmodified timed one-network controls retain the original deployment budget.
        for j in jobs:
            for code in ['000','111']:search(j,code,0,timed=True)
        combos=[''.join(map(str,c)) for c in itertools.product([0,1],repeat=3)]
        total=0
        for j in jobs:
            for budget in ([2048,4096] if j['index'] in FAILURES else [4096]):
                order=combos[j['index']%8:]+combos[:j['index']%8]
                for code in order:
                    search(j,code,budget,capture=budget==4096 and code in ['000','111']);total+=1
                    status('factorial',completed=total,total=120,index=j['index'],code=code,budget=budget)
        for j in jobs:
            if j['index'] not in FAILURES:continue
            for route in ['old_leaf_FPU','old_confidence','old_value_Q_seed','old_V_root_input_only']:
                search(j,'111',4096,route=route);status('value-routing',index=j['index'],route=route)
            for code in ['112','113']:search(j,code,4096);status('value-weight-location',index=j['index'],code=code)
        # Common teachers score EVERY chosen intervention move, never just a preselected answer.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures={}
            for j in jobs:
                old=read(AUDIT/'teachers'/f"{j['index']:03d}.json")
                candidates=list(dict.fromkeys([x['move'] for x in (old['confirmation_teacher'] or old['initial_teacher'])]+[r['move'] for r in rows if r['index']==j['index']]))
                futures[pool.submit(teacher_case,out,j,candidates)]=j['index']
            for f in concurrent.futures.as_completed(futures):f.result();status('deep-teacher',index=futures[f],done=sum(f.done() for f in futures),total=len(jobs))
        teachers={j['index']:read(out/'teachers'/f"{j['index']:03d}.json") for j in jobs}
        for r in rows:
            ref=teachers[r['index']]['final'];scores={a['move']:a['sort_score'] for a in ref}
            r['regret']=max(scores.values())-scores[r['move']]
            r['teacher_cp_only']=all(x['cp'] is not None for x in ref);r['teacher_depth24_exact']=teachers[r['index']]['depth24_exact']
        effects={}
        for j in jobs:
            cases=[r for r in rows if r['index']==j['index'] and r['budget']==4096 and r['route']=='normal' and r['code'] in combos and not r['timed']]
            bycode={r['code']:r for r in cases};heads={}
            for k,h in enumerate(['Policy','Q','Value']):
                paired=[]
                for code,a in bycode.items():
                    if code[k]!='0':continue
                    other=code[:k]+'1'+code[k+1:];b=bycode[other]
                    paired.append(dict(other_heads=code,a=code,b=other,regret_delta=b['regret']-a['regret']))
                heads[h]=dict(mean_regret_delta_new_minus_old=float(np.mean([a['regret_delta'] for a in paired])),contexts=paired)
            effects[str(j['index'])]=heads
        verify_inputs()
        report=dict(completed=True,jobs=jobs,rows=rows,effects=effects,teachers=teachers,
            main_training_resumed=False,optimizer_steps=0,automatic_promotion=False,
            limitations=plan['scope'],seconds=time.time()-started)
        write(out/'results.json',report)
        write(out/'completion.json',dict(completed=True,report_sha256=sha(out/'results.json'),files_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix not in ['.log','.lock','.tmp'] and p.name not in ['status.json','completion.json']},training_resumed=False))
        status('complete',searches=len(rows),roots=len(jobs))
    except BaseException:
        write(out/'failure.json',dict(traceback=traceback.format_exc(),at=time.time(),training_resumed=False));status('failed');raise

if __name__=='__main__':main()
