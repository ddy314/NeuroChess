"""History-aware, versioned teacher evidence and bounded dynamic mistake replay.

Canonical board keys guard split leakage; full-history IDs identify teacher
states. Derived positions never increase the count of new mistake roots.
No model, optimizer, cloud service or background monitor is started here.
"""
import hashlib,json,sqlite3,time
from pathlib import Path
import chess
import numpy as np
from .encoding import position_key,encode
from .targets import TARGET_SPEC

SCHEMA=2
DYNAMIC_TARGET_SPEC=dict(TARGET_SPEC,schema=2,history_free=False,missing_policy='zero-vector with explicit sidecar mask',q_scope='independently-stable-annotated-actions-only')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def teacher_spec_id(spec):
    # Scheduling the unresolved heads separately does not invalidate already
    # qualified targets from the same independent searches and numeric contract.
    return digest({k:v for k,v in spec.items() if k!='early_stop_when_any_target_is_stable'})


def board_for(job):
    b=chess.Board(job['initial_fen'])
    if b.chess960 or job.get('source_game_headers',{}).get('Variant','Standard')!='Standard':raise ValueError('Unsupported variant')
    for move in job['history']:
        parsed=b.parse_uci(move)
        if not parsed:raise ValueError('Null moves are not legal source-game moves')
        b.push(parsed)
    if not b.is_valid() or position_key(b).hex()!=job['key']:raise ValueError('Invalid state identity/history')
    return b


def encode_jobs_grouped(jobs):
    """Reuse exact legal history prefixes without dropping repetition/clocks.

    Output order matches the caller. Only board reconstruction is shared;
    every position still receives its own canonical-key and validity checks.
    """
    tokens=np.empty((len(jobs),64),np.uint8);meta=np.empty((len(jobs),12),np.float32)
    order=sorted(range(len(jobs)),key=lambda i:(jobs[i]['initial_fen'],jobs[i]['group'],len(jobs[i]['history'])))
    previous=None;history=[];board=None
    for i in order:
        job=jobs[i];identity=(job['initial_fen'],job['group'])
        if identity!=previous:board=chess.Board(job['initial_fen']);history=[];previous=identity
        common=0
        for a,b in zip(history,job['history']):
            if a!=b:break
            common+=1
        while len(board.move_stack)>common:board.pop()
        for move in job['history'][common:]:
            parsed=board.parse_uci(move)
            if not parsed:raise ValueError('Null moves are not legal source-game moves')
            board.push(parsed)
        if not board.is_valid() or board.chess960 or job.get('source_game_headers',{}).get('Variant','Standard')!='Standard' or position_key(board).hex()!=job['key']:raise ValueError('Invalid grouped state identity/history')
        tokens[i],meta[i]=encode(board);history=job['history']
    return tokens,meta


def state_id(job):return digest([job['initial_fen'],job['history']])


def state_job(parent,board,branch,depth):
    j=dict(parent,key=position_key(board).hex(),history=[m.uci() for m in board.move_stack],
           role='descendant',branch=branch,depth=depth,root_id=parent.get('root_id',state_id(parent)))
    j.pop('case_id',None);return j


class Store:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path);self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS families(group_id TEXT PRIMARY KEY,split TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS exclusions(position_key TEXT PRIMARY KEY,reason TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS states(
          case_id TEXT PRIMARY KEY,position_key TEXT NOT NULL,input_key TEXT NOT NULL,group_id TEXT NOT NULL,
          split TEXT NOT NULL,role TEXT NOT NULL,root_id TEXT NOT NULL,depth INTEGER NOT NULL,
          terminal INTEGER NOT NULL,quarantine INTEGER NOT NULL DEFAULT 0,job TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS state_position ON states(position_key,split);
        CREATE INDEX IF NOT EXISTS state_role ON states(split,role,quarantine);
        CREATE TABLE IF NOT EXISTS root_pools(case_id TEXT NOT NULL,pool TEXT NOT NULL,PRIMARY KEY(case_id,pool));
        CREATE TABLE IF NOT EXISTS edges(
          parent_id TEXT NOT NULL,move TEXT NOT NULL,child_id TEXT NOT NULL,branch TEXT NOT NULL,
          PRIMARY KEY(parent_id,move,child_id,branch));
        CREATE TABLE IF NOT EXISTS rollout_visits(
          root_id TEXT NOT NULL,case_id TEXT NOT NULL,branch TEXT NOT NULL,depth INTEGER NOT NULL,model_sha TEXT NOT NULL,
          PRIMARY KEY(root_id,case_id,branch,depth,model_sha));
        CREATE TABLE IF NOT EXISTS models(sha TEXT PRIMARY KEY,step INTEGER NOT NULL,path TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS predictions(
          case_id TEXT NOT NULL,model_sha TEXT NOT NULL,prediction TEXT NOT NULL,created REAL NOT NULL,
          PRIMARY KEY(case_id,model_sha),FOREIGN KEY(case_id) REFERENCES states(case_id));
        CREATE TABLE IF NOT EXISTS teachers(
          label_id TEXT PRIMARY KEY,case_id TEXT NOT NULL,spec_sha TEXT NOT NULL,candidates_sha TEXT NOT NULL,
          stable INTEGER NOT NULL,metadata TEXT NOT NULL,payload BLOB,created REAL NOT NULL,
          FOREIGN KEY(case_id) REFERENCES states(case_id));
        CREATE INDEX IF NOT EXISTS label_case ON teachers(case_id,stable);
        CREATE TABLE IF NOT EXISTS decisions(
          case_id TEXT NOT NULL,model_sha TEXT NOT NULL,label_id TEXT NOT NULL,active INTEGER NOT NULL,
          priority REAL NOT NULL,decision TEXT NOT NULL,PRIMARY KEY(case_id,model_sha),
          FOREIGN KEY(label_id) REFERENCES teachers(label_id));
        CREATE TABLE IF NOT EXISTS replay_usage(
          run_id TEXT NOT NULL,case_id TEXT NOT NULL,total INTEGER NOT NULL,PRIMARY KEY(run_id,case_id));
        CREATE TABLE IF NOT EXISTS draws(
          run_id TEXT NOT NULL,update_id INTEGER NOT NULL,pool TEXT NOT NULL,model_sha TEXT NOT NULL,case_ids TEXT NOT NULL,
          PRIMARY KEY(run_id,update_id,pool));
        CREATE TABLE IF NOT EXISTS replay_runs(
          run_id TEXT PRIMARY KEY,seed INTEGER NOT NULL,max_repeats INTEGER NOT NULL);
        ''')
        row=self.db.execute("SELECT value FROM settings WHERE key='schema'").fetchone()
        if row and json.loads(row[0])!=SCHEMA:raise ValueError('Incompatible curriculum schema')
        self.db.execute("INSERT OR IGNORE INTO settings VALUES ('schema',?)",(json.dumps(SCHEMA),));self.db.commit()

    def exclude(self,keys,reason):
        self.db.executemany('INSERT OR IGNORE INTO exclusions VALUES (?,?)',((k,reason) for k in keys))
        self.db.execute('UPDATE states SET quarantine=1 WHERE position_key IN (SELECT position_key FROM exclusions)')

    def add(self,job):
        b=board_for(job);case=state_id(job);group=job['group'];split=job['split']
        if split not in ('train','val','test'):raise ValueError('Unknown split')
        previous=self.db.execute('SELECT split FROM families WHERE group_id=?',(group,)).fetchone()
        if previous and previous[0]!=split:raise ValueError('Source game crosses train/validation/test')
        self.db.execute('INSERT OR IGNORE INTO families VALUES (?,?)',(group,split))
        collision=self.db.execute('SELECT 1 FROM states WHERE position_key=? AND split!=?',(job['key'],split)).fetchone()
        if collision:self.exclude([job['key']],'canonical cross-split collision')
        forbidden=self.db.execute('SELECT 1 FROM exclusions WHERE position_key=?',(job['key'],)).fetchone() is not None
        tokens,meta=encode(b);input_key=hashlib.sha256(tokens.tobytes()+meta.tobytes()).hexdigest()
        terminal=b.is_game_over(claim_draw=False);claimable=not terminal and b.can_claim_draw()
        j=dict(job,case_id=case,fen=b.fen(en_passant='fen'),side_to_move='white' if b.turn else 'black',schema=SCHEMA,claimable_draw=claimable)
        role=job.get('role','root');root=job.get('root_id',case)
        self.db.execute('INSERT OR IGNORE INTO states VALUES (?,?,?,?,?,?,?,?,?,?,?)',(case,job['key'],input_key,group,split,role,root,int(job.get('depth',0)),int(terminal),int(forbidden),json.dumps(j)))
        if job.get('replay_anchor') and role=='root':self.db.execute("INSERT OR IGNORE INTO root_pools VALUES (?,'general')",(case,))
        if claimable:self.db.execute('UPDATE states SET quarantine=1 WHERE case_id=?',(case,))
        if terminal:
            outcome=b.outcome(claim_draw=False);value=0 if outcome.winner is None else 1 if outcome.winner==b.turn else -1
            j.update(terminal=True,terminal_value=value,terminal_reason=outcome.termination.name)
            self.db.execute('UPDATE states SET job=? WHERE case_id=?',(json.dumps(j),case))
        return case

    def exact_terminal(self,case,model_sha):
        from .data import DTYPE
        j=self.job(case);b=board_for(j);outcome=b.outcome(claim_draw=False)
        if outcome is None:raise ValueError('Optional draw claim is not an exact terminal target')
        if not self.db.execute('SELECT 1 FROM models WHERE sha=?',(model_sha,)).fetchone():raise ValueError('Unregistered model')
        row=np.zeros((),dtype=DTYPE);row['tokens'],row['meta']=encode(b);row['moves']=-1
        v=0 if outcome.winner is None else 1 if outcome.winner==b.turn else -1
        row['value'][0 if v==1 else 2 if v==-1 else 1]=1.;row['value_weight']=1.;row['key']=np.frombuffer(position_key(b),np.uint8)
        spec=dict(kind='exact_automatic_chess_terminal',schema=SCHEMA,target_spec=DYNAMIC_TARGET_SPEC,claim_draw=False);identity=digest([case,spec])
        meta=dict(job=j,terminal=True,terminal_reason=outcome.termination.name,stable=True,supervision=dict(policy=False,value=True,q_moves=[]))
        self.db.execute('INSERT OR IGNORE INTO teachers VALUES (?,?,?,?,?,?,?,?)',(identity,case,digest(spec),digest([]),1,json.dumps(meta),row.tobytes(),time.time()))
        self.db.execute('INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?)',(case,model_sha,identity,0,.1,json.dumps(dict(terminal=True,value_target=v))))
        return identity

    def link(self,parent,move,child,branch,model_sha=None,root_id=None,depth=None):
        p=self.job(parent);c=self.job(child)
        if p['group']!=c['group'] or p['split']!=c['split']:raise ValueError('Edge crosses source family/split')
        b=board_for(p);b.push_uci(move)
        if state_id(dict(p,history=[m.uci() for m in b.move_stack]))!=child:raise ValueError('Edge is not an exact legal successor')
        self.db.execute('INSERT OR IGNORE INTO edges VALUES (?,?,?,?)',(parent,move,child,branch))
        if model_sha is not None:
            root=self.job(root_id)
            if c['initial_fen']!=root['initial_fen'] or c['history'][:len(root['history'])]!=root['history'] or len(c['history'])-len(root['history'])!=depth:raise ValueError('Rollout provenance differs from legal root prefix/depth')
            self.db.execute('INSERT OR IGNORE INTO rollout_visits VALUES (?,?,?,?,?)',(root_id,child,branch,depth,model_sha))

    def job(self,case):
        row=self.db.execute('SELECT job FROM states WHERE case_id=?',(case,)).fetchone()
        if not row:raise KeyError(case)
        return json.loads(row[0])

    def register_model(self,sha,step,path):
        self.db.execute('INSERT OR IGNORE INTO models VALUES (?,?,?)',(sha,step,str(path)))

    def predict(self,case,prediction):
        if not self.db.execute('SELECT 1 FROM models WHERE sha=?',(prediction['model_sha256'],)).fetchone():raise ValueError('Unregistered model SHA')
        self.db.execute('INSERT OR REPLACE INTO predictions VALUES (?,?,?,?)',(case,prediction['model_sha256'],json.dumps(prediction),time.time()))

    def save_label(self,case,spec,metadata):
        from .data import record
        from .targets import assign_targets
        if spec['target_spec']!=DYNAMIC_TARGET_SPEC:raise ValueError('WDL/policy/context target convention changed')
        job=self.job(case);b=board_for(job)
        if b.is_game_over(claim_draw=True):raise ValueError('Terminal states use explicit terminal targets')
        if state_id(metadata['job'])!=case:raise ValueError('Teacher labelled a different history')
        moves=sorted(p['line'].split()[0] for p in metadata['teacher'])
        if len(moves)!=len(set(moves)):raise ValueError('Duplicate teacher candidate')
        stable=metadata.get('stable') is True
        stages=metadata.get('attempts',[])
        if not stages or stages[-1]['teacher']!=metadata['teacher']:raise ValueError('Teacher target differs from final independent attempt')
        for pv in metadata['teacher']:
            line=b.copy()
            for move in pv['line'].split():line.push_uci(move)
        if stable and (len(stages)<2 or not all(s.get('independent') for s in stages[-2:])):raise ValueError('Stable labels require two independent budgets')
        if stable and max(s['budget'] for s in stages)<spec['minimum_confirm_nodes']:raise ValueError('Teacher below quality floor')
        payload=None
        if stable:
            r,legal=record(b);assign_targets(r,b,legal,metadata['teacher'])
            if metadata.get('supervision') is not None:
                from .teacher_stability import qualify_guarded
                verified=qualify_guarded(stages[-2]['teacher'],stages[-1]['teacher'],b.turn,spec,metadata.get('value_audit'))
                if verified!=metadata['supervision']:raise ValueError('Forged or stale target qualification')
                for stage in metadata.get('value_audit',[]):
                    for pv in stage['teacher']:
                        line=b.copy()
                        for move in pv['line'].split():line.push_uci(move)
                if stages[-1]['teacher']!=metadata['teacher']:raise ValueError('Teacher target differs from final independent attempt')
                if not verified['policy']:r['policy']=0
                r['value_weight']=float(verified['value'])
                r['qmask']=False
                for move in verified['q_moves']:r['qmask'][legal.index(b.parse_uci(move))]=True
            payload=r.tobytes()
        spec_sha=teacher_spec_id(spec);candidate_sha=digest(moves);identity=digest([case,spec_sha,candidate_sha,stages,metadata.get('supervision'),metadata.get('value_audit')])
        self.db.execute('INSERT OR IGNORE INTO teachers VALUES (?,?,?,?,?,?,?,?)',(identity,case,spec_sha,candidate_sha,int(stable),json.dumps(metadata),payload,time.time()))
        return identity

    def refresh(self,case,model_sha,label_id):
        from scripts.targeted_mistakes import classify,needs_repair
        p=self.db.execute('SELECT prediction FROM predictions WHERE case_id=? AND model_sha=?',(case,model_sha)).fetchone()
        label=self.db.execute('SELECT case_id,stable,metadata FROM teachers WHERE label_id=?',(label_id,)).fetchone()
        if not p or not label or label[0]!=case:raise ValueError('Prediction/label scope mismatch')
        pred=json.loads(p[0]);meta=json.loads(label[2]);job=self.job(case);b=board_for(job)
        if pred['move'] not in {pv['line'].split()[0] for pv in meta['teacher']}:raise ValueError('Student move missing: queue teacher expansion')
        decision=classify(meta['teacher'],pred['move'],pred['value'],b.turn)
        if meta.get('supervision') is not None:
            from .teacher_stability import qualified_decision
            decision=qualified_decision(meta['attempts'][-2]['teacher'],meta['teacher'],pred,b.turn,meta['supervision'])
        active=label[1] and decision.get('usable_error',needs_repair(decision))
        decision['teacher_pending_heads']=meta.get('pending_heads',[])
        decision['status']='qualified_error' if active else 'pending_teacher' if meta.get('pending_heads') else 'no_error_on_qualified_targets'
        priority=min(4.,max(decision['value_error']/.25,(decision['regret_cp'] or (100 if decision['policy_error'] else 0))/100)) if active else .05
        self.db.execute('INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?)',(case,model_sha,label_id,int(active),priority,json.dumps(decision)))
        return decision

    def cached_label(self,case,prediction,spec):
        """Reuse teacher evidence only for the exact history/protocol and support."""
        required=set(prediction['top_moves'])|{prediction['move']}
        for identity,encoded in self.db.execute('SELECT label_id,metadata FROM teachers WHERE case_id=? AND spec_sha=? AND stable=1 ORDER BY created DESC',(case,teacher_spec_id(spec))):
            meta=json.loads(encoded)
            if required.issubset({p['line'].split()[0] for p in meta['teacher']}):return identity,meta
        return None

    def cached_value_audit(self,case,spec):
        # Independent of the student version and its changed policy support.
        for encoded, in self.db.execute('SELECT metadata FROM teachers WHERE case_id=? AND spec_sha=? ORDER BY created DESC',(case,teacher_spec_id(spec))):
            audit=json.loads(encoded).get('value_audit')
            if audit:return audit
        return None

    def sample(self,model_sha,n,run_id,update_id,pool='mistake',seed=0,max_repeats=8):
        if n<1 or max_repeats<1:raise ValueError('Positive replay limits required')
        self.db.commit();self.db.execute('BEGIN IMMEDIATE')
        try:
            settings=self.db.execute('SELECT seed,max_repeats FROM replay_runs WHERE run_id=?',(run_id,)).fetchone()
            if settings and settings!=(seed,max_repeats):raise ValueError('Replay run seed/exposure cap changed on resume')
            self.db.execute('INSERT OR IGNORE INTO replay_runs VALUES (?,?,?)',(run_id,seed,max_repeats))
            prior=self.db.execute('SELECT case_ids,model_sha FROM draws WHERE run_id=? AND update_id=? AND pool=?',(run_id,update_id,pool)).fetchone()
            if prior:
                ids=json.loads(prior[0])
                if prior[1]!=model_sha or len(ids)!=n:raise ValueError('Resume draw differs in model or batch size')
                for case in ids:
                    row=self.db.execute('SELECT quarantine FROM states WHERE case_id=?',(case,)).fetchone()
                    if not row or row[0]:raise ValueError('Previously reserved example is now quarantined; invalidate this training update')
                self.db.commit();return ids
            condition="d.active=1 AND s.role='root'" if pool=='mistake' else "s.role='descendant'" if pool=='frontier' else "s.role='root' AND EXISTS(SELECT 1 FROM root_pools rp WHERE rp.case_id=s.case_id AND rp.pool='general')" if pool=='new' else None
            if condition is None:raise ValueError('Unknown dynamic pool')
            rows=self.db.execute(f'''SELECT s.case_id,s.position_key,d.priority FROM states s JOIN decisions d ON s.case_id=d.case_id
                 JOIN teachers t ON d.label_id=t.label_id LEFT JOIN replay_usage u ON u.run_id=? AND s.case_id=u.case_id
                 WHERE d.model_sha=? AND s.split='train' AND s.quarantine=0 AND t.stable=1 AND COALESCE(u.total,0)<? AND {condition}''',(run_id,model_sha,max_repeats)).fetchall()
            # Canonical key is the exposure unit even when several histories are retained.
            used={};eligible=[]
            all_usage=dict(self.db.execute('SELECT s.position_key,SUM(u.total) FROM replay_usage u JOIN states s ON s.case_id=u.case_id WHERE u.run_id=? GROUP BY s.position_key',(run_id,)))
            in_update=set()
            for encoded, in self.db.execute('SELECT case_ids FROM draws WHERE run_id=? AND update_id=?',(run_id,update_id)):
                for other in json.loads(encoded):in_update.add(self.db.execute('SELECT position_key FROM states WHERE case_id=?',(other,)).fetchone()[0])
            for row in rows:
                if row[1] not in used and row[1] not in in_update and all_usage.get(row[1],0)<max_repeats:eligible.append(row);used[row[1]]=True
            if len(eligible)<n:raise RuntimeError(f'Insufficient refreshed {pool} examples under exposure cap: {len(eligible)} < {n}; replenish data')
            rng=np.random.default_rng(np.random.SeedSequence([seed,update_id,int(digest(pool)[:8],16)]));weights=np.array([max(.05,r[2]) if pool=='mistake' else 1. for r in eligible]);weights/=weights.sum()
            ids=[eligible[i][0] for i in rng.choice(len(eligible),n,replace=False,p=weights)]
            for case in ids:self.db.execute('INSERT INTO replay_usage VALUES (?,?,1) ON CONFLICT(run_id,case_id) DO UPDATE SET total=total+1',(run_id,case))
            self.db.execute('INSERT INTO draws VALUES (?,?,?,?,?)',(run_id,update_id,pool,model_sha,json.dumps(ids)));self.db.commit();return ids
        except BaseException:self.db.rollback();raise

    def counts(self,model_sha,spec_sha=None):
        q=self.db.execute('''SELECT COUNT(DISTINCT s.position_key),COUNT(DISTINCT CASE WHEN s.role='root' THEN s.position_key END),
            COUNT(DISTINCT CASE WHEN s.role='descendant' THEN s.case_id END)
            FROM states s JOIN decisions d ON s.case_id=d.case_id JOIN teachers t ON t.label_id=d.label_id
            WHERE s.split='train' AND s.quarantine=0 AND d.model_sha=? AND d.active=1 AND t.stable=1 AND (? IS NULL OR t.spec_sha=?)''',(model_sha,spec_sha,spec_sha)).fetchone()
        primary=self.db.execute('''SELECT COUNT(DISTINCT s.position_key) FROM states s JOIN decisions d ON s.case_id=d.case_id
            JOIN teachers t ON t.label_id=d.label_id WHERE s.split='train' AND s.role='root' AND s.quarantine=0 AND d.model_sha=?
            AND d.active=1 AND t.stable=1 AND (? IS NULL OR t.spec_sha=?)
            AND (EXISTS(SELECT 1 FROM json_each(d.decision,'$.repair_heads') WHERE value IN ('policy','value'))
                 OR json_extract(d.decision,'$.repair_heads') IS NULL)''',(model_sha,spec_sha,spec_sha)).fetchone()[0]
        general=self.db.execute('''SELECT COUNT(DISTINCT s.position_key) FROM states s JOIN decisions d ON s.case_id=d.case_id JOIN teachers t ON t.label_id=d.label_id
            JOIN root_pools rp ON rp.case_id=s.case_id AND rp.pool='general' WHERE s.split='train' AND s.role='root' AND s.quarantine=0 AND d.model_sha=? AND t.stable=1''',(model_sha,)).fetchone()[0]
        frontier=self.db.execute('''SELECT COUNT(DISTINCT s.position_key) FROM states s JOIN decisions d ON s.case_id=d.case_id JOIN teachers t ON t.label_id=d.label_id
            WHERE s.split='train' AND s.role='descendant' AND s.quarantine=0 AND d.model_sha=? AND t.stable=1''',(model_sha,)).fetchone()[0]
        return dict(active_unique_mistake_positions=q[0],verified_unique_mistake_roots=q[1],verified_policy_value_mistake_roots=primary,
                    Q_only_mistake_roots=q[1]-primary,active_descendant_states=q[2],qualified_general_replay_roots=general,qualified_frontier_positions=frontier)

    def close(self):self.db.commit();self.db.close()
