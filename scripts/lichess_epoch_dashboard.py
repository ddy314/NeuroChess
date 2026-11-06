"""Adapt guarded three-head logs to the existing, read-only training dashboard."""
import argparse
from functools import partial
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import time
from urllib.parse import urlparse

from neurochess.dashboard import Handler as OriginalHandler, WEB, read_json, read_rows


def snapshot(root):
    root = Path(root)
    plan = read_json(root / 'plan.json')
    review = read_json(root / 'progress-review.json')
    gate = read_json(root / 'latest-gate.json')
    status = dict(read_json(root / 'status.json'))
    restore = read_json(root / 'restore-verification.json')
    proof = read_json(root / 'checkpoint-verification.json')
    if proof:
        review['durable_checkpoint'] = proof
    if restore and status.get('phase') == 'preparing':
        status.update(preparing_step=status.get('step'), step=restore['step'],
                      new_positions=restore['new_positions'], replay_positions=restore['replay_positions'],
                      epoch_fraction=restore['new_positions']/9800000)
    if review.get('runtime_state') == 'unavailable':
        status.update(phase='stopped', note='当前无活动实例；以下为 Drive 保存日志。中断原因未确认，单轮训练尚未完成。')
    metrics = []
    for row in read_rows(root / 'metrics.jsonl'):
        m = dict(row)
        policy, q, value, anchor = row['loss']
        m.update(loss=[policy + .5*q + value + anchor, policy, q, value],
                 anchor_loss=anchor, positions_per_second=row['new_positions_per_second'],
                 session_positions=row['new_positions'] + row['replay_positions'],
                 batch_positions=1024, q_weight=.5, value_weight=1.)
        if restore and row['timestamp'] < restore['verified_at']:
            m['positions_per_second'] = None
        metrics.append(m)
    # Keep every split: the standard dashboard's step-only deduplication would
    # collapse three validation populations into one.
    history = sorted({(v['step'],v['split']):v for v in read_rows(root / 'validation.jsonl')}.values(),
                     key=lambda v:(v['step'],v['split']))
    validation = [dict(step=v['step'], positions=v['positions'],
                       **{'val_'+key: v[key] for key in
                          ['loss', 'policy_loss', 'q_loss', 'value_loss', 'top1', 'q_mae']})
                  for v in history if v['split'] == 'new']
    runtime = read_json(root / 'runtime.json')
    runtime['device'] = runtime.get('gpu')
    if restore:
        runtime['started_at'] = restore['verified_at']
    search = review.get('search', {})
    search_files = sorted(root.glob('search-gate-*.json'))
    if search_files:
        d = json.loads(search_files[-1].read_text())
        search = dict(step=d['step'], passed=d['passed'], roots=len(d['rows']),
                      mean_regret_delta_cp=d['mean_regret_delta_cp'],
                      introduced_blunders=d['introduced_blunders'],
                      changed_choices=sum(v['move']!=v['baseline_move'] for v in d['rows']))
    return dict(run='.', config=dict(plan, steps=12761, batch_size=64, accum=16),
                status=status, dataset=dict(counts=dict(train=9800000, val=100000),
                provenance=dict(kind='Lichess depth ≥24 / knodes ≥1000 · 已审计 1000 万局面')),
                replay_dataset={}, runtime=runtime, persistence={},
                validation_dataset=dict(counts=dict(val=8192)), checkpoints={},
                lineage=read_json(root / 'lineage.json'), metrics=metrics,
                validation=validation, validation_splits=history, arena=[],
                gate=gate, review=review, restore=restore, search_screen=search, server_time=time.time())


PANEL = '''<section class="panel"><div class="section-head"><h2>验证门控与恢复状态</h2><span id="epoch-comparison-label" class="subtle">固定验证样本</span></div>
<p id="epoch-summary"></p><p id="epoch-eta" class="footnote"></p>
<div class="table-scroll"><table><thead><tr><th>验证集 / 局面数</th><th>Policy CE ↓</th><th>Top-1 ↑</th><th>Q MAE ↓</th><th>Value CE ↓</th><th>Value MAE ↓</th></tr></thead><tbody id="epoch-comparison"></tbody></table></div>
<details><summary>Value 分层与门控明细</summary><div class="table-scroll"><table><thead><tr><th>验证集</th><th>优势 MAE ↓</th><th>均势 MAE ↓</th><th>劣势 MAE ↓</th></tr></thead><tbody id="epoch-strata"></tbody></table></div><pre id="epoch-guards"></pre></details>
<p class="footnote" id="epoch-source"></p><a href="api/evidence?file=progress-review.json">查看状态核验记录</a> · <a href="api/evidence?file=latest-gate.json">查看最新门控</a></section>'''

EXTRA = '''
const originalEpochRender=render;
render=function(){originalEpochRender();const g=data.gate,r=data.review;if(!g?.baseline)return;
 const label={new:'新 Lichess',old_core:'旧核心',old_general:'旧通用'}, names=Object.keys(label);
 const pair=(a,b,p=false)=>p?`${(a*100).toFixed(2)}% → ${(b*100).toFixed(2)}% (${((b-a)*100).toFixed(3)} pp)`:`${a.toFixed(6)} → ${b.toFixed(6)} (${b-a>=0?'+':''}${(b-a).toFixed(6)})`;
 const stamp=t=>new Date(t*1000).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false});
 const live=r.runtime_state==='live',search=data.search_screen,phase=data.status.phase;
 if(!live){txt('connection','已保存快照 · 无活动实例');$('dot').className='';txt('phase','实例已不可用 · 单轮训练未完成');txt('notice',`最后保存日志：${stamp(data.status.updated_at)}（北京时间），第 ${data.status.step} 步。当前查询无活动实例；没有正常完成或停机记录，中断原因未确认。`)}
 else if(phase==='preparing'&&data.server_time-data.status.updated_at<180){txt('phase','恢复训练数据与隔离检查');txt('notice',`已挂载 Drive，从第 100 步恢复。数据分片准备 ${data.status.prepared_shards??0} / ${data.status.total_shards??880}；完成核验后继续更新。`)}
 else if(phase==='guard-stopped'){txt('phase','门控未通过，已停止更新');txt('notice',`第 ${data.status.step} 步未通过门控；保存并评估已通过检查的最佳模型。`)}
 else if(phase==='search-regression')txt('phase','固定搜索回归检查');
 txt('epoch-comparison-label',`固定验证样本 · 初始模型 → 第 ${g.step} 步`);
 txt('epoch-summary',`100 步 pilot 已通过，三个头与共享层均有参数更新。最新第 ${g.step} 步：${Object.values(g.checks).filter(Boolean).length}/${Object.keys(g.checks).length} 项指标检查通过。搜索检查：第 ${search.step??100} 步，${search.roots} 个固定局面，${search.passed?'通过':'未通过'}；平均 regret 变化 ${search.mean_regret_delta_cp} cp，走法变化 ${search.changed_choices} 个。`);
 const etaText=live?`本次恢复初步预算约 10–12 小时；实际用时受吞吐、验证和保存影响，门控失败会提前停止。`:`当前没有完成倒计时；若恢复同配置 L4 并通过后续门控，预计另需约 10–12 小时（纯训练约 ${r.conditional_pure_training_hours.toFixed(1)} 小时）。`;
 txt('epoch-eta',`当前日志：${data.status.step} / 12,761 步，${data.status.new_positions.toLocaleString()} / 9,800,000 个新训练局面（${(100*data.status.epoch_fraction).toFixed(2)}%）；另有 ${data.status.replay_positions.toLocaleString()} 个旧回放局面。已校验可恢复检查点：第 ${r.durable_checkpoint.step} 步，${r.durable_checkpoint.new_positions.toLocaleString()} 个新局面。${etaText}`);
 const body=$('epoch-comparison');body.replaceChildren();
 const strata=$('epoch-strata');strata.replaceChildren();
 for(const name of names){const a=g.baseline[name],b=g.current[name],row=body.insertRow();
  for(const v of [`${label[name]} / ${b.positions.toLocaleString()}`,pair(a.policy_loss,b.policy_loss),pair(a.top1,b.top1,true),pair(a.q_mae,b.q_mae),pair(a.value_loss,b.value_loss),pair(a.value_mae,b.value_mae)])row.insertCell().textContent=v;
  const sr=strata.insertRow();for(const v of [label[name],...['winning','neutral','losing'].map(k=>pair(a.value_strata[k].mae,b.value_strata[k].mae))])sr.insertCell().textContent=v;
 }
 txt('epoch-guards',JSON.stringify(g.checks,null,2));
 txt('epoch-source',`验证：新 8,192 / 旧核心 8,192 / 旧通用 4,096。后续每 250 步复查；最新验证为第 ${g.step} 步。旧 Top-1 容许下降 0.5 pp，旧 Q/Value MAE 容许增加 0.005，旧 Value 分层容许增加 0.01。搜索为有限开发集检查，尚不能据此确认实战棋力。`);
 const link=document.createElement('a');link.href=r.source_folder;link.textContent='打开 Drive 原始记录';link.target='_blank';link.rel='noopener';$('epoch-source').append(' ',link);
 $('follow-control').style.display='none';
 const history=data.validation_splits;
 chart('accuracy-chart',names.map((k,i)=>series(label[k],visible(history.filter(x=>x.split===k)),x=>x.top1,colors[i])),{percent:true});
 chart('q-chart',names.map((k,i)=>series(label[k],visible(history.filter(x=>x.split===k)),x=>x.q_mae,colors[i])));
 chart('value-chart',names.map((k,i)=>series(label[k],visible(history.filter(x=>x.split===k)),x=>x.value_mae,colors[i])));
 for(const id of ['accuracy-chart','q-chart','value-chart']){$(id+'-legend')?.remove();const legend=document.createElement('div');legend.id=id+'-legend';legend.className='legend';for(const [i,name]of names.entries()){const item=document.createElement('span');item.style.setProperty('--color',colors[i]);item.textContent=label[name];legend.append(item)}$(id).before(legend)}
 txt('source','来源：metrics.jsonl、validation.jsonl、latest-gate.json、search-gate-*.json、checkpoint-verification.json。训练总 loss = Policy +0.5×Q +Value +旧回放锚定项；验证分布未重新平衡。'+(live?'通过认证 SSH 隧道读取运行实例本地日志，每 5 秒刷新；超过 3 分钟未更新会提示。':'页面为本次下载快照，刷新不会查询云端。'));
 txt('score',search.passed?'通过':'未通过');txt('score-note',`第 ${search.step??100} 步 / ${search.roots} 个固定根局面`);txt('arena-plan','固定 2 秒 PUCT，SF 100 万节点候选集评分');
 txt('updated',live?`运行日志更新于 ${stamp(data.status.updated_at)} · 每 5 秒读取 · 只读`:`快照核验于 ${stamp(r.checked_at)} · 页面只读`);
};
'''


class Handler(OriginalHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/run':
            return self.send(json.dumps(snapshot(self.root)).encode())
        if path == '/api/runs':
            return self.send(b'{"runs":["."]}')
        if path == '/api/evidence':
            from urllib.parse import parse_qs
            name = parse_qs(urlparse(self.path).query).get('file', [''])[0]
            if name not in ['progress-review.json', 'latest-gate.json']:
                return self.send(b'{}', status=404)
            return self.send((self.root / name).read_bytes())
        if path in ['/', '/index.html']:
            html = (WEB / 'index.html').read_text().replace('<title>NeuroChess · 训练监控</title>', '<title>NeuroChess · Lichess 联合训练 R2</title>')
            html = html.replace('<section class="panel">', PANEL+'<section class="panel">', 1)
            html = html.replace('<section class="panel"><div class="section-head"><h2>验证记录</h2>', '<section class="panel"><div class="section-head"><h2>Value MAE</h2></div><div id="value-chart" class="chart compact" aria-label="三个验证集 Value MAE"></div></section><section class="panel"><div class="section-head"><h2>新验证集记录</h2>')
            html = html.replace('最近配对得分率', 'Pilot 搜索门控').replace('平均训练吞吐', '累计新局面吞吐')
            return self.send(html.encode(), 'text/html; charset=utf-8')
        if path == '/app.js':
            return self.send(((WEB / 'app.js').read_text()+EXTRA).encode(), 'text/javascript; charset=utf-8')
        return super().do_GET()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--port', type=int, default=18769)
    args = parser.parse_args()
    ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, root=args.root)).serve_forever()
