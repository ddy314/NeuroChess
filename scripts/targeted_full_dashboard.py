"""Use the original full dashboard for targeted distillation's actual logs."""
import argparse,json,time
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from neurochess.dashboard import Handler as OriginalHandler,WEB,read_json,snapshot as original_snapshot
from scripts.targeted_dashboard import snapshot


def full_snapshot(root):
 d=snapshot(root);p=d['plan'];r=root;metrics=[]
 scale=root.parent/'targeted-scale'
 if (scale/'joint-training/run.json').exists():
  result=original_snapshot(scale,'joint-training');result.update(screening=d)
  return result
 scale_status=read_json(scale/'status.json');scale_pipeline=read_json(scale/'pipeline-status.json')
 if scale_status and d['status'].get('phase')=='paused':
  counts=scale_status.get('counts',{});phase=scale_pipeline.get('phase',scale_status.get('phase'))
  d['status']['note']=f"2000步试训已保存，实例保留。新增精英候选：训练{counts.get('train',0):,} / 验证{counts.get('val',0):,} / 测试{counts.get('test',0):,}；扩量阶段：{phase}。"
  d['status']['updated_at']=max(scale_status.get('updated_at',0),scale_pipeline.get('updated_at',0))
 for row in d['metrics']:
  m=dict(row);m['loss']=[row['loss'],None,None,row['loss']];m['lr']=p['lr'];m['batch_positions']=p['effective_batch'];m['cumulative_step']=1000+m['step'];m['value_weight']=1.;m['session_positions']=m['step']*p['effective_batch'];metrics.append(m)
 for i,m in enumerate(metrics):
  if i and m['timestamp']>metrics[i-1]['timestamp']:
   m['positions_per_second']=(m['step']-metrics[i-1]['step'])*p['effective_batch']/(m['timestamp']-metrics[i-1]['timestamp'])
 validation=[]
 for v in ([dict(step=0,**d['initial_validation'])] if d['initial_validation'] else [])+d['validations']:
  validation.append(dict(step=v['step'],val_loss=v['ce'],val_value_loss=v['ce'],value_mae=v['mae'],positions=v['n']))
 config=dict(p,steps=p['steps'],batch_size=p['microbatch'],accum=5,validation_data='4096 fixed general validation positions',arena_every=0,trainable=p['trainable'])
 return dict(run='.',config=config,status=d['status'],pipeline=None,screening=d,
  dataset=dict(counts=dict(train=5000000+int(d['audit'].get('counts',{}).get('train',0)),val=4096),provenance=dict(kind='80% broad teacher replay +20% verified mistakes; sampling ratio differs from unique data counts')),
  replay_dataset={},runtime=dict(origin_step=1000,device='NVIDIA L4',precision='BF16 forward / FP32 optimizer'),persistence={},lineage={},
  validation_dataset=dict(counts=dict(val=4096)),metrics=metrics,validation=validation,arena=[],
  checkpoints={name:dict(bytes=(r/name).stat().st_size,modified_at=(r/name).stat().st_mtime) for name in ['last.pt','best.pt'] if (r/name).exists()},server_time=time.time())

PANEL='''<section class="panel"><div class="section-head"><h2>错题筛查与数据准备</h2><span class="subtle">已筛局面包含双方后续4个半回合</span></div><div id="screening-info" class="data-progress">正在读取筛查统计…</div><p class="footnote">确认错题按训练／验证／测试分开；争议标签不进入错题训练。完整训练窗口为2000步，阶段验证不自动晋升模型。</p></section>'''
EXTRA='''
const originalTargetedRender=render;
render=function(){originalTargetedRender();const s=data.screening;if(!s)return;
 const n=x=>Number(x||0).toLocaleString();const b=s.confirmed_by_split;
 txt('screening-info',`筛查 ${n(s.screened)} / ${n(s.goal)} 局面 · 确认错题 ${n(s.confirmed)}（训练 ${n(b.train)} / 验证 ${n(b.val)} / 测试 ${n(b.test)}）· 争议标签 ${n(s.unstable)} · 来源棋局 ${n(s.source_families)}`);
 if(s.status.phase==='screening')txt('phase','筛查和教师复核中');
 if(s.status.phase==='search-evaluation')txt('phase','重复搜索评测中');
 if(s.screened>=s.goal&&s.status.phase==='screening')txt('notice','筛查目标已达到，正在收尾和准备数据审计。');
 chart('q-chart',[series('通用 Value MAE',visible(data.validation),r=>r.value_mae,colors[6])]);
 txt('arena-plan','本轮采用新局面 × 3次重复搜索，无完整对局');txt('score-note',`重复搜索已完成 ${n(s.evaluation_pairs)} / 96 组`);
 txt('notice',(s.status.phase==='training'?`训练 ${n(s.status.step)} / ${n(s.plan.steps)} 步 · `:'')+($('notice').textContent));
 txt('source','来源：当前云端 status.json、metrics.jsonl、validation-*.json、screened.jsonl。未记录的策略/Q损失、准确率和梯度范数保持空白。');
 const dg=$('details-grid');const card=document.createElement('article');
 for(const [tag,text]of [['div','确认训练错题 / 通用回放'],['strong',`${n(b.train)} / 5,000,000`],['small','每批80%通用回放 +20%错题；验证与测试错题不混入训练']]){const e=document.createElement(tag);e.textContent=text;card.append(e)}dg.append(card);
};
'''

class Handler(OriginalHandler):
 def do_GET(self):
  path=urlparse(self.path).path
  if path=='/api/run':
   try:return self.send(json.dumps(full_snapshot(self.root)).encode())
   except (ValueError,KeyError,OSError):return self.send(b'{"error":"Snapshot changing; retry"}',status=503)
  if path=='/api/runs':return self.send(b'{"runs":["."]}')
  if path=='/api/progress':return self.send(json.dumps(snapshot(self.root)).encode())
  if path in ['/','/index.html']:
   html=(WEB/'index.html').read_text().replace('<section class="panel">',PANEL+'<section class="panel">',1)
   html=html.replace('<h2>Q 预测误差</h2>','<h2>通用 Value 预测误差</h2>').replace('仅已标注候选 · MAE','独立通用验证集 · MAE')
   return self.send(html.encode(),'text/html; charset=utf-8')
  if path=='/app.js':return self.send(((WEB/'app.js').read_text()+EXTRA).encode(),'text/javascript; charset=utf-8')
  return super().do_GET()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--port',type=int,default=8765);a=p.parse_args()
 ThreadingHTTPServer(('127.0.0.1',a.port),partial(Handler,root=a.root)).serve_forever()
