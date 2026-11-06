"""Read-only screening and training dashboard for targeted distillation."""
import argparse,json,time
from collections import Counter
from functools import partial
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path


def read(path):
 try:return json.loads(path.read_text())
 except (OSError,ValueError):return {}


def rows(path):
 try:lines=path.read_text().splitlines()
 except OSError:return []
 result=[]
 for line in lines:
  try:result.append(json.loads(line))
  except ValueError:pass
 return result


def snapshot(root):
 status=read(root/'status.json');plan=read(root/'plan.json');data=rows(root/'screened.jsonl')
 wrong=[m for m in data if m.get('repair')]
 confirmed=Counter(m['job']['split'] for m in wrong)
 phase=status.get('phase','setup')
 validations=[]
 for p in sorted(root.glob('validation-*.json'),key=lambda p:int(p.stem.split('-')[-1])):
  validations.append(dict(step=int(p.stem.split('-')[-1]),**read(p)))
 return dict(server_time=time.time(),status=status,plan=plan,screened=len(data),goal=plan.get('screen_states',10000),
  confirmed=len(wrong),confirmed_by_split=confirmed,by_split=Counter(m['job']['split'] for m in data),
  by_depth=Counter(str(m['job']['depth']) for m in data),source_families=len({m['job']['group'] for m in data}),
  unstable=sum(m.get('stable') is False for m in data),
  policy_errors=sum(bool(m['decision']['policy_error']) for m in data),value_errors=sum(bool(m['decision']['value_mistake']) for m in data),
  metrics=rows(root/'metrics.jsonl')[-5000:],validations=validations,initial_validation=read(root/'initial-validation.json'),
  audit=read(root/'data-audit.json'),completion=read(root/'completion.json'),failure=read(root/'failure.json'),
  evaluation_pairs=len(read(root/'search-evaluation.json').get('rows',[]))//2,
  updated_at=status.get('updated_at'),training_started=phase in ['training','search-evaluation','complete'] or bool(validations))

HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NeuroChess · 错题蒸馏进度</title>
<style>body{background:#10161f;color:#e6edf5;font:16px system-ui;margin:0}main{max-width:1150px;margin:40px auto;padding:0 24px}h1{font-size:29px;margin-bottom:8px}p{color:#a9b8c9;line-height:1.7}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px}.card,section{background:#192330;border:1px solid #2b3b4e;border-radius:14px;padding:20px;margin-bottom:18px}.card span{display:block;color:#a9b8c9;font-size:13px}.card b{font-size:31px;display:block;margin-top:8px}.track{background:#304155;border-radius:8px;overflow:hidden;height:12px;margin:16px 0}.bar{background:#57d9af;height:100%;transition:width .4s}h2{font-size:19px;margin:0 0 14px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #304155}th{color:#a9b8c9;font-weight:500}.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}.tag{color:#57d9af}canvas{width:100%;height:220px}#error{color:#ffb78a;white-space:pre-wrap}.small{font-size:13px}button{background:#304155;color:white;border:0;padding:8px 14px;border-radius:8px;cursor:pointer}@media(max-width:700px){.columns{grid-template-columns:1fr}}</style>
<main><h1>NeuroChess · 错题蒸馏</h1><p><span class="tag" id="phase">连接中…</span>　<span id="fresh"></span>　<button onclick="update()">立即刷新</button></p><div id="error"></div>
<div class="cards"><div class="card"><span>已筛查的不同局面</span><b id="screened">—</b></div><div class="card"><span>确认错题 · 全部分组</span><b id="confirmed">—</b></div><div class="card"><span>可用于训练的错题</span><b id="trainwrong">—</b></div><div class="card"><span>教师复核不稳定 · 不进入错题训练</span><b id="unstable">—</b></div><div class="card"><span>已覆盖来源棋局</span><b id="families">—</b></div></div>
<section><h2 id="screenlabel">局面筛查</h2><div class="track"><div class="bar" id="screenbar" style="width:0"></div></div><p id="rootprogress"></p><p class="small">同一局面只计一次。模型与教师走法不同不一定算错；可训练错题需要更深预算复核稳定。每个来源棋局只属于一个分组。</p></section>
<div class="columns"><section><h2>数据划分</h2><table><thead><tr><th>用途</th><th>已筛查</th><th>确认错题</th></tr></thead><tbody id="splits"></tbody></table><p class="small">验证和测试错题留作检查，不混入训练。</p></section><section><h2>双方后续分支</h2><table><thead><tr><th>距原题面</th><th>局面数</th></tr></thead><tbody id="depths"></tbody></table><p class="small">包含教师与模型连续走出的分支；深度单位为半回合。</p></section></div>
<section><h2>混合训练</h2><p id="training">筛查与审计完成后开始训练。</p><div class="track"><div class="bar" id="trainbar" style="width:0"></div></div><p class="small" id="mixture"></p><canvas id="loss"></canvas><p id="lossempty" class="small"></p></section>
<section><h2>验证与重复搜索</h2><p id="eval"></p><table><thead><tr><th>训练步数</th><th>通用 CE</th><th>通用 value MAE</th></tr></thead><tbody id="validation"></tbody></table><p class="small">2000步训练窗口内记录变化；重复搜索与新验证题用于综合评估。没有从loss自动晋升模型。</p></section><p class="small">每5秒读取当前实例文件；筛查统计按完成批次更新。实例释放后连接会中断。</p></main>
<script>
const $=id=>document.getElementById(id),num=n=>Number(n||0).toLocaleString('zh-CN');
const phases={setup:'准备输入',screening:'正在筛查和复核错题',auditing:'正在审计数据',training:'正在混合训练','search-evaluation':'正在重复搜索评测',complete:'流程完成',failed:'流程失败 · 已保存诊断'};
function cells(id,rs){$(id).replaceChildren(...rs.map(r=>{let tr=document.createElement('tr');r.forEach(s=>{let td=document.createElement('td');td.textContent=s;tr.append(td)});return tr}))}
function plot(ms){let c=$('loss'),dpr=devicePixelRatio||1,w=c.clientWidth,h=220;c.width=w*dpr;c.height=h*dpr;let x=c.getContext('2d');x.scale(dpr,dpr);x.clearRect(0,0,w,h);let a=ms.filter(m=>Number.isFinite(Number(m.loss)));$('lossempty').textContent=a.length?'训练目标随步数变化；请结合通用验证与搜索结果判断。':'尚无训练曲线；训练开始后自动显示，不填充模拟数据。';if(!a.length)return;let lo=Math.min(...a.map(m=>+m.loss)),hi=Math.max(...a.map(m=>+m.loss));let max=Math.max(1,...a.map(m=>m.step));x.strokeStyle='#57d9af';x.lineWidth=2;x.beginPath();a.forEach((m,i)=>{let px=55+(w-80)*m.step/max,py=20+170*(1-(+m.loss-lo)/Math.max(hi-lo,.00001));i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();x.fillStyle='#a9b8c9';x.fillText(hi.toFixed(4),2,25);x.fillText(lo.toFixed(4),2,190);x.fillText('0',55,215);x.fillText(max+' 步',w-65,215)}
async function update(){try{let r=await fetch('/api/progress',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);let d=await r.json(),s=d.status,p=d.plan;let age=d.updated_at?Math.max(0,d.server_time-d.updated_at):null;$('phase').textContent=phases[s.phase]||s.phase||'准备输入';$('fresh').textContent=age===null?'等待首批记录':'记录更新于 '+Math.round(age)+' 秒前';$('error').textContent=d.failure.traceback||'';$('screened').textContent=num(d.screened);$('confirmed').textContent=num(d.confirmed);$('trainwrong').textContent=num(d.confirmed_by_split.train);$('unstable').textContent=num(d.unstable);$('families').textContent=num(d.source_families);$('screenlabel').textContent='局面筛查 · '+num(d.screened)+' / '+num(d.goal);$('screenbar').style.width=Math.min(100,100*d.screened/d.goal)+'%';$('rootprogress').textContent=s.roots_completed?'来源种子：'+num(s.roots_completed)+' / '+num(s.roots_total)+'；筛查目标完成 '+(100*d.screened/d.goal).toFixed(1)+'%':'筛查计数保留，当前阶段见上方状态。';cells('splits',[['训练','train'],['验证','val'],['测试','test']].map(([n,k])=>[n,num(d.by_split[k]),num(d.confirmed_by_split[k])]));cells('depths',[0,1,2,3,4].map(k=>[k===0?'原题面':k+' 个半回合',num(d.by_depth[k])]));$('training').textContent=d.training_started?'已执行 '+num(s.step)+' / '+num(p.steps)+' 步':'尚未开始；等待筛查和数据审计通过。';$('trainbar').style.width=Math.min(100,100*(s.step||0)/(p.steps||2000))+'%';$('mixture').textContent='通用回放 '+Math.round(100*p.replay_fraction)+'% · 错题 '+Math.round(100*p.mistake_fraction)+'% · 有效批量 '+num(p.effective_batch)+' · 当前训练 value 编码器与输出头';plot(d.metrics);let vs=d.initial_validation.n?[{step:'起点',...d.initial_validation},...d.validations]:d.validations;cells('validation',vs.map(v=>[v.step,Number(v.ce).toFixed(5),Number(v.mae).toFixed(5)]));$('eval').textContent='重复搜索：'+num(d.evaluation_pairs)+' / 96 组比较（32个新局面 × 3次重复）。';}catch(e){$('error').textContent='连接暂时不可用：'+e.message+'。实例停机后将无法继续刷新。'}}update();setInterval(update,5000);
</script></html>'''

class Handler(BaseHTTPRequestHandler):
 def __init__(self,*args,root,**kwargs):self.root=Path(root);super().__init__(*args,**kwargs)
 def log_message(self,*args):pass
 def do_GET(self):
  if self.path=='/api/runs':body=json.dumps({'runs':['targeted-distill']}).encode();ctype='application/json'
  elif self.path=='/api/progress':
   try:body=json.dumps(snapshot(self.root)).encode();ctype='application/json'
   except (OSError,ValueError,KeyError):self.send_error(503,'Progress snapshot changing; retry');return
  elif self.path=='/':body=HTML.encode();ctype='text/html; charset=utf-8'
  else:self.send_error(404);return
  self.send_response(200);self.send_header('Content-Type',ctype);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--port',type=int,default=8765);a=p.parse_args()
 ThreadingHTTPServer(('127.0.0.1',a.port),partial(Handler,root=a.root)).serve_forever()
