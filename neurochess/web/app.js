const $ = id => document.getElementById(id);
let paused=false, data=null, busy=false;
const colors=['#19664c','#b47834','#637fbb','#92769d','#db6959','#247d9a','#aa4778','#747528'];
const fmt=(x,n=3)=>Number.isFinite(x)?x.toFixed(n):'—';
const txt=(id,value)=>$(id).textContent=value;
const phaseNames={copying_data:'复制数据分片',setup_evaluator:'准备评测引擎',preparing:'读取远程棋谱并转换分片',prepared:'数据准备完成',auditing:'检查数据质量',prepare_failed:'数据准备失败',failed:'流程失败',training:'训练中',validating:'验证中',saving:'保存检查点',evaluating:'对局评测中',complete:'训练完成',paused:'已按计划暂停',stopped:'训练提前退出'};
function pipelineLabel(stage){
 const names={'stage-a':'基础续训','sf1000-evaluation':'最终评测：SF 1000 nodes，无搜索与搜索各 64 局',complete:'训练和最终评测已完成',setup:'恢复数据与权重'};
 if(names[stage])return names[stage];
 const n=stage?.match(/(\d+)$/)?.[1];
 if(stage?.startsWith('filtering-old-hard-labels-'))return `准备第 ${n} 批难例：剔除旧标签并校验数据`;
 if(stage?.startsWith('waiting-for-hard-wave-'))return `等待第 ${n} 批难例交付`;
 if(stage?.startsWith('hard-wave-'))return `第 ${n} 批难例训练`;
 return stage||'等待流水线状态';
}

function series(name,rows,key,color){return {name,color,points:rows.map(r=>({x:r.step,y:key(r)})).filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y))};}
function visible(rows){const n=Number($('range').value);return n?rows.slice(-n):rows;}
function ema(points){let v;return points.map(p=>({x:p.x,y:v=v===undefined?p.y:.2*p.y+.8*v}));}
function chart(id,items,{percent=false,scientific=false,smooth=false}={}){
 const host=$(id);host.replaceChildren();
 const available=items.filter(s=>s.points.length);
 if(!available.length){const el=document.createElement('div');el.className='empty';el.textContent='暂无记录 · 数据产生后自动显示';host.append(el);return;}
 const w=Math.max(300,host.clientWidth),h=host.clientHeight,pad={l:54,r:18,t:16,b:32};
 const all=available.flatMap(s=>s.points);let xmin=Math.min(...all.map(p=>p.x)),xmax=Math.max(...all.map(p=>p.x));
 let ymin=percent?0:Math.min(...all.map(p=>p.y)),ymax=percent?1:Math.max(...all.map(p=>p.y));
 if(xmin===xmax){xmin-=1;xmax+=1;}if(ymin===ymax){const d=Math.abs(ymin)*.1||.1;ymin-=d;ymax+=d;}
 if(!percent){const d=(ymax-ymin)*.08;ymin-=d;ymax+=d;}
 const X=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),Y=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
 const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.setAttribute('role','img');
 svg.setAttribute('aria-label',host.getAttribute('aria-label')||id);
 function el(tag,attrs,text){const n=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;svg.append(n);return n;}
 for(let i=0;i<=4;i++){const y=ymin+(ymax-ymin)*i/4;el('line',{x1:pad.l,y1:Y(y),x2:w-pad.r,y2:Y(y),stroke:'#eaf0e7','stroke-width':1});el('text',{x:pad.l-9,y:Y(y)+4,fill:'#869187','font-size':10,'text-anchor':'end'},percent?`${Math.round(y*100)}%`:scientific?y.toExponential(1):y.toFixed(2));}
 for(let i=0;i<=4;i++){const x=xmin+(xmax-xmin)*i/4;el('text',{x:X(x),y:h-8,fill:'#869187','font-size':10,'text-anchor':'middle'},Math.round(x).toLocaleString());}
 for(const s of available){const pts=smooth?ema(s.points):s.points;el('path',{d:pts.map((p,i)=>`${i?'L':'M'}${X(p.x)},${Y(p.y)}`).join(' '),fill:'none',stroke:s.color,'stroke-width':2,'stroke-linejoin':'round'});if(pts.length<30)for(const p of pts)el('circle',{cx:X(p.x),cy:Y(p.y),r:2.5,fill:s.color});}
 const cross=el('line',{x1:0,y1:pad.t,x2:0,y2:h-pad.b,stroke:'#8d9c90','stroke-dasharray':'3 3',visibility:'hidden'});
 svg.addEventListener('mousemove',e=>{const rect=svg.getBoundingClientRect();const x=xmin+((e.clientX-rect.left)/rect.width*w-pad.l)/(w-pad.l-pad.r)*(xmax-xmin);let lines=[];
  for(const s of available){const p=s.points.reduce((a,b)=>Math.abs(a.x-x)<Math.abs(b.x-x)?a:b);lines.push(`${s.name} · step ${p.x}: ${percent?(p.y*100).toFixed(1)+'%':scientific?p.y.toExponential(3):p.y.toFixed(4)}`);}
  cross.setAttribute('x1',X(x));cross.setAttribute('x2',X(x));cross.setAttribute('visibility','visible');const tip=$('tooltip');tip.textContent=lines.join('\n')+(smooth?'\n曲线为 EMA；以上为原始值':'');tip.hidden=false;tip.style.left=Math.max(4,Math.min(e.clientX+15,innerWidth-tip.offsetWidth-10))+'px';tip.style.top=Math.max(4,Math.min(e.clientY+15,innerHeight-tip.offsetHeight-10))+'px';
 });svg.addEventListener('mouseleave',()=>{cross.setAttribute('visibility','hidden');$('tooltip').hidden=true;});host.append(svg);
}
function renderDetails(latest,validation,m,v,status,config){
 const missing='未记录', number=(x,n=3)=>Number.isFinite(x)?x.toFixed(n):missing;
 const count=x=>Number.isFinite(x)?x.toLocaleString():missing;
 const stamp=x=>Number.isFinite(x)?new Date(x*1000).toLocaleString():missing;
 const rt=data.runtime||{},ck=data.checkpoints||{},sync=data.persistence||{};
 const best=data.validation.reduce((a,b)=>Number.isFinite(b.val_loss)&&(!a||b.val_loss<a.val_loss)?b:a,null);
 const recent=data.metrics.filter(r=>!rt.started_at||r.timestamp>=rt.started_at).slice(-20),first=recent[0],last=recent.at(-1);
 const stepRate=last&&first&&last.timestamp>first.timestamp?(last.step-first.step)/(last.timestamp-first.timestamp):0;
 const remaining=(config.steps??0)-(status.step??latest?.step??0);
 const eta=status.phase==='complete'?'已完成':status.phase==='training'&&stepRate>0&&data.server_time-(status.updated_at||0)<180?`${(remaining/stepRate/3600).toFixed(1)} 小时`:'—';
 const cards=[
 ['训练 Policy / Value / Q',[latest?.loss?.[1],latest?.loss?.[3],latest?.loss?.[2]].map(x=>number(x)).join(' / '),'未加权分量 · 最新记录'],
 ['验证 Top-1',Number.isFinite(validation?.val_top1)?(validation.val_top1*100).toFixed(2)+'%':missing,`验证 step ${validation?.step??'—'}`],
 ['验证 Q MAE',number(validation?.val_q_mae),`${count(validation?.q_labels)} 个有效候选标签`],
 ['最佳验证 loss',number(best?.val_loss),`当前保留日志中 step ${best?.step??'—'}`],
 ['学习率',Number.isFinite(latest?.lr)?latest.lr.toExponential(3):missing,`Q / Value 权重 ${number(latest?.q_weight)} / ${number(latest?.value_weight)}`],
 ['裁剪前梯度范数',number(latest?.gradient_norm),latest?.optimizer_step_skipped?'本步 FP16 溢出，已跳过更新':`裁剪阈值 1.0 · FP16 scale ${number(latest?.scaler_scale,0)}`],
 ['GPU 已分配 / 预留',[latest?.gpu_allocated_gib,latest?.gpu_reserved_gib].map(x=>number(x,2)).join(' / ')+' GiB',`峰值 ${number(latest?.gpu_peak_gib,2)} / 总显存 ${number(rt.gpu_total_gib,2)} GiB`],
 ['设备 / 精度',`${rt.device??missing} · ${rt.precision??missing}`,`${count(rt.parameters)} 参数`],
 ['实际批量 / Q 标签',`${count(latest?.batch_positions)} / ${count(latest?.q_labels)}`,`配置 ${config.batch_size??'—'} × ${config.accum??'—'} · epoch ${latest?.epoch??'—'}（从 0 计）`],
 ['本次启动已处理局面',count(latest?.session_positions),`训练循环累计 ${Number.isFinite(latest?.session_seconds)?(latest.session_seconds/3600).toFixed(2)+' 小时':missing}`],
 ['预计剩余时间',eta,'最近 20 条记录估算；保存和评测会影响耗时'],
 ['数据规模',`${count(data.dataset?.counts?.train)} / ${count(data.dataset?.counts?.val)}`,'引擎/主数据：训练 / 验证局面'],
 ['当前验证集',count(data.validation_dataset?.counts?.val??data.dataset?.counts?.val),config.validation_data?'沿用扩容前的固定验证集':'主数据验证集'],
 ['累计引擎更新',count(latest?.cumulative_step??latest?.step),rt.origin_step?`接续第 ${count(rt.origin_step)} 步；图表显示本阶段步数`:'当前阶段'],
 ['人类回放',count(data.replay_dataset?.counts?.train),config.replay_data?`每 ${config.replay_every??8} 个微批次约 1 个 · 仅策略`:'未启用回放'],
 ['最新 checkpoint',stamp(ck['last.pt']?.modified_at),ck['last.pt']?`${(ck['last.pt'].bytes/2**30).toFixed(2)} GiB · 可续训`:'尚无 last.pt'],
 ['最佳 checkpoint',stamp(ck['best.pt']?.modified_at),ck['best.pt']?'best.pt · 仅权重':'尚无 best.pt'],
 ['最近成功同步 Drive',stamp(sync.last_sync),sync.destination??'旧启动器未记录同步状态']
 ];
 $('details-grid').replaceChildren(...cards.map(([label,value,note])=>{const card=document.createElement('article');for(const [tag,text]of [['div',label],['strong',value],['small',note]]){const el=document.createElement(tag);el.textContent=text;card.append(el);}return card;}));
 chart('accuracy-chart',[series('验证 Top-1',v,r=>r.val_top1,colors[0])],{percent:true});
 chart('q-chart',[series('验证 Q MAE',v,r=>r.val_q_mae,colors[6])]);
 chart('speed-chart',[series('平均吞吐',m,r=>r.positions_per_second,colors[0])]);
 chart('gradient-chart',[series('梯度范数',m,r=>r.gradient_norm,colors[4])]);
 const body=$('validations');body.replaceChildren();
 for(const r of [...data.validation].slice(-20).reverse()){
   const row=body.insertRow();for(const value of [r.step,number(r.val_loss),number(r.val_policy_loss),number(r.val_q_loss),number(r.val_value_loss),Number.isFinite(r.val_top1)?(100*r.val_top1).toFixed(2)+'%':missing,number(r.val_q_mae),`${count(r.positions)} / ${count(r.q_labels)}`])row.insertCell().textContent=value;
 }
 if(!data.validation.length){const cell=body.insertRow().insertCell();cell.colSpan=8;cell.textContent='尚未运行验证';}
}
function render(){if(!data)return;
 const pipeline=data.pipeline;$('pipeline-progress').hidden=!pipeline;$('follow-control').hidden=!pipeline;
 const m=visible(data.metrics),v=visible(data.validation),latest=data.metrics.at(-1),validation=data.validation.at(-1),status=data.status,config=data.config;
 const s=status.step??latest?.step??0,total=config.steps??0;
 txt('step',`${s.toLocaleString()} / ${Number(total).toLocaleString()}`);$('progress').style.width=(total?Math.min(100,s/total*100):0)+'%';
 txt('phase',phaseNames[status.phase]||'无实时状态');txt('loss',fmt(latest?.loss?.[0]));txt('val',`验证 loss ${fmt(validation?.val_loss)}`);txt('speed',fmt(latest?.positions_per_second,1));txt('lr-label',latest?.lr?.toExponential(2)||'—');
 const age=status.updated_at?Math.max(0,data.server_time-status.updated_at):null;
 const active=['training','evaluating','validating','saving','preparing','auditing'].includes(status.phase);const stale=!status.phase||(active&&(age===null||age>180));
 txt('connection',paused?'已暂停刷新':stale?'日志未更新':'日志已连接');$('dot').className=!paused&&!stale?'online':'';
 txt('notice',stale?`日志超过 3 分钟未更新或没有状态记录；不能确认进程仍在运行。`:`${phaseNames[status.phase]||'等待状态'}${status.phase==='evaluating'?` · 对局 ${status.games_done??0} / ${status.games_total??'—'}`:''} · ${Math.round(age)} 秒前更新${active?' · 训练与棋力评测轮流使用 GPU':''}`);
 const prep=status.data?.source?status.data:null,meta=data.dataset;const dp=$('data-progress');dp.hidden=!prep&&!meta?.counts;
 if(prep)dp.textContent=`${phaseNames[status.phase]||status.phase} · 已读取 ${(prep.compressed_bytes_read/1048576).toFixed(1)} MiB 压缩数据 · 筛选 ${prep.games} 盘 · 生成 ${prep.positions.toLocaleString()} / ${prep.target_positions||'不限'} 局面 · 双方等级分 ≥ ${prep.min_elo} · ${prep.source}${prep.error?' · '+prep.error:''}`;
 else if(meta?.counts)dp.textContent=`实际数据：${meta.counts.train.toLocaleString()} 训练局面 / ${meta.counts.val.toLocaleString()} 验证局面 · 来源 ${meta.provenance?.source||meta.provenance?.profile?.name||meta.provenance?.kind||'已处理分片'}${meta.provenance?.sources?' · '+meta.provenance.sources.map(s=>`${s.name||s.url}: ${s.positions.toLocaleString()} 局面`).join(' / '):''}`;
 if(status.note)txt('notice',`${phaseNames[status.phase]||status.phase} · ${status.note}`);
 if(pipeline){
  const label=pipelineLabel(pipeline.stage),age=Math.max(0,data.server_time-(pipeline.updated_at||0));
  txt('pipeline-progress',`整轮进度 ${pipeline.completed_steps.toLocaleString()} / ${Number(pipeline.total_steps).toLocaleString()} 步 · ${label} · ${Math.round(age)} 秒前更新`);
  if(status.phase==='complete'&&pipeline.stage!=='complete')txt('phase','本阶段完成，整轮仍在进行');
  if($('follow').checked&&!pipeline.active_run&&pipeline.stage!=='complete')txt('notice',`${label}。当前保留上一阶段的曲线，下一阶段开始后自动切换。`);
  if(pipeline.error?.error){txt('notice',`流水线出错：${pipeline.error.error}`);txt('connection','流水线出错');$('dot').className='';}
  else if(age>180&&pipeline.stage!=='complete'){txt('notice','流水线状态超过 3 分钟未更新，不能确认任务仍在运行。');txt('connection','日志未更新');$('dot').className='';}
 }

 const names=['总 loss','策略 loss','Q loss','WDL loss'];const lines=names.map((name,i)=>series(name,m,r=>r.loss?.[i],colors[i]));lines.push(series('验证总 loss',v,r=>r.val_loss,colors[4]),series('验证策略 loss',v,r=>r.val_policy_loss,colors[5]),series('验证 Q loss',v,r=>r.val_q_loss,colors[6]),series('验证 Value loss',v,r=>r.val_value_loss,colors[7]));
 $('loss-legend').replaceChildren(...lines.map(s=>{const span=document.createElement('span');span.style.setProperty('--color',s.color);span.textContent=s.name;return span;}));
 chart('loss-chart',lines,{smooth:$('smooth').checked});chart('lr-chart',[series('学习率',m,r=>r.lr,colors[0])],{scientific:true});
 renderDetails(latest,validation,m,v,status,config);
 const arena=data.arena.filter(r=>!r.error),last=arena.at(-1);txt('score',Number.isFinite(last?.paired_score)?(last.paired_score*100).toFixed(1)+'%':'—');
 txt('score-note',last?`${last.independent_pairs??0} 个完整开局对 · 截断 ${last.truncated??0} 局`:'等待周期对局');
 txt('arena-plan',config.arena_every?`每 ${config.arena_every} step · ${config.arena_pairs} 对 · 最多 ${config.arena_max_seconds}s`:'未启用周期评测');
 const groups=new Map();for(const r of arena){const key=JSON.stringify([r.opponent?.name,r.settings?.nodes,r.settings?.openings_sha256,r.settings?.mode,r.settings?.max_plies,r.settings?.max_seconds]);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(r);}
 chart('arena-chart',[...groups.values()].map((rs,i)=>series(`${rs[0].opponent?.name||'Stockfish'} / ${rs[0].settings?.nodes} nodes`,visible(rs),r=>r.paired_score,colors[i%colors.length])),{percent:true});
 const tbody=$('matches');tbody.replaceChildren();
 if(!data.arena.length){const row=tbody.insertRow(),cell=row.insertCell();cell.colSpan=8;cell.textContent='尚无对局结果。启用周期评测后，这里会保存每轮结果和 PGN。';}
 for(const r of [...data.arena].reverse()){
  const row=tbody.insertRow();if(r.error){const cell=row.insertCell();cell.colSpan=8;cell.className='error';cell.textContent=`Step ${r.step} · 评测失败：${r.error}（训练继续）`;continue;}
  const interval=r.paired_score_bootstrap_95;const score=Number.isFinite(r.paired_score)?`${(r.paired_score*100).toFixed(1)}%${interval?' ['+interval.map(x=>(100*x).toFixed(0)).join(', ')+']':''}`:'—';
  const cells=[r.step,`${r.opponent?.name||'Stockfish'} / ${r.settings?.nodes??'—'}`,`${r.wins} / ${r.draws} / ${r.losses}`,score,`${r.completed} / ${r.scheduled??r.completed+r.truncated}`,`${r.truncated} / ${r.unplayed??0}`,`${fmt(r.seconds,1)}s`];
  for(const value of cells)row.insertCell().textContent=value;
  const cell=row.insertCell();if(r.pgn){const a=document.createElement('a');a.href=`api/pgn?run=${encodeURIComponent(data.run)}&file=${encodeURIComponent(r.pgn)}`;a.textContent='下载 PGN';a.download=`step-${r.step}.pgn`;cell.append(a);}
 }
 txt('config',JSON.stringify({training:config,dataset:meta},null,2));txt('source',`来源：${data.run}/metrics.jsonl、validation.jsonl、arena.jsonl。最多读取最近 5000 条指标；断点重跑的相同 step 显示最后一次记录。`);txt('updated',`读取于 ${new Date().toLocaleTimeString()} · 每 5 秒刷新 · 只读`);
}
async function get(url){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json();}
async function refresh(){if(paused||busy)return;busy=true;try{
 const list=await get('api/runs'),select=$('runs'),old=select.value;select.replaceChildren(...list.runs.map(name=>{const op=document.createElement('option');op.value=name;op.textContent=name==='.'?'当前实验':name;return op;}));
 if(list.runs.includes(old))select.value=old;
 else {const requested=new URLSearchParams(location.search).get('run');if(list.runs.includes(requested))select.value=requested;}
 if(!list.runs.length){txt('notice','还没有实验。启动训练后，页面会自动发现日志。');txt('connection','等待实验');$('dot').className='';data=null;return;}
 data=await get(`api/run?run=${encodeURIComponent(select.value)}`);
 const next=data.pipeline?.active_run;
 if($('follow').checked&&next&&next!==select.value&&list.runs.includes(next)){
  select.value=next;data=await get(`api/run?run=${encodeURIComponent(next)}`);
  const url=new URL(location.href);url.searchParams.set('run',next);history.replaceState(null,'',url);
 }
 render();
 }catch(error){txt('notice',`无法读取日志：${error.message}。页面保留上次数据，正在等待重新连接。`);txt('connection','连接断开');$('dot').className='';}finally{busy=false;}}
$('follow').addEventListener('change',()=>{if($('follow').checked)refresh();else render();});
$('runs').addEventListener('change',()=>{$('follow').checked=false;data=null;const was=paused;paused=false;refresh().finally(()=>{paused=was;});});
$('range').addEventListener('change',render);$('smooth').addEventListener('change',render);
$('pause').addEventListener('click',()=>{paused=!paused;txt('pause',paused?'恢复刷新':'暂停刷新');if(paused){txt('connection','已暂停刷新');$('dot').className='';}else refresh();});
window.addEventListener('resize',()=>{if(data)render();});refresh();setInterval(refresh,5000);
