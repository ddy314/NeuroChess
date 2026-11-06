"""Human-facing, score-domain-safe receipt and scientific figure."""
from pathlib import Path
import json,hashlib,time,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/lichess-five-failure-mechanisms-20261105'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()

def main():
    verification={}
    for folder in [OUT,OUT/'review',OUT/'manual-weight-probes']:
        c=read(folder/'completion.json');assert sha(folder/'results.json')==c['report_sha256']
        for name,h in c['files_sha256'].items():assert sha(folder/name)==h,name
        verification[str(folder.relative_to(OUT))]=dict(files=len(c['files_sha256']),report_sha256=c['report_sha256'])
    report=read(OUT/'review/results.json');safe={}
    for index,z in report['cases'].items():
        ref=read(OUT/'review'/f'teacher-{int(index):03d}.json')['final'];pure=all(x['cp'] is not None for x in ref)
        a=dict(z,score_domain='centipawns' if pure else 'mate_ordering_surrogate',all_exact=not any(x['lowerbound'] or x['upperbound'] for x in ref))
        if not pure:
            a['surrogate_shapley_allocation']=a.pop('shapley_allocation');a['shapley_allocation']=None
            a['warning']='Do not interpret delta or surrogate allocation as CP or proof parent branch avoids defeat.'
        safe[index]=a
    node=[a for a in report['visited_node_probes'] if not a['teacher']['lowerbound'] and not a['teacher']['upperbound']]
    trials=read(OUT/'manual-weight-probes/results.json')['candidates']
    summary=dict(completed=True,cases=safe,manual_weight_trials=[dict(alpha=r['alpha'],path=r['path'],sha256=r['sha256'],
        metric_gate_original=r['metric_gate_original']['passed'],metric_gate_local_parent=r['metric_gate_local_parent']['passed'],
        remaining_regressions=r['five_failure_regressions'],retained_control_regressions=r['retained_control_regressions'],
        fresh_validation=r['fresh_validation'],adopted=False) for r in trials],
        selected_node_probe_exact=dict(n=len(node),parent_mae=float(np.mean([r['parent_error'] for r in node])),candidate_mae=float(np.mean([r['candidate_error'] for r in node])),scope='Maximum-drift selected visited FENs,1M finite teacher, not independent population or outcomes'),
        offline_root_rule= {k:read(OUT/'offline-root-selection.json')[k] for k in ['method','random','hard']},
        manual_candidates_adopted=False,optimizer_steps=0,main_training_resumed=False,default_promoted=False,
        qualification_change='147:5M111cp gap became20M77cp, below original absolute100cp error criterion. 420:160M complete-depth25 found candidate mate-20, parent still large negative CP; no CP subtraction or claim parent is nonlosing.',
        manifests=verification)
    (OUT/'final-analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (OUT/'manual-weight-probes/REJECTED.json').write_text(json.dumps(dict(adopted=False,reason='All three candidates still fail diagnostic search checks386 and420, despite original supervised metric gates passing. Not approved for play or automatic resume.',trials=[r['path'] for r in trials]),indent=2)+'\n')
    plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':10})
    fig,(ax,traceax)=plt.subplots(1,2,figsize=(13.6,5.8),gridspec_kw={'width_ratios':[1.4,1],'wspace':.28})
    codes=['000','001','010','011','100','101','110','111'];indices=['147','296','302','386','420']
    vals=np.zeros((5,8));labels=[]
    for i,index in enumerate(indices):
        z=safe[index];base=z['factorial_cells']['000']['regret'];row=[]
        for j,code in enumerate(codes):
            d=z['factorial_cells'][code]['regret']-base
            if z['score_domain']=='centipawns':vals[i,j]=d;row.append(str(d))
            else:vals[i,j]=np.nan if code!='000' else 0;row.append('M' if code!='000' else '0')
        labels.append(row)
    cmap=plt.get_cmap('YlOrRd').copy();cmap.set_bad('#d0d1d1');im=ax.imshow(vals,vmin=0,vmax=600,cmap=cmap,aspect='auto')
    ax.set_xticks(range(8),codes);ax.set_yticks(range(5),[f"{i}号 · {safe[i]['budget']}次" for i in indices]);ax.set_xlabel('P / Q / V 来源：0=冻结父模型，1=4000步模型')
    ax.set_title('同一搜索量下的头部交互：相对全旧输出的额外损失',loc='left',fontweight='bold')
    for i in range(5):
        for j in range(8):ax.text(j,i,labels[i][j],ha='center',va='center',color='white' if np.isfinite(vals[i,j]) and vals[i,j]>350 else '#222222')
    fig.colorbar(im,ax=ax,fraction=.035,pad=.025,label='cp；M表示涉及将杀，不作cp换算')
    r=read(OUT/'matched-budget/302-111-normal.json');samples=r['trace']+[dict(roots=[r['root']])];seen=set();x=[];good=[];bad=[]
    for s in samples:
        root=s['roots'][0];n=root['new_simulations'];leaders={a['move']:a for a in root['leaders']}
        if n in seen or not {'d5b5','a7a6'}<=set(leaders):continue
        seen.add(n);x.append(n);good.append(leaders['d5b5']['visits']);bad.append(leaders['a7a6']['visits'])
    traceax.plot(x,good,color='#28736e',linewidth=2,label='较好着法 d5b5');traceax.plot(x,bad,color='#ad665a',linewidth=2,label='坏着法 a7a6')
    traceax.set_title('302号：较好分支的访问数未及时追上',loc='left',fontweight='bold');traceax.set_xlabel('已完成模拟数');traceax.set_ylabel('根分支累计访问数');traceax.legend(loc='upper left');traceax.spines[['top','right']].set_visible(False)
    end={a['move']:a for a in r['root']['leaders']};traceax.text(.04,.98,f"停止时Q：较好 {end['d5b5']['q']:+.3f} / 坏 {end['a7a6']['q']:+.3f}\n最终仍按累计访问数选择坏着法",transform=traceax.transAxes,va='top',fontsize=9)
    traceax.legend(loc='lower right')
    fig.subplots_adjust(bottom=.25,top=.86,left=.09,right=.97)
    fig.text(.09,.09,'每个局面的所有组合均使用相同模拟数、双网络前向和原搜索算法。属于事后选择的机制对照，不是速度或棋力测量。',fontsize=9,color='#555555')
    fig.text(.09,.045,'参考：depth≥24完整无边界教师。147号新参考差距77cp，已低于原100cp门槛；420号将杀分数单独处理。',fontsize=9,color='#555555')
    fig.savefig(OUT/'mechanism-diagnosis.png',dpi=180)
    (OUT/'delivery-verification.json').write_text(json.dumps(dict(verified_at=time.time(),completed=True,manifests=verification,final_analysis_sha256=sha(OUT/'final-analysis.json'),image_sha256=sha(OUT/'mechanism-diagnosis.png'),weights_unchanged=True,manual_candidates_adopted=False,training_resumed=False),indent=2)+'\n')
    print('complete',OUT/'final-analysis.json')

if __name__=='__main__':main()
