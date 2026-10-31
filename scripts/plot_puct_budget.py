"""Export local PUCT time-budget measurements as a standalone figure."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=Path(a.root)
    pilot=json.loads((r/'pilot/result.json').read_text())['summary'];data=json.loads((r/'analysis.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    counts=[512,1024,2048,3072,4096]
    for batch in [32,64,128,256]:axes[0].plot(counts,[pilot[f'n{n}-b{batch}']['median_seconds'] for n in counts],marker='o',label=f'batch {batch}')
    axes[0].axhline(2,color='#dc2626',ls='--');axes[0].set(title='Pilot:4 roots, eager inference',xlabel='Additional simulations',ylabel='Median seconds / move');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    ns=['fixed-1792','fixed-2048','cached-2048'];labels=['1792\neager','2048\neager','2048\ncached'];s=data['summary'];x=np.arange(3)
    axes[1].bar(x-.18,[s[n]['median_seconds'] for n in ns],.36,label='median',color='#2563eb');axes[1].bar(x+.18,[s[n]['p95_seconds'] for n in ns],.36,label='P95',color='#93c5fd')
    axes[1].set_xticks(x,labels);axes[1].axhline(2,color='#dc2626',ls='--');axes[1].set(title='New32 roots:fixed-count latency',ylabel='Seconds / move');axes[1].legend();axes[1].grid(axis='y',alpha=.2)
    f=data['final_validation']['summary'];names=['guard-2s','optimized-2s'];vals=[f[n]['simulations_median'] for n in names]
    axes[2].bar([0,1],vals,color=['#64748b','#16a34a']);axes[2].set_xticks([0,1],['Guard, no cache','Guard + cache']);axes[2].set(title='New16 roots:same2-second budget',ylabel='Median completed simulations')
    for i,v in enumerate(vals):axes[2].text(i,v+30,f'{v:.0f}',ha='center')
    axes[2].set_ylim(0,max(vals)*1.16);axes[2].grid(axis='y',alpha=.2)
    fig.suptitle('RTX4060Laptop / frozen155M model / one game slot; simulation counts are not independent neural observations',fontsize=11)
    fig.savefig(r/'latency-budget.png',dpi=180)

if __name__=='__main__':main()
