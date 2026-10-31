"""Export source-backed scientific figures for the causal search report."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    specs=[('wdl_mse','Validation WDL MSE'),('rank_accuracy','Validation sibling rank accuracy'),('cp_mae','Validation CP MAE'),('mean_sibling_cp_regret','Validation sibling CP regret')]
    for name,path,color in [('128 / 32 (107k)',Path('output/rank-frontier-20261031/cp-rank'),'#2563eb'),('512 / 64 (460k)',root/'capacity-512','#d97706')]:
        rows=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()];best=min(rows,key=lambda r:r['selection_score'])
        for ax,(key,title) in zip(axes.flat,specs):
            ax.plot([r['step'] for r in rows],[r['validation'][key] for r in rows],label=name,color=color)
            ax.scatter([best['step']],[best['validation'][key]],color=color,s=45,zorder=3)
            ax.set(title=title,xlabel='GPU training steps');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Same game-grouped data and training recipe; dots = selected checkpoint\nOne matched seed; larger capacity does not remove validation error',fontsize=12)
    fig.savefig(root/'capacity-validation.png',dpi=180);plt.close(fig)
    report=json.loads((root/'analysis.json').read_text())['heldout'];names=[n for n in report if n not in ['reuse','legacy']]
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for y,n in enumerate(names):
        r=report[n];m=r['delta_cp_vs_legacy'];lo,hi=r['delta_cp_bootstrap95'];ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt='o',color='#2563eb' if n.startswith('puct') else '#d97706',capsize=3)
    ax.axvline(0,color='#64748b',lw=1);ax.set_yticks(range(len(names)),names);ax.invert_yaxis()
    ax.set(xlabel='Mean CP regret change vs legacy (negative is better)',title='64 new cold-tree roots: paired bootstrap 95% intervals\nFinite candidate-union Stockfish labels; these intervals are not Elo')
    ax.grid(axis='x',alpha=.2);fig.savefig(root/'heldout-ablations.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
