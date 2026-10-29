"""Render descriptive figures from bundled completed reports; no new tests or fits.
Run with numpy, scipy and matplotlib installed. All marks are training-run summaries.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_style import use_style, save, WIDTH, BLUE, ORANGE, TEAL
use_style()
ROOT=Path(__file__).resolve().parents[1]
def read(name): return json.loads((ROOT/'analysis'/f'{name}.json').read_text())
colors=[BLUE,ORANGE]
x=read('ral-evidence-v2'); seeds=x['config']['training_seeds']
f=read('f1-candidate-audit')['report']['conditions']
fig,ax=plt.subplots(1,3,figsize=(WIDTH,2.75),layout='constrained')
for j,cond in enumerate(['wam_demo','wam_branch']):
    # Same horizon and state coordinates in both prediction conditions.
    vals=[]
    for s in seeds:
        p=x['runs'][f'{s}/{cond}']['test/prediction']
        vals.append([np.array(p['factual']['state_nmse_per_dim'])[10,4:8].mean(),np.array(p['counterfactual']['wam_nmse_per_dim'])[9,4:8].mean()])
    vals=np.array(vals)
    for row in vals: ax[0].plot([0,1],row,color=colors[j],alpha=.18,lw=.8)
    ax[0].plot([0,1],vals.mean(0),'o-',color=colors[j],label=['Demos','+ branches'][j],lw=2)
    regrets=f[f'{cond}/common/regret/all']['per_run']
    ax[1].scatter(np.full(len(regrets),j),regrets,s=18,color=colors[j],alpha=.7)
    ax[1].scatter(j,np.mean(regrets),marker='_',s=380,color='black',zorder=4)
    plans=[x['runs'][f'{s}/{cond}']['test/restart']['plan_checks'] for s in seeds]
    ax[2].scatter([p['expected_gain_px'] for p in plans],[p['real_gain_px'] for p in plans],s=25,color=colors[j],label=['Demos','+ branches'][j])
for a,b in zip(f['wam_demo/common/regret/all']['per_run'],f['wam_branch/common/regret/all']['per_run']): ax[1].plot([0,1],[a,b],color='#777777',alpha=.3,lw=.7,zorder=0)
ax[0].set(xticks=[0,1],xticklabels=['Recorded','Random'],ylabel='Block nMSE at 10 steps',title='(a) Prediction')

ax[1].set(xticks=[0,1],xticklabels=['Demos','+ branches'],ylabel='Selection regret (cost units)',title='(b) Candidate ranking',xlim=(-.35,1.35))
ax[2].plot([-5,45],[-5,45],ls='--',color='#999999',lw=1)
ax[2].axhline(0,color='#aaaaaa',lw=.7)
ax[2].set(xlabel='Predicted progress (px)',ylabel='Realized progress (px)',title='(c) Plan check')
fig.legend(*ax[0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
save(fig, ROOT/'generated/mechanism-panels.pdf'); plt.close(fig)
c=read('s4-imitation-followup')['report']['contrasts']
fig,axes=plt.subplots(1,2,figsize=(WIDTH,2.7),layout='constrained',sharey=True)
for ax,metric,title in zip(axes,['mean_max_coverage','success_rate'],['Coverage difference (pp)','Success difference (pp)']):
    for j,planner in enumerate(['latent_branch_restart','oracle_restart']):
        for i,baseline in enumerate(['bc','history_bc','diffusion_policy']):
            r=c[f'{planner} minus {baseline}, {metric}']; m=r['mean']; lo,hi=r['ci95']; y=2-i+(j-.5)*.23
            ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt='o',capsize=3,color=[BLUE,TEAL][j],label=['Learned restart','Exact restart'][j] if i==0 else None)
    ax.axvline(0,color='#777777',lw=1,ls='--'); ax.set(xlabel=title,ylim=(-.5,2.6),yticks=[2,1,0],yticklabels=['BC (15 seeds)','Tuned BC (15 seeds)','Diffusion Policy (5 seeds)'])
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2)
save(fig, ROOT/'generated/imitation-contrasts.pdf'); plt.close(fig)
print('Rendered mechanism and same-start imitation figures from completed reports.')
