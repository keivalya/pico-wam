"""Render the verified proposal/scoring study, preserving endpoint labels."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_style import use_style, save, WIDTH, BLUE, ORANGE, TEAL
use_style()
R=Path(__file__).resolve().parents[1]
r=json.loads((R/'analysis/tmlr-factorial-v1.json').read_text())['report']
fig,axes=plt.subplots(1,2,figsize=(WIDTH,2.8),layout='constrained',gridspec_kw={'width_ratios':[1,1.5]})
mat=np.array([[r['evaluations'][a+b]['mean_max_coverage']['mean'] for b in 'LE'] for a in 'LE'])
axes[0].imshow(mat,cmap='Blues',vmin=70,vmax=100)
for i,a in enumerate('LE'):
 for j,b in enumerate('LE'):
  v=r['evaluations'][a+b]['mean_max_coverage'];lo,hi=v['ci95']
  axes[0].text(j,i,f'{v["mean"]:.1f}%\n[{lo:.1f}, {hi:.1f}]',ha='center',va='center',color='black',fontsize=9)
axes[0].set(xticks=[0,1],yticks=[0,1],xticklabels=['Learned','Exact'],yticklabels=['Learned','Exact'],xlabel='Scoring',ylabel='Proposal pipeline',title='(a) Coverage [95% CI]')
keys=['LE minus LL, mean_max_coverage','EE minus EL, mean_max_coverage','EL minus LL, mean_max_coverage','EE minus LE, mean_max_coverage']
labels=['LL → LE*','EL → EE','LL → EL','LE → EE']
for i,k in enumerate(keys):
 v=r['contrasts'][k];m=v['mean'];lo,hi=v['ci95'];axes[1].errorbar(m,3-i,xerr=[[m-lo],[hi-m]],fmt='o',capsize=3,color=TEAL if i<2 else ORANGE)
axes[1].axvline(0,color='gray',ls='--',lw=1);axes[1].set(yticks=[3,2,1,0],yticklabels=labels,xlabel='Paired coverage difference (pp)',title='(b) Substitutions',ylim=(-.5,3.5))
fig.legend(handles=[plt.Line2D([],[],marker='o',color=TEAL,label='Change scoring'),plt.Line2D([],[],marker='o',color=ORANGE,label='Change proposals')],loc='outside lower center',ncol=2)
save(fig, R/'generated/factorial.pdf');plt.close(fig)
# Values stay tied to the report rather than being typed into prose.
mac=[]
def macro(name,x):mac.append('\\newcommand{\\'+name+'}{'+x+'}')
for key,name in [('LE minus LL, mean_max_coverage','FxScoring'),('EL minus LL, mean_max_coverage','FxProposal'),('EE minus LE, mean_max_coverage','FxExactProposal'),('LE minus LL, success_rate','FxSuccess')]:
 v=r['contrasts'][key]
 for suffix,n in [('Mean',v['mean']),('Low',v['ci95'][0]),('High',v['ci95'][1])]:macro(name+suffix,f'{n:.1f}'.replace('-',r'\ensuremath{-}'))
for cell in ['LL','LE','EL','EE']:macro('Fx'+cell+'Coverage',f'{r["evaluations"][cell]["mean_max_coverage"]["mean"]:.1f}')
(R/'generated/factorial-statistics.tex').write_text('\n'.join(mac)+'\n')
