"""Plot verified door-task results; intervals condition on the fixed data/start banks."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_style import use_style, save, WIDTH, BLUE, ORANGE, TEAL
use_style()
from scipy.stats import t
R=Path(__file__).resolve().parents[1];x=json.loads((R/'analysis/tmlr-contact-v1.json').read_text());r=x['report.json']['data']
seeds=x['manifest.json']['data']['config']['training_seeds'];assert len(seeds)==10
v=np.array([[x[f'seed-{s}/result.json']['data']['ranking'][name]['mean_selection_regret'] for name in ['demo','mixed']] for s in seeds])
np.testing.assert_allclose(v[:,1]-v[:,0],r['primary_result']['per_run'],rtol=0,atol=1e-15)
fig,axes=plt.subplots(1,2,figsize=(WIDTH,2.8),layout='constrained',gridspec_kw={'width_ratios':[1,1.5]})
for row in v*1e4:axes[0].plot([0,1],row,'o-',color='#777777',alpha=.4,lw=.8,markersize=3)
axes[0].plot([0,1],v.mean(0)*1e4,'o-',color='#176f86',lw=2)
axes[0].set(xticks=[0,1],xticklabels=['Demo','Mixed'],ylabel='Selection regret (cost units × $10^{-4}$)',title='(a) Candidate selection',xlim=(-.25,1.25))
for i,(key,label) in enumerate([('mixed_minus_demo_success_pp','Mixed − demo'),('mixed_minus_bc_success_pp','Mixed − BC'),('demo_minus_bc_success_pp','Demo − BC')]):
 q=r['secondary'][key];m=q['mean'];lo,hi=q['ci95'];axes[1].errorbar(m,2-i,xerr=[[m-lo],[hi-m]],fmt='o',capsize=3,color='#176f86')
axes[1].axvline(0,color='gray',ls='--',lw=1);axes[1].set(yticks=[2,1,0],yticklabels=['Mixed − demo','Mixed − BC','Demo − BC'],ylim=(-.5,2.5),xlim=(-33,22),xlabel='Success difference (pp)',title='(b) Control contrasts')
save(fig, R/'generated/contact.pdf');plt.close(fig)
lines=[]
def macro(n,z):lines.append('\\newcommand{\\'+n+'}{'+f'{z:.2f}'.replace('-',r'\ensuremath{-}')+'}')
for suffix,z in [('Mean',r['primary_result']['mean']),('Low',r['primary_result']['ci95'][0]),('High',r['primary_result']['ci95'][1])]:macro('CtRegret'+suffix,z*1e4)
for name in ['bc','demo','mixed']:
 a=[100*x[f'seed-{s}/result.json']['data']['control'][name]['success_rate'] for s in seeds];macro('Ct'+name.title()+'Success',np.mean(a))
(R/'generated/contact-statistics.tex').write_text('\n'.join(lines)+'\n')
print('Verified seed-level regret differences and rendered contact figure.')
