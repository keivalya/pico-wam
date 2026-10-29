"""Render frozen-state geometry and recorded-pose coverage diagnostics."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_style import use_style, save, WIDTH, BLUE, ORANGE, TEAL
use_style()
from matplotlib.patches import Polygon,Circle,Patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parent))
from envs.pusht_env import from_state,to_world,T_OUTLINE,GOAL_POSE
x=json.loads((R/'analysis/tmlr-example.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(WIDTH,2.65),layout='constrained')
start=np.array(x['start_state']);phys=from_state(start)
def block(ax,s,color,alpha=.4,ls='-',fill=True):
 p=from_state(np.array(s));ax.add_patch(Polygon(to_world(T_OUTLINE,p[4:7]),facecolor=color if fill else 'none',edgecolor=color,alpha=alpha,lw=1.5,ls=ls))
for ax in axes:
 ax.add_patch(Polygon(to_world(T_OUTLINE,GOAL_POSE),color='#64a675',alpha=.35));block(ax,start,'#6d7784',.7)
 ax.set(xlim=(0,512),ylim=(512,0),aspect='equal',xticks=[],yticks=[])
 for spine in ax.spines.values():spine.set_color('#cccccc')
axes[0].add_patch(Circle(phys[:2],15,color='#3d74c5'));axes[0].set_title('(a) Shared start')
axes[0].set_xlabel('Push toward the target')
for ax,cond,title in zip(axes[1:],['wam_demo','wam_branch'],['(b) Demo-trained','(c) Branch-trained']):
 row=x['conditions'][cond]
 for key,color,ls in [('predicted_states','#b95735','--'),('true_states','#176f86','-')]:
  states=np.array(row[key]);block(ax,states[-1],color,.95,ls,fill=False)
  pp=np.array([from_state(z) for z in np.vstack([start,states])]);ax.plot(pp[:,4],pp[:,5],color=color,ls=ls,lw=1)
  ax.plot(pp[:,0],pp[:,1],color=color,ls=ls,lw=.8,alpha=.6);ax.add_patch(Circle(pp[-1,:2],15,fill=False,edgecolor=color,ls=ls))
 ax.set_title(title);ax.set_xlabel(f'Selection regret: {row["regret"]:.2f}')
fig.legend(handles=[Patch(facecolor='#64a675',alpha=.35,label='Target'),Patch(facecolor='#6d7784',label='Start'),Patch(facecolor='none',edgecolor='#b95735',ls='--',label='Predicted endpoint'),Patch(facecolor='none',edgecolor='#176f86',label='Simulator endpoint')],loc='outside lower center',ncol=2,frameon=False,fontsize=8.5)
save(fig, R/'generated/task-example.pdf');plt.close(fig)
c=json.loads((R/'analysis/tmlr-coverage.json').read_text());v=np.array(c['original_episode_max'])*100
fig,ax=plt.subplots(figsize=(WIDTH*.75,2.35),layout='constrained');ax.hist(v,bins=np.arange(0,101,5),color='#176f86',edgecolor='white');ax.axvline(95,color='#b95735',ls='--',label='Success threshold (95%)');ax.set(xlabel='Maximum recorded-pose coverage (%)',ylabel='Demonstrations',xlim=(0,100));ax.spines[['top','right']].set_visible(False);fig.legend(*ax.get_legend_handles_labels(),loc='outside upper center');save(fig, R/'generated/demo-coverage.pdf');plt.close(fig)
