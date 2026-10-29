"""Vector explanation of the evaluated configuration; no model changes."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from paper_style import use_style, save, WIDTH, BLUE, ORANGE, TEAL
use_style()
R=Path(__file__).resolve().parents[1]
fig,ax=plt.subplots(figsize=(WIDTH,6.1))
fig.subplots_adjust(left=.015,right=.985,bottom=.015,top=.99)
ax.set(xlim=(0,100),ylim=(-3,109));ax.axis('off')
def box(x,y,w,h,text,fill='#f1f5f8',edge=BLUE):
 ax.add_patch(FancyBboxPatch((x,y-h/2),w,h,boxstyle='round,pad=.35,rounding_size=.6',facecolor=fill,edgecolor=edge,lw=.8))
 ax.text(x+w/2,y,text,ha='center',va='center',fontsize=9,linespacing=1.18)
def arrow(points,color='#475467',dashed=False):
 for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=.85,ls='--' if dashed else '-')
 ax.annotate('',xy=points[-1],xytext=points[-2],arrowprops=dict(arrowstyle='-|>',color=color,lw=.85,linestyle='--' if dashed else '-',shrinkA=0,shrinkB=1))
def heading(y,text):
 ax.text(1,y,text,fontweight='bold',fontsize=10,va='center')
heading(105,'(a) Encode image, instruction and supplied state')
for y,label in [(96,'Image: 64 × 64\nCNN + 16 keypoints'),(86,'Fixed instruction\nEmbedding + GRU'),(76,'Supplied state: 8-D\nState MLP')]:
 box(1,y,27,8,label);arrow([(28.5,y),(31,y),(31,86),(35,86)])
box(35,86,17,11,'Fusion MLP\n+ LayerNorm')
box(58,86,12,11,'Shared $z_t$\n128-D',fill='#e2f1ef',edge=TEAL)
arrow([(52.5,86),(57.5,86)])
box(77,95,22,10,'Policy $\\pi(z_t)$\n3 × 512 MLP\n8 × 2 actions',fill='#fff2e9',edge=ORANGE)
box(77,77,22,10,'State readout $R$\nLinear: 128 → 8',fill='#e2f1ef',edge=TEAL)
arrow([(70.5,86),(73.5,86),(73.5,95),(76.5,95)])
arrow([(70.5,86),(73.5,86),(73.5,77),(76.5,77)])
ax.text(1,67,'Full physical state is supplied at inference; image decoding is disabled.',fontsize=8.5,color='#505050')
ax.plot([1,99],[64,64],color='#d6dce2',lw=.8)
heading(61,'(b) Predict action-conditioned futures; supervise the shared model')
box(1,51,15,9,'Latent\n$z_t$',fill='#e2f1ef',edge=TEAL)
box(24,51,25,11,'Dynamics $f_\\theta(z_t,a_t)$\nResidual MLP (2 × 256)\n+ LayerNorm')
box(57,51,15,9,'Predicted latent\n$\\hat z_{t+1}$',fill='#e2f1ef',edge=TEAL)
box(80,51,19,9,'Readout $R$\nPredicted state',fill='#e2f1ef',edge=TEAL)
for a,b in [(16.5,23.5),(49.5,56.5),(72.5,79.5)]:arrow([(a,51),(b,51)])
ax.text(36.5,41,'Supplied action $a_t$ (2-D)',ha='center',va='center',fontsize=8.5)
arrow([(36.5,43),(36.5,45.5)])
ax.text(1,35,'Training: expert actions → policy loss; states → current / future readout losses.',fontsize=8.5)
ax.text(1,31,'Future encodings are stop-gradient targets for latent consistency.',fontsize=8.5)
ax.plot([1,99],[27.5,27.5],color='#d6dce2',lw=.8)
heading(24,'(c) Plan with the learned components; execute in the environment')
for x,w,label in [(1,22,'Propose\nPolicy prior + noise\n256 candidates'),(27,22,'Imagine\nRoll dynamics\n12 steps + readout'),(53,22,'Score\nGeometric cost\non predicted states'),(79,20,'Execute\nFirst action of\nlowest-cost plan')]:box(x,13,w,11,label,fill='#f7f8fa',edge='#687787')
for a,b in [(23.5,26.5),(49.5,52.5),(75.5,78.5)]:arrow([(a,13),(b,13)])
arrow([(89,7),(89,3),(12,3),(12,7)])
ax.text(50,3,'Observe again and re-encode',ha='center',va='center',backgroundcolor='white',fontsize=8.5)
ax.text(1,-2,'Simulator replay checks predictions; it is not used to score learned-model plans.',fontsize=8.5,color='#505050')
save(fig,R/'generated/architecture.pdf')
