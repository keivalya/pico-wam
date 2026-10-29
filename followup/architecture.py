"""Architecture follow-up without changes to the frozen study's execution code."""
import hashlib
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from models.wam_policy import PicoWAMPolicy
from scripts.evidence_protocol import digest,sample_indices
from followup import study


class AblatedWAM(PicoWAMPolicy):
    def __init__(self, variant, **kwargs):
        super().__init__(**kwargs)
        if variant not in ('full','state_only','image_only','no_text','no_future','no_latent','detach_aux'):
            raise ValueError(variant)
        self.variant=variant

    def encode_obs(self,image,text_tokens,state):
        if self.variant in ('full','no_future','no_latent','detach_aux'):
            return super().encode_obs(image,text_tokens,state)
        zero=image.new_zeros((len(image),self.config['d_model']))
        img=zero if self.variant=='state_only' else self.img_encoder(image)
        txt=zero  # all three observation variants remove the constant instruction
        st=zero if self.variant=='image_only' else self.state_encoder(self.normalize_state(state))
        return self.fusion(img,txt,st)


def objective(model,b,a,c):
    z=model.encode_obs(b['image'],b['text_ids'],b['state'])
    action=model.policy_head.loss(b['action_chunk'].flatten(1),z)
    za=model.encode_obs(a['image'],a['text_ids'],a['state'])
    if model.variant=='detach_aux':za=za.detach()
    loss=action+c['state_weight']*F.mse_loss(model.state_readout(za),model.normalize_state(a['state']))
    zp=model.dynamics.rollout(za,a['actions'])
    if model.variant!='no_future':
        loss=loss+c['state_weight']*F.mse_loss(model.state_readout(zp),model.normalize_state(a['future_states']))
    if model.variant!='no_latent':
        with torch.no_grad():target=model.encode_sequence(a['future_images'],a['text_ids'],a['future_states'])
        loss=loss+c['latent_weight']*F.mse_loss(zp,target)
    return loss,action


def trainer(path,m,src,spec,seed,out,device):
    from scripts.train import TrainingDataset
    c=src[1]['protocol']['config']; variant=spec['variant']
    for d in src[1]['datasets'].values():
        if digest(d['path'])!=d['sha256']:raise ValueError('Training data changed.')
    data=TrainingDataset(src[1]['datasets']['branches']['path'],horizon=c['horizon'],action_chunk=c['action_chunk'])
    ids=src[1]['protocol']['splits']['train'];starts=data.starts_in(ids)
    mask=np.isin(data.episode_ids,ids)&(data.is_expert>.5)
    ds=starts[data.is_expert[starts]>.5];bs=starts[data.is_expert[starts]<.5]
    torch.manual_seed(seed);np.random.seed(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    data.to(device)
    model=AblatedWAM(variant,vocab_size=max(data.vocab.values())+1,state_dim=8,action_dim=2,
        d_model=c['d_model'],action_chunk=c['action_chunk'],policy_hidden=c['policy_hidden'],policy_layers=c['policy_layers'],
        predict_image=False,predict_state=True,use_state_input=True).to(device)
    model.set_state_stats(data.states[mask].mean(0),data.states[mask].std(0))
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    init=h.hexdigest();opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
    first=0;history=[];ckpt=out/'model.pt'
    if ckpt.exists():
        ck=torch.load(ckpt,map_location=device,weights_only=False)
        if (ck['manifest_sha256'],ck['variant'],ck['seed'],ck['initialization_sha256'])!=(digest(path/'manifest.json'),variant,seed,init):
            raise ValueError('Resume provenance mismatch.')
        model.load_state_dict(ck['model_state_dict']);opt.load_state_dict(ck['optimizer']);first=ck['steps'];history=ck['history']
    begin=time.time();model.train()
    for step in range(first,c['steps']):
        torch.manual_seed(seed*1000000+step)
        ai,xi=sample_indices(ds,bs,c,seed,step,'wam_branch')
        b=next(data.batches(ai,c['batch_size'],False));a=next(data.batches(xi,c['batch_size'],False))
        loss,action=objective(model,b,a,c)
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite step {step}')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if (step+1)%c['save_every']==0 or step+1==c['steps']:
            history.append({'step':step+1,'loss':float(loss.detach()),'action_loss':float(action.detach())})
            ck={'model_config':model.config,'variant':variant,'model_state_dict':model.state_dict(),'optimizer':opt.state_dict(),
                'steps':step+1,'seed':seed,'initialization_sha256':init,'manifest_sha256':digest(path/'manifest.json'),
                'vocab':data.vocab,'history':history}
            torch.save(ck,out/'model.tmp');(out/'model.tmp').replace(ckpt)
            print(variant,seed,history[-1],flush=True)
    return {'steps':c['steps'],'variant':variant,'initialization_sha256':init,'this_invocation_seconds':time.time()-begin,
            'parameter_count':sum(p.numel() for p in model.parameters()),
            'parameters_by_module':{name:sum(p.numel() for p in child.parameters()) for name,child in model.named_children()},
            'note':'Allocated parameters matched; masked encoders inactive. Updates/sample streams matched, FLOPs not matched.'}


def builder(m,src,spec,seed,control,out,device):
    from followup.common import ChunkPolicyAgent,PlannerAgent,instruction_tokens
    from scripts.plan import LatentPlanner
    ck=torch.load(out/'model.pt',map_location=device,weights_only=False)
    if ck['variant']!=spec['variant']:raise ValueError('Wrong ablation checkpoint.')
    model=AblatedWAM(ck['variant'],**ck['model_config']).to(device)
    model.load_state_dict(ck['model_state_dict']);model.eval()
    agent=ChunkPolicyAgent(model) if control=='policy' else PlannerAgent(LatentPlanner(model,warm_start=control=='warm',**src[1]['protocol']['config']['planner']))
    return agent,instruction_tokens(ck['vocab'],device),{'checkpoint_sha256':digest(out/'model.pt')}


study.TRAINERS['architecture']=trainer
study.AGENT_BUILDERS['architecture']=builder
if __name__=='__main__':study.main()
