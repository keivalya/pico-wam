"""Prospective state-space replication on Meta-World door opening.

Separate from pico-WAM architecture tests. No renderer, language, or hardware claims.
All data are uninterrupted expert/noisy-expert rollouts; no arbitrary-state training resets.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scripts.evidence_protocol import digest, object_hash, write_json, exclusive

FILE = Path(__file__).resolve()


def make_env(seed):
    import gymnasium as gym
    import metaworld
    from metaworld.policies import ENV_POLICY_MAP
    env=gym.make('Meta-World/MT1',env_name='door-open-v3',seed=int(seed))
    obs,_=env.reset(seed=int(seed))
    return env,ENV_POLICY_MAP['door-open-v3'](),np.asarray(obs,dtype=np.float32)


def summary(v):
    from scipy.stats import t
    v=np.asarray(v,dtype=float); h=t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
    return {'per_run':v.tolist(),'mean':float(v.mean()),'ci95':[float(v.mean()-h),float(v.mean()+h)]}


def seal(path,name,obj):
    f=path/name
    if f.exists():raise FileExistsError(f)
    write_json(f,obj);f.with_suffix(f.suffix+'.sha256').write_text(digest(f)+'\n')


def read_sealed(path,name):
    f=path/name
    if digest(f)!=f.with_suffix(f.suffix+'.sha256').read_text().strip():raise ValueError(f'Changed {name}')
    return json.loads(f.read_text())


def load(path):
    path=Path(path);m=read_sealed(path,'manifest.json')
    if m['code_sha256']!=digest(FILE):raise ValueError('Code changed since registration')
    return path,m['config']


def cost(states):
    # Handle to target plus gripper approach; task-specific surrogate, not reward/success.
    return (np.linalg.norm(states[...,4:7]-states[...,36:39],axis=-1)+
            .1*np.linalg.norm(states[...,0:3]-states[...,4:7],axis=-1)).mean(axis=-1)


def replay(env,actions):
    branch=copy.deepcopy(env)
    # Meta-World pickle restoration reloads the XML and only restores qpos/qvel.
    # Preserve randomized model parameters and the full integration/mocap state.
    branch.unwrapped.model=copy.deepcopy(env.unwrapped.model)
    branch.unwrapped.data=copy.deepcopy(env.unwrapped.data)
    states=[]
    try:
        for a in actions:
            s,_,terminated,truncated,_=branch.step(a)
            states.append(np.asarray(s,dtype=np.float32))
            if terminated or truncated:raise ValueError('Candidate crosses episode boundary')
    finally:branch.close()
    return np.asarray(states)


def prepare(a):
    c=json.loads(Path(a.config).read_text());path=Path(a.study);path.mkdir(parents=True,exist_ok=False)
    if len(set(c['training_seeds']))!=len(c['training_seeds']):raise ValueError('Duplicate seeds')
    import importlib.metadata as md
    seal(path,'manifest.json',{'config':c,'code_sha256':digest(FILE),'created_unix':time.time(),
         'packages':{k:md.version(k) for k in ['torch','numpy','metaworld','mujoco','gymnasium']}})
    print('Registered',path,flush=True)


def collect(a):
    path,c=load(a.study)
    with exclusive(path/'collection.lock'):
        if (path/'data.json').exists():raise FileExistsError('Data already collected')
        # Feasibility gate is execution/replay fidelity, not comparative model performance.
        env,policy,obs=make_env(c['pilot_seed'])
        for _ in range(30):obs,*_=env.step(np.clip(policy.get_action(obs),-1,1))
        actions=np.tile(np.clip(policy.get_action(obs),-1,1),(c['horizon'],1))
        r1=replay(env,actions);r2=replay(env,actions)
        err=float(np.max(np.abs(r1-r2)))
        # Compare clone against actual continuation of the original environment.
        actual=[]
        for action in actions:
            s,*_=env.step(action);actual.append(s)
        err=max(err,float(np.max(np.abs(r1-np.array(actual)))))
        env.close()
        if err>1e-6:raise ValueError(f'Replay fidelity failed: {err}')
        counts={};successes={};H=c['horizon']
        for mode,noise in [('demo',0.),('noise',c['collection_noise'])]:
            states=[];acts=[];futures=[];epids=[];wins=[]
            for ep in range(c['data_episodes']):
                seed=c['data_seed_start']+(0 if mode=='demo' else 10000)+ep
                env,policy,obs=make_env(seed);rng=np.random.default_rng(seed);ss=[obs.copy()];aa=[];success=False
                for step in range(c['max_steps']):
                    action=np.clip(policy.get_action(obs)+rng.normal(0,noise,4),-1,1).astype(np.float32)
                    obs,_,term,trunc,info=env.step(action);obs=np.asarray(obs,dtype=np.float32)
                    aa.append(action);ss.append(obs.copy());success|=bool(info['success'])
                    if term or trunc:break
                env.close();wins.append(success)
                for j in range(0,len(aa)-H+1,2):
                    states.append(ss[j]);acts.append(aa[j:j+H]);futures.append(ss[j+1:j+H+1]);epids.append(seed)
            arr={'state':np.asarray(states,np.float32),'actions':np.asarray(acts,np.float32),'future':np.asarray(futures,np.float32),'episode':np.asarray(epids)}
            if not len(states) or not all(np.isfinite(v).all() for v in arr.values()):raise ValueError('Invalid dataset')
            np.savez_compressed(path/f'{mode}.npz',**arr);counts[mode]=len(states);successes[mode]=float(np.mean(wins))
        seal(path,'data.json',{'hashes':{k:digest(path/f'{k}.npz') for k in ['demo','noise']},'windows':counts,
             'expert_and_noisy_expert_success':successes,'replay_max_error':err})
        print('Data ready',counts,successes,flush=True)


class Model(nn.Module):
    def __init__(self,kind,mean,std,width=256):
        super().__init__();self.kind=kind
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32).clamp_min(.01))
        self.net=nn.Sequential(nn.Linear(39+(4 if kind=='dynamics' else 0),width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,39 if kind=='dynamics' else 4))
    def forward(self,s,a=None):
        n=(s-self.mean)/self.std
        if self.kind=='policy':return self.net(n).tanh()
        pred=s+self.std*self.net(torch.cat([n,a],-1))
        return torch.cat([pred[...,:36],s[...,36:39]],-1)
    def rollout(self,s,actions):
        out=[]
        for k in range(actions.shape[1]):s=self(s,actions[:,k]);out.append(s)
        return torch.stack(out,1)


def arrays(path,c,device):
    meta=read_sealed(path,'data.json')
    for k,h in meta['hashes'].items():
        if digest(path/f'{k}.npz')!=h:raise ValueError('Training data changed')
    return [{k:torch.as_tensor(v,device=device) for k,v in np.load(path/f'{mode}.npz').items()} for mode in ['demo','noise']]


def train(a):
    path,c=load(a.study);seed=a.seed
    if seed not in c['training_seeds']:raise ValueError('Unregistered seed')
    if (path/'freeze.json').exists():raise ValueError('Already frozen')
    out=path/f'seed-{seed}';out.mkdir(exist_ok=True)
    with exclusive(out/'train.lock'):
        if (out/'complete.json').exists():raise FileExistsError('Training complete')
        device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(2)
        demo,noise=arrays(path,c,device);mean=demo['state'].mean(0);std=demo['state'].std(0).clamp_min(.01)
        histories={};begin=time.time()
        for name in ['bc','demo','mixed']:
            torch.manual_seed(seed);np.random.seed(seed)
            model=Model('policy' if name=='bc' else 'dynamics',mean,std,c['width']).to(device)
            initial=object_hash({k:v.detach().cpu().tolist() for k,v in model.state_dict().items()})
            opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate']);rng=np.random.default_rng(seed); noise_rng=np.random.default_rng(seed+100000)
            for step in range(c['updates']):
                idx=rng.integers(len(demo['state']),size=c['batch_size']);b={k:v[idx] for k,v in demo.items() if k!='episode'}
                if name=='mixed':
                    ni=noise_rng.integers(len(noise['state']),size=c['batch_size']//2)
                    for k in b:b[k]=torch.cat([b[k][:c['batch_size']//2],noise[k][ni]])
                if name=='bc':loss=(model(b['state'])-b['actions'][:,0]).square().mean()
                else:loss=((model.rollout(b['state'],b['actions'])-b['future'])/model.std).square().mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                opt.zero_grad();loss.backward();opt.step()
            torch.save({'model':model.state_dict(),'kind':model.kind,'width':c['width']},out/f'{name}.pt')
            histories[name]={'final_training_loss':float(loss.detach()),'initialization_sha256':initial}
        if histories['demo']['initialization_sha256']!=histories['mixed']['initialization_sha256']:raise ValueError('Unmatched dynamics initializations')
        seal(out,'complete.json',{'manifest_sha256':digest(path/'manifest.json'),'checkpoints':{k:digest(out/f'{k}.pt') for k in ['bc','demo','mixed']},'training':histories,'seconds':time.time()-begin})
        print('Trained',seed,flush=True)


def freeze(a):
    path,c=load(a.study);entries={}
    arrays(path,c,'cpu')
    for seed in c['training_seeds']:
        out=path/f'seed-{seed}';r=read_sealed(out,'complete.json')
        if r['manifest_sha256']!=digest(path/'manifest.json'):raise ValueError('Manifest mismatch')
        for name,h in r['checkpoints'].items():
            if digest(out/f'{name}.pt')!=h:raise ValueError('Checkpoint changed')
            entries[f'{seed}/{name}']=h
    seal(path,'freeze.json',{'checkpoints':entries,'timestamp':time.time(),'manifest_sha256':digest(path/'manifest.json')})


def bank(a):
    path,c=load(a.study);read_sealed(path,'freeze.json');states=[];candidates=[];truth=[];future=[]
    with exclusive(path/'bank.lock'):
        if (path/'bank.json').exists():raise FileExistsError('Bank exists')
        for ep in range(c['bank_episodes']):
            seed=c['bank_seed_start']+ep;env,policy,obs=make_env(seed);rng=np.random.default_rng(seed); noise_rng=np.random.default_rng(seed+100000)
            for step in range(c['max_steps']):
                if step in c['bank_steps']:
                    base=np.tile(np.clip(policy.get_action(obs),-1,1),(c['horizon'],1))
                    seq=np.clip(base[None]+rng.normal(0,c['candidate_noise'],(c['candidates'],c['horizon'],4)),-1,1).astype(np.float32);seq[0]=base
                    fut=np.stack([replay(env,q) for q in seq]);states.append(obs.copy());candidates.append(seq);truth.append(cost(fut));future.append(fut)
                action=np.clip(policy.get_action(obs),-1,1);obs,_,term,trunc,_=env.step(action);obs=np.asarray(obs,dtype=np.float32)
                if term or trunc:break
            env.close()
        np.savez_compressed(path/'bank.npz',state=np.array(states),actions=np.array(candidates),cost=np.array(truth),future=np.array(future))
        seal(path,'bank.json',{'sha256':digest(path/'bank.npz'),'states':len(states),'created_unix':time.time()})
        print('Bank ready',len(states),flush=True)


def load_models(path,c,seed):
    frozen=read_sealed(path,'freeze.json');models={}
    for name in ['bc','demo','mixed']:
        p=path/f'seed-{seed}'/f'{name}.pt'
        if digest(p)!=frozen['checkpoints'][f'{seed}/{name}']:raise ValueError('Not frozen checkpoint')
        ck=torch.load(p,map_location='cpu',weights_only=False);sd=ck['model']
        model=Model(ck['kind'],sd['mean'],sd['std'],ck['width']);model.load_state_dict(sd);model.eval();models[name]=model
    return models


def evaluate(a):
    path,c=load(a.study);seed=a.seed
    if seed not in c['training_seeds']:raise ValueError('Unregistered seed')
    out=path/f'seed-{seed}';torch.set_num_threads(2)
    with exclusive(out/'evaluation.lock'),torch.no_grad():
        if (out/'result.json').exists():raise FileExistsError('Final result exists')
        models=load_models(path,c,seed);bmeta=read_sealed(path,'bank.json')
        if digest(path/'bank.npz')!=bmeta['sha256']:raise ValueError('Bank changed')
        b=np.load(path/'bank.npz');scores={};controls={};begin=time.time()
        for name in ['demo','mixed']:
            regrets=[];errors=[];factual=[]
            for s,seq,truth,fut in zip(b['state'],b['actions'],b['cost'],b['future']):
                pred=models[name].rollout(torch.tensor(s).expand(len(seq),-1),torch.tensor(seq)).numpy()
                chosen=cost(pred).argmin();regrets.append(float(truth[chosen]-truth.min()))
                errors.append(float(np.mean((pred[...,4:7]-fut[...,4:7])**2)))
                factual.append(float(np.mean((pred[0,:,4:7]-fut[0,:,4:7])**2)))
            scores[name]={'mean_selection_regret':float(np.mean(regrets)),'handle_mse':float(np.mean(errors)),
                          'unperturbed_sequence_handle_mse':float(np.mean(factual)),'regret_by_state':regrets}
        for name in ['bc','demo','mixed']:
            wins=[];returns=[];steps=[]
            for ep in range(c['test_episodes']):
                es=c['test_seed_start']+ep;env,_,obs=make_env(es);rng=np.random.default_rng(es);win=False;ret=0.
                for step in range(c['max_steps']):
                    s=torch.tensor(obs)[None];base=models['bc'](s)[0].numpy()
                    if name=='bc':action=base
                    else:
                        seq=np.clip(base[None,None]+rng.normal(0,c['candidate_noise'],(c['candidates'],c['horizon'],4)),-1,1).astype(np.float32);seq[0]=base
                        pred=models[name].rollout(s.expand(len(seq),-1),torch.tensor(seq)).numpy();action=seq[cost(pred).argmin(),0]
                    obs,reward,term,trunc,info=env.step(action);obs=np.asarray(obs,dtype=np.float32);ret+=reward;win|=bool(info['success'])
                    if win or term or trunc:break
                env.close();wins.append(win);returns.append(ret);steps.append(step+1)
            controls[name]={'success_rate':float(np.mean(wins)),'success':wins,'return':returns,'lengths':steps,'episode_seeds':list(range(c['test_seed_start'],c['test_seed_start']+c['test_episodes']))}
        seal(out,'result.json',{'manifest_sha256':digest(path/'manifest.json'),'freeze_sha256':digest(path/'freeze.json'),'bank_sha256':bmeta['sha256'],'ranking':scores,'control':controls,'seconds':time.time()-begin})
        print('Evaluated',seed,flush=True)


def report(a):
    path,c=load(a.study);rows=[]
    for seed in c['training_seeds']:
        load_models(path,c,seed);r=read_sealed(path/f'seed-{seed}','result.json')
        if r['manifest_sha256']!=digest(path/'manifest.json') or r['freeze_sha256']!=digest(path/'freeze.json') or r['bank_sha256']!=digest(path/'bank.npz'):raise ValueError('Provenance mismatch')
        rows.append(r)
    primary=summary([r['ranking']['mixed']['mean_selection_regret']-r['ranking']['demo']['mean_selection_regret'] for r in rows])
    secondary={f'{a}_minus_{b}_success_pp':summary([100*(r['control'][a]['success_rate']-r['control'][b]['success_rate']) for r in rows]) for a,b in [('mixed','demo'),('mixed','bc'),('demo','bc')]}
    out={'primary':'mixed minus demo selected-sequence regret on common states (negative favors mixed)','primary_result':primary,'secondary':secondary,
         'scope':'Door-open-v3; state-space MLP and expert/noisy-expert rollouts. Not the shared pico-WAM architecture, human demonstrations, image-only control, or a matched replication of the Push-T factorial. Conditional on one fixed dataset and state bank; intervals over training seeds only.',
         'manifest_sha256':digest(path/'manifest.json')}
    write_json(path/'report.json',out);(path/'report.md').write_text('# Door-opening replication\n\n'+json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','collect','train','freeze','bank','evaluate','report']);p.add_argument('--study',required=True);p.add_argument('--config');p.add_argument('--seed',type=int)
    a=p.parse_args();globals()[a.stage](a)
if __name__=='__main__':main()
