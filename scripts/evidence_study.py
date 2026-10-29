"""Matched evidence study. See experiments/README.md for the staged workflow."""
import argparse
import json
import os
from pathlib import Path
import platform
from importlib.metadata import version
import signal
import sys
import time
import numpy as np
from scripts.evidence_protocol import (ROOT, CONDITIONS, digest, object_hash, write_json,
    source_hashes, validate_config, control_banks, split_ids, validate_datasets,
    sample_indices, exclusive)


def load_study(path, check_data=True):
    path = Path(path).resolve()
    m = json.loads((path/'manifest.json').read_text())
    if digest(path/'manifest.json') != (path/'manifest.sha256').read_text().strip():
        raise ValueError('Study manifest changed after preparation.')
    if object_hash(m['protocol']) != m['protocol_sha256']:
        raise ValueError('Protocol changed after preparation; prepare a new study.')
    if source_hashes() != m['source_sha256']:
        raise ValueError('Experiment source changed after preparation; prepare a new study.')
    if check_data:
        for data in m['datasets'].values():
            if digest(data['path']) != data['sha256']:
                raise ValueError('Dataset content changed after preparation.')
    return path,m


def run_dir(path, condition, seed):
    return path/'runs'/f'seed-{seed}'/condition


def resource_usage(device):
    """Peak memory of this process: compute accounting only, not a study outcome."""
    import resource
    import torch
    usage={'peak_host_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           'host':platform.node(),'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    if device.type=='cuda':
        usage.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                     peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),
                     cuda_device=torch.cuda.get_device_name(device))
    return usage


def validate_run(m, condition, seed):
    if condition not in m['protocol']['config']['conditions'] or seed not in m['protocol']['config']['training_seeds']:
        raise ValueError('Condition or seed is not part of the frozen protocol.')


def prepare(args):
    c = json.loads(Path(args.config).read_text()); validate_config(c)
    with np.load(args.demos,allow_pickle=True) as f: demo=dict(f)
    with np.load(args.branches,allow_pickle=True) as f: branch=dict(f)
    validate_datasets(demo,branch)
    splits=split_ids(demo['episode_ids'],c)
    # Validate all training windows before reserving an output directory.
    from scripts.train import TrainingDataset
    dataset=TrainingDataset(args.branches,horizon=c['horizon'],action_chunk=c['action_chunk'])
    starts=dataset.starts_in(splits['train']); n=len(demo['states'])
    if not np.any(starts<n) or not np.any(starts>=n):
        raise ValueError('Training split lacks valid demonstration or branch windows.')
    protocol={'config':c,'splits':splits,'control_banks':control_banks(c)}
    out=Path(args.study).resolve(); out.mkdir(parents=True,exist_ok=False)
    manifest={'protocol':protocol,'protocol_sha256':object_hash(protocol),
              'source_sha256':source_hashes(),
              'datasets':{k:{'path':str(Path(v).resolve()),'sha256':digest(v)}
                          for k,v in [('demos',args.demos),('branches',args.branches)]},
              'created_unix':time.time(), 'python':platform.python_version(),
              'training_demo_windows':int(np.sum(starts<n)),
              'training_branch_windows':int(np.sum(starts>=n)),
              'status':'Protocol registered locally; test remains gated until freeze. Not public preregistration.'}
    write_json(out/'manifest.json',manifest)
    (out/'manifest.sha256').write_text(digest(out/'manifest.json')+'\n')
    write_json(out/'PROTOCOL.json',protocol)
    print(f'Prepared {out}: {len(c["training_seeds"])*len(c["conditions"])} training jobs.')
    print('Split counts:',{k:len(v) for k,v in splits.items()})


def checkpoint(path, model, optimizer, data, m, seed, condition, step, history, init_hash):
    import torch
    ck={'model_state_dict':model.state_dict(),'model_config':model.config,
        'vocab':data.vocab,'state_labels':data.state_labels,'env':'pusht','task':'push',
        'train_episodes':m['protocol']['splits']['train'],
        'val_episodes':m['protocol']['splits']['validation'],
        'test_episodes':m['protocol']['splits']['test'],
        'dataset_path':m['datasets']['demos']['path'],'resize_to':64,
        'horizon':m['protocol']['config']['horizon'],'history':history,
        'optimizer':optimizer.state_dict(),'completed_steps':step,
        'condition':condition,'training_seed':seed,'protocol_sha256':m['protocol_sha256'],
        'initialization_sha256':init_hash,
        'trained_dynamics':condition.startswith('wam_'),
        'note':'Only policy outputs are meaningful for bc/current conditions; unused dynamics remain untrained.'}
    temp=path.with_suffix('.tmp'); torch.save(ck,temp); temp.replace(path)


def train(args):
    import hashlib
    import torch
    import torch.nn.functional as F
    from scripts.train import TrainingDataset, get_device
    from models.wam_policy import PicoWAMPolicy
    path,m=load_study(args.study); c=m['protocol']['config']
    validate_run(m,args.condition,args.seed)
    if (path/'freeze.json').exists(): raise ValueError('Frozen studies cannot be retrained.')
    out=run_dir(path,args.condition,args.seed)
    if out.exists() and not args.resume: raise FileExistsError(f'{out}: use --resume for an interrupted run.')
    out.mkdir(parents=True,exist_ok=True)
    with exclusive(out/'training.lock'):
        if (out/'complete.json').exists():
            print('Already complete; checkpoint preserved.'); return
        device=get_device(args.device); torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        data=TrainingDataset(m['datasets']['branches']['path'],horizon=c['horizon'],action_chunk=c['action_chunk'])
        train_ids=m['protocol']['splits']['train']; starts=data.starts_in(train_ids)
        demo_mask=np.isin(data.episode_ids,train_ids) & (data.is_expert>.5)
        demo_starts=starts[data.is_expert[starts]>.5]
        branch_starts=starts[data.is_expert[starts]<.5]
        data.to(device)
        model=PicoWAMPolicy(vocab_size=max(data.vocab.values())+1,state_dim=8,action_dim=2,
            d_model=c['d_model'],action_chunk=c['action_chunk'],policy_hidden=c['policy_hidden'],
            policy_layers=c['policy_layers'],predict_image=False,predict_state=True,
            use_state_input=c['use_state_input']).to(device)
        model.set_state_stats(data.states[demo_mask].mean(0),data.states[demo_mask].std(0))
        h=hashlib.sha256()
        for key,value in model.state_dict().items(): h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
        init_hash=h.hexdigest()
        optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
        first,history=0,[]
        if args.resume and (out/'model.pt').exists():
            ck=torch.load(out/'model.pt',map_location=device,weights_only=False)
            if (ck['protocol_sha256'],ck['condition'],ck['training_seed'],ck['initialization_sha256']) != (m['protocol_sha256'],args.condition,args.seed,init_hash):
                raise ValueError('Resume checkpoint does not match this run.')
            model.load_state_dict(ck['model_state_dict']); optimizer.load_state_dict(ck['optimizer'])
            first=ck['completed_steps']; history=ck['history']
        cuda_device=torch.cuda.get_device_name(device) if device.type=='cuda' else None
        earlier=json.loads((out/'run.json').read_text()).get('invocations',[]) if (out/'run.json').exists() else []
        write_json(out/'run.json',{'condition':args.condition,'seed':args.seed,'protocol_sha256':m['protocol_sha256'],
            'initialization_sha256':init_hash,'device':str(device),'torch':torch.__version__,
            'cuda_device':cuda_device,
            'packages':{name:version(name) for name in ['torch','numpy','pymunk','scipy']},
            'python':platform.python_version(),'cuda_runtime':torch.version.cuda,
            'normalization':'training demonstration transitions only, identical across conditions',
            'state_mean':model.state_mean.cpu().tolist(),'state_std':model.state_std.cpu().tolist(),
            'invocations':earlier+[{'start_update':first,'cuda_device':cuda_device,'host':platform.node(),
                'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'started_unix':time.time()}]})
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        model.train(); begin=time.time()
        for step in range(first,c['steps']):
            # Stateless step seeds make resumption and between-condition ordering explicit.
            torch.manual_seed(args.seed*1000000+step)
            action_idx,aux_idx=sample_indices(demo_starts,branch_starts,c,args.seed,step,args.condition)
            b=next(data.batches(action_idx,c['batch_size'],False))
            z=model.encode_obs(b['image'],b['text_ids'],b['state'])
            target=b['action_chunk'].flatten(1) if c['action_chunk']>1 else b['action']
            action_loss=model.policy_head.loss(target,z); loss=action_loss
            if args.condition!='bc':
                a=next(data.batches(aux_idx,c['batch_size'],False))
                za=model.encode_obs(a['image'],a['text_ids'],a['state'])
                loss=loss+c['state_weight']*F.mse_loss(model.state_readout(za),model.normalize_state(a['state']))
                if args.condition.startswith('wam_'):
                    zp=model.dynamics.rollout(za,a['actions'])
                    loss=loss+c['state_weight']*F.mse_loss(model.state_readout(zp),model.normalize_state(a['future_states']))
                    with torch.no_grad(): ztarget=model.encode_sequence(a['future_images'],a['text_ids'],a['future_states'])
                    loss=loss+c['latent_weight']*F.mse_loss(zp,ztarget)
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at update {step+1}')
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if (step+1)%c['save_every']==0 or step+1==c['steps']:
                history.append({'step':step+1,'loss':float(loss.detach()),'action_loss':float(action_loss.detach())})
                checkpoint(out/'model.pt',model,optimizer,data,m,args.seed,args.condition,step+1,history,init_hash)
                print(args.condition,args.seed,step+1,history[-1],flush=True)
        seconds=time.time()-begin
        write_json(out/'complete.json',{'completed_steps':c['steps'],'checkpoint_sha256':digest(out/'model.pt'),
            'protocol_sha256':m['protocol_sha256'],'initialization_sha256':init_hash,
            'this_invocation_seconds':seconds,'this_invocation_updates':c['steps']-first,
            'this_invocation_updates_per_second':(c['steps']-first)/max(seconds,1e-9),
            'resume_count':len(earlier),**resource_usage(device)})


def verify_checkpoint(out,m):
    complete=json.loads((out/'complete.json').read_text())
    if complete['completed_steps'] != m['protocol']['config']['steps'] or complete['protocol_sha256'] != m['protocol_sha256']:
        raise ValueError('Training is incomplete or belongs to another protocol.')
    if digest(out/'model.pt')!=complete['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after completion.')
    return complete['checkpoint_sha256']


def freeze(args):
    path,m=load_study(args.study); c=m['protocol']['config']; entries={}
    if (path/'freeze.json').exists(): raise FileExistsError('Already frozen.')
    for seed in c['training_seeds']:
        initializations=set()
        for condition in c['conditions']:
            out=run_dir(path,condition,seed); sha=verify_checkpoint(out,m)
            run=json.loads((out/'run.json').read_text()); initializations.add(run['initialization_sha256'])
            dev=json.loads((out/'dev'/'complete.json').read_text())
            if dev['checkpoint_sha256']!=sha or dev['protocol_sha256']!=m['protocol_sha256']:
                raise ValueError('Development evaluation is missing or stale.')
            for name,h in dev['outputs'].items():
                if digest(out/'dev'/name)!=h: raise ValueError('Development results changed.')
            entries[f'{seed}/{condition}']=sha
        if len(initializations)!=1: raise ValueError('Conditions have unmatched initial parameters/normalization.')
    with exclusive(path/'freeze.lock'):
        if (path/'freeze.json').exists(): raise FileExistsError('Already frozen.')
        write_json(path/'freeze.json',{'protocol_sha256':m['protocol_sha256'],'checkpoints':entries,
            'source_sha256':m['source_sha256'],'timestamp':time.time(),
            'primary_estimand':c['primary_estimand'],
            'note':'Local audit boundary. No model selection or retraining after test results are opened.'})
    print('Frozen. Test evaluation is now enabled.')


def dev_precision(args):
    """Predeclared seed-count check from development outputs; uses variance only, never test results."""
    from scipy.stats import t
    path,m=load_study(args.study,check_data=False); c=m['protocol']['config']; seeds=c['training_seeds']
    runs,episode_var=[],[]
    for seed in seeds:
        coverage={}
        for condition in ['wam_branch','wam_demo']:
            out=run_dir(path,condition,seed); sha=verify_checkpoint(out,m)
            for method in ['warm','restart']:
                r=json.loads((out/'dev'/f'{method}.json').read_text())
                if r['checkpoint_sha256']!=sha or r['stage']!='dev' or r['episode_seeds']!=m['protocol']['control_banks']['dev'][str(seed)]:
                    raise ValueError('Development result is stale or unpaired.')
                coverage[condition,method]=np.array(r['max_coverage'])*100
        per_episode=(coverage['wam_branch','restart']-coverage['wam_branch','warm'])-(coverage['wam_demo','restart']-coverage['wam_demo','warm'])
        runs.append(per_episode.mean()); episode_var.append(per_episode.var(ddof=1))
    runs,episode_var=np.array(runs),np.array(episode_var)
    n_dev,n_test=c['dev_episodes'],c['test_episodes']
    between=max(0.0,runs.var(ddof=1)-episode_var.mean()/n_dev)
    per_run_test=between+episode_var.mean()/n_test
    half=lambda n:float(t.ppf(.975,n-1)*np.sqrt(per_run_test/n))
    needed=next((n for n in range(2,c['max_training_seeds']+1) if half(n)<=c['precision_target_pp']),None)
    result={'between_run_variance_pp2':between,'mean_episode_variance_pp2':float(episode_var.mean()),
        'projected_test_half_width_pp':{str(n):half(n) for n in range(len(seeds),c['max_training_seeds']+1)},
        'target_pp':c['precision_target_pp'],'current_seeds':len(seeds),'seeds_needed':needed,
        'decision':'proceed to freeze' if half(len(seeds))<=c['precision_target_pp'] else
                   f'register a new study with {needed or c["max_training_seeds"]} seeds before any test evaluation',
        'development_per_run_interaction_pp':runs.tolist(),'rule':c.get('precision_rule'),
        'note':'Variance components only drive the decision; development effects are recorded for transparency.'}
    write_json(path/'dev_precision.json',result); print(json.dumps(result,indent=2))


def evaluate_control(model,vocab,c,episode_seeds,method,device):
    import torch
    from envs.pusht_env import PushTEnv
    from scripts.plan import LatentPlanner
    from scripts.test import to_tensor
    from utils.tokenizer import SimpleTokenizer
    tok=SimpleTokenizer(vocab=vocab); env=PushTEnv(image_size=64); sim=PushTEnv(image_size=64)
    planner=None if method=='policy' else LatentPlanner(model,warm_start=method=='warm',**c['planner'])
    success_flags=[]; coverages=[]; lengths=[]; checks=[]
    model.eval(); begin=time.time()
    with torch.no_grad():
        for episode_seed in episode_seeds:
            # Match random streams per episode; different episode lengths do not shift the next stream.
            torch.manual_seed(episode_seed)
            image,state,info=env.reset(seed=episode_seed)
            tokens=torch.tensor([tok.encode(info['instruction'])],device=device)
            best=float(info.get('coverage',0)); success=False; queue=[]; episode_checks=[]
            if planner:
                planner.reset(); planner.generator.manual_seed(episode_seed+123456)
            for step in range(c['max_steps']):
                im=to_tensor(image,64,device); st=torch.as_tensor(state,dtype=torch.float32,device=device)[None]
                if planner:
                    action=planner.act(im,tokens,st)[0].cpu().numpy()
                    if step%c['check_plans_every']==0:
                        check=planner.check_last_plan(sim,np.asarray(state))
                        episode_checks.append({'step':step,**check})
                else:
                    if not queue: queue=list(model.act_chunk(im,tokens,st)[0,:max(1,c['action_chunk']//2)].cpu().numpy())
                    action=queue.pop(0)
                image,state,_,done,info=env.step(action)
                best=max(best,float(info['coverage'])); success=success or bool(info.get('success',False))
                if done or success: break
            success_flags.append(success); coverages.append(best); lengths.append(step+1)
            checks.append({'episode_seed':episode_seed,'checks':episode_checks})
    env.close(); sim.close()
    return {'method':method,'episode_seeds':episode_seeds,'success':success_flags,'max_coverage':coverages,
        'lengths':lengths,'success_rate':float(np.mean(success_flags)),'mean_max_coverage':float(np.mean(coverages)),
        'plan_checks_by_episode':checks,'wall_seconds':time.time()-begin,
        'coverage_definition':'maximum over the reset state and every post-action state of the episode',
        'audit_note':'Policy-dependent visited states; these are not a common-state candidate-ranking audit.'}


def evaluate_run(args):
    import torch
    from models.wam_policy import PicoWAMPolicy
    from scripts.train import get_device
    from scripts.evaluate import evaluate, evaluate_counterfactuals
    path,m=load_study(args.study); c=m['protocol']['config']; validate_run(m,args.condition,args.seed)
    out=run_dir(path,args.condition,args.seed); sha=verify_checkpoint(out,m)
    if args.stage=='test':
        frozen=json.loads((path/'freeze.json').read_text())
        if frozen['protocol_sha256']!=m['protocol_sha256'] or frozen['checkpoints'].get(f'{args.seed}/{args.condition}')!=sha:
            raise ValueError('Test checkpoint or protocol does not match freeze.')
    stage=out/args.stage; stage.mkdir(exist_ok=True)
    with exclusive(stage/'evaluation.lock'):
        device=get_device(args.device); torch.set_num_threads(args.threads); begin=time.time()
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        model,ck=PicoWAMPolicy.from_checkpoint(out/'model.pt',device)
        seeds=m['protocol']['control_banks'][args.stage][str(args.seed)]
        outputs={}
        for method in (['policy','warm','restart'] if ck['trained_dynamics'] else ['policy']):
            target=stage/f'{method}.json'
            if target.exists():
                result=json.loads(target.read_text())
                if result.get('checkpoint_sha256')!=sha or result.get('protocol_sha256')!=m['protocol_sha256'] or result['episode_seeds']!=seeds:
                    raise ValueError('Existing evaluation is stale; refusing to overwrite.')
            else:
                result=evaluate_control(model,ck['vocab'],c,seeds,method,device)
                result.update(checkpoint_sha256=sha,protocol_sha256=m['protocol_sha256'],stage=args.stage)
                write_json(target,result)
            outputs[target.name]=digest(target)
        if ck['trained_dynamics']:
            target=stage/'prediction.json'
            episodes=m['protocol']['splits']['validation' if args.stage=='dev' else 'test']
            if target.exists():
                result=json.loads(target.read_text())
                if result.get('checkpoint_sha256')!=sha or result.get('protocol_sha256')!=m['protocol_sha256'] or result['episodes']!=episodes:
                    raise ValueError('Existing prediction evaluation is stale.')
            else:
                prediction_begin=time.time()
                with np.load(m['datasets']['demos']['path'],allow_pickle=True) as f: data=dict(f)
                with torch.no_grad():
                    factual=evaluate(model,data,episodes,c['prediction_horizon'])
                    cf=evaluate_counterfactuals(model,data,episodes,'pusht',c['counterfactual_horizon'],
                        c['counterfactual_plans'],c['counterfactual_stride'],seed=424242 if args.stage=='dev' else 424243)
                # Same fixed normalization permits physical position RMSE from per-coordinate nMSE.
                mse=np.array(cf['wam_nmse_per_dim'])*model.state_std.cpu().numpy()**2
                cf['block_position_rmse_px']=(512*np.sqrt(mse[:,4:6].sum(1))).tolist()
                result={'factual':factual,'counterfactual':cf,'episodes':episodes,'stage':args.stage,
                    'checkpoint_sha256':sha,'protocol_sha256':m['protocol_sha256'],
                    'prediction_status':c['prediction_status'],'wall_seconds':time.time()-prediction_begin}
                write_json(target,result)
            outputs[target.name]=digest(target)
        write_json(stage/'complete.json',{'checkpoint_sha256':sha,'protocol_sha256':m['protocol_sha256'],'outputs':outputs,
            'this_invocation_seconds':time.time()-begin,**resource_usage(device)})
    print('Evaluated',args.stage,args.seed,args.condition)


def report(args):
    from scipy.stats import t
    path,m=load_study(args.study,check_data=False); c=m['protocol']['config']
    frozen=json.loads((path/'freeze.json').read_text())
    if frozen['protocol_sha256']!=m['protocol_sha256']: raise ValueError('Freeze/protocol mismatch.')
    rows={}; predictions={}; train_cost={}; test_cost={}
    for seed in c['training_seeds']:
        for condition in c['conditions']:
            out=run_dir(path,condition,seed); sha=verify_checkpoint(out,m)
            if frozen['checkpoints'][f'{seed}/{condition}']!=sha: raise ValueError('Frozen checkpoint mismatch.')
            complete=json.loads((out/'test/complete.json').read_text())
            if complete['checkpoint_sha256']!=sha or complete['protocol_sha256']!=m['protocol_sha256']:
                raise ValueError('Stale evaluation manifest.')
            for name,h in complete['outputs'].items():
                if digest(out/'test'/name)!=h: raise ValueError('Result changed after evaluation.')
            for method in (['policy','warm','restart'] if condition.startswith('wam_') else ['policy']):
                r=json.loads((out/'test'/f'{method}.json').read_text())
                if r['checkpoint_sha256']!=sha or r['protocol_sha256']!=m['protocol_sha256'] or r['stage']!='test':
                    raise ValueError('Result provenance does not match this test.')
                if r['episode_seeds']!=m['protocol']['control_banks']['test'][str(seed)]: raise ValueError('Unpaired episode bank.')
                if len(r['success'])!=len(r['episode_seeds']) or len(r['max_coverage'])!=len(r['episode_seeds']):
                    raise ValueError('Missing episode outcomes.')
                if not np.isclose(np.mean(r['success']),r['success_rate']) or not np.isclose(np.mean(r['max_coverage']),r['mean_max_coverage']):
                    raise ValueError('Aggregate metrics disagree with episode outcomes.')
                rows[f'{seed}/{condition}/{method}']=r
            if condition.startswith('wam_'):
                r=json.loads((out/'test'/'prediction.json').read_text())
                if r['checkpoint_sha256']!=sha or r['protocol_sha256']!=m['protocol_sha256'] or r['stage']!='test' or r['episodes']!=m['protocol']['splits']['test']:
                    raise ValueError('Prediction provenance does not match this test.')
                predictions[f'{seed}/{condition}']=r
            train_cost[f'{seed}/{condition}']=json.loads((out/'complete.json').read_text())
            test_cost[f'{seed}/{condition}']=complete
    def values(condition,method,metric): return np.array([rows[f'{s}/{condition}/{method}'][metric]*100 for s in c['training_seeds']])
    effects={}
    for metric in ['mean_max_coverage','success_rate']:
        b=values('wam_branch','restart',metric)-values('wam_branch','warm',metric)
        d=values('wam_demo','restart',metric)-values('wam_demo','warm',metric)
        contrasts={'interaction':b-d,'restart_branch_effect':values('wam_branch','restart',metric)-values('wam_demo','restart',metric),
            'future_vs_current_branch_policy':values('wam_branch','policy',metric)-values('current_branch','policy',metric),
            'current_branch_vs_demo_policy':values('current_branch','policy',metric)-values('current_demo','policy',metric),
            'restart_branch_vs_bc':values('wam_branch','restart',metric)-values('bc','policy',metric)}
        for name,v in contrasts.items():
            if len(v)<2: raise ValueError('Need at least two training runs for intervals.')
            se=v.std(ddof=1)/np.sqrt(len(v)); radius=t.ppf(.975,len(v)-1)*se
            effects[f'{metric}/{name}']={'per_run_percentage_points':v.tolist(),'mean':float(v.mean()),
                'sample_sd':float(v.std(ddof=1)),'ci95':[float(v.mean()-radius),float(v.mean()+radius)],
                'primary':metric==c['primary_metric'] and name=='interaction'}
    for r in effects.values(): r['runs_above_zero']=int(np.sum(np.array(r['per_run_percentage_points'])>0))
    def summary(v):
        v=np.asarray(v,dtype=float); radius=t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
        return {'per_run':v.tolist(),'mean':float(v.mean()),'sample_sd':float(v.std(ddof=1)),
                'ci95':[float(v.mean()-radius),float(v.mean()+radius)]}
    # Descriptive outcomes: every condition, method and training run, plus selected-plan checks.
    methods=[(condition,method) for condition in c['conditions']
             for method in (['policy','warm','restart'] if condition.startswith('wam_') else ['policy'])]
    by_condition={}
    for condition,method in methods:
        entry={}
        for metric in ['success_rate','mean_max_coverage']: entry[metric]=summary(values(condition,method,metric))
        entry['mean_length']=summary([np.mean(rows[f'{s}/{condition}/{method}']['lengths']) for s in c['training_seeds']])
        entry['successes']=[int(np.sum(rows[f'{s}/{condition}/{method}']['success'])) for s in c['training_seeds']]
        if method!='policy':
            for key in ['plan_error_px','expected_gain_px','real_gain_px']:
                entry[key]=summary([np.mean([x[key] for e in rows[f'{s}/{condition}/{method}']['plan_checks_by_episode'] for x in e['checks']])
                                    for s in c['training_seeds']])
        by_condition[f'{condition}/{method}']=entry
    # Prediction is measured only for conditions whose dynamics were trained.
    def prediction_metric(condition,name):
        out=[]
        for s in c['training_seeds']:
            r=predictions[f'{s}/{condition}']; f,cf=r['factual'],r['counterfactual']
            k={'factual_nmse_k1':(f['state_nmse'],1),'factual_nmse_k15':(f['state_nmse'],c['prediction_horizon']),
               'persistence_nmse_k15':(f['state_nmse_copy'],c['prediction_horizon']),
               'counterfactual_nmse_k10':(cf['wam_nmse'],c['counterfactual_horizon']-1),
               'zero_action_nmse_k10':(cf['noop_nmse'],c['counterfactual_horizon']-1),
               'counterfactual_block_rmse_px_k10':(cf['block_position_rmse_px'],c['counterfactual_horizon']-1)}[name]
            out.append(k[0][k[1]])
        return np.array(out)
    prediction_names=['factual_nmse_k1','factual_nmse_k15','persistence_nmse_k15','counterfactual_nmse_k10',
                      'zero_action_nmse_k10','counterfactual_block_rmse_px_k10']
    prediction={f'{condition}/{name}':summary(prediction_metric(condition,name))
                for condition in ['wam_demo','wam_branch'] for name in prediction_names}
    for name in prediction_names:
        prediction[f'branch_minus_demo/{name}']=summary(prediction_metric('wam_branch',name)-prediction_metric('wam_demo',name))
    compute={'training_seconds':{k:v['this_invocation_seconds'] for k,v in train_cost.items()},
             'training_updates_per_second':{k:v.get('this_invocation_updates_per_second') for k,v in train_cost.items()},
             'training_peak_cuda_allocated_bytes':{k:v.get('peak_cuda_allocated_bytes') for k,v in train_cost.items()},
             'test_evaluation_seconds':{k:v.get('this_invocation_seconds') for k,v in test_cost.items()},
             'note':'Wall time of the final invocation in each stage; resumed runs also spent time in earlier invocations.'}
    result={'protocol_sha256':m['protocol_sha256'],'training_seeds':c['training_seeds'],'effects':effects,
        'conditions':by_condition,'prediction':prediction,'compute':compute,
        'inference':'Paired training-run Student-t intervals. Secondary intervals are descriptive and unadjusted. Normality is uncertain with few runs.',
        'descriptive_note':'Condition summaries, plan checks and prediction metrics are descriptive; the predeclared contrasts are in effects.',
        'prediction_status':c['prediction_status']}
    write_json(path/'report.json',result)
    fmt=lambda r,scale=1:f'{r["mean"]*scale:.2f} [{r["ci95"][0]*scale:.2f}, {r["ci95"][1]*scale:.2f}]'
    lines=['# Evidence study results','','Primary estimand: '+c['primary_estimand'], '',
        '| Metric / contrast | Mean (pp) | 95% interval | Runs > 0 |','|---|---:|---:|---:|']
    for key,r in effects.items(): lines.append(f'| {key}'+(' (primary)' if r['primary'] else '')+f' | {r["mean"]:.2f} | [{r["ci95"][0]:.2f}, {r["ci95"][1]:.2f}] | {r["runs_above_zero"]}/{len(c["training_seeds"])} |')
    seeds=c['training_seeds']
    lines+=['','## Every condition and training run','',
        '| Condition / method | Coverage, % | Success, % | Successes per run of '+str(c['test_episodes'])+' | Per-run coverage, % |','|---|---:|---:|---:|---|']
    for key,r in by_condition.items():
        lines.append(f'| {key} | {fmt(r["mean_max_coverage"])} | {fmt(r["success_rate"])} | {", ".join(map(str,r["successes"]))} | '
                     +', '.join(f'{x:.1f}' for x in r['mean_max_coverage']['per_run'])+' |')
    lines+=['','## Selected-plan checks along each policy\'s own trajectory','',
        '| Condition / planner | Plan error, px | Expected gain, px | Real gain, px |','|---|---:|---:|---:|']
    for key,r in by_condition.items():
        if 'plan_error_px' in r: lines.append(f'| {key} | {fmt(r["plan_error_px"])} | {fmt(r["expected_gain_px"])} | {fmt(r["real_gain_px"])} |')
    lines+=['','## Prediction on the test demonstrations','','| Metric | Value, mean [95% interval] over runs |','|---|---:|']
    for key,r in prediction.items(): lines.append(f'| {key} | {fmt(r)} |')
    lines+=['',f'Seeds: {seeds}. Intervals use the training run as the unit.','',result['inference'],'',result['descriptive_note'],'',c['prediction_status']]
    (path/'report.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--config',default='experiments/ral_evidence.json'); a.add_argument('--demos',required=True); a.add_argument('--branches',required=True)
    for command in ['train','evaluate','dev-precision','freeze','report']: sub.add_parser(command)
    for command,parser in sub.choices.items(): parser.add_argument('--study',required=True)
    for command in ['train','evaluate']:
        a=sub.choices[command]; a.add_argument('--seed',type=int,required=True); a.add_argument('--condition',choices=CONDITIONS,required=True)
        a.add_argument('--device',default='cpu',choices=['cpu','cuda','mps','auto']); a.add_argument('--threads',type=int,default=2)
    sub.choices['train'].add_argument('--resume',action='store_true')
    sub.choices['evaluate'].add_argument('--stage',choices=['dev','test'],required=True)
    args=p.parse_args()
    # Slurm sends SIGTERM at its time limit; exiting normally releases this run's lock.
    signal.signal(signal.SIGTERM,lambda signum,frame:sys.exit(128+signum))
    {'prepare':prepare,'train':train,'evaluate':evaluate_run,'dev-precision':dev_precision,'freeze':freeze,'report':report}[args.command](args)


if __name__=='__main__': main()
