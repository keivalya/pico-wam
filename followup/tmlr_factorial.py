"""Frozen-checkpoint proposal/scoring factorial; does not modify earlier study modules."""
import argparse
import json
from pathlib import Path
import numpy as np
from followup import study
from scripts.evidence_protocol import digest, write_json, exclusive


def mixed_agent(m, src, spec, seed, control, out, device):
    from followup.common import GeneralPlanner, PlannerAgent, instruction_tokens, latent_functions, oracle_functions
    model, ck, sha = study.frozen_model(src, spec['policy_from'], seed, device)
    pc = src[1]['protocol']['config']['planner']
    learned = latent_functions(model, pc['horizon'])
    exact = oracle_functions(model, pc['horizon'], device)
    prior = {'learned': learned[0], 'exact': exact[0]}[spec['proposal']]
    scoring = {'learned': learned[1], 'exact': exact[1]}[spec['scoring']]
    return (PlannerAgent(GeneralPlanner(prior, scoring, device, warm_start=False, **pc)),
            instruction_tokens(ck['vocab'], device), {'frozen_policy_sha256': sha})

study.AGENT_BUILDERS['crossed_planner'] = mixed_agent


def common(args):
    """Both scorers see identical union candidates; proposal effects use exact best costs."""
    import torch
    from envs.pusht_env import PushTEnv
    from followup.common import latent_functions, oracle_functions, instruction_tokens, render_tensor, GeneralPlanner
    path, m, src = study.load(args.study); c = m['config']; seed = args.seed
    if seed not in c['training_seeds']: raise ValueError('Unregistered seed')
    target = path / f'common-{seed}.json'
    if target.exists(): raise FileExistsError(target)
    torch.set_num_threads(2)
    with exclusive(path / f'common-{seed}.lock'):
        model, ck, sha = study.frozen_model(src, 'wam_branch', seed, 'cpu')
        pc = src[1]['protocol']['config']['planner']; H = pc['horizon']; n = c['common_candidates_per_prior']
        learned = latent_functions(model, H); exact = oracle_functions(model, H, 'cpu')
        tokens = instruction_tokens(ck['vocab'], 'cpu')
        cost = GeneralPlanner(*learned, 'cpu', **pc).cost
        env = PushTEnv(image_size=64); states = []
        for es in range(c['common_seed_start'], c['common_seed_start'] + c['common_episodes']):
            _, state, _ = env.reset(seed=es)
            for step in range(c['common_steps']):
                if step % c['common_stride'] == 0: states.append(state.copy())
                _, state, _, done, _ = env.step(env.expert_action(noise=.2))
                if done: break
        rows = []
        with torch.no_grad():
            for i, s in enumerate(states):
                image = render_tensor(env, s, 'cpu'); st = torch.tensor(s)[None]
                torch.manual_seed(seed * 10000 + i)
                lp, ep = learned[0](image,tokens,st), exact[0](image,tokens,st)
                rng = torch.Generator().manual_seed(9000000 + seed * 10000 + i)
                eps = torch.randn((n,H,2), generator=rng) * pc['noise']; eps[0] = 0
                a, b = (lp[None]+eps).clamp(-1,1), (ep[None]+eps).clamp(-1,1)
                seqs = torch.cat([a,b]); truth = cost(exact[1](image,tokens,st,seqs))
                pred = cost(learned[1](image,tokens,st,seqs))
                best = truth.min(); chosen = pred.argmin()
                rows.append({'learned_selection_regret':float(truth[chosen]-best),
                             'exact_selection_regret':0.,
                             'exact_minus_learned_proposal_best_cost':float(truth[n:].min()-truth[:n].min()),
                             'selected_optimism':float(truth[chosen]-pred[chosen]),
                             'candidate_sha256':__import__('hashlib').sha256(seqs.numpy().tobytes()).hexdigest()})
        write_json(target, {'manifest_sha256':digest(path/'manifest.json'), 'checkpoint_sha256':sha,
                            'state_count':len(rows), 'rows':rows})
        target.with_suffix('.sha256').write_text(digest(target)+'\n')


def report(args):
    from scipy.stats import t
    path,m,src = study.load(args.study); seeds=m['config']['training_seeds']
    def summary(v):
        v=np.asarray(v); h=t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
        return {'per_run':v.tolist(),'mean':float(v.mean()),'ci95':[float(v.mean()-h),float(v.mean()+h)]}
    rows={}
    for seed in seeds:
        f=path/f'common-{seed}.json'
        if digest(f)!=f.with_suffix('.sha256').read_text().strip(): raise ValueError('Common result changed')
        r=json.loads(f.read_text())
        if r['manifest_sha256']!=digest(path/'manifest.json') or r['checkpoint_sha256']!=src[2]['checkpoints'][f'{seed}/wam_branch']:
            raise ValueError('Common provenance mismatch')
        rows[seed]=r
    study.report(args)
    report_path=path/'report.json'; out=json.loads(report_path.read_text())
    out['common_state_secondary']={key:summary([np.mean([r[key] for r in rows[s]['rows']]) for s in seeds])
          for key in ('learned_selection_regret','exact_minus_learned_proposal_best_cost','selected_optimism')}
    e=out['evaluations']; vals=lambda name:np.array(e[name]['mean_max_coverage']['per_run'])
    out['interaction_secondary']=summary(vals('EE')-vals('EL')-vals('LE')+vals('LL'))
    out['interpretation']='L/E designate learned/exact proposal pipeline first, scoring second. Secondary intervals unadjusted. Exact proposals re-encode rendered states; this is a pipeline intervention, not dynamics alone.'
    write_json(report_path,out)
    with (path/'report.md').open('a') as f:
        f.write('\n## Common-state diagnostics and interaction (secondary)\n\n'+json.dumps({k:out[k] for k in ('common_state_secondary','interaction_secondary','interpretation')},indent=2)+'\n')


def main():
    import sys
    if len(sys.argv)>1 and sys.argv[1] in ('common','final-report'):
        p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--study',required=True);p.add_argument('--seed',type=int)
        a=p.parse_args(); (common if a.stage=='common' else report)(a)
    else: study.main()
if __name__=='__main__': main()
