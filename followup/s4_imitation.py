"""Exploratory frozen-imitation comparison on the already inspected S4 bank.

A separate sealed manifest; never modifies S4 or its source studies. No training/tuning.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scripts.evidence_protocol import ROOT, digest, write_json, exclusive
from followup import study

LABEL = 'Exploratory follow-up motivated by observed S4 results; bank already inspected. Not an independent confirmatory test.'
SOURCES = {'s4': 'results/s4-planners', 'bc': 'results/ral-evidence-v2',
           'history_bc': 'results/s3a-history-bc', 'diffusion_policy': 'results/s3b-diffusion-policy'}


def read_checked(path):
    if digest(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError(f'Changed result: {path}')
    return json.loads(path.read_text())


def validate_result(r, bank):
    if r['episode_seeds'] != bank or r['stage'] != 'test' or 'note' in r:
        raise ValueError('Unpaired or truncated results.')
    for key, metric in [('max_coverage', 'mean_max_coverage'), ('success', 'success_rate')]:
        values = np.asarray(r[key], float)
        if len(values) != len(bank) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Invalid episode outcomes.')
        if not np.isclose(values.mean(), r[metric], rtol=0, atol=1e-12):
            raise ValueError('Aggregate does not match raw outcomes.')


def prepare(args):
    s4, sm, src = study.load(SOURCES['s4'])
    files = {}
    def retain(p):
        p = Path(p).resolve(); files[str(p)] = digest(p); return str(p)
    configs = [sm['config']]
    for name in ('history_bc', 'diffusion_policy'):
        _, m, _ = study.load(SOURCES[name]); configs.append(m['config'])
    for c in configs:
        for rel in study.code_hashes(c): retain(ROOT / rel)
    retain(Path(__file__))
    for name, loc in SOURCES.items():
        for fn in ('manifest.json', 'manifest.sha256', 'freeze.json'): retain(Path(loc) / fn)
    items = []
    for name in ('bc', 'history_bc', 'diffusion_policy'):
        seeds = list(range(10, 15 if name == 'diffusion_policy' else 25))
        root = Path(SOURCES[name]).resolve()
        frozen = json.loads((root / 'freeze.json').read_text())
        for seed in seeds:
            ck = root / 'runs' / f'seed-{seed}' / name / 'model.pt'
            sha = digest(ck)
            if sha != frozen['checkpoints'][f'{seed}/{name}']: raise ValueError('Checkpoint differs from source freeze.')
            retain(ck)
            refs = {}
            for method, evaluation in [('oracle', 'oracle_restart'), ('latent_branch', 'latent_branch_restart')]:
                f = s4 / 'runs' / f'seed-{seed}' / method / 'test' / f'{evaluation}.json'
                r = read_checked(f); validate_result(r, sm['control_banks']['test'][str(seed)])
                if r['manifest_sha256'] != digest(s4/'manifest.json'): raise ValueError('S4 manifest mismatch.')
                if r['provenance']['frozen_policy_sha256'] != src[2]['checkpoints'][f'{seed}/wam_branch']:
                    raise ValueError('S4 policy provenance mismatch.')
                refs[evaluation] = retain(f); retain(f.with_suffix('.sha256'))
            items.append({'method': name, 'seed': seed, 'checkpoint': str(ck), 'checkpoint_sha256': sha,
                          'episode_seeds': sm['control_banks']['test'][str(seed)], 'references': refs})
    out = Path(args.study); out.mkdir(parents=True, exist_ok=False)
    m = {'label': LABEL, 'created_unix': time.time(), 'files': files, 'items': items,
         'sources': {k:str(Path(v).resolve()) for k,v in SOURCES.items()},
         'max_steps': sm['config']['max_steps'], 'check_plans_every': sm['config']['check_plans_every'],
         'analysis': 'All oracle/latent restart minus each baseline contrasts for coverage and success. Paired run-level t intervals; descriptive, unadjusted; no confirmatory primary. DP restricted to shared seeds 10-14.'}
    write_json(out/'manifest.json', m)
    (out/'manifest.sha256').write_text(digest(out/'manifest.json')+'\n')
    print(f'Registered {len(items)} frozen evaluations. {LABEL}')


def load(path):
    path = Path(path).resolve(); m = read_checked(path/'manifest.json')
    for f, sha in m['files'].items():
        if digest(f) != sha: raise ValueError(f'Source changed: {f}')
    return path,m


def evaluate(args):
    import torch
    from followup.common import run_episodes
    path,m = load(args.study); item = m['items'][args.index]
    out = path/'evaluations'; out.mkdir(exist_ok=True)
    target = out/f'{item["method"]}-{item["seed"]}.json'
    if target.exists():
        r=read_checked(target); validate_result(r,item['episode_seeds'])
        if r['manifest_sha256'] != digest(path/'manifest.json'): raise ValueError('Result manifest mismatch.')
        print('Existing valid result preserved.'); return
    torch.set_num_threads(args.threads)
    with exclusive(target.with_suffix('.lock')):
        name,seed = item['method'],item['seed']
        if name == 'bc':
            src=study.source({'source_study':m['sources']['bc']})
            agent,tokens,prov=study.build_agent({},src,{'type':'frozen','policy_from':'bc'},seed,'policy',None,args.device)
        else:
            sp,sm,src=study.load(m['sources'][name])
            agent,tokens,prov=study.build_agent(sm,src,sm['config']['methods'][name],seed,'policy',study.run_dir(sp,seed,name),args.device)
        r=run_episodes(agent,tokens,item['episode_seeds'],m['max_steps'],m['check_plans_every'],args.device)
        r.update(stage='test',label=LABEL,manifest_sha256=digest(path/'manifest.json'),
                 checkpoint_sha256=item['checkpoint_sha256'],provenance=prov,method=name,seed=seed)
        validate_result(r,item['episode_seeds'])
        write_json(target,r); target.with_suffix('.sha256').write_text(digest(target)+'\n')
    print(name,seed,r['mean_max_coverage'],r['success_rate'])


def report(args):
    from scipy.stats import t
    path,m=load(args.study); results={}; contrasts={}
    for item in m['items']:
        f=path/'evaluations'/f'{item["method"]}-{item["seed"]}.json'
        r=read_checked(f); validate_result(r,item['episode_seeds'])
        if r['checkpoint_sha256'] != item['checkpoint_sha256'] or r['manifest_sha256'] != digest(path/'manifest.json'):
            raise ValueError('Evaluation provenance mismatch.')
        results[item['method'],item['seed']]=r
    for name in ('bc','history_bc','diffusion_policy'):
        items=[i for i in m['items'] if i['method']==name]
        for ref in ('oracle_restart','latent_branch_restart'):
            for metric in ('mean_max_coverage','success_rate'):
                vals=[]
                for i in items:
                    r=read_checked(Path(i['references'][ref])); validate_result(r,i['episode_seeds'])
                    vals.append(100*(r[metric]-results[name,i['seed']][metric]))
                v=np.asarray(vals); half=t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
                contrasts[f'{ref} minus {name}, {metric}']={'seeds':[i['seed'] for i in items],
                    'per_run':vals,'mean':float(v.mean()),'ci95':[float(v.mean()-half),float(v.mean()+half)]}
    report={'label':LABEL,'manifest_sha256':digest(path/'manifest.json'),'contrasts':contrasts}
    write_json(path/'report.json',report)
    lines=['# S4 frozen imitation follow-up','',LABEL,'',m['analysis'],'',
           '| Contrast | Runs | Difference, points [95% CI] |','|---|---:|---:|']
    for k,r in contrasts.items(): lines.append(f'| {k} | {len(r["seeds"])} | {r["mean"]:.2f} [{r["ci95"][0]:.2f}, {r["ci95"][1]:.2f}] |')
    (path/'report.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    for name in ('prepare','evaluate','report'):
        a=sub.add_parser(name); a.add_argument('--study',required=True)
        if name=='evaluate':
            a.add_argument('--index',type=int,required=True);a.add_argument('--device',default='cpu');a.add_argument('--threads',type=int,default=2)
    args=p.parse_args();globals()[args.command](args)
if __name__=='__main__': main()
