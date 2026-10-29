"""Registered follow-up studies: prepare -> [tune] -> train -> evaluate dev -> freeze -> evaluate test -> report.

The same guarantees as scripts/evidence_study.py: a hashed protocol and code snapshot, outputs
never overwritten, one start bank per training seed shared by every method, a freeze gate that
records checkpoint hashes before the final bank can be evaluated, and paired run-level intervals.
Frozen evidence-study checkpoints are verified against that study's freeze file on every load.

    python -m followup.study prepare  --config experiments/s4_planners.json --study results/s4-planners
    python -m followup.study train    --study S --seed 10 --method state_branch
    python -m followup.study evaluate --study S --seed 10 --evaluation state_branch_restart --stage dev
    python -m followup.study freeze   --study S
    python -m followup.study report   --study S
"""
import argparse
import json
from pathlib import Path
import platform
import signal
import sys
import time

import numpy as np

from scripts.evidence_protocol import ROOT, control_banks, digest, exclusive, object_hash, write_json

SHARED_CODE = ['scripts/plan.py', 'scripts/test.py', 'scripts/train.py', 'scripts/evidence_protocol.py',
               'scripts/evidence_study.py'] + [str(p.relative_to(ROOT)) for d in ('models', 'envs', 'utils')
                                               for p in sorted((ROOT / d).glob('*.py'))]


def code_hashes(c):
    """The study's declared follow-up files plus the shared modules; other follow-ups may change freely."""
    files = sorted(set(c['code_files']) | {'followup/__init__.py', 'followup/study.py', 'followup/common.py'})
    return {p: digest(ROOT / p) for p in files + SHARED_CODE}


# ------------------------------------------------------------------ registration


def validate(c):
    seeds = c['training_seeds']
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Training seeds must be unique.')
    for e in c['evaluations']:
        if e['method'] not in c['methods'] or e['control'] not in ('policy', 'warm', 'restart'):
            raise ValueError(f'Bad evaluation {e}.')
    unused = set(c['methods']) - {e['method'] for e in c['evaluations']}
    if unused:
        raise ValueError(f'Methods without an evaluation: {sorted(unused)}.')
    names = {e['name'] for e in c['evaluations']}
    if len(names) != len(c['evaluations']):
        raise ValueError('Evaluation names must be unique.')
    for a, b, metric in c['contrasts']:
        if a not in names or b not in names or metric not in ('mean_max_coverage', 'success_rate'):
            raise ValueError(f'Bad contrast {a}, {b}, {metric}.')
    if c['primary'] not in [f'{a} minus {b}, {m}' for a, b, m in c['contrasts']]:
        raise ValueError('The primary endpoint must be one of the contrasts.')
    banks = control_banks(c)
    if {s for v in banks['dev'].values() for s in v} & {s for v in banks['test'].values() for s in v}:
        raise ValueError('Development and final banks overlap.')
    for start in c.get('reserved_seed_ranges', []):
        lo, hi = start
        for stage in ('dev', 'test'):
            if any(lo <= s <= hi for v in banks[stage].values() for s in v):
                raise ValueError(f'{stage} bank overlaps the reserved range {start}.')


def source(c_or_m):
    """The frozen evidence study this follow-up builds on; integrity re-checked on every call."""
    from scripts.evidence_study import load_study
    spath, sm = load_study(c_or_m['source_study'], check_data=False)
    frozen = json.loads((spath / 'freeze.json').read_text())
    return spath, sm, frozen


def prepare(args):
    c = json.loads(Path(args.config).read_text()); validate(c)
    spath, sm, frozen = source(c)
    out = Path(args.study).resolve(); out.mkdir(parents=True, exist_ok=False)
    m = {'config': c, 'config_sha256': object_hash(c), 'code_sha256': code_hashes(c),
         'source': {'study': str(spath), 'protocol_sha256': sm['protocol_sha256'], 'freeze_sha256': digest(spath / 'freeze.json')},
         'control_banks': control_banks(c), 'created_unix': time.time(), 'python': platform.python_version(),
         'status': 'Registered before any method of this study was trained or evaluated.'}
    m['source_study'] = c['source_study']
    write_json(out / 'manifest.json', m); (out / 'manifest.sha256').write_text(digest(out / 'manifest.json') + '\n')
    print(f'Prepared {out}: {len(c["training_seeds"])} seeds x {len(c["evaluations"])} evaluations.')


def load(path):
    path = Path(path).resolve()
    m = json.loads((path / 'manifest.json').read_text())
    if digest(path / 'manifest.json') != (path / 'manifest.sha256').read_text().strip():
        raise ValueError('Study manifest changed after preparation.')
    if object_hash(m['config']) != m['config_sha256']:
        raise ValueError('Config changed after preparation.')
    if code_hashes(m['config']) != m['code_sha256']:
        raise ValueError('Follow-up or shared code changed after preparation; prepare a new study.')
    spath, sm, frozen = source(m)
    if sm['protocol_sha256'] != m['source']['protocol_sha256'] or digest(spath / 'freeze.json') != m['source']['freeze_sha256']:
        raise ValueError('Source study changed since registration.')
    return path, m, (spath, sm, frozen)


def run_dir(path, seed, method):
    return path / 'runs' / f'seed-{seed}' / method


# ------------------------------------------------------------------ methods


def frozen_model(src, condition, seed, device):
    from models.wam_policy import PicoWAMPolicy
    spath, sm, frozen = src
    ckpt = spath / 'runs' / f'seed-{seed}' / condition / 'model.pt'
    sha = digest(ckpt)
    if frozen['checkpoints'][f'{seed}/{condition}'] != sha:
        raise ValueError(f'{seed}/{condition} does not match the source freeze.')
    model, ck = PicoWAMPolicy.from_checkpoint(ckpt, device)
    return model, ck, sha


def trainable(spec):
    return spec['type'] in TRAINERS


def train_state_planner(path, m, src, spec, seed, out, device):
    import torch
    from followup.planners import train_state_model
    model, info = train_state_model(src[1], spec['data'], seed, device=device)
    torch.save({'state_dict': model.state_dict(), 'config': model.config, 'info': info}, out / 'model.tmp'); (out / 'model.tmp').replace(out / 'model.pt')
    return info


def build_agent(m, src, spec, seed, control, out, device):
    """Returns (agent, tokens, provenance) for one evaluation."""
    import torch
    from followup.common import (ChunkPolicyAgent, GeneralPlanner, PlannerAgent, instruction_tokens,
                                 latent_functions, oracle_functions, state_model_functions)
    pc = src[1]['protocol']['config']['planner']
    t = spec['type']
    if t in ('frozen', 'state_planner', 'oracle_planner'):
        model, ck, sha = frozen_model(src, spec['policy_from'], seed, device)
        tokens = instruction_tokens(ck['vocab'], device); prov = {'frozen_policy_sha256': sha}
        if control == 'policy':
            if t != 'frozen': raise ValueError('Planner methods need a warm or restart control.')
            return ChunkPolicyAgent(model), tokens, prov
        if t == 'frozen':
            fns = latent_functions(model, pc['horizon'])
        elif t == 'oracle_planner':
            fns = oracle_functions(model, pc['horizon'], device)
        else:
            from followup.planners import StateDynamicsMLP
            ckpt = out / 'model.pt'; saved = torch.load(ckpt, map_location=device, weights_only=False)
            sm = StateDynamicsMLP(**saved['config']).to(device); sm.load_state_dict(saved['state_dict']); sm.eval()
            fns = state_model_functions(model, sm, pc['horizon'], device); prov['checkpoint_sha256'] = digest(ckpt)
        return PlannerAgent(GeneralPlanner(*fns, device, warm_start=control == 'warm', **pc)), tokens, prov
    if t in AGENT_BUILDERS:
        return AGENT_BUILDERS[t](m, src, spec, seed, control, out, device)
    raise ValueError(f'Unknown method type {t}.')


TRAINERS = {'state_planner': train_state_planner}
AGENT_BUILDERS = {}
# Study 3's baselines export their own trainers and agent builders, which are merged here so this
# file never changes when they do. Environments without their extra dependencies simply lack them.
try:
    from followup.baselines import AGENT_BUILDERS as _baseline_agents, TRAINERS as _baseline_trainers
    TRAINERS.update(_baseline_trainers); AGENT_BUILDERS.update(_baseline_agents)
except ImportError as e:  # pragma: no cover
    BASELINES_UNAVAILABLE = repr(e)


# ------------------------------------------------------------------ commands


def train(args):
    import torch
    from scripts.train import get_device
    path, m, src = load(args.study); c = m['config']
    spec = c['methods'][args.method]
    if args.seed not in c['training_seeds'] or not trainable(spec):
        raise ValueError('Seed or method is not trainable in this study.')
    if (path / 'freeze.json').exists():
        raise ValueError('Frozen studies cannot be retrained.')
    out = run_dir(path, args.seed, args.method); out.mkdir(parents=True, exist_ok=True)
    with exclusive(out / 'training.lock'):
        if (out / 'complete.json').exists():
            print('Already complete; preserved.'); return
        device = get_device(args.device); torch.set_num_threads(args.threads)
        info = TRAINERS[spec['type']](path, m, src, spec, args.seed, out, device)
        write_json(out / 'complete.json', {'checkpoint_sha256': digest(out / 'model.pt'), 'manifest_sha256': digest(path / 'manifest.json'),
                   'info': info, 'device': str(device), 'host': platform.node(), 'torch': torch.__version__})
    print('Trained', args.seed, args.method)


def evaluate(args):
    import torch
    from followup.common import run_episodes
    from scripts.train import get_device
    path, m, src = load(args.study); c = m['config']
    e = next(x for x in c['evaluations'] if x['name'] == args.evaluation)
    spec = c['methods'][e['method']]
    if args.stage == 'test' and not (path / 'freeze.json').exists():
        raise ValueError('Freeze the study before evaluating the final bank.')
    out = run_dir(path, args.seed, e['method'])
    if trainable(spec):
        done = json.loads((out / 'complete.json').read_text())
        if digest(out / 'model.pt') != done['checkpoint_sha256']: raise ValueError('Checkpoint changed after training.')
        if args.stage == 'test' and json.loads((path / 'freeze.json').read_text())['checkpoints'][f'{args.seed}/{e["method"]}'] != done['checkpoint_sha256']:
            raise ValueError('Checkpoint does not match the freeze.')
    # Truncated software-check runs never land where freeze or report look.
    stage = out / ('software-check' if args.limit else args.stage); stage.mkdir(parents=True, exist_ok=True)
    target = stage / f'{e["name"]}.json'
    if target.exists() and not args.limit:
        print('Already evaluated; preserved.'); return
    with exclusive(stage / f'{e["name"]}.lock'):
        device = get_device(args.device); torch.set_num_threads(args.threads)
        agent, tokens, prov = build_agent(m, src, spec, args.seed, e['control'], out, device)
        seeds = m['control_banks'][args.stage][str(args.seed)][:args.limit or None]
        r = run_episodes(agent, tokens, seeds, c['max_steps'], c['check_plans_every'], device)
        r.update(evaluation=e, stage=args.stage, manifest_sha256=digest(path / 'manifest.json'), provenance=prov, host=platform.node())
        if args.limit:
            r['note'] = f'Truncated to {args.limit} episodes for a software check; not a study result.'
        write_json(target, r); target.with_suffix('.sha256').write_text(digest(target) + '\n')
    print('Evaluated', args.stage, args.seed, e['name'], f'{r["mean_max_coverage"]:.3f}')


def freeze(args):
    path, m, src = load(args.study); c = m['config']
    if (path / 'freeze.json').exists(): raise FileExistsError('Already frozen.')
    entries, missing = {}, []
    for seed in c['training_seeds']:
        for name, spec in c['methods'].items():
            if trainable(spec):
                out = run_dir(path, seed, name); done = json.loads((out / 'complete.json').read_text())
                if digest(out / 'model.pt') != done['checkpoint_sha256']: raise ValueError(f'{seed}/{name} checkpoint changed.')
                entries[f'{seed}/{name}'] = done['checkpoint_sha256']
        for e in c['evaluations']:
            f = run_dir(path, seed, e['method']) / 'dev' / f'{e["name"]}.json'
            if not f.exists() or 'note' in json.loads(f.read_text()): missing.append(f'{seed}/{e["name"]}')
            elif digest(f) != f.with_suffix('.sha256').read_text().strip(): raise ValueError(f'{f} changed after evaluation.')
    if missing: raise ValueError(f'Development evaluations missing: {missing[:8]}')
    with exclusive(path / 'freeze.lock'):
        write_json(path / 'freeze.json', {'checkpoints': entries, 'manifest_sha256': digest(path / 'manifest.json'),
                   'timestamp': time.time(), 'note': 'No retraining or method changes after the final bank is opened.'})
    print('Frozen.')


def report(args):
    from scipy.stats import t
    path, m, src = load(args.study); c = m['config']
    frozen = json.loads((path / 'freeze.json').read_text()); seeds = c['training_seeds']
    rows = {}
    for seed in seeds:
        for e in c['evaluations']:
            f = run_dir(path, seed, e['method']) / 'test' / f'{e["name"]}.json'
            if digest(f) != f.with_suffix('.sha256').read_text().strip(): raise ValueError(f'{f} changed after evaluation.')
            r = json.loads(f.read_text())
            if r['episode_seeds'] != m['control_banks']['test'][str(seed)] or r['stage'] != 'test' or 'note' in r:
                raise ValueError(f'{seed}/{e["name"]}: unpaired, truncated or not a final-bank result.')
            spec = c['methods'][e['method']]
            if trainable(spec) and r['provenance'].get('checkpoint_sha256') != frozen['checkpoints'][f'{seed}/{e["method"]}']:
                raise ValueError(f'{seed}/{e["name"]}: evaluated checkpoint is not the frozen one.')
            if 'policy_from' in spec and r['provenance'].get('frozen_policy_sha256') != src[2]['checkpoints'][f'{seed}/{spec["policy_from"]}']:
                raise ValueError(f'{seed}/{e["name"]}: policy is not the source study\'s frozen checkpoint.')
            rows[seed, e['name']] = r
    def summary(v):
        v = 100 * np.asarray(v, float); h = t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return {'per_run': v.tolist(), 'mean': float(v.mean()), 'sample_sd': float(v.std(ddof=1)),
                'ci95': [float(v.mean() - h), float(v.mean() + h)], 'runs_above_zero': int((v > 0).sum())}
    out = {'manifest_sha256': digest(path / 'manifest.json'), 'primary': c['primary'], 'evaluations': {}, 'contrasts': {}}
    for e in c['evaluations']:
        out['evaluations'][e['name']] = {k: summary([rows[s, e['name']][k] for s in seeds]) for k in ('mean_max_coverage', 'success_rate')}
    for a, b, metric in c['contrasts']:
        key = f'{a} minus {b}, {metric}'
        out['contrasts'][key] = summary([rows[s, a][metric] - rows[s, b][metric] for s in seeds])
        out['contrasts'][key]['primary'] = key == c['primary']
    write_json(path / 'report.json', out)
    lines = [f'# {c["name"]}', '', f'Primary: {c["primary"]}', '', '| Evaluation | Coverage, % | Success, % |', '|---|---:|---:|']
    f = lambda r: f'{r["mean"]:.1f} [{r["ci95"][0]:.1f}, {r["ci95"][1]:.1f}]'
    for name, r in out['evaluations'].items(): lines.append(f'| {name} | {f(r["mean_max_coverage"])} | {f(r["success_rate"])} |')
    lines += ['', '| Contrast | Mean, points [95% interval] | Runs > 0 |', '|---|---:|---:|']
    for key, r in out['contrasts'].items(): lines.append(f'| {key}{" (primary)" if r["primary"] else ""} | {f(r)} | {r["runs_above_zero"]}/{len(seeds)} |')
    lines += ['', f'{len(seeds)} training runs; paired Student-t intervals over runs; secondary intervals unadjusted.']
    (path / 'report.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--config', required=True); a.add_argument('--study', required=True)
    for name in ['train', 'evaluate', 'freeze', 'report']:
        a = sub.add_parser(name); a.add_argument('--study', required=True)
    for name in ['train', 'evaluate']:
        a = sub.choices[name]; a.add_argument('--seed', type=int, required=True)
        a.add_argument('--device', default='cpu'); a.add_argument('--threads', type=int, default=2)
    sub.choices['train'].add_argument('--method', required=True)
    sub.choices['evaluate'].add_argument('--evaluation', required=True)
    sub.choices['evaluate'].add_argument('--stage', choices=['dev', 'test'], required=True)
    sub.choices['evaluate'].add_argument('--limit', type=int, default=0, help='Software checks only: first N episodes.')
    args = p.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    {'prepare': prepare, 'train': train, 'evaluate': evaluate, 'freeze': freeze, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
