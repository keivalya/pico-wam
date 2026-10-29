"""Data splits, provenance and sampling for the prospective control study."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import numpy as np

CONDITIONS = ('bc', 'current_demo', 'current_branch', 'wam_demo', 'wam_branch')
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def source_hashes():
    paths = [p for folder in ('models', 'envs', 'utils') for p in (ROOT/folder).glob('*.py')]
    paths += [ROOT/'scripts'/n for n in ('train.py', 'evaluate.py', 'plan.py', 'test.py',
               'evidence_protocol.py', 'evidence_study.py')]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def validate_config(c):
    if c['conditions'] != list(CONDITIONS):
        raise ValueError('Use all five predeclared conditions, in the documented order.')
    seeds = c['training_seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Training seeds must be unique nonnegative integers.')
    for name in ('steps', 'batch_size', 'save_every', 'horizon', 'action_chunk',
                 'dev_episodes', 'test_episodes', 'max_steps', 'prediction_horizon',
                 'counterfactual_horizon', 'counterfactual_plans', 'counterfactual_stride',
                 'check_plans_every', 'd_model', 'policy_hidden', 'policy_layers'):
        if type(c[name]) is not int or c[name] <= 0:
            raise ValueError(f'{name} must be a positive integer.')
    if not 0 < c['validation_fraction'] < 1 or not 0 < c['test_fraction'] < 1:
        raise ValueError('Validation and test fractions must be positive.')
    if c['validation_fraction'] + c['test_fraction'] >= 1:
        raise ValueError('The split must leave training episodes.')
    if not 0 < c['branch_probability'] <= 1:
        raise ValueError('branch_probability must be in (0, 1].')
    if c['learning_rate'] <= 0 or min(c['state_weight'],c['latent_weight']) < 0:
        raise ValueError('Invalid loss weights or learning rate.')
    banks = control_banks(c)
    a = {s for bank in banks['dev'].values() for s in bank}
    b = {s for bank in banks['test'].values() for s in bank}
    if a & b:
        raise ValueError('Development and test control banks overlap.')
    if any(type(c[k]) is not int or c[k] < 0 for k in ('dev_seed_start','test_seed_start')):
        raise ValueError('Control seed starts must be nonnegative integers.')
    if c['primary_metric'] != 'mean_max_coverage':
        raise ValueError('This suite predeclares coverage as the primary metric.')
    planner = c['planner']
    if 'warm_start' in planner or 'seed' in planner:
        raise ValueError('The suite controls planner warm_start and seed.')
    if planner['horizon'] <= 0 or planner['samples'] < 2 or not 2 <= planner['elites'] <= planner['samples']:
        raise ValueError('Invalid planner horizon/sample/elite count.')
    if planner['iterations'] < 1 or planner['noise'] <= 0:
        raise ValueError('Invalid planner search configuration.')
    if not c['precision_target_pp'] > 0 or type(c['max_training_seeds']) is not int or c['max_training_seeds'] < len(seeds):
        raise ValueError('Invalid predeclared precision rule.')


def control_banks(c):
    return {stage: {str(seed): list(range(c[f'{stage}_seed_start'] + i*c[f'{stage}_episodes'],
                                            c[f'{stage}_seed_start'] + (i+1)*c[f'{stage}_episodes']))
                    for i, seed in enumerate(c['training_seeds'])}
            for stage in ('dev', 'test')}


def split_ids(ids, c):
    ids = np.unique(ids).copy()
    np.random.default_rng(c['split_seed']).shuffle(ids)
    nv = max(1, round(len(ids)*c['validation_fraction']))
    nt = max(1, round(len(ids)*c['test_fraction']))
    if nv + nt >= len(ids):
        raise ValueError('Too few episodes for three nonempty splits.')
    return {'train': sorted(ids[nv+nt:].tolist()), 'validation': sorted(ids[:nv].tolist()),
            'test': sorted(ids[nv:nv+nt].tolist())}


def validate_datasets(demo, augmented):
    """Reject mismatched prefixes, branch labels, and source-group leakage."""
    n = len(demo['states'])
    if len(augmented['states']) <= n:
        raise ValueError('Augmented dataset must contain additional branch transitions.')
    for key in ('images','states','actions','next_images','next_states','text_ids','episode_ids','is_expert'):
        if not np.array_equal(demo[key], augmented[key][:n]):
            raise ValueError(f'Augmented prefix does not match demos: {key}')
    if not np.all(demo['is_expert'] > .5) or not np.all(augmented['is_expert'][n:] == 0):
        raise ValueError('Only original demonstration transitions may have expert action labels.')
    source = augmented['source_episodes']
    eps = augmented['episode_ids']
    if not np.array_equal(source[:n],demo['episode_ids']):
        raise ValueError('Demonstration source IDs must match their episode IDs.')
    if not set(source.tolist()) <= set(demo['episode_ids'].tolist()):
        raise ValueError('Unknown branch source episode.')
    if set(eps[n:].tolist()) & set(demo['episode_ids'].tolist()):
        raise ValueError('Branch episodes must have new IDs.')
    for e in np.unique(eps):
        if len(np.unique(source[eps == e])) != 1:
            raise ValueError('An episode crosses source groups.')
    if demo['states'].shape[1] != 8 or str(demo['env']) != 'pusht':
        raise ValueError('This study targets modified Push-T.')


def sample_indices(demo_starts, branch_starts, c, seed, step, condition):
    """Independent RNG streams keep action labels identical across all conditions."""
    rng = lambda stream: np.random.default_rng(np.random.SeedSequence([seed, step, stream]))
    n = c['batch_size']
    action = rng(0).choice(demo_starts,n,replace=True)
    aux = rng(1).choice(demo_starts,n,replace=True)
    if condition.endswith('_branch'):
        if len(branch_starts) == 0:
            raise ValueError('No eligible training branch windows.')
        mask = rng(2).random(n) < c['branch_probability']
        aux[mask] = rng(3).choice(branch_starts,int(mask.sum()),replace=True)
    return action, aux


@contextmanager
def exclusive(path):
    """Owner details let a stale lock be checked against the scheduler before removal."""
    import os, socket, time
    path = Path(path)
    with path.open('x') as f:
        f.write(json.dumps({'pid': os.getpid(), 'host': socket.gethostname(), 'created_unix': time.time(),
                            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                            'slurm_array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'),
                            'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID')}) + '\n')
    try:
        yield
    finally:
        path.unlink()
