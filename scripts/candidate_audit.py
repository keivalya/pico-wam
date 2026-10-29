"""Common-state candidate ranking audit of frozen evidence-study checkpoints (follow-up F1).

The evidence study's plan checks follow each policy's own trajectory and replay only the chosen
plan. This audit instead scores every model on the same physical states and the same candidate
action sequences, and compares the models' rankings with the simulator's:

    prepare : register the protocol, build the fixed state bank and the common candidates
              with their true costs (CPU; no model is loaded)
    score   : one frozen checkpoint -> model costs on the common candidates, plus its own
              restart-planner candidates and their true costs (CPU)
    report  : paired run-level contrasts between conditions

Costs use the planner's own cost function (scripts/plan.py) on predicted or simulated states,
over the planner's 12-step horizon. See experiments/FOLLOWUP_PROTOCOLS.md, F1.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.evidence_protocol import ROOT, digest, object_hash, write_json

CODE = ['scripts/candidate_audit.py', 'scripts/plan.py', 'envs/pusht_env.py', 'envs/pusht_expert.py',
        'models/wam_policy.py', 'models/dynamics_head.py', 'models/action_heads.py', 'models/encoders.py',
        'models/fusion.py', 'utils/tokenizer.py']


def code_hashes():
    return {p: digest(ROOT / p) for p in CODE}


class _Stub:
    """Lets the planner's cost function run on simulated states without a model."""
    is_wam = predict_state = True

    def __init__(self):
        import torch
        self.state_mean = torch.zeros(8)


def cost_fn(approach_weight):
    import torch
    from scripts.plan import LatentPlanner
    planner = LatentPlanner(_Stub(), approach_weight=approach_weight)
    return lambda states: planner.cost(torch.as_tensor(np.asarray(states), dtype=torch.float32)).numpy()


def true_costs(env, state, seqs, cost):
    return cost(np.stack([env.simulate(state, s) for s in seqs]))


def load_audit(path, check_bank=True):
    path = Path(path).resolve()
    m = json.loads((path / 'manifest.json').read_text())
    if digest(path / 'manifest.json') != (path / 'manifest.sha256').read_text().strip():
        raise ValueError('Audit manifest changed after preparation.')
    if code_hashes() != m['code_sha256']:
        raise ValueError('Audit code changed after preparation; prepare a new audit.')
    if check_bank and any(digest(path / n) != h for n, h in m['files'].items()):
        raise ValueError('State bank or common candidates changed after preparation.')
    return path, m


def source_study(m):
    from scripts.evidence_study import load_study
    spath, sm = load_study(m['source']['study'], check_data=False)
    frozen = json.loads((spath / 'freeze.json').read_text())
    if sm['protocol_sha256'] != m['source']['protocol_sha256'] or digest(spath / 'freeze.json') != m['source']['freeze_sha256']:
        raise ValueError('Source study changed since the audit was registered.')
    return spath, sm, frozen


def prepare(args):
    from envs.pusht_env import PushTEnv
    from scripts.evidence_study import load_study
    c = json.loads(Path(args.config).read_text())
    spath, sm = load_study(c['source_study'], check_data=False)
    if not (spath / 'freeze.json').exists():
        raise ValueError('The source study must be frozen.')
    pc = sm['protocol']['config']['planner']
    H, N = pc['horizon'], c['candidates']
    out = Path(args.audit).resolve(); out.mkdir(parents=True, exist_ok=False)
    env = PushTEnv(image_size=64)
    states, actions, sources, labels = [], [], [], []
    # 1. Unseen test-split demonstration states with at least H recorded actions left.
    with np.load(sm['datasets']['demos']['path'], allow_pickle=True) as f:
        demo = {k: f[k] for k in ['states', 'actions', 'episode_ids']}
    for e in sm['protocol']['splits']['test']:
        idx = np.nonzero(demo['episode_ids'] == e)[0]
        for t in range(0, len(idx) - H + 1, c['demo_state_stride']):
            states.append(demo['states'][idx[t]]); actions.append(demo['actions'][idx[t:t + H]])
            sources.append(0); labels.append(f'demo/{e}/{t}')
    # 2. States visited by the noisy scripted expert from fresh simulator seeds; model-independent.
    for i in range(c['expert_episodes']):
        seed = c['expert_seed_start'] + i
        _, s, _ = env.reset(seed=seed); traj, acts = [s], []
        for _ in range(c['max_steps']):
            a = env.expert_action(noise=c['expert_noise']); _, s, _, done, _ = env.step(a)
            acts.append(a); traj.append(s)
            if done: break
        for t in range(0, len(acts) - H + 1, c['expert_state_stride']):
            states.append(traj[t]); actions.append(np.asarray(acts[t:t + H])); sources.append(1); labels.append(f'expert/{seed}/{t}')
    states = np.asarray(states, np.float32); actions = np.asarray(actions, np.float32)
    images = np.stack([env._render(s) for s in states])
    # Common candidates: half uniform at the counterfactual-evaluation scale, half the recorded
    # trajectory's next H actions plus noise. Every model sees exactly these sequences.
    rng = np.random.default_rng(c['candidate_seed'])
    nu = c['uniform_candidates']
    uni = rng.uniform(-c['uniform_scale'], c['uniform_scale'], size=(len(states), nu, H, 2))
    near = actions[:, None] + c['trajectory_noise'] * rng.standard_normal((len(states), N - nu, H, 2))
    cands = np.clip(np.concatenate([uni, near], 1), -1, 1).astype(np.float32)
    cost = cost_fn(pc['approach_weight']); begin = time.time()
    truth = np.stack([true_costs(env, s, q, cost) for s, q in zip(states, cands)]).astype(np.float32)
    np.savez_compressed(out / 'bank.npz', states=states, images=images, actions=actions,
                        sources=np.asarray(sources), labels=np.asarray(labels))
    np.savez_compressed(out / 'common.npz', candidates=cands, true_cost=truth)
    manifest = {'config': c, 'config_sha256': object_hash(c), 'code_sha256': code_hashes(),
                'source': {'study': str(spath), 'protocol_sha256': sm['protocol_sha256'],
                           'freeze_sha256': digest(spath / 'freeze.json'), 'planner': pc},
                'files': {n: digest(out / n) for n in ['bank.npz', 'common.npz']},
                'n_states': int(len(states)), 'n_demo_states': int(sources.count(0)),
                'n_expert_states': int(sources.count(1)), 'truth_seconds': time.time() - begin,
                'created_unix': time.time(),
                'status': 'Registered before any checkpoint was scored on this bank.'}
    write_json(out / 'manifest.json', manifest)
    (out / 'manifest.sha256').write_text(digest(out / 'manifest.json') + '\n')
    print(f'Prepared {out}: {len(states)} states ({sources.count(0)} demo, {sources.count(1)} expert).')


def rank_metrics(pred, true, prior=None, elites=32):
    """Per-state metrics for one candidate bank. pred, true: (S, N) costs, lower is better."""
    from scipy.stats import spearmanr
    S = len(pred); sel = pred.argmin(1); rows = np.arange(S)
    rho = np.array([spearmanr(p, t).statistic for p, t in zip(pred, true)])
    top_p = np.argsort(pred, 1)[:, :elites]; top_t = np.argsort(true, 1)[:, :elites]
    overlap = np.array([len(set(a) & set(b)) / elites for a, b in zip(top_p, top_t)])
    out = {'spearman': rho, 'regret': true[rows, sel] - true.min(1), 'elite_overlap': overlap,
           'optimism': true[rows, sel] - pred[rows, sel]}
    if prior is not None:
        out['gain_over_prior'] = true[rows, prior] - true[rows, sel]
    return out


def score(args):
    import torch
    from envs.pusht_env import PushTEnv
    from models.wam_policy import PicoWAMPolicy
    from utils.tokenizer import SimpleTokenizer
    path, m = load_audit(args.audit)
    spath, sm, frozen = source_study(m)
    c, pc = m['config'], m['source']['planner']
    if args.condition not in c['conditions'] or args.seed not in sm['protocol']['config']['training_seeds']:
        raise ValueError('Condition or seed is not part of this audit.')
    ckpt = spath / 'runs' / f'seed-{args.seed}' / args.condition / 'model.pt'
    sha = digest(ckpt)
    if frozen['checkpoints'][f'{args.seed}/{args.condition}'] != sha:
        raise ValueError('Checkpoint does not match the frozen study.')
    target = path / 'scores' / f'seed-{args.seed}-{args.condition}.npz'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        print('Already scored; preserved.'); return
    torch.set_num_threads(args.threads); begin = time.time()
    model, ck = PicoWAMPolicy.from_checkpoint(ckpt, 'cpu')
    if not ck['trained_dynamics']:
        raise ValueError('Only conditions with trained dynamics can be audited.')
    with np.load(path / 'bank.npz') as f: bank = dict(f)
    with np.load(path / 'common.npz') as f: common = dict(f)
    tok = SimpleTokenizer(vocab=ck['vocab']); env = PushTEnv(image_size=64)
    tokens = torch.tensor([tok.encode(PushTEnv.instruction)])
    cost = cost_fn(pc['approach_weight']); H, N = pc['horizon'], c['candidates']
    pred_common, own, pred_own, true_own = [], [], [], []
    with torch.no_grad():
        for i, (s, im) in enumerate(zip(bank['states'], bank['images'])):
            img = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().div(255)[None]
            st = torch.from_numpy(s)[None]
            z0 = model.encode_obs(img, tokens, st)
            roll = lambda q: cost(model.denormalize_state(model.state_readout(model.dynamics.rollout(z0.expand(len(q), -1), q))))
            pred_common.append(roll(torch.from_numpy(common['candidates'][i])))
            # The restart planner's first round: the policy's imagined plan plus noise, seeded per state.
            prior = model.imagine(img, tokens, st, H)['actions'][0]
            g = torch.Generator().manual_seed(c['candidate_seed'] + 1 + i)
            q = (prior + pc['noise'] * torch.randn((N, H, prior.shape[-1]), generator=g)).clamp(-1, 1)
            q[0], q[1] = prior, prior
            own.append(q.numpy()); pred_own.append(roll(q)); true_own.append(true_costs(env, s, q.numpy(), cost))
    np.savez_compressed(target, pred_common=np.asarray(pred_common, np.float32), own_candidates=np.asarray(own, np.float32),
                        pred_own=np.asarray(pred_own, np.float32), true_own=np.asarray(true_own, np.float32))
    write_json(target.with_suffix('.json'), {'seed': args.seed, 'condition': args.condition, 'checkpoint_sha256': sha,
               'scores_sha256': digest(target), 'seconds': time.time() - begin, 'audit_manifest_sha256': digest(path / 'manifest.json')})
    print('Scored', args.seed, args.condition)


def report(args):
    from scipy.stats import t
    path, m = load_audit(args.audit)
    spath, sm, frozen = source_study(m)
    c = m['config']; seeds = sm['protocol']['config']['training_seeds']
    with np.load(path / 'bank.npz') as f: sources = f['sources']
    with np.load(path / 'common.npz') as f: truth = f['true_cost']
    per_run = {}
    for seed in seeds:
        for cond in c['conditions']:
            target = path / 'scores' / f'seed-{seed}-{cond}.npz'
            info = json.loads(target.with_suffix('.json').read_text())
            if info['scores_sha256'] != digest(target) or info['checkpoint_sha256'] != frozen['checkpoints'][f'{seed}/{cond}']:
                raise ValueError(f'Score file for {seed}/{cond} changed or belongs to another checkpoint.')
            with np.load(target) as f: s = dict(f)
            mc = rank_metrics(s['pred_common'], truth, elites=c['elites'])
            mo = rank_metrics(s['pred_own'], s['true_own'], prior=np.ones(len(truth), int), elites=c['elites'])
            row = {}
            for name, mask in [('all', np.ones(len(sources), bool)), ('demo', sources == 0), ('expert', sources == 1)]:
                # nanmean: a rank correlation is undefined for a state whose costs are all tied.
                for k, v in mc.items(): row[f'common/{k}/{name}'] = float(np.nanmean(v[mask]))
                for k, v in mo.items(): row[f'own/{k}/{name}'] = float(np.nanmean(v[mask]))
            row['common/undefined_spearman_states'] = int(np.isnan(mc['spearman']).sum())
            per_run[f'{seed}/{cond}'] = row
    def summary(v):
        v = np.asarray(v, float); h = t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return {'per_run': v.tolist(), 'mean': float(v.mean()), 'ci95': [float(v.mean() - h), float(v.mean() + h)],
                'runs_above_zero': int((v > 0).sum())}
    keys = list(next(iter(per_run.values())))
    a, b = c['conditions']
    result = {'audit_manifest_sha256': digest(path / 'manifest.json'), 'source_protocol_sha256': sm['protocol_sha256'],
              'n_states': m['n_states'], 'n_demo_states': m['n_demo_states'], 'n_expert_states': m['n_expert_states'],
              'primary': c['primary'], 'conditions': {}, 'contrasts': {}}
    for k in keys:
        for cond in c['conditions']:
            result['conditions'][f'{cond}/{k}'] = summary([per_run[f'{s}/{cond}'][k] for s in seeds])
        result['contrasts'][f'{b}_minus_{a}/{k}'] = summary([per_run[f'{s}/{b}'][k] - per_run[f'{s}/{a}'][k] for s in seeds])
    result['contrasts'][f'{b}_minus_{a}/common/regret/all']['primary'] = True
    write_json(path / 'report.json', result)
    fmt = lambda r: f'{r["mean"]:.3f} [{r["ci95"][0]:.3f}, {r["ci95"][1]:.3f}]'
    lines = ['# Common-state candidate audit', '', f'Primary: {c["primary"]}', '',
             f'{m["n_states"]} states: {m["n_demo_states"]} unseen demonstration, {m["n_expert_states"]} noisy-expert. '
             f'{len(seeds)} training runs. Paired Student-t intervals over runs; secondary intervals unadjusted.', '',
             f'| Metric | {a} | {b} | {b} minus {a} |', '|---|---:|---:|---:|']
    for k in keys:
        lines.append(f'| {k} | {fmt(result["conditions"][f"{a}/{k}"])} | {fmt(result["conditions"][f"{b}/{k}"])} | '
                     f'{fmt(result["contrasts"][f"{b}_minus_{a}/{k}"])} |')
    (path / 'report.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--config', required=True); a.add_argument('--audit', required=True)
    a = sub.add_parser('score'); a.add_argument('--audit', required=True); a.add_argument('--seed', type=int, required=True)
    a.add_argument('--condition', required=True); a.add_argument('--threads', type=int, default=2)
    a = sub.add_parser('report'); a.add_argument('--audit', required=True)
    args = p.parse_args()
    {'prepare': prepare, 'score': score, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
