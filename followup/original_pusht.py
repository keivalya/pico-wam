"""Study 5a: frozen evidence-study models under the original gym-pusht dynamics and success measure.

The letter's simulator replays Push-T with relative actions and a physics space rebuilt every control
step. This study runs the same frozen checkpoints in LeRobot's gym-pusht 0.1.8, the original Push-T
implementation, unchanged: absolute PD targets, one persistent pymunk space, its own coverage and
success (> 0.95), and its own start distribution. The adapter only translates. Each relative action
becomes the absolute target that envs/pusht_env.physics_step would set, and the gym's bodies become
this repository's 8-D state. Model images come from this repository's renderer, as in every other
evaluation. Plan checks need the exact replay simulator and are not recorded here.

Runs in its own environment (gym-pusht needs pymunk < 7): FOLLOWUP_PUSHT_PY.

    prepare  : register; record the replay-fidelity check of unseen demonstrations
    evaluate : one (seed, evaluation) on the final bank
    report   : paired run-level contrasts
"""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from scripts.evidence_protocol import ROOT, digest, object_hash, write_json

CODE = ['followup/original_pusht.py', 'followup/common.py', 'followup/study.py', 'scripts/plan.py', 'envs/pusht_env.py',
        'models/wam_policy.py', 'models/dynamics_head.py', 'models/action_heads.py', 'models/encoders.py', 'models/fusion.py']


def code_hashes():
    return {p: digest(ROOT / p) for p in CODE}


class OriginalPushT:
    """gym-pusht behind this repository's state and relative-action interface."""

    def __init__(self):
        import gymnasium as gym
        import gym_pusht  # noqa: F401
        from importlib.metadata import version
        self.env = gym.make('gym_pusht/PushT-v0', obs_type='state', render_mode='rgb_array').unwrapped
        self.version = version('gym-pusht')

    def state(self):
        from envs.pusht_env import to_state
        e = self.env
        return to_state(np.array([*e.agent.position, *e.agent.velocity, *e.block.position, e.block.angle], dtype=np.float64))

    def set_phys(self, phys):
        """Place agent and block exactly at a physics state: angle before position, as physics_step does."""
        ax, ay, vx, vy, bx, by, th = phys; e = self.env
        e.agent.position, e.agent.velocity = (ax, ay), (vx, vy)
        e.block.angle = th; e.block.position = (bx, by); e.block.velocity = (0.0, 0.0); e.block.angular_velocity = 0.0

    def reset(self, seed=None, reset_to_state=None):
        _, info = self.env.reset(seed=seed, options=None if reset_to_state is None else {'reset_to_state': reset_to_state})
        return self.state(), float(self.env._get_coverage())

    def target(self, action):
        """The absolute PD target that envs/pusht_env.physics_step sets for a relative action."""
        from envs.pusht_env import STEP, WORKSPACE
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        ax, ay = self.env.agent.position
        return np.clip([ax + STEP * a[0], ay - STEP * a[1]], 0.0, WORKSPACE)

    def step(self, action):
        _, _, terminated, _, info = self.env.step(self.target(action))
        return self.state(), float(info['coverage']), bool(info['is_success'])


def run_episodes_original(agent, tokens, episode_seeds, max_steps, device):
    import torch
    from envs.pusht_env import PushTEnv
    from followup.common import render_tensor
    env, renderer = OriginalPushT(), PushTEnv(image_size=64)
    success_flags, coverages, rewards, lengths, initial = [], [], [], [], []
    begin = time.time()
    with torch.no_grad():
        for episode_seed in episode_seeds:
            agent.reset(episode_seed)
            state, cov = env.reset(seed=episode_seed); initial.append(cov)
            best, success = cov, False
            for step in range(max_steps):
                st = torch.as_tensor(state, dtype=torch.float32, device=device)[None]
                action = agent.act(render_tensor(renderer, state, device), tokens, st)
                state, cov, done = env.step(action)
                best = max(best, cov); success = success or done
                if success: break
            success_flags.append(success); coverages.append(best); lengths.append(step + 1)
            rewards.append(float(np.clip(best / 0.95, 0.0, 1.0)))
    return {'episode_seeds': list(episode_seeds), 'success': success_flags, 'max_coverage': coverages, 'max_reward': rewards,
            'initial_coverage': initial, 'lengths': lengths, 'success_rate': float(np.mean(success_flags)),
            'mean_max_coverage': float(np.mean(coverages)), 'mean_max_reward': float(np.mean(rewards)),
            'wall_seconds': time.time() - begin, 'gym_pusht': env.version,
            'coverage_definition': 'gym-pusht coverage, maximum over the reset state and every post-action state; reward = clip(coverage / 0.95, 0, 1)'}


def replay_fidelity(sm, max_episodes=None):
    """Replay unseen demonstrations' recorded actions in gym-pusht from their first state; block divergence in px."""
    from envs.pusht_env import from_state
    with np.load(sm['datasets']['demos']['path'], allow_pickle=True) as f:
        d = {k: f[k] for k in ['states', 'actions', 'next_states', 'episode_ids']}
    env = OriginalPushT(); one_step, drift = [], []
    for e in sm['protocol']['splits']['test'][:max_episodes]:
        idx = np.nonzero(d['episode_ids'] == e)[0]
        for i in idx:  # one-step agreement from each recorded state under its recorded action
            env.reset(seed=0); env.set_phys(from_state(d['states'][i]))
            s1, _, _ = env.step(d['actions'][i])
            one_step.append(float(np.linalg.norm(from_state(s1)[4:6] - from_state(d['next_states'][i])[4:6])))
        env.reset(seed=0); env.set_phys(from_state(d['states'][idx[0]]))
        for i in idx:  # open-loop drift over the whole episode in one persistent space
            s, _, _ = env.step(d['actions'][i])
        drift.append(float(np.linalg.norm(from_state(s)[4:6] - from_state(d['next_states'][idx[-1]])[4:6])))
    q = lambda v, x: float(np.percentile(v, x))
    return {'episodes': len(drift), 'transitions': len(one_step), 'one_step_block_px_median': q(one_step, 50),
            'one_step_block_px_p99': q(one_step, 99), 'episode_final_block_drift_px_median': q(drift, 50),
            'note': 'Descriptive check of the adapter and of how far the two simulators diverge on human actions.'}


def prepare(args):
    from scripts.evidence_protocol import control_banks
    from followup.study import source
    c = json.loads(Path(args.config).read_text())
    spath, sm, frozen = source(c)
    out = Path(args.study).resolve(); out.mkdir(parents=True, exist_ok=False)
    m = {'config': c, 'config_sha256': object_hash(c), 'code_sha256': code_hashes(), 'control_banks': control_banks(c),
         'source': {'study': str(spath), 'protocol_sha256': sm['protocol_sha256'], 'freeze_sha256': digest(spath / 'freeze.json')},
         'replay_fidelity': replay_fidelity(sm, args.fidelity_episodes or None), 'gym_pusht': OriginalPushT().version,
         'created_unix': time.time(), 'status': 'Registered before any model was evaluated in gym-pusht.'}
    write_json(out / 'manifest.json', m); (out / 'manifest.sha256').write_text(digest(out / 'manifest.json') + '\n')
    print(f'Prepared {out}. Replay fidelity:', m['replay_fidelity'])


def load(path):
    from followup.study import source
    path = Path(path).resolve(); m = json.loads((path / 'manifest.json').read_text())
    if digest(path / 'manifest.json') != (path / 'manifest.sha256').read_text().strip(): raise ValueError('Manifest changed.')
    if code_hashes() != m['code_sha256']: raise ValueError('Code changed after registration.')
    src = source({'source_study': m['source']['study']})
    if digest(src[0] / 'freeze.json') != m['source']['freeze_sha256']: raise ValueError('Source study changed.')
    return path, m, src


def evaluate(args):
    import torch
    from followup.common import ChunkPolicyAgent, GeneralPlanner, PlannerAgent, instruction_tokens, latent_functions
    from followup.study import frozen_model
    path, m, src = load(args.study); c = m['config']
    e = next(x for x in c['evaluations'] if x['name'] == args.evaluation)
    target = path / 'runs' / f'seed-{args.seed}' / f'{e["name"]}.json'; target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists(): print('Already evaluated; preserved.'); return
    torch.set_num_threads(args.threads); device = 'cpu'
    model, ck, sha = frozen_model(src, e['policy_from'], args.seed, device)
    pc = src[1]['protocol']['config']['planner']
    agent = ChunkPolicyAgent(model) if e['control'] == 'policy' else PlannerAgent(GeneralPlanner(*latent_functions(model, pc['horizon']), device, warm_start=e['control'] == 'warm', **pc))
    r = run_episodes_original(agent, instruction_tokens(ck['vocab'], device), m['control_banks']['test'][str(args.seed)], c['max_steps'], device)
    r.update(evaluation=e, frozen_policy_sha256=sha, manifest_sha256=digest(path / 'manifest.json'), host=platform.node())
    write_json(target, r); target.with_suffix('.sha256').write_text(digest(target) + '\n')
    print('Evaluated', args.seed, e['name'], f'{r["mean_max_coverage"]:.3f}')


def report(args):
    from scipy.stats import t
    path, m, src = load(args.study); c = m['config']; seeds = c['training_seeds']; rows = {}
    for s in seeds:
        for e in c['evaluations']:
            f = path / 'runs' / f'seed-{s}' / f'{e["name"]}.json'
            if digest(f) != f.with_suffix('.sha256').read_text().strip(): raise ValueError(f'{f} changed.')
            r = json.loads(f.read_text())
            if r['episode_seeds'] != m['control_banks']['test'][str(s)] or r['frozen_policy_sha256'] != src[2]['checkpoints'][f'{s}/{e["policy_from"]}']:
                raise ValueError(f'{f}: unpaired episodes or not the frozen checkpoint.')
            rows[s, e['name']] = r
    def summary(v):
        v = 100 * np.asarray(v, float); h = t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return {'per_run': v.tolist(), 'mean': float(v.mean()), 'ci95': [float(v.mean() - h), float(v.mean() + h)], 'runs_above_zero': int((v > 0).sum())}
    metrics = ['mean_max_coverage', 'success_rate', 'mean_max_reward']
    out = {'primary': c['primary'], 'replay_fidelity': m['replay_fidelity'],
           'evaluations': {e['name']: {k: summary([rows[s, e['name']][k] for s in seeds]) for k in metrics} for e in c['evaluations']},
           'contrasts': {}}
    for a, b, k in c['contrasts']:
        key = f'{a} minus {b}, {k}'; out['contrasts'][key] = summary([rows[s, a][k] - rows[s, b][k] for s in seeds]); out['contrasts'][key]['primary'] = key == c['primary']
    write_json(path / 'report.json', out)
    f = lambda r: f'{r["mean"]:.1f} [{r["ci95"][0]:.1f}, {r["ci95"][1]:.1f}]'
    lines = [f'# {c["name"]}', '', f'gym-pusht {m["gym_pusht"]}. Replay fidelity: {m["replay_fidelity"]}', '',
             '| Evaluation | Coverage, % | Success, % | Reward, % |', '|---|---:|---:|---:|']
    for name, r in out['evaluations'].items(): lines.append(f'| {name} | {f(r["mean_max_coverage"])} | {f(r["success_rate"])} | {f(r["mean_max_reward"])} |')
    lines += ['', '| Contrast | Mean [95% interval] | Runs > 0 |', '|---|---:|---:|']
    for key, r in out['contrasts'].items(): lines.append(f'| {key}{" (primary)" if r["primary"] else ""} | {f(r)} | {r["runs_above_zero"]}/{len(seeds)} |')
    (path / 'report.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--config', required=True); a.add_argument('--study', required=True)
    a.add_argument('--fidelity-episodes', type=int, default=0)
    a = sub.add_parser('evaluate'); a.add_argument('--study', required=True); a.add_argument('--seed', type=int, required=True)
    a.add_argument('--evaluation', required=True); a.add_argument('--threads', type=int, default=2)
    a = sub.add_parser('report'); a.add_argument('--study', required=True)
    args = p.parse_args(); {'prepare': prepare, 'evaluate': evaluate, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
