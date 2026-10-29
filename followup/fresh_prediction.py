"""Study 5b: prediction on freshly collected episodes that no analysis has touched.

The evidence study's prediction test reuses a previously inspected demonstration collection
under a new split. This study collects new episodes with the scripted expert (envs/pusht_expert.py)
plus action noise, from simulator seeds never used elsewhere. It registers them by hash before any
model is scored, then measures the frozen WAM checkpoints with exactly the evidence study's
prediction code. The episodes come from a scripted pusher, not humans: a distribution shift that
the results must state.

    prepare : collect and register episodes (CPU)
    score   : one frozen checkpoint -> factual and counterfactual prediction (CPU)
    report  : paired run-level contrasts
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.evidence_protocol import ROOT, digest, object_hash, write_json

CODE = ['followup/fresh_prediction.py', 'scripts/evaluate.py', 'envs/pusht_env.py', 'envs/pusht_expert.py',
        'models/wam_policy.py', 'models/dynamics_head.py', 'utils/tokenizer.py']


def code_hashes():
    return {p: digest(ROOT / p) for p in CODE}


def prepare(args):
    import torch
    from envs.pusht_env import PushTEnv, STATE_LABELS
    from followup.study import source
    from utils.tokenizer import SimpleTokenizer
    c = json.loads(Path(args.config).read_text())
    spath, sm, frozen = source(c)
    out = Path(args.study).resolve(); out.mkdir(parents=True, exist_ok=False)
    vocab = torch.load(spath / 'runs' / f'seed-{sm["protocol"]["config"]["training_seeds"][0]}' / 'wam_branch' / 'model.pt',
                       map_location='cpu', weights_only=False)['vocab']
    tok = SimpleTokenizer(vocab=vocab); ids = tok.encode(PushTEnv.instruction)
    env = PushTEnv(image_size=64); rows = {k: [] for k in ['images', 'states', 'actions', 'next_images', 'next_states', 'episode_ids', 'timesteps']}
    lengths = []
    for e in range(c['episodes']):
        im, st, _ = env.reset(seed=c['seed_start'] + e)
        for t in range(c['max_steps']):
            a = env.expert_action(noise=c['expert_noise']); ni, ns, _, done, _ = env.step(a)
            for k, v in zip(rows, [im, st, a, ni, ns, e, t]): rows[k].append(v)
            im, st = ni, ns
            if done: break
        lengths.append(t + 1)
    data = {k: np.asarray(v) for k, v in rows.items()}
    data.update(text_ids=np.repeat(np.asarray([ids]), len(data['states']), 0), texts=np.asarray([PushTEnv.instruction] * len(data['states'])),
                is_expert=np.ones(len(data['states']), np.float32), vocab=np.array(vocab, dtype=object),
                state_labels=np.array(STATE_LABELS), env=np.array('pusht'))
    np.savez_compressed(out / 'episodes.npz', **data)
    m = {'config': c, 'config_sha256': object_hash(c), 'code_sha256': code_hashes(), 'episodes_sha256': digest(out / 'episodes.npz'),
         'source': {'study': str(spath), 'protocol_sha256': sm['protocol_sha256'], 'freeze_sha256': digest(spath / 'freeze.json')},
         'episode_lengths': lengths, 'transitions': int(len(data['states'])), 'created_unix': time.time(),
         'status': 'Collected and registered before any checkpoint was scored on these episodes.'}
    write_json(out / 'manifest.json', m); (out / 'manifest.sha256').write_text(digest(out / 'manifest.json') + '\n')
    print(f'Prepared {out}: {c["episodes"]} episodes, {len(data["states"])} transitions, mean length {np.mean(lengths):.1f}.')


def load(path):
    from followup.study import source
    path = Path(path).resolve(); m = json.loads((path / 'manifest.json').read_text())
    if digest(path / 'manifest.json') != (path / 'manifest.sha256').read_text().strip(): raise ValueError('Manifest changed.')
    if code_hashes() != m['code_sha256']: raise ValueError('Code changed after registration.')
    if digest(path / 'episodes.npz') != m['episodes_sha256']: raise ValueError('Episodes changed after registration.')
    spath, sm, frozen = source({'source_study': m['source']['study']})
    if digest(spath / 'freeze.json') != m['source']['freeze_sha256']: raise ValueError('Source study changed.')
    return path, m, (spath, sm, frozen)


def score(args):
    import torch
    from followup.study import frozen_model
    from scripts.evaluate import evaluate, evaluate_counterfactuals
    path, m, src = load(args.study); c = m['config']
    if args.condition not in c['conditions']: raise ValueError('Condition not in this study.')
    target = path / 'scores' / f'seed-{args.seed}-{args.condition}.json'; target.parent.mkdir(exist_ok=True)
    if target.exists(): print('Already scored; preserved.'); return
    torch.set_num_threads(args.threads); begin = time.time()
    model, ck, sha = frozen_model(src, args.condition, args.seed, 'cpu')
    with np.load(path / 'episodes.npz', allow_pickle=True) as f: data = dict(f)
    episodes = sorted(set(data['episode_ids'].tolist()))
    with torch.no_grad():
        factual = evaluate(model, data, episodes, c['prediction_horizon'])
        cf = evaluate_counterfactuals(model, data, episodes, 'pusht', c['counterfactual_horizon'], c['counterfactual_plans'],
                                      c['counterfactual_stride'], seed=c['counterfactual_seed'])
    mse = np.array(cf['wam_nmse_per_dim']) * model.state_std.cpu().numpy() ** 2
    cf['block_position_rmse_px'] = (512 * np.sqrt(mse[:, 4:6].sum(1))).tolist()
    write_json(target, {'seed': args.seed, 'condition': args.condition, 'checkpoint_sha256': sha, 'factual': factual,
                        'counterfactual': cf, 'seconds': time.time() - begin, 'manifest_sha256': digest(path / 'manifest.json')})
    target.with_suffix('.sha256').write_text(digest(target) + '\n')
    print('Scored', args.seed, args.condition)


def report(args):
    from scipy.stats import t
    path, m, src = load(args.study); c = m['config']; seeds = src[1]['protocol']['config']['training_seeds']
    H, K = c['prediction_horizon'], c['counterfactual_horizon'] - 1
    def metrics(r):
        f, cf = r['factual'], r['counterfactual']
        block = lambda key: float(np.asarray(cf[key])[K, 4:8].mean())
        return {'factual_nmse_k1': f['state_nmse'][1], f'factual_nmse_k{H}': f['state_nmse'][H], f'persistence_nmse_k{H}': f['state_nmse_copy'][H],
                f'counterfactual_nmse_k{K + 1}': cf['wam_nmse'][K], f'zero_action_nmse_k{K + 1}': cf['noop_nmse'][K],
                f'counterfactual_block_nmse_k{K + 1}': block('wam_nmse_per_dim'), f'zero_action_block_nmse_k{K + 1}': block('noop_nmse_per_dim'),
                f'block_position_rmse_px_k{K + 1}': cf['block_position_rmse_px'][K]}
    rows = {}
    for s in seeds:
        for k in c['conditions']:
            f = path / 'scores' / f'seed-{s}-{k}.json'
            if digest(f) != f.with_suffix('.sha256').read_text().strip(): raise ValueError(f'{f} changed.')
            r = json.loads(f.read_text())
            if r['checkpoint_sha256'] != src[2]['checkpoints'][f'{s}/{k}']: raise ValueError(f'{f} is not the frozen checkpoint.')
            rows[s, k] = metrics(r)
    def summary(v):
        v = np.asarray(v, float); h = t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return {'per_run': v.tolist(), 'mean': float(v.mean()), 'ci95': [float(v.mean() - h), float(v.mean() + h)]}
    a, b = c['conditions']; keys = list(rows[seeds[0], a])
    out = {'primary': c['primary'], 'transitions': m['transitions'], 'episodes': c['episodes'],
           'conditions': {f'{k}/{name}': summary([rows[s, k][name] for s in seeds]) for k in c['conditions'] for name in keys},
           'contrasts': {f'{b}_minus_{a}/{name}': summary([rows[s, b][name] - rows[s, a][name] for s in seeds]) for name in keys},
           'distribution_note': c['distribution_note']}
    write_json(path / 'report.json', out)
    fmt = lambda r: f'{r["mean"]:.4f} [{r["ci95"][0]:.4f}, {r["ci95"][1]:.4f}]'
    lines = [f'# {c["name"]}', '', c['distribution_note'], '', f'| Metric | {a} | {b} | {b} minus {a} |', '|---|---:|---:|---:|']
    for name in keys:
        lines.append(f'| {name} | {fmt(out["conditions"][f"{a}/{name}"])} | {fmt(out["conditions"][f"{b}/{name}"])} | {fmt(out["contrasts"][f"{b}_minus_{a}/{name}"])} |')
    (path / 'report.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--config', required=True); a.add_argument('--study', required=True)
    a = sub.add_parser('score'); a.add_argument('--study', required=True); a.add_argument('--seed', type=int, required=True)
    a.add_argument('--condition', required=True); a.add_argument('--threads', type=int, default=2)
    a = sub.add_parser('report'); a.add_argument('--study', required=True)
    args = p.parse_args(); {'prepare': prepare, 'score': score, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
