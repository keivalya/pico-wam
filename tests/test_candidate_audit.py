"""Candidate audit on a tiny frozen smoke study: software checks only, never evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from scripts.evidence_protocol import CONDITIONS, ROOT, write_json
from scripts import evidence_study as study
from scripts import candidate_audit as audit


def smoke_study(root):
    import torch
    from envs.pusht_env import PushTEnv, STATE_LABELS
    from utils.tokenizer import SimpleTokenizer
    torch.set_num_threads(1)
    env = PushTEnv(image_size=64); rng = np.random.default_rng(5)
    rows = {k: [] for k in ['images', 'states', 'actions', 'next_images', 'next_states', 'episode_ids', 'is_expert', 'texts', 'timesteps']}
    for e in range(10):
        im, st, info = env.reset(seed=7100000 + e)
        for t in range(16):
            a = rng.uniform(-.2, .2, 2).astype(np.float32); ni, ns, _, _, _ = env.step(a)
            for key, value in zip(rows, [im, st, a, ni, ns, e, 1., info['instruction'], t]): rows[key].append(value)
            im, st = ni, ns
    tok = SimpleTokenizer(); tok.build_from_texts(rows['texts'])
    data = {k: np.asarray(v) for k, v in rows.items()}
    # Random actions are labeled expert only to exercise code paths.
    data.update(text_ids=np.array([tok.encode(s) for s in rows['texts']]), vocab=np.array(tok.vocab, dtype=object),
                state_labels=np.array(STATE_LABELS), env=np.array('pusht'))
    demos, branches = root / 'demos.npz', root / 'branches.npz'; np.savez_compressed(demos, **data)
    subprocess.run([sys.executable, '-m', 'scripts.branch_demos', str(demos), '--output-path', str(branches), '--length', '4'], cwd=ROOT, check=True)
    c = json.loads((ROOT / 'experiments/ral_evidence.json').read_text())
    c.update(name='SMOKE-ONLY', training_seeds=[10, 11], steps=2, batch_size=2, save_every=1, horizon=2, action_chunk=2,
             d_model=16, policy_hidden=16, policy_layers=1, dev_episodes=2, test_episodes=1, max_steps=2,
             prediction_horizon=2, counterfactual_horizon=2, counterfactual_plans=1, counterfactual_stride=8)
    c['planner'].update(horizon=3, samples=4, elites=2)
    write_json(root / 'config.json', c); path = root / 'study'
    study.prepare(argparse.Namespace(config=str(root / 'config.json'), demos=str(demos), branches=str(branches), study=str(path)))
    for seed in c['training_seeds']:
        for k in CONDITIONS:
            study.train(argparse.Namespace(study=str(path), seed=seed, condition=k, device='cpu', threads=1, resume=False))
            study.evaluate_run(argparse.Namespace(study=str(path), seed=seed, condition=k, device='cpu', threads=1, stage='dev'))
    study.freeze(argparse.Namespace(study=str(path)))
    return path


class CandidateAuditTests(unittest.TestCase):
    def test_prepare_score_report_and_guards(self):
        with tempfile.TemporaryDirectory(prefix='pwam-audit-smoke-') as tmp:
            root = Path(tmp); source = smoke_study(root)
            c = json.loads((ROOT / 'experiments/f1_candidate_ranking.json').read_text())
            c.update(source_study=str(source), demo_state_stride=6, expert_episodes=2, expert_state_stride=20, max_steps=30,
                     candidates=8, uniform_candidates=4, elites=2)
            write_json(root / 'f1.json', c); path = root / 'audit'
            audit.prepare(argparse.Namespace(config=str(root / 'f1.json'), audit=str(path)))
            with self.assertRaises(FileExistsError):
                audit.prepare(argparse.Namespace(config=str(root / 'f1.json'), audit=str(path)))
            m = json.loads((path / 'manifest.json').read_text())
            self.assertGreater(m['n_demo_states'], 0); self.assertGreater(m['n_expert_states'], 0)
            with np.load(path / 'common.npz') as f:
                self.assertEqual(f['candidates'].shape[1:], (8, 3, 2)); self.assertTrue(np.all(np.isfinite(f['true_cost'])))
            with self.assertRaises(ValueError):
                audit.score(argparse.Namespace(audit=str(path), seed=10, condition='bc', threads=1))
            for seed in [10, 11]:
                for k in ['wam_demo', 'wam_branch']:
                    audit.score(argparse.Namespace(audit=str(path), seed=seed, condition=k, threads=1))
            audit.report(argparse.Namespace(audit=str(path)))
            r = json.loads((path / 'report.json').read_text())
            self.assertTrue(r['contrasts']['wam_branch_minus_wam_demo/common/regret/all']['primary'])
            self.assertGreaterEqual(min(r['conditions']['wam_demo/common/regret/all']['per_run']), 0)
            # The selected sequence can never beat the best candidate, and the own bank contains the prior.
            with np.load(path / 'scores/seed-10-wam_demo.npz') as f:
                np.testing.assert_array_equal(f['own_candidates'][:, 0], f['own_candidates'][:, 1])
            # Tampering with the bank or a score file is rejected.
            s = path / 'scores/seed-10-wam_demo.json'; s.write_text(s.read_text().replace('"seconds"', '"seconds_"'))
            f = path / 'scores/seed-10-wam_demo.npz'; f.write_bytes(f.read_bytes() + b'0')
            with self.assertRaises(ValueError): audit.report(argparse.Namespace(audit=str(path)))
            b = path / 'bank.npz'; b.write_bytes(b.read_bytes() + b'0')
            with self.assertRaises(ValueError): audit.load_audit(path)


if __name__ == '__main__':
    unittest.main()
