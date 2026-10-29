"""Study-3 baselines on a tiny frozen smoke study: software checks only, never evidence."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from scripts.evidence_protocol import ROOT, write_json
from tests.test_candidate_audit import smoke_study


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='pwam-baselines-smoke-')
        cls.root = Path(cls.tmp.name); cls.path = smoke_study(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def src(self):
        from followup.study import source
        return source({'source_study': str(self.path)})

    def test_diffusion_windows_follow_chi_et_al(self):
        from followup.baselines import LimitsNormalizer, dp_windows
        src = self.src(); data, starts, first, last = dp_windows(src, 16, 2, 8, 'cpu')
        train = src[1]['protocol']['splits']['train']
        lengths = [int(np.sum(data.episode_ids == e)) for e in train]
        self.assertEqual(len(starts), sum(L - 16 + 1 + 7 + 1 for L in lengths))  # pad_before 1, pad_after 7
        x = torch.randn(50, 3) * torch.tensor([1., 5., 0.]) + 2
        n = LimitsNormalizer(x); y = n.normalize(x)
        self.assertAlmostEqual(float(y[:, 0].min()), -1, places=5); self.assertAlmostEqual(float(y[:, 0].max()), 1, places=5)
        torch.testing.assert_close(n.unnormalize(y), x)

    def test_study_three_cycle(self):
        from followup import baselines, study as fs
        c = json.loads((ROOT / 'experiments/s3_baselines.json').read_text())
        c.update(source_study=str(self.path), training_seeds=[10, 11], dev_episodes=1, test_episodes=1, max_steps=12)
        c['methods']['history_bc'].update(steps=3, batch_size=4)
        c['methods']['diffusion_policy']['recipe'].update(down_dims=[16, 32], diffusion_step_embed_dim=16, n_groups=4,
                                                          num_train_timesteps=10, num_inference_steps=3, num_epochs=1, batch_size=8)
        c['tuning'].update(bank_episodes=1, grid={'lr': [1e-3], 'width': [16, 32], 'history': [2], 'chunk': [4]})
        write_json(self.root / 's3.json', c); path = self.root / 's3'
        ns = lambda **k: argparse.Namespace(study=str(path), device='cpu', threads=1, limit=0, **k)
        fs.prepare(argparse.Namespace(config=str(self.root / 's3.json'), study=str(path)))
        with self.assertRaises(ValueError): fs.train(ns(seed=10, method='history_bc'))  # tuning must come first
        for i in range(2): baselines.tune(argparse.Namespace(study=str(path), index=i, device='cpu'))
        baselines.select(argparse.Namespace(study=str(path)))
        with self.assertRaises(FileExistsError): baselines.select(argparse.Namespace(study=str(path)))
        for seed in [10, 11]:
            for method in ['history_bc', 'diffusion_policy']: fs.train(ns(seed=seed, method=method))
            for e in c['evaluations']: fs.evaluate(ns(seed=seed, evaluation=e['name'], stage='dev'))
        fs.freeze(argparse.Namespace(study=str(path)))
        for seed in [10, 11]:
            for e in c['evaluations']: fs.evaluate(ns(seed=seed, evaluation=e['name'], stage='test'))
        fs.report(argparse.Namespace(study=str(path)))
        r = json.loads((path / 'report.json').read_text())
        self.assertIn('v2_wam_branch_restart minus diffusion_policy, mean_max_coverage', r['contrasts'])
        saved = torch.load(path / 'runs/seed-10/diffusion_policy/model.pt', weights_only=False)
        self.assertGreater(saved['info']['steps'], 0); self.assertIn('ema_state_dict', saved)

    def test_diffusion_resume_is_exact(self):
        from followup.baselines import train_diffusion_policy
        c = json.loads((ROOT / 'experiments/s3_baselines.json').read_text())
        spec = copy.deepcopy(c['methods']['diffusion_policy'])
        spec['recipe'].update(down_dims=[16, 32], diffusion_step_embed_dim=16, n_groups=4, num_train_timesteps=10, num_epochs=3, batch_size=8)
        src = self.src()
        full, _, _, info = train_diffusion_policy(src, spec, 10, 'cpu')
        ck = self.root / 'dp-partial.pt'
        per_epoch = info['steps'] // 3
        train_diffusion_policy(src, spec, 10, 'cpu', max_steps=2 * per_epoch, checkpoint=ck, checkpoint_every=1)  # "interrupted" after epoch 1
        self.assertTrue(ck.exists())
        resumed, _, _, info2 = train_diffusion_policy(src, spec, 10, 'cpu', checkpoint=ck, checkpoint_every=1)
        self.assertTrue(info2['resumed']); self.assertEqual(info2['steps'], info['steps'])
        for (k, a), b in zip(full.state_dict().items(), resumed.state_dict().values()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=k)


if __name__ == '__main__':
    unittest.main()
