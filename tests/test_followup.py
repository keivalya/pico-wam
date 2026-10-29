"""Follow-up harness checks on a tiny frozen smoke study: software checks only, never evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from scripts import evidence_study as study
from tests.test_candidate_audit import smoke_study


class FollowupHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='pwam-followup-smoke-')
        cls.path = smoke_study(Path(cls.tmp.name))
        cls.manifest = json.loads((cls.path / 'manifest.json').read_text())
        cls.c = copy.deepcopy(cls.manifest['protocol']['config'])
        cls.c['max_steps'] = 25  # long enough for several chunks and plan checks

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def load(self, condition, seed=10):
        from models.wam_policy import PicoWAMPolicy
        return PicoWAMPolicy.from_checkpoint(self.path / f'runs/seed-{seed}/{condition}/model.pt', 'cpu')

    def test_runner_reproduces_evaluate_control(self):
        from followup.common import (ChunkPolicyAgent, GeneralPlanner, PlannerAgent, instruction_tokens,
                                     latent_functions, run_episodes)
        seeds = [2000000, 2000001]
        for condition, method in [('bc', 'policy'), ('wam_branch', 'policy'), ('wam_branch', 'warm'), ('wam_demo', 'restart')]:
            model, ck = self.load(condition)
            ref = study.evaluate_control(model, ck['vocab'], self.c, seeds, method, torch.device('cpu'))
            tokens = instruction_tokens(ck['vocab'], 'cpu')
            if method == 'policy':
                agent = ChunkPolicyAgent(model)
            else:
                pc = self.c['planner']
                agent = PlannerAgent(GeneralPlanner(*latent_functions(model, pc['horizon']), 'cpu', warm_start=method == 'warm', **pc))
            got = run_episodes(agent, tokens, seeds, self.c['max_steps'], self.c['check_plans_every'], 'cpu')
            for key in ['success', 'max_coverage', 'lengths', 'plan_checks_by_episode']:
                self.assertEqual(got[key], ref[key], f'{condition}/{method}/{key}')

    def test_state_model_and_alternative_planners_run(self):
        from followup.common import (GeneralPlanner, PlannerAgent, instruction_tokens, oracle_functions,
                                     run_episodes, state_model_functions)
        from followup.planners import train_state_model
        model, ck = self.load('wam_branch')
        tokens = instruction_tokens(ck['vocab'], 'cpu'); pc = self.c['planner']
        for condition in ['wam_demo', 'wam_branch']:
            sm, info = train_state_model(self.manifest, condition, 10, steps=3)
            self.assertEqual(info['steps'], 3)
            torch.testing.assert_close(sm.state_std, model.state_std)  # same demonstration statistics
        fns = {'state': state_model_functions(model, sm, pc['horizon'], 'cpu'), 'oracle': oracle_functions(model, pc['horizon'], 'cpu')}
        for name, (prior_fn, predict_fn) in fns.items():
            r = run_episodes(PlannerAgent(GeneralPlanner(prior_fn, predict_fn, 'cpu', **pc)), tokens, [2000000], 6, 3, 'cpu')
            self.assertEqual(len(r['max_coverage']), 1); self.assertTrue(r['plan_checks_by_episode'][0]['checks'])
        # The oracle predicts the simulator exactly, so its chosen plans have no endpoint error.
        r = run_episodes(PlannerAgent(GeneralPlanner(*fns['oracle'], 'cpu', **pc)), tokens, [2000001], 6, 2, 'cpu')
        self.assertTrue(all(x['plan_error_px'] < 1e-3 for x in r['plan_checks_by_episode'][0]['checks']))

    def test_registered_study_workflow(self):
        import argparse
        from followup import study as fs
        from scripts.evidence_protocol import write_json
        c = json.loads((Path(__file__).resolve().parents[1] / 'experiments/s4_planners.json').read_text())
        c.update(source_study=str(self.path), training_seeds=[10, 11], dev_episodes=1, test_episodes=1, max_steps=6, check_plans_every=3,
                 evaluations=[e for e in c['evaluations'] if e['name'] in ('latent_branch_restart', 'state_branch_restart', 'oracle_restart')],
                 contrasts=[['state_branch_restart', 'latent_branch_restart', 'mean_max_coverage'], ['oracle_restart', 'latent_branch_restart', 'success_rate']])
        c['methods'] = {k: v for k, v in c['methods'].items() if k in ('latent_branch', 'state_branch', 'oracle')}
        root = Path(self.tmp.name); write_json(root / 's4.json', c); path = root / 's4'
        ns = lambda **k: argparse.Namespace(study=str(path), device='cpu', threads=1, limit=0, **k)
        fs.prepare(argparse.Namespace(config=str(root / 's4.json'), study=str(path)))
        with self.assertRaises(FileExistsError): fs.prepare(argparse.Namespace(config=str(root / 's4.json'), study=str(path)))
        with self.assertRaises(ValueError): fs.evaluate(ns(seed=10, evaluation='oracle_restart', stage='test'))
        for seed in [10, 11]:
            fs.train(ns(seed=seed, method='state_branch'))
            for e in c['evaluations']: fs.evaluate(ns(seed=seed, evaluation=e['name'], stage='dev'))
        fs.freeze(argparse.Namespace(study=str(path)))
        with self.assertRaises(ValueError): fs.train(ns(seed=10, method='state_branch'))
        for seed in [10, 11]:
            for e in c['evaluations']: fs.evaluate(ns(seed=seed, evaluation=e['name'], stage='test'))
        fs.report(argparse.Namespace(study=str(path)))
        r = json.loads((path / 'report.json').read_text())
        self.assertEqual(sum(v['primary'] for v in r['contrasts'].values()), 1)
        f = path / 'runs/seed-10/oracle/test/oracle_restart.json'; f.write_text(f.read_text() + ' ')
        with self.assertRaises(ValueError): fs.report(argparse.Namespace(study=str(path)))

    def test_fresh_prediction_cycle(self):
        import argparse
        from followup import fresh_prediction as fp
        from scripts.evidence_protocol import write_json
        c = json.loads((Path(__file__).resolve().parents[1] / 'experiments/s5b_fresh_prediction.json').read_text())
        c.update(source_study=str(self.path), episodes=2, max_steps=20, prediction_horizon=3, counterfactual_horizon=2, counterfactual_plans=1, counterfactual_stride=8)
        root = Path(self.tmp.name); write_json(root / 's5b.json', c); path = root / 's5b'
        fp.prepare(argparse.Namespace(config=str(root / 's5b.json'), study=str(path)))
        for seed in [10, 11]:
            for k in c['conditions']: fp.score(argparse.Namespace(study=str(path), seed=seed, condition=k, threads=1))
        fp.report(argparse.Namespace(study=str(path)))
        r = json.loads((path / 'report.json').read_text())
        self.assertIn('wam_branch_minus_wam_demo/block_position_rmse_px_k2', r['contrasts'])
        (path / 'episodes.npz').write_bytes((path / 'episodes.npz').read_bytes() + b'0')
        with self.assertRaises(ValueError): fp.load(path)


if __name__ == '__main__':
    unittest.main()
