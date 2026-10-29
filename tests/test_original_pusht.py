"""Study-5a adapter and cycle on a tiny frozen smoke study; runs only where gym-pusht is installed."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.evidence_protocol import ROOT, write_json
try:
    import gym_pusht  # noqa: F401
except ImportError:
    gym_pusht = None


@unittest.skipIf(gym_pusht is None, 'gym-pusht is installed only in the study-5a environment')
class OriginalPushTTests(unittest.TestCase):
    def test_adapter_agrees_with_this_simulator_where_it_should(self):
        from envs.pusht_env import GOAL_POSE, PushTEnv, coverage, from_state, physics_step
        from followup.original_pusht import OriginalPushT
        g, ours = OriginalPushT(), PushTEnv(image_size=64)
        _, s, _ = ours.reset(seed=3); g.reset(seed=0); g.set_phys(from_state(s))
        np.testing.assert_allclose(g.state(), s, atol=1e-6)
        phys = np.array([60., 60., 0., 0., GOAL_POSE[0] + 10, GOAL_POSE[1] - 5, GOAL_POSE[2] + .1])
        g.reset(seed=0); g.set_phys(phys); self.assertAlmostEqual(coverage(phys[4:7]), g.env._get_coverage(), places=6)
        free = phys.copy(); free[4:6] = [400., 400.]; g.reset(seed=0); g.set_phys(free)
        nxt, a = physics_step(free, [.3, -.2]), np.array([.3, -.2]); sg, _, _ = g.step(a)
        np.testing.assert_allclose(from_state(sg)[:2], nxt[:2], atol=1e-3)  # identical PD control off contact

    def test_cycle(self):
        from followup import original_pusht as op
        from tests.test_candidate_audit import smoke_study
        with tempfile.TemporaryDirectory(prefix='pwam-s5a-smoke-') as tmp:
            root = Path(tmp); source = smoke_study(root)
            c = json.loads((ROOT / 'experiments/s5a_original_pusht.json').read_text())
            c.update(source_study=str(source), training_seeds=[10, 11], test_episodes=1, max_steps=8)
            write_json(root / 's5a.json', c); path = root / 's5a'
            op.prepare(argparse.Namespace(config=str(root / 's5a.json'), study=str(path), fidelity_episodes=1))
            for seed in [10, 11]:
                for e in c['evaluations']: op.evaluate(argparse.Namespace(study=str(path), seed=seed, evaluation=e['name'], threads=1))
            op.report(argparse.Namespace(study=str(path)))
            r = json.loads((path / 'report.json').read_text())
            self.assertTrue(r['contrasts']['wam_branch_restart minus bc_policy, mean_max_coverage']['primary'])


if __name__ == '__main__':
    unittest.main()
