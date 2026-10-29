"""End-to-end miniature factorial: routing, paired banks, freeze and provenance."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from tests.test_candidate_audit import smoke_study
from scripts.evidence_protocol import write_json
from followup import tmlr_factorial as f

class FactorialTests(unittest.TestCase):
    def test_complete_cycle_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=smoke_study(root)
            c=json.loads(Path('experiments/tmlr_factorial_v1.json').read_text())
            c.update(source_study=str(source),training_seeds=[10,11],dev_episodes=1,test_episodes=1,max_steps=3,common_episodes=1,common_steps=3,common_stride=2,common_candidates_per_prior=3)
            write_json(root/'config.json',c);path=root/'factorial'
            ns=lambda **k:argparse.Namespace(study=str(path),**k)
            f.study.prepare(ns(config=str(root/'config.json')))
            with self.assertRaises(ValueError):f.study.freeze(ns())
            for seed in c['training_seeds']:
                for e in c['evaluations']:f.study.evaluate(ns(seed=seed,evaluation=e['name'],stage='dev',device='cpu',threads=1,limit=0))
                f.common(ns(seed=seed))
            f.study.freeze(ns())
            for seed in c['training_seeds']:
                for e in c['evaluations']:f.study.evaluate(ns(seed=seed,evaluation=e['name'],stage='test',device='cpu',threads=1,limit=0))
            f.report(ns());r=json.loads((path/'report.json').read_text())
            self.assertIn('interaction_secondary',r)
            self.assertEqual(sum(x['primary'] for x in r['contrasts'].values()),1)
            self.assertGreaterEqual(r['common_state_secondary']['learned_selection_regret']['mean'],0)
            (path/'common-10.json').write_text((path/'common-10.json').read_text()+' ')
            with self.assertRaises(ValueError):f.report(ns())

if __name__=='__main__':unittest.main()
