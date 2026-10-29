"""Actual MuJoCo integration check; tiny runs are never paper evidence."""
import argparse
import json
from pathlib import Path
import tempfile
from followup import tmlr_contact as f

def main():
    with tempfile.TemporaryDirectory(prefix='tmlr-contact-smoke-') as tmp:
        root=Path(tmp);c=json.loads(Path('experiments/tmlr_contact_v1.json').read_text())
        c.update(training_seeds=[1,2],data_episodes=2,max_steps=40,horizon=2,width=16,updates=2,batch_size=4,bank_episodes=1,bank_steps=[30],candidates=4,test_episodes=1)
        (root/'config.json').write_text(json.dumps(c));path=root/'study'
        ns=lambda **k:argparse.Namespace(study=str(path),**k)
        f.prepare(ns(config=str(root/'config.json')));f.collect(ns())
        for s in c['training_seeds']:f.train(ns(seed=s))
        f.freeze(ns());f.bank(ns())
        for s in c['training_seeds']:f.evaluate(ns(seed=s))
        f.report(ns())
        r=json.loads((path/'report.json').read_text());assert len(r['primary_result']['per_run'])==2
        (path/'bank.npz').write_bytes((path/'bank.npz').read_bytes()+b'0')
        try:f.report(ns())
        except ValueError:pass
        else:raise AssertionError('Changed bank accepted')
        print('CONTACT INTEGRATION PASSED',flush=True)
if __name__=='__main__':main()
