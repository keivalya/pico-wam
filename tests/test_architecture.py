import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from followup.architecture import AblatedWAM,objective,study
from models.wam_policy import PicoWAMPolicy
from scripts.evidence_protocol import ROOT,write_json
from tests.test_candidate_audit import smoke_study


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.kw=dict(vocab_size=8,state_dim=8,action_dim=2,d_model=16,action_chunk=2,policy_hidden=16,policy_layers=1,predict_image=False)
        self.im=torch.rand(2,3,64,64);self.txt=torch.ones(2,4,dtype=torch.long);self.st=torch.rand(2,8)

    def test_full_is_identical_and_masks_remove_input_dependence(self):
        torch.manual_seed(9);base=PicoWAMPolicy(**self.kw)
        torch.manual_seed(9);model=AblatedWAM('full',**self.kw)
        torch.testing.assert_close(base.encode_obs(self.im,self.txt,self.st),model.encode_obs(self.im,self.txt,self.st),rtol=0,atol=0)
        for variant in ('state_only','image_only','no_text'):
            model=AblatedWAM(variant,**self.kw)
            a=model.encode_obs(self.im,self.txt,self.st)
            b=model.encode_obs(self.im+10 if variant=='state_only' else self.im,self.txt+1,
                               self.st+10 if variant=='image_only' else self.st)
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_detach_aux_leaves_only_policy_encoder_gradient(self):
        b=dict(image=self.im,text_ids=self.txt,state=self.st,action_chunk=torch.rand(2,2,2))
        a=dict(b,actions=torch.rand(2,2,2),future_images=torch.rand(2,2,3,64,64),future_states=torch.rand(2,2,8))
        model=AblatedWAM('detach_aux',**self.kw)
        loss,action=objective(model,b,a,{'state_weight':.5,'latent_weight':1})
        params=list(model.fusion.parameters())
        g1=torch.autograd.grad(loss,params,retain_graph=True)
        g2=torch.autograd.grad(action,params)
        for x,y in zip(g1,g2):torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_full_cycle_all_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=smoke_study(root)
            c=json.loads((ROOT/'experiments/architecture_v1.json').read_text())
            c.update(source_study=str(source),training_seeds=[30,31],dev_episodes=1,test_episodes=1,max_steps=2)
            write_json(root/'config.json',c);out=root/'architecture'
            ns=lambda **kw:argparse.Namespace(study=str(out),device='cpu',threads=1,limit=0,**kw)
            study.prepare(argparse.Namespace(config=str(root/'config.json'),study=str(out)))
            hashes={}
            for seed in c['training_seeds']:
                hashes[seed]=set()
                for method in c['methods']:
                    study.train(ns(seed=seed,method=method))
                    ck=torch.load(out/f'runs/seed-{seed}/{method}/model.pt',weights_only=False)
                    hashes[seed].add(ck['initialization_sha256'])
                for e in c['evaluations']:study.evaluate(ns(seed=seed,evaluation=e['name'],stage='dev'))
            self.assertTrue(all(len(v)==1 for v in hashes.values()))
            study.freeze(ns())
            for seed in c['training_seeds']:
                for e in c['evaluations']:study.evaluate(ns(seed=seed,evaluation=e['name'],stage='test'))
            study.report(ns())
            result=json.loads((out/'report.json').read_text())
            self.assertEqual(len(result['contrasts']),24)

if __name__=='__main__':unittest.main()
