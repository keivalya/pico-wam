"""Protocol invariants and a tiny end-to-end smoke study (no scientific results)."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts.evidence_protocol import (CONDITIONS, ROOT, sample_indices, split_ids,
    validate_config, validate_datasets, control_banks, write_json)
from scripts import evidence_study as study


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((ROOT/'experiments/ral_evidence.json').read_text())

    def test_disjoint_episode_and_control_splits(self):
        split=split_ids(np.arange(206),self.c)
        self.assertEqual(set(sum(split.values(),[])),set(range(206)))
        self.assertEqual(sum(map(len,split.values())),206)
        self.assertEqual(split,split_ids(np.arange(206),self.c))
        banks=control_banks(self.c)
        a=set(sum(banks['dev'].values(),[])); b=set(sum(banks['test'].values(),[]))
        self.assertFalse(a & b)
        self.c['test_seed_start']=self.c['dev_seed_start']
        with self.assertRaises(ValueError): validate_config(self.c)

    def test_action_minibatches_match_even_when_auxiliary_data_changes(self):
        self.c['batch_size']=128
        pairs=[sample_indices(np.arange(10),np.arange(100,120),self.c,10,50,c) for c in CONDITIONS]
        for action,_ in pairs: np.testing.assert_array_equal(action,pairs[0][0])
        np.testing.assert_array_equal(pairs[2][1],pairs[4][1])
        np.testing.assert_array_equal(pairs[1][1],pairs[3][1])
        self.assertTrue(np.any(pairs[4][1]>=100))
        self.assertTrue(np.all(pairs[3][1]<10))
        with self.assertRaises(ValueError): sample_indices(np.arange(10),[],self.c,10,50,'wam_branch')

    def test_malformed_branch_sources_and_prefix_rejected(self):
        demo={'states':np.zeros((4,8)),'env':np.array('pusht')}
        for key in ['images','actions','next_images','next_states','text_ids']: demo[key]=np.zeros((4,2))
        demo['episode_ids']=np.array([0,0,1,1]);demo['is_expert']=np.ones(4)
        aug={k:np.concatenate([v,v[:2]]) for k,v in demo.items() if k!='env'}
        aug['env']=demo['env'];aug['episode_ids']=np.array([0,0,1,1,2,2]);aug['source_episodes']=np.array([0,0,1,1,0,0]);aug['is_expert'][4:]=0
        validate_datasets(demo,aug)
        bad=copy.deepcopy(aug);bad['source_episodes'][-1]=1
        with self.assertRaises(ValueError): validate_datasets(demo,bad)
        bad=copy.deepcopy(aug);bad['actions'][0,0]=1
        with self.assertRaises(ValueError): validate_datasets(demo,bad)
        bad=copy.deepcopy(aug);bad['is_expert'][-1]=1
        with self.assertRaises(ValueError): validate_datasets(demo,bad)


class SmokeStudyTests(unittest.TestCase):
    def test_end_to_end_and_resume(self):
        import torch
        from envs.pusht_env import PushTEnv, STATE_LABELS
        from utils.tokenizer import SimpleTokenizer
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='pwam-evidence-smoke-') as tmp:
            root=Path(tmp); env=PushTEnv(image_size=64); rng=np.random.default_rng(123)
            rows={k:[] for k in ['images','states','actions','next_images','next_states','episode_ids','is_expert','texts','timesteps']}
            for e in range(10):
                im,st,info=env.reset(seed=7000000+e)
                for t in range(8):
                    a=rng.uniform(-.2,.2,2).astype(np.float32); ni,ns,_,_,_=env.step(a)
                    vals=[im,st,a,ni,ns,e,1.,info['instruction'],t]
                    for key,value in zip(rows,vals): rows[key].append(value)
                    im,st=ni,ns
            env.close(); tok=SimpleTokenizer();tok.build_from_texts(rows['texts'])
            data={k:np.asarray(v) for k,v in rows.items()}
            # Random actions are labeled expert only to exercise code paths, never as study evidence.
            data.update(text_ids=np.array([tok.encode(s) for s in rows['texts']]),vocab=np.array(tok.vocab,dtype=object),
                        state_labels=np.array(STATE_LABELS),env=np.array('pusht'))
            demos=root/'demos.npz';branches=root/'branches.npz'; np.savez_compressed(demos,**data)
            subprocess.run([sys.executable,'-m','scripts.branch_demos',str(demos),'--output-path',str(branches),'--length','4'],cwd=ROOT,check=True)
            c=json.loads((ROOT/'experiments/ral_evidence.json').read_text())
            c.update(name='SMOKE-ONLY',training_seeds=[10,11],steps=2,batch_size=2,save_every=1,horizon=2,
                     action_chunk=2,d_model=16,policy_hidden=16,policy_layers=1,dev_episodes=2,test_episodes=1,
                     max_steps=2,prediction_horizon=2,counterfactual_horizon=2,counterfactual_plans=1,counterfactual_stride=8)
            c['planner'].update(horizon=2,samples=4,elites=2)
            config=root/'config.json';write_json(config,c)
            path=root/'study'
            study.prepare(argparse.Namespace(config=str(config),demos=str(demos),branches=str(branches),study=str(path)))
            manifest=path/'manifest.json'; original_manifest=manifest.read_text()
            manifest.write_text(original_manifest+' ')
            with self.assertRaisesRegex(ValueError,'manifest changed'):study.load_study(path)
            manifest.write_text(original_manifest)
            with self.assertRaises(FileExistsError):
                study.prepare(argparse.Namespace(config=str(config),demos=str(demos),branches=str(branches),study=str(path)))
            train_args=lambda seed,condition,resume=False:argparse.Namespace(study=str(path),seed=seed,condition=condition,device='cpu',threads=1,resume=resume)
            eval_args=lambda seed,condition,stage:argparse.Namespace(study=str(path),seed=seed,condition=condition,device='cpu',threads=1,stage=stage)
            with self.assertRaises(FileNotFoundError):study.freeze(argparse.Namespace(study=str(path)))
            # Simulate interruption immediately after the first checkpoint, then resume it.
            original=study.checkpoint
            def interrupted(*args,**kwargs):
                original(*args,**kwargs)
                raise RuntimeError('simulated interruption')
            with patch.object(study,'checkpoint',interrupted):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):study.train(train_args(10,'bc'))
            self.assertFalse((path/'runs/seed-10/bc/training.lock').exists())
            study.train(train_args(10,'bc',True))
            with self.assertRaises(FileNotFoundError):study.evaluate_run(eval_args(10,'bc','test'))
            init_hashes={};state_stats=[]
            for seed in c['training_seeds']:
                init_hashes[seed]=set()
                for condition in CONDITIONS:
                    if (seed,condition)!=(10,'bc'):study.train(train_args(seed,condition))
                    ck=torch.load(path/f'runs/seed-{seed}/{condition}/model.pt',map_location='cpu',weights_only=False)
                    self.assertEqual(ck['completed_steps'],2)
                    self.assertEqual(ck['trained_dynamics'],condition.startswith('wam_'))
                    init_hashes[seed].add(ck['initialization_sha256']); state_stats.append(ck['model_state_dict']['state_std'])
                    study.evaluate_run(eval_args(seed,condition,'dev'))
            self.assertTrue(all(len(h)==1 for h in init_hashes.values()))
            for value in state_stats:torch.testing.assert_close(value,state_stats[0],rtol=0,atol=0)
            study.dev_precision(argparse.Namespace(study=str(path)))
            precision=json.loads((path/'dev_precision.json').read_text())
            self.assertEqual(len(precision['development_per_run_interaction_pp']),2)
            self.assertFalse(any((path/f'runs/seed-{s}/{k}/test').exists() for s in c['training_seeds'] for k in CONDITIONS))
            study.freeze(argparse.Namespace(study=str(path)))
            with self.assertRaises(ValueError):study.train(train_args(10,'bc',True))
            for seed in c['training_seeds']:
                for condition in CONDITIONS:study.evaluate_run(eval_args(seed,condition,'test'))
            study.report(argparse.Namespace(study=str(path)))
            result=json.loads((path/'report.json').read_text())
            self.assertEqual(sum(v['primary'] for v in result['effects'].values()),1)
            self.assertEqual(len(result['conditions']),9)
            self.assertIn('branch_minus_demo/counterfactual_block_rmse_px_k10',result['prediction'])
            self.assertEqual(len(result['compute']['training_seconds']),10)
            # Resumption is exactly equivalent to the uninterrupted fixed-update run.
            second=root/'resume-reference'
            study.prepare(argparse.Namespace(config=str(config),demos=str(demos),branches=str(branches),study=str(second)))
            a=train_args(10,'bc');a.study=str(second);study.train(a)
            left=torch.load(path/'runs/seed-10/bc/model.pt',map_location='cpu',weights_only=False)
            right=torch.load(second/'runs/seed-10/bc/model.pt',map_location='cpu',weights_only=False)
            for key in left['model_state_dict']:torch.testing.assert_close(left['model_state_dict'][key],right['model_state_dict'][key],rtol=0,atol=0)
            # Mutated result files cannot silently enter a report.
            f=path/'runs/seed-10/bc/test/policy.json'; f.write_text(f.read_text()+' ')
            with self.assertRaises(ValueError):study.report(argparse.Namespace(study=str(path)))


if __name__=='__main__':unittest.main()
