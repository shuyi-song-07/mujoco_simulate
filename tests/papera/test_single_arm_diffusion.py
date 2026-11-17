"""Same DP engine for S9/A7: causal windows, updates, resume and rollout tensors."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tasks.data_schema import SINGLE, DUAL
from tasks.registry import get
from tasks.stages import stage_adapter
from tests.papera.test_safe_stitch_data import make_pack
from training.diffusion.data import prepare_data, DiffusionDataset, Normalizer, _fit_statistics, restore_data
from training.diffusion.model import DiffusionConfig, DiffusionPolicy
from training.diffusion.train import ROOT, run
from rollout.common.evaluation_settings import task_evaluation
from rollout.common.diffusion_policy import DiffusionRolloutPolicy
from rollout.common.rollout import validate_policy_settings


class SingleArmDiffusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = ROOT / '.local/checks/push_training_adaptation_20261117/tmp'
        folder.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=folder)
        cls.root = Path(cls.temp.name)
        cls.pack = make_pack(cls.root, task_id='push_t')
        cls.binding = prepare_data(cls.pack, 'push_t', dict(observation_horizon=2,
            prediction_horizon=30, action_horizon=8, image_size=[64,48]), engineering=True)
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        cls.temp.cleanup()

    def test_same_reader_causal_seam_hard_boundary_and_windows_workers(self):
        data = DiffusionDataset(self.binding)
        self.addCleanup(data.close)
        self.assertEqual(data[0]['state'].shape, (2,9))
        self.assertEqual(data[0]['action'].shape, (30,7))
        self.assertEqual(data[0]['gesture'].shape, (1,))
        self.assertEqual(data[0]['target_control_tick'][0], data[0]['source_control_tick']+1)
        self.assertEqual(data[8]['target_raw_frame_index'][:5].tolist(), [8,9,40,41,42])
        self.assertEqual(data[10]['observation_raw_frame_index'].tolist(), [9,40])
        self.assertEqual(data[23]['target_raw_frame_index'][:3].tolist(), [53,54,-1])
        self.assertEqual(data[25]['observation_raw_frame_index'].tolist(), [55,55])
        self.assertEqual(int(data[23]['is_pad'].sum()), 28)
        expected = next(iter(torch.utils.data.DataLoader(data, batch_size=2, num_workers=0)))
        loader = torch.utils.data.DataLoader(data, batch_size=2, num_workers=2, multiprocessing_context='spawn')
        iterator = iter(loader)
        try:
            actual = next(iterator)
            for key in ('state','images','action','is_pad','gesture'):
                torch.testing.assert_close(actual[key], expected[key])
        finally:
            iterator._shutdown_workers()

    def test_declared_schema_normalization_circular_angles_and_restore(self):
        arrays = {'train': {'state': np.arange(27).reshape(3,9)/100.,
                           'action': np.zeros((3,7)), 'action_mask': np.ones(3,bool)}}
        arrays['train']['action'][:,3] = [3.13,-3.13,3.12]
        stats = _fit_statistics(arrays, ['train'], schema=SINGLE)
        norm = Normalizer(stats)
        for name in ('state','action'):
            v = arrays['train'][name]
            np.testing.assert_allclose(getattr(norm,'decode_'+name)(getattr(norm,'encode_'+name)(v)),v,atol=1e-6)
        with self.assertRaises(ValueError): _fit_statistics(arrays, ['train'], schema=DUAL)
        bad = dict(stats);bad.pop('data_schema')
        with self.assertRaises(ValueError): Normalizer(bad)
        with patch('training.diffusion.data._fit_statistics', side_effect=AssertionError('refit')):
            restored = restore_data(json.loads(json.dumps(self.binding)))
        self.assertEqual(restored['stats'], self.binding['stats'])

    def test_single_hand_head_padding_and_same_official_network(self):
        config = DiffusionConfig(state_dim=9,action_dim=7,image_size=(64,48),feature_dim=16,
            time_dim=16,down_dims=(16,32),diffusion_steps=8,inference_steps=2,crop_shape=(42,56),variant='gesture')
        model = DiffusionPolicy(config).train()
        images,state = torch.rand(2,2,1,3,48,64),torch.rand(2,2,9)
        action = torch.rand(2,30,7);pad=torch.zeros(2,30,dtype=torch.bool);pad[:,15:]=True
        labels=dict(gesture=torch.tensor([[0],[2]]),gesture_mask=torch.ones(2,1,dtype=torch.bool))
        before = model.denoiser.network.final_conv[-1].weight.detach().clone()
        # eval disables the random spatial crop while retaining gradient flow.
        model.eval()
        first=model.compute_loss(images,state,action,pad,generator=torch.Generator().manual_seed(31),**labels)
        action[pad]=float('nan')
        second=model.compute_loss(images,state,action,pad,generator=torch.Generator().manual_seed(31),**labels)
        torch.testing.assert_close(first['loss'],second['loss'],rtol=0,atol=0)
        self.assertEqual(first['gesture_logits'].shape,(2,1,3))
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        second['loss'].backward();optimizer.step()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertFalse(torch.equal(before,model.denoiser.network.final_conv[-1].weight))

    def test_full_checkpoint_resume_and_single_arm_rollout_history(self):
        config=json.loads((ROOT/'config/training_diffusion.json').read_text())
        config.pop('experiment_protocol',None)
        config.update(device='cpu',precision='fp32',steps=3,batch_size=2,num_workers=0,cpu_threads=2,
            warmup_steps=1,validation_every=1,validation_batches=1,validation_sample_batches=1,
            save_every=1,log_every=1,image_size=[64,48],candidate_save_every=None)
        config['model'].update(feature_dim=16,time_dim=16,down_dims=[16,32],diffusion_steps=8,
            inference_steps=2,backbone_weights=None,crop_shape=[42,56])
        full=run(config,self.pack,'push_t',output=self.root/'full',engineering=True)
        resumed=run(config,self.pack,'push_t',output=self.root/'resumed',engineering=True,stop_after=1)
        saved=torch.load(resumed/'checkpoints/last.pt',weights_only=False,map_location='cpu')
        self.assertEqual((saved['model_config']['state_dim'],saved['model_config']['action_dim']),(9,7))
        run(saved['config'],self.pack,'push_t',resume=resumed/'checkpoints/last.pt',engineering=True)
        a=torch.load(full/'checkpoints/last.pt',weights_only=False,map_location='cpu')
        b=torch.load(resumed/'checkpoints/last.pt',weights_only=False,map_location='cpu')
        for section in ('model','ema'):
            for key in a[section]:torch.testing.assert_close(a[section][key],b[section][key],rtol=0,atol=0)
        self.assertEqual(a['scheduler'],b['scheduler']);self.assertEqual(a['ema_state'],b['ema_state'])
        # Synthetic Pack has no physical snapshots. Real physics is checked by
        # the separate recorder -> Pack -> preview engineering run.
        with patch('rollout.common.protocol.load_scene_provenance',return_value={'oblique_render_sources':{}}):
            policy=DiffusionRolloutPolicy(resumed,device='cpu',allow_engineering=True)
        image={'oblique':np.zeros((480,640,3),np.uint8)}
        state=np.zeros(9,np.float32)
        policy.observe(image,state)
        self.assertEqual(policy.predict(image,state).shape,(30,7))
        self.assertEqual(len(policy.history),2)
        with self.assertRaises(ValueError):policy.observe(image,np.zeros(16))
        policy.reset(4);self.assertFalse(policy.history)

    def test_task_execution_profiles_preserve_each_success_criterion(self):
        from scripts.run_baseline_campaign import validate_evaluations
        for task_id in ('push_t','h_frame_push_align','push_r','tray_to_platform'):
            task=get(task_id);adapter=stage_adapter(task,'training')
            adapter.validate_model('act');adapter.validate_model('diffusion')
            act,dp=task_evaluation(task),task_evaluation(task,'diffusion')
            validate_evaluations(act,dp)
            self.assertEqual(act['version'],dp['version'])
            self.assertEqual(act['task_success_criterion'],dp['task_success_criterion'])
            self.assertEqual(act['max_seconds'],dp['max_seconds'])
            stage_adapter(task,'rollout').validate_success(dict(dp,task_id=task_id))
            validate_policy_settings({'algorithm':'diffusion_policy','action_horizon':8},dp)
            self.assertEqual((dp['execution'],dp['n_action_steps'],dp['expected_checkpoint_step']),('chunked',8,100000))

    def test_explicit_shared_act_config_is_adapted_like_vscode_new_run(self):
        from training.act import train as act
        with patch.object(act,'run',return_value=Path('not_started')) as execute:
            act.main(['--task','push_t','--pack',str(self.pack),'--variant','baseline',
                      '--config',str(ROOT/'config/training_baseline.json')])
        config=execute.call_args.args[0]
        self.assertEqual((config['model']['state_dim'],config['model']['action_dim']),(9,7))
        self.assertEqual(config['steps'],60000)


if __name__ == '__main__': unittest.main()
