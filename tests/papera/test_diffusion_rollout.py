"""DP EMA selection and observation-clock history on the shared rollout path."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.papera.test_safe_stitch_data import make_pack
from training.diffusion.train import ROOT, run
from rollout.diffusion_policy import DiffusionRolloutPolicy, verify_run, inference_metadata
from rollout.policy import read_run_identity, load_policy, verify_run_provenance
from rollout.rollout import validate_policy_settings, validate_settings
from rollout.action_chunks import ActionChunkScheduler


class DiffusionRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'.local/checks/diffusion').mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT/'.local/checks/diffusion')
        cls.root = Path(cls.temp.name)
        cls.pack = make_pack(cls.root)
        cls.config = json.loads((ROOT/'config/training_diffusion.json').read_text())
        cls.config.update(device='cpu', precision='fp32', steps=2, batch_size=2,
            num_workers=0, cpu_threads=2, warmup_steps=0, validation_every=1,
            validation_batches=1, validation_sample_batches=0, save_every=1,
            log_every=1, image_size=[64,48], candidate_save_every=1)
        cls.config['model'].update(feature_dim=16, time_dim=16, down_dims=[16,32],
            diffusion_steps=8, inference_steps=2, backbone_weights=None, observation_horizon=2,
            crop_shape=[42,56])
        cls.run_dir = run(cls.config, cls.pack, 'tray_transfer', output=cls.root/'trained', engineering=True)
        cls.scene = {'oblique_render_sources': {'native': {'camera': {'test': 'frozen'}}}}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def policy(self):
        with patch('rollout.protocol.load_scene_provenance', return_value=self.scene):
            return load_policy(self.run_dir, device='cpu', allow_overfit=True)

    def test_ema_best_dispatch_and_scope(self):
        identity = read_run_identity(self.run_dir)
        self.assertEqual(identity['variant'], 'diffusion')
        self.assertEqual(identity['task_id'], 'tray_transfer')
        with self.assertRaisesRegex(ValueError, 'Engineering'):
            verify_run(self.run_dir)
        with patch('rollout.protocol.load_scene_provenance', return_value=self.scene):
            metadata = verify_run_provenance(self.run_dir, verify_cache=False, allow_overfit=True)
        self.assertEqual(metadata['weights'], 'ema')
        self.assertEqual(metadata['action_horizon'], 8)
        saved = torch.load(self.run_dir/'checkpoints/best.pt', map_location='cpu', weights_only=False)
        policy = self.policy()
        for key, value in policy.model.state_dict().items():
            torch.testing.assert_close(value, saved['ema'][key], rtol=0, atol=0)
        self.assertFalse(policy.model.training)

    def test_gesture_identity_routes_separately_from_historical_baseline(self):
        path = self.run_dir/'run.json'
        original = path.read_text(encoding='utf-8')
        run = json.loads(original)
        run['config']['variant'] = 'gesture'
        run['config']['model']['variant'] = 'gesture'
        try:
            path.write_text(json.dumps(run), encoding='utf-8')
            identity = read_run_identity(self.run_dir)
            self.assertEqual(identity['variant'], 'diffusion_gesture')
            self.assertEqual(identity['training_variant'], 'gesture')
        finally:
            path.write_text(original,encoding='utf-8')

    def test_bad_checkpoint_step_rejected_before_inference(self):
        path = self.run_dir/'checkpoints/best.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        bad = dict(saved, step=0)
        try:
            torch.save(bad, path)
            with self.assertRaisesRegex(ValueError, 'selection'):
                verify_run(self.run_dir, allow_engineering=True)
        finally:
            torch.save(saved, path)

    def test_retained_step_uses_its_own_ema_without_changing_best_last(self):
        with patch('rollout.protocol.load_scene_provenance', return_value=self.scene):
            metadata, saved, _ = verify_run(self.run_dir, 'step_000001', allow_engineering=True)
        self.assertEqual(metadata['checkpoint_step'], 1)
        self.assertEqual(metadata['selected_by'], 'training/retained_candidate')
        self.assertEqual(metadata['weights'], 'ema')
        self.assertEqual(saved['ema_updates'], 1)
        index = json.loads((self.run_dir/'checkpoints/candidates.json').read_text(encoding='utf-8'))
        entry = next(row for row in index['candidates'] if row['saved_step'] == 1)
        self.assertEqual(entry['checkpoint'], 'step_000001.pt')
        self.assertEqual(entry['validation']['step'], 1)
        self.assertEqual(entry['file_size_bytes'], (self.run_dir/'checkpoints/step_000001.pt').stat().st_size)
        for name in ('best', 'last'):
            with patch('rollout.protocol.load_scene_provenance', return_value=self.scene):
                self.assertTrue(verify_run(self.run_dir, name, allow_engineering=True)[0]['weights'] == 'ema')

    def test_historical_best_last_do_not_require_index_or_candidate_signature(self):
        index = self.run_dir/'checkpoints/candidates.json'
        backup = index.with_suffix('.test-backup')
        index.rename(backup)
        try:
            for name in ('best', 'last'):
                path = self.run_dir/'checkpoints'/f'{name}.pt'
                saved = torch.load(path, map_location='cpu', weights_only=False)
                legacy = dict(saved)
                legacy.pop('candidate_signature', None)
                try:
                    torch.save(legacy, path)
                    with patch('rollout.protocol.load_scene_provenance', return_value=self.scene):
                        self.assertEqual(verify_run(self.run_dir, name, allow_engineering=True)[0]['weights'], 'ema')
                finally:
                    torch.save(saved, path)
        finally:
            backup.rename(index)

    def test_candidate_payload_identity_step_and_completion_are_required(self):
        path = self.run_dir/'checkpoints/step_000001.pt'
        original = torch.load(path, map_location='cpu', weights_only=False)
        index_path = self.run_dir/'checkpoints/candidates.json'
        index_text = index_path.read_text(encoding='utf-8')
        index = json.loads(index_text)
        mutations = [dict(original, candidate_signature='another-run'),
                     dict(original, step=2), dict(original, ema_updates=0),
                     dict(original, source_sha256={'different': 'source'})]
        try:
            for saved in mutations:
                torch.save(saved, path)
                # Keep file-size evidence current to exercise payload checks,
                # not merely the earlier truncated-file rejection.
                index['candidates'][0]['file_size_bytes'] = path.stat().st_size
                index_path.write_text(json.dumps(index), encoding='utf-8')
                with self.assertRaises(ValueError):
                    verify_run(self.run_dir, 'step_000001', allow_engineering=True)
        finally:
            torch.save(original, path)
            index['candidates'][0]['file_size_bytes'] = path.stat().st_size
            index_path.write_text(json.dumps(index), encoding='utf-8')
        completion_path = self.run_dir/'completion.json'
        completion_text = completion_path.read_text(encoding='utf-8')
        completion = json.loads(completion_text)
        try:
            completion['status'] = 'stopped_after_requested_updates'
            completion_path.write_text(json.dumps(completion), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'completed training budget'):
                verify_run(self.run_dir, 'step_000001', allow_engineering=True)
        finally:
            completion_path.write_text(completion_text, encoding='utf-8')

    def test_legacy_baseline_model_defaults_remain_loadable(self):
        run_path = self.run_dir/'run.json'
        checkpoint_path = self.run_dir/'checkpoints/best.pt'
        original_run = run_path.read_text(encoding='utf-8')
        original_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        legacy_run = json.loads(original_run)
        legacy_checkpoint = copy.deepcopy(original_checkpoint)
        for field in ('variant', 'gesture_weight'):
            legacy_run['model_config'].pop(field, None)
            legacy_checkpoint['model_config'].pop(field, None)
        try:
            run_path.write_text(json.dumps(legacy_run), encoding='utf-8')
            torch.save(legacy_checkpoint, checkpoint_path)
            policy = self.policy()
            self.assertEqual(policy.model.config.variant, 'baseline')
            self.assertFalse(hasattr(policy.model, 'gesture_head'))
        finally:
            run_path.write_text(original_run, encoding='utf-8')
            torch.save(original_checkpoint, checkpoint_path)

    def test_observation_history_advances_between_queries_and_resets(self):
        policy = self.policy()
        observed = []
        def fake_predict(images, states, generator):
            observed.append((images.clone(), states.clone()))
            return torch.arange(30, dtype=torch.float32).view(1,30,1).expand(1,30,14)/30
        policy.model.predict_action = fake_predict
        settings = {'execution':'chunked', 'n_action_steps':8, 'chunk_size':30}
        scheduler = ActionChunkScheduler.from_settings(settings)
        for frame in range(9):
            image = {'oblique':np.full((480,640,3),frame,np.uint8)}
            state = np.full(16, frame, np.float32)
            policy.observe(image,state)
            action, meta = scheduler.next_action(frame, lambda: policy.predict(image,state))
            self.assertEqual(meta['selected_offset'], frame % 8)
            self.assertEqual(action.shape,(14,))
        self.assertEqual(len(observed),2)
        np.testing.assert_allclose(observed[0][0][0,:,0,0,0,0].numpy(), [0,0])
        np.testing.assert_allclose(observed[1][0][0,:,0,0,0,0].numpy(), np.array([7,8])/255)
        np.testing.assert_allclose(observed[1][1][0].numpy(),
            policy.normalizer.encode_state(np.stack([np.full(16,7),np.full(16,8)])))
        policy.reset(12)
        self.assertEqual(len(policy.history),0)
        with self.assertRaisesRegex(RuntimeError,'history'):
            policy.predict(image,state)

    def test_real_ddpm_is_reproducible_after_seed_reset(self):
        policy = self.policy()
        rgb = {'oblique':np.zeros((480,640,3),np.uint8)}
        state=np.zeros(16,np.float32)
        results=[]
        for _ in range(2):
            policy.reset(100)
            policy.observe(rgb,state)
            results.append(policy.predict(rgb,state))
        np.testing.assert_array_equal(*results)
        self.assertEqual(results[0].shape,(30,14))
        self.assertTrue(np.isfinite(results[0]).all())

    def test_inference_overrides_are_explicit_bounded_and_reproducible(self):
        policy = self.policy()
        settings = dict(diffusion_inference_steps=4, diffusion_action_horizon=4,
                        inference_selection_reason='fixed independent validation candidate')
        policy.configure_inference(settings)
        self.assertEqual(policy.metadata['inference_steps'], 4)
        self.assertEqual(policy.metadata['action_horizon'], 4)
        self.assertEqual(policy.metadata['training_inference_steps'], 2)
        self.assertEqual(policy.model.config.inference_steps, 2)
        self.assertEqual(policy.model.config.action_horizon, 8)
        image = {'oblique': np.zeros((480,640,3),np.uint8)}
        state = np.zeros(16,np.float32)
        policy.observe(image,state)
        observed = []
        def predict(images, states, generator, num_inference_steps):
            observed.append(num_inference_steps)
            return torch.zeros(1,30,14)
        policy.model.predict_action = predict
        policy.predict(image,state)
        self.assertEqual(observed, [4])
        for override in ({'diffusion_inference_steps': 9}, {'diffusion_action_horizon':31},
                         {'diffusion_inference_steps': True}, {'diffusion_inference_steps':3}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                inference_metadata(policy.metadata, override)

    def test_shared_settings_validate_explicit_dp_overrides(self):
        policy = self.policy()
        settings = json.loads((ROOT/'config/evaluation_diffusion.json').read_text())
        settings.update(diffusion_inference_steps=4, diffusion_action_horizon=4,n_action_steps=4,
                        inference_selection_reason='Independent validation candidate')
        validate_policy_settings(policy.metadata,settings)
        with self.assertRaisesRegex(ValueError,'DP needs'):
            validate_policy_settings(policy.metadata,dict(settings,n_action_steps=8))
        with self.assertRaisesRegex(ValueError,'ACT'):
            validate_policy_settings({},dict(settings,policy_algorithm='act'))

    def test_dp_settings_require_frozen_horizon_and_physics_matches_act(self):
        settings=json.loads((ROOT/'config/evaluation_diffusion.json').read_text())
        original=json.loads((ROOT/'tasks/tray_to_platform/evaluation.json').read_text())
        allowed={'policy_algorithm','execution','n_action_steps','checkpoint_selection'}
        self.assertEqual({k:v for k,v in settings.items() if k not in allowed},
                         {k:v for k,v in original.items() if k not in allowed})
        validate_settings(settings)
        meta={'algorithm':'diffusion_policy','action_horizon':8}
        validate_policy_settings(meta,settings)
        for field,value in [('n_action_steps',1),('execution','temporal_ensemble')]:
            with self.assertRaisesRegex(ValueError,'DP needs'):
                validate_policy_settings(meta,dict(settings,**{field:value}))
        with self.assertRaisesRegex(ValueError,'ACT'):
            validate_policy_settings({},settings)

    def test_bad_inputs_and_unfinished_run_rejected(self):
        policy=self.policy()
        with self.assertRaisesRegex(ValueError,'exactly'):
            policy.observe({'top':np.zeros((480,640,3),np.uint8)},np.zeros(16))
        with self.assertRaisesRegex(ValueError,'S16'):
            policy.observe({'oblique':np.zeros((480,640,3),np.uint8)},np.full(16,np.nan))
        path=self.run_dir/'completion.json'
        original=path.read_text()
        try:
            completion=json.loads(original)
            completion['status']='interrupted'
            path.write_text(json.dumps(completion))
            with self.assertRaisesRegex(ValueError,'completed'):
                verify_run(self.run_dir,allow_engineering=True)
        finally:
            path.write_text(original)


if __name__=='__main__':
    unittest.main()
