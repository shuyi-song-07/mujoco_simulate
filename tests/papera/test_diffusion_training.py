"""Complete updates, checkpoints and exact CPU continuation on real Pack tensors."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.papera.test_safe_stitch_data import make_pack
from training.diffusion.train import (ROOT, EMA, DEFAULT_EMA, CHECKPOINT_VERSION,
                                     learning_rate_factor, run, validate_config)
from training.diffusion.official.ema_model import EMAModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from training.diffusion.inference import load_policy
from training.diffusion.data import DiffusionDataset
from training.diffusion.candidates import (run_signature, candidate_due, retain_candidate,
    candidate_record, reconcile_resume_candidates)


class DiffusionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / '.local/checks/diffusion').mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT / '.local/checks/diffusion')
        cls.root = Path(cls.temp.name)
        cls.pack = make_pack(cls.root)
        cls.config = json.loads((ROOT / 'config/training_diffusion.json').read_text(encoding='utf-8'))
        cls.config.update(device='cpu', precision='fp32', steps=4, batch_size=2,
            num_workers=0, cpu_threads=2, warmup_steps=1, validation_every=2,
            validation_batches=1, validation_sample_batches=1, save_every=2,
            log_every=1, image_size=[64,48])
        cls.config['model'].update(feature_dim=16, time_dim=16, down_dims=[16,32],
            diffusion_steps=8, inference_steps=2, backbone_weights=None, observation_horizon=1,
            crop_shape=[42,56])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_update_ema_checkpoint_and_cpu_resume_equal_uninterrupted(self):
        a = run(self.config, self.pack, 'tray_transfer', output=self.root / 'uninterrupted', engineering=True)
        b = run(self.config, self.pack, 'tray_transfer', output=self.root / 'resumed', stop_after=2, engineering=True)
        initial = torch.load(b / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
        self.assertEqual(initial['step'], 2)
        self.assertEqual(initial['ema_updates'], 2)
        self.assertEqual(initial['version'], CHECKPOINT_VERSION)
        self.assertEqual(initial['ema_state']['optimization_step'], 2)
        self.assertEqual(initial['ema_state']['config'], self.config['ema'])
        # The original official EMA copies the first two parameter updates.
        self.assertEqual(initial['ema_state']['decay'], 0.)
        self.assertTrue(all(torch.equal(initial['model'][key], value)
                            for key, value in initial['ema'].items()))
        run(self.config, self.pack, 'tray_transfer', resume=b / 'checkpoints/last.pt', engineering=True)
        full = torch.load(a / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
        resumed = torch.load(b / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
        self.assertEqual(resumed['step'], 4)
        self.assertEqual(resumed['sampler']['cursor'], full['sampler']['cursor'])
        for section in ('model', 'ema'):
            for key in full[section]:
                torch.testing.assert_close(full[section][key], resumed[section][key], rtol=0, atol=0)
        self.assertEqual(resumed['scheduler'], full['scheduler'])
        self.assertEqual(resumed['ema_state'], full['ema_state'])
        self.assertEqual(json.loads((b / 'completion.json').read_text())['status'], 'complete')
        self.assertTrue((b / 'tensorboard').is_dir())
        self.assertTrue((b / 'checkpoints/best.pt').is_file())
        self.assertIn('training/diffusion/official/ema_model.py', full['source_sha256'])
        policy, normalizer, metadata = load_policy(b / 'checkpoints/best.pt')
        self.assertEqual(metadata['task_id'], 'tray_transfer')
        with torch.no_grad():
            pred = policy.predict_action(torch.zeros(1,1,1,3,48,64), torch.zeros(1,1,16),
                                         generator=torch.Generator().manual_seed(12))
        self.assertEqual(normalizer.decode_action(pred.numpy()).shape, (1,30,14))

    def test_check_only_creates_no_training_run(self):
        output = self.root / 'check_only_must_not_exist'
        result = run(self.config, self.pack, 'tray_transfer', output=output, check_only=True, engineering=True)
        self.assertEqual(result['state'], [1,16])
        self.assertEqual(result['action'], [30,14])
        self.assertFalse(output.exists())

    def test_config_rejects_invalid_budget_and_nonfinite_optimizer_values(self):
        for key, value in (('steps', 0), ('num_workers', -1), ('learning_rate', float('nan')),
                           ('ema', dict(DEFAULT_EMA, max_value=1.)), ('betas', [.9, 1.]),
                           ('gradient_clip_norm', 0.), ('gradient_clip_norm', True)):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(config)

    def test_official_ema_updates_and_scheduler_state_restore_match_reference(self):
        model = torch.nn.Linear(3, 2)
        ema = EMA(model, DEFAULT_EMA)
        reference = EMAModel(copy.deepcopy(model), **DEFAULT_EMA)
        for index in range(20):
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(.1 * (index + 1))
            ema.update(model)
            reference.step(model)
            self.assertEqual(ema.decay, reference.decay)
            self.assertEqual(ema.updates, reference.optimization_step)
            for actual, expected in zip(ema.model.parameters(), reference.averaged_model.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if index == 8:
                restored = EMA(model, DEFAULT_EMA)
                restored.model.load_state_dict(ema.model.state_dict())
                restored.load_state_dict(ema.state_dict())
                ema = restored

    def test_lr_uses_original_diffusers_warmup_and_zero_terminal_lr(self):
        config = dict(steps=40000, warmup_steps=500)
        optimizer = torch.optim.SGD([torch.zeros((), requires_grad=True)], lr=1.)
        schedule = get_cosine_schedule_with_warmup(optimizer, 500, 40000)
        for step in (0,1,499,500,1000,20000,39999,40000):
            self.assertEqual(learning_rate_factor(step, config), schedule.lr_lambdas[0](step))
        self.assertEqual(learning_rate_factor(0, config), 0.)
        self.assertEqual(learning_rate_factor(40000, config), 0.)

    def test_old_local_checkpoint_and_config_are_rejected_explicitly(self):
        old = self.root/'legacy_local.pt'
        torch.save({'version':'papera-diffusion-checkpoint-v1', 'task_id':'tray_transfer'}, old)
        with self.assertRaisesRegex(ValueError, 'older local DP'):
            run(self.config, self.pack, 'tray_transfer', resume=old, engineering=True)
        for name, value in (('ema_decay', .999), ('minimum_lr_fraction', .1)):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Legacy local DP'):
                validate_config(dict(self.config, **{name:value}))

    def test_gesture_training_records_masked_metrics_and_candidate(self):
        config = copy.deepcopy(self.config)
        config.update(steps=2, variant='gesture', candidate_save_every=2)
        config['model']['variant'] = 'gesture'
        original_getitem = DiffusionDataset.__getitem__
        def labelled_engineering_item(dataset, index):
            item = original_getitem(dataset, index)
            # The scripted physical fixture has no human records. Inject known
            # synthetic labels at the dataset seam to test trainer plumbing.
            item['gesture'] = np.array([index % 3, (index+1) % 3], dtype=np.int64)
            item['gesture_mask'] = np.array([True, index % 2 == 0], dtype=bool)
            return item
        with patch.object(DiffusionDataset, '__getitem__', labelled_engineering_item):
            output = run(config, self.pack, 'tray_transfer', output=self.root/'gesture', engineering=True)
        saved = torch.load(output/'checkpoints/last.pt', map_location='cpu', weights_only=False)
        self.assertTrue(any(key.startswith('gesture_head.') for key in saved['ema']))
        self.assertTrue((output/'checkpoints/step_000002.pt').is_file())
        self.assertIn('training/gesture_auxiliary.py', saved['source_sha256'])
        metrics = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
        validation = [row for row in metrics if row['event'] == 'validation'][-1]
        self.assertGreater(validation['gesture_valid_hands'], 0)
        self.assertEqual(sum(sum(sum(row) for row in side) for side in validation['gesture_confusion']),
                         validation['gesture_valid_hands'])
        self.assertGreater(validation['gesture_loss'], 0)
        model, _, _ = load_policy(output/'checkpoints/best.pt')
        with torch.no_grad():
            action = model.predict_action(torch.zeros(1,1,1,3,48,64), torch.zeros(1,1,16),
                generator=torch.Generator().manual_seed(12))
        self.assertEqual(action.shape, (1,30,14))

    def test_retention_leaves_updates_rng_and_resume_exactly_unchanged(self):
        plain = copy.deepcopy(self.config)
        plain['candidate_save_every'] = None
        retained = dict(plain, candidate_save_every=2)
        a = run(plain, self.pack, 'tray_transfer', output=self.root/'plain_retention_control', engineering=True)
        b = run(retained, self.pack, 'tray_transfer', output=self.root/'retention_resume',
                stop_after=2, engineering=True)
        run(retained, self.pack, 'tray_transfer', resume=b/'checkpoints/last.pt', engineering=True)
        expected = torch.load(a/'checkpoints/last.pt', map_location='cpu', weights_only=False)
        actual = torch.load(b/'checkpoints/last.pt', map_location='cpu', weights_only=False)

        def exact(left, right):
            if torch.is_tensor(left):
                self.assertTrue(torch.equal(left, right))
            elif isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    exact(left[key], right[key])
            elif isinstance(left, (tuple, list)):
                self.assertEqual(len(left), len(right))
                for x, y in zip(left, right):
                    exact(x, y)
            else:
                self.assertEqual(left, right)

        for name in ('model', 'ema', 'optimizer', 'scheduler', 'sampler', 'rng',
                     'loader_generator', 'ema_state'):
            exact(expected[name], actual[name])
        index = json.loads((b/'checkpoints/candidates.json').read_text(encoding='utf-8'))
        self.assertEqual([row['saved_step'] for row in index['candidates']], [2, 4])


class DiffusionCandidateIndexTests(unittest.TestCase):
    """Run-bound candidate bookkeeping; no network or training fixtures."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'checkpoints').mkdir()
        self.config = {'steps': 5, 'candidate_save_every': 2}
        self.sources = {'training/diffusion/model.py': 'source-identity'}
        self.signature = run_signature(self.root, 'task', self.root/'pack', self.config, self.sources)
        self.saved = []

    def save(self, name):
        self.saved.append(name)
        (self.root/'checkpoints'/name).write_bytes(b'complete-checkpoint-fixture')

    def record(self, name='step_000002'):
        return candidate_record(self.root, name, self.signature, self.config)

    def test_cadence_final_step_and_validation_are_not_invented(self):
        self.assertIsNone(retain_candidate(self.root, self.save, self.config, 1, self.signature))
        retain_candidate(self.root, self.save, self.config, 2, self.signature)
        self.assertIsNone(self.record()['validation'])
        validation = {'step': 4, 'event': 'validation', 'ema': True, 'noise_mse': .2}
        retain_candidate(self.root, self.save, self.config, 4, self.signature, validation)
        retain_candidate(self.root, self.save, self.config, 5, self.signature)
        self.assertEqual(self.saved, ['step_000002.pt', 'step_000004.pt', 'step_000005.pt'])
        self.assertEqual(self.record('step_000004')['validation'], validation)
        self.assertTrue(candidate_due(self.config, 5))
        self.assertFalse(candidate_due(dict(self.config, candidate_save_every=None), 5))

    def test_signature_changes_for_run_task_pack_config_or_source(self):
        args = [self.root, 'task', self.root/'pack', self.config, self.sources]
        for index, different in enumerate([self.root/'other', 'other', self.root/'other_pack',
                                          dict(self.config, steps=6), {'model.py': 'other'}]):
            values = args.copy()
            values[index] = different
            self.assertNotEqual(run_signature(*values), self.signature)

    def test_invalid_name_index_size_and_duplicate_are_rejected(self):
        retain_candidate(self.root, self.save, self.config, 2, self.signature)
        for name in ('../last', 'step_000002.pt', 'step_000000', 'step_000001',
                     'step_000006', 'step_000004'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.record(name)
        with self.assertRaises(ValueError):
            candidate_record(self.root, 'step_000002', 'another-run', self.config)
        path = self.root/'checkpoints/step_000002.pt'
        path.write_bytes(b'truncated')
        with self.assertRaisesRegex(ValueError, 'size'):
            self.record()
        self.save(path.name)
        index_path = self.root/'checkpoints/candidates.json'
        index = json.loads(index_path.read_text(encoding='utf-8'))
        index['candidates'].append(copy.deepcopy(index['candidates'][0]))
        index_path.write_text(json.dumps(index), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self.record()

    def test_wrong_step_and_stale_validation_are_rejected(self):
        retain_candidate(self.root, self.save, self.config, 2, self.signature)
        path = self.root/'checkpoints/candidates.json'
        original = json.loads(path.read_text(encoding='utf-8'))
        mutations = [lambda row: row.update(saved_step=4),
                     lambda row: row.update(validation_step=2),
                     lambda row: row.update(validation={'step': 1, 'event': 'validation', 'ema': True},
                                             validation_step=1, validation_source='metrics.jsonl:validation')]
        for mutate in mutations:
            changed = copy.deepcopy(original)
            mutate(changed['candidates'][0])
            path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaises(ValueError):
                self.record()
        with self.assertRaises(ValueError):
            retain_candidate(self.root, self.save, self.config, 4, self.signature,
                             {'step': 2, 'event': 'validation', 'ema': True})

    def test_resume_withdraws_future_index_entries_without_deleting_files(self):
        for step in (2, 4):
            retain_candidate(self.root, self.save, self.config, step, self.signature)
        reconcile_resume_candidates(self.root, self.signature, 2)
        self.assertEqual(self.record()['saved_step'], 2)
        self.assertTrue((self.root/'checkpoints/step_000004.pt').is_file())
        with self.assertRaises(ValueError):
            self.record('step_000004')
        with self.assertRaisesRegex(ValueError, 'overwritten'):
            retain_candidate(self.root, self.save, self.config, 4, self.signature)
        for name in ('best', 'last'):
            self.assertIsNone(candidate_record(self.root, name, 'legacy-no-index', {}))


if __name__ == '__main__':
    unittest.main()
