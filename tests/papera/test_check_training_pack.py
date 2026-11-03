"""The common Pack check reads boundaries and restores real policy state."""
import copy
from pathlib import Path
import tempfile
import unittest

from tests.check_training_pack import (ROOT, check_pack, engineering_updates,
    inspect_sample, load_config)
from tests.papera.test_safe_stitch_data import make_pack
from training.papera.pack_data import prepare_pack, SafeStitchDataset
from training.diffusion.data import prepare_data, DiffusionDataset


class TrainingPackCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = ROOT / '.local/checks/training-pack-tests'
        folder.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=folder)
        cls.root = Path(cls.temp.name)
        cls.pack = make_pack(cls.root)
        cls.act = load_config('training_baseline.json')
        cls.dp = load_config('training_diffusion.json')
        for config in (cls.act, cls.dp):
            config.update(image_size=[64, 48], cpu_threads=2, precision='fp32', batch_size=2)
            config['model']['backbone_weights'] = None
        cls.act['model'].update(hidden_dim=32, heads=4, feedforward_dim=64,
            encoder_layers=1, decoder_layers=1, posterior_layers=1, latent_dim=8)
        cls.dp['model'].update(feature_dim=16, time_dim=16, down_dims=[16, 32], observation_horizon=2,
            diffusion_steps=8, inference_steps=2)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_read_only_checks_both_histories_and_keeps_pack_files_unchanged(self):
        before = {str(path.relative_to(self.pack)): (path.stat().st_size, path.stat().st_mtime_ns)
                  for path in self.pack.rglob('*') if path.is_file()}
        report = check_pack(self.pack, 'tray_transfer', act_config=self.act, dp_config=self.dp)
        self.assertFalse(report['formal_training_started'])
        self.assertFalse(report['training_started'])
        self.assertEqual(report['act']['train']['action'], [30, 14])
        self.assertEqual(report['dp']['H1']['train']['images'], [1, 1, 3, 48, 64])
        self.assertEqual(report['dp']['H2']['train']['images'], [2, 1, 3, 48, 64])
        self.assertGreater(report['counts']['train']['safe_stitches'], 0)
        self.assertGreater(report['counts']['train']['hard_boundaries'], 0)
        after = {str(path.relative_to(self.pack)): (path.stat().st_size, path.stat().st_mtime_ns)
                 for path in self.pack.rglob('*') if path.is_file()}
        self.assertEqual(before, after)

    def test_wrong_task_and_inconsistent_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Cross-task'):
            check_pack(self.pack, 'another-task', act_config=self.act, dp_config=self.dp)
        bad = copy.deepcopy(self.dp)
        bad['model']['action_dim'] = 13
        with self.assertRaisesRegex(ValueError, 'S16/A14'):
            check_pack(self.pack, 'tray_transfer', act_config=self.act, dp_config=bad)

    def test_padding_cannot_carry_targets_and_provenance_cannot_cross_boundary(self):
        dataset = SafeStitchDataset(prepare_pack(self.pack, image_size=[64, 48]))
        self.addCleanup(dataset.close)
        index = len(dataset) - 1
        sample = copy.deepcopy(dataset[index])
        sample['action'][-1, 0] = 1
        with self.assertRaisesRegex(ValueError, 'Padding carries'):
            inspect_sample(dataset, index, sample=sample)
        sample = copy.deepcopy(dataset[index])
        sample['target_raw_frame_index'][-1] = 999
        with self.assertRaises(AssertionError):
            inspect_sample(dataset, index, sample=sample)

    def test_history_cannot_read_before_the_hard_boundary(self):
        binding = prepare_data(self.pack, 'tray_transfer', {'observation_horizon': 2,
            'prediction_horizon': 30, 'action_horizon': 8, 'image_size': [64, 48]}, engineering=True)
        dataset = DiffusionDataset(binding)
        self.addCleanup(dataset.close)
        sample = copy.deepcopy(dataset[0])
        sample['observation_raw_frame_index'][0] = -1
        with self.assertRaises(AssertionError):
            inspect_sample(dataset, 0, history=2, sample=sample)

    def test_engineering_budget_and_output_guard_reject_long_or_formal_runs(self):
        for count in (4, 40000, -1, True):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, '0..3'):
                check_pack(self.pack, 'tray_transfer', updates=count)
        with self.assertRaisesRegex(ValueError, 'Read-only'):
            check_pack(self.pack, 'tray_transfer', output=self.root / 'unused')
        with self.assertRaisesRegex(ValueError, '.local/checks'):
            engineering_updates('act', None, self.act, ROOT / 'outputs/invalid-check',
                                updates=1, device='cpu')

    def test_real_act_and_dp_updates_restore_checkpoint_then_continue(self):
        report = check_pack(self.pack, 'tray_transfer', act_config=self.act, dp_config=self.dp,
            updates=2, output=self.root / 'updates', device='cpu')
        for algorithm in ('act', 'dp'):
            evidence = report[algorithm]['engineering']
            self.assertEqual(evidence['updates'], 2)
            self.assertEqual(evidence['resumed_updates'], 1)
            self.assertTrue(evidence['checkpoint_restore_equal'])
            self.assertTrue(Path(evidence['checkpoint']).is_file())
            self.assertFalse(evidence['deployable'])
            self.assertEqual(evidence['batch_size'], 2)
        self.assertEqual(report['dp']['engineering']['observation_horizon'], 2)
        self.assertEqual(report['act']['engineering']['observation_horizon'], 1)
        self.assertEqual(report['configured_models']['dp']['observation_horizon'], 2)
        self.assertTrue(report['training_started'])
        self.assertFalse(report['formal_training_started'])


if __name__ == '__main__':
    unittest.main()
