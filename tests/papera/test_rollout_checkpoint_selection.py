"""Formal checkpoint rules stay frozen independently of diagnostic comparisons."""
import argparse
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from rollout.rollout import validate_batch_checkpoint_selection, validate_policy_settings, checkpoint_argument
from rollout.policy import verify_inference_sources

ROOT = Path(__file__).resolve().parents[2]


class CheckpointSelectionTests(unittest.TestCase):
    def setUp(self):
        self.best = json.loads((ROOT / 'config/evaluation_baseline.json').read_text(encoding='utf-8'))
        self.last = dict(self.best, checkpoint_selection='fixed_last_training_step',
                         checkpoint_selection_reason='Chosen on independent validation seeds, before the fixed test set.',
                         expected_checkpoint_step=40000)

    def test_default_best_rule_and_file_are_preserved(self):
        record = validate_batch_checkpoint_selection(self.best, 'best',
            {'checkpoint': 'run/checkpoints/best.pt', 'checkpoint_step': 11000})
        self.assertEqual(record['rule'], 'best_validation_prior_action_l1')
        self.assertEqual(record['actual_step'], 11000)
        with self.assertRaisesRegex(ValueError, '--checkpoint best'):
            validate_batch_checkpoint_selection(self.best, 'last')

    def test_last_config_freezes_step_reason_and_actual_file(self):
        frozen = validate_batch_checkpoint_selection(self.last)
        self.assertEqual(frozen['expected_step'], 40000)
        record = validate_batch_checkpoint_selection(self.last, 'last',
            {'checkpoint': 'run/checkpoints/last.pt', 'checkpoint_step': 40000})
        self.assertEqual(record['reason'], self.last['checkpoint_selection_reason'])
        self.assertEqual(record['actual_step'], frozen['expected_step'])
        with self.assertRaisesRegex(ValueError, '--checkpoint last'):
            validate_batch_checkpoint_selection(self.last, 'best')
        with self.assertRaisesRegex(ValueError, 'differs from loaded step'):
            validate_batch_checkpoint_selection(self.last, 'last',
                {'checkpoint': 'run/checkpoints/last.pt', 'checkpoint_step': 30000})
        with self.assertRaisesRegex(ValueError, 'file differs'):
            validate_batch_checkpoint_selection(self.last, 'last',
                {'checkpoint': 'run/checkpoints/best.pt', 'checkpoint_step': 40000})

    def test_last_requires_explicit_valid_reason_and_integer_step(self):
        for reason in (None, '', '  ', 123):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, 'checkpoint_selection_reason'):
                validate_batch_checkpoint_selection(dict(self.last, checkpoint_selection_reason=reason))
        for step in (None, True, 0, -1, 40000., '40000'):
            with self.subTest(step=step), self.assertRaisesRegex(ValueError, 'expected_checkpoint_step'):
                validate_batch_checkpoint_selection(dict(self.last, expected_checkpoint_step=step))
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            validate_batch_checkpoint_selection(dict(self.last, checkpoint_selection='rollout_best'))
        with self.assertRaisesRegex(ValueError, 'only valid'):
            validate_batch_checkpoint_selection(dict(self.best, expected_checkpoint_step=40000))

    def test_dp_default_is_unchanged_and_last_is_not_implicitly_enabled(self):
        dp = json.loads((ROOT / 'config/evaluation_diffusion.json').read_text(encoding='utf-8'))
        self.assertEqual(validate_batch_checkpoint_selection(dp, 'best')['rule'], 'best_validation_ema_noise_mse')
        with self.assertRaisesRegex(ValueError, 'checkpoint_selection_reason'):
            validate_batch_checkpoint_selection(dict(dp, checkpoint_selection='fixed_last_training_step'))
        last = dict(dp, checkpoint_selection='fixed_last_training_step', expected_checkpoint_step=40000,
                    checkpoint_selection_reason='Frozen using independent validation seeds')
        policy = dict(algorithm='diffusion_policy', checkpoint='run/checkpoints/last.pt', checkpoint_step=40000,
                      action_horizon=8, weights='ema')
        self.assertEqual(validate_batch_checkpoint_selection(last, 'last', policy)['actual_step'],40000)
        validate_policy_settings(policy, last)
        with self.assertRaisesRegex(ValueError, '--checkpoint best'):
            validate_batch_checkpoint_selection(dp, 'last', policy)

    def test_retained_act_selection_freezes_index_step_reason_and_file(self):
        settings = dict(self.best, checkpoint_selection='fixed_retained_training_step',
                        expected_checkpoint_step=45000, checkpoint_selection_reason='Independent validation winner')
        policy = dict(checkpoint='run/checkpoints/step_045000.pt', checkpoint_step=45000,
                      selected_by='training/retained_candidate')
        selected = validate_batch_checkpoint_selection(settings,'step_045000',policy)
        self.assertEqual(selected['expected_step'],45000)
        for changed in (dict(policy,checkpoint_step=50000),
                        dict(policy,checkpoint='run/checkpoints/last.pt'),
                        dict(policy,selected_by='training/completed_step')):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_batch_checkpoint_selection(settings,'step_045000',changed)
        with self.assertRaisesRegex(ValueError, 'ACT only'):
            validate_batch_checkpoint_selection(dict(settings,policy_algorithm='diffusion_policy'))

    def test_cli_accepts_only_explicit_checkpoint_names(self):
        for value in ('best','last','step_045000','step_060000'):
            self.assertEqual(checkpoint_argument(value),value)
        for value in ('../last','last.pt','step_45000','step_000000','step_1000000'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                checkpoint_argument(value)

    def test_diagnostic_api_does_not_enforce_formal_checkpoint_pairing(self):
        validate_policy_settings({'checkpoint': 'run/checkpoints/last.pt', 'checkpoint_step': 40000}, self.best)

    def test_reviewed_trainer_sources_keep_the_frozen_source_record(self):
        frozen = dict(data_contract='papera-safe-stitch-act-v1', model_source_sha256='model',
                      reader_source_sha256='reader', trainer_source_sha256='old-trainer')
        report = dict(compatibility='reviewed-checkpoint-retention-only',
                      sha256={'act_model.py':'model','pack_data.py':'reader','train.py':'new-trainer'})
        with patch('training.papera.source_compatibility.verify_training_sources', return_value=report):
            mode,current,historical=verify_inference_sources(frozen)
        self.assertEqual(mode,'reviewed-checkpoint-retention-only')
        self.assertEqual(current,('model','reader','new-trainer'))
        self.assertEqual(historical,('model','reader','old-trainer'))

    def test_exact_source_report_retains_its_existing_name(self):
        frozen = dict(model_source_sha256='model', reader_source_sha256='reader', trainer_source_sha256='trainer')
        report = dict(compatibility='exact-current-sources',
                      sha256={'act_model.py':'model','data.py':'reader','train.py':'trainer'})
        with patch('training.papera.source_compatibility.verify_training_sources', return_value=report):
            mode,current,historical=verify_inference_sources(frozen)
        self.assertEqual(mode,'exact-current-training-sources')
        self.assertEqual(current,historical)


if __name__ == '__main__':
    unittest.main()
