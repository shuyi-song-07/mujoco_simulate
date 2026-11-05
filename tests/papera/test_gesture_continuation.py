"""Continuation preserves a run's frozen schedule and checkpoint learning rate.

The 40k-cosine fixture is a compatibility case, not the current ACT default.
"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from training.papera.train import (CHECKPOINT_VERSION, canonical_hash, json_write,
                                  load_extension_source, lr_multiplier, run)


class GestureContinuationTests(unittest.TestCase):
    def source(self, directory, variant='gesture', overfit=False):
        directory = Path(directory)
        (directory / 'checkpoints').mkdir()
        config = {'steps': 40000}
        binding = {'split': {'train': ['a'], 'val': ['b']}, 'stats': {'train': 'a'}}
        provenance = {'config': config, 'variant': variant, 'pack_binding': binding,
                      'engineering_overfit': overfit}
        signature = canonical_hash(provenance)
        doc = {'signature': signature, 'provenance': provenance}
        for name, value in [('run', doc), ('config', config), ('pack_binding', binding),
                            ('split', binding['split']), ('stats', binding['stats']),
                            ('completion', {'completed': True, 'step': 40000, 'target_steps': 40000})]:
            json_write(directory / f'{name}.json', value)
        checkpoint = {'version': CHECKPOINT_VERSION, 'signature': signature, 'provenance': provenance,
                      'step': 40000, 'best_prior_action_l1': .1,
                      'optimizer': {'param_groups': [{'lr': 1e-5}, {'lr': 1e-6}]}}
        torch.save(checkpoint, directory / 'checkpoints/last.pt')
        torch.save({**checkpoint, 'step': 11000}, directory / 'checkpoints/best.pt')
        return directory / 'checkpoints/last.pt', doc

    @patch('training.papera.source_compatibility.verify_training_sources')
    def test_both_full_data_variants_preserve_source_checkpoint_and_rates(self, _verify):
        for variant in ('baseline', 'gesture'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                path, doc = self.source(directory, variant)
                before = path.read_bytes()
                result = load_extension_source(path, 60000)
                self.assertEqual(result['run_doc'], doc)
                self.assertEqual(result['start_lrs'], [1e-5, 1e-6])
                self.assertEqual(result['checkpoint']['step'], 40000)
                self.assertEqual(path.read_bytes(), before)

    @patch('training.papera.source_compatibility.verify_training_sources')
    def test_unknown_or_overfit_variants_and_incomplete_budget_are_rejected(self, _verify):
        for variant, overfit in [('unsupported', False), ('gesture', True)]:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                path, _ = self.source(directory, variant, overfit)
                with self.assertRaisesRegex(ValueError, 'full-data'):
                    load_extension_source(path, 60000)
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.source(directory)
            with self.assertRaisesRegex(ValueError, 'greater'):
                load_extension_source(path, 40000)

    def test_extension_cannot_switch_model_variant(self):
        source = {'run_doc': {'provenance': {'variant': 'baseline'}}}
        with self.assertRaisesRegex(ValueError, 'cannot change.*variant'):
            run({'steps': 60000}, variant='gesture', continue_from='unused', extension_source=source)

    def test_fair_schedule_uses_original_40k_cosine_and_frozen_minimum(self):
        original = {'steps': 40000, 'warmup_steps': 500, 'minimum_lr_fraction': .1}
        gesture_first_phase = copy.deepcopy(original)
        self.assertEqual([lr_multiplier(s, original) for s in range(40001)],
                         [lr_multiplier(s, gesture_first_phase) for s in range(40001)])
        held = {**original, 'steps': 60000,
                'lr_schedule': {'kind': 'constant_checkpoint_lr', 'start_step': 40000,
                                'start_lrs': [1e-5, 1e-6]}}
        self.assertTrue(all(lr_multiplier(s, held) == 1. for s in range(20001)))
        self.assertNotEqual(lr_multiplier(20000, original),
                            lr_multiplier(20000, {**original, 'steps': 60000}))


if __name__ == '__main__':
    unittest.main()
