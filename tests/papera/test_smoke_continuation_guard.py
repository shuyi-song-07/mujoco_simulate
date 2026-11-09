"""Completed ACT engineering smoke cannot become full-data continuation.

These are small verifier fixtures, not trained policies or physical records.
They call the real extension loader and validate the current source byte hashes;
no source verifier, checkpoint loader, label, or data reader is mocked.
"""
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from training.papera.train import (
    CHECKPOINT_VERSION, canonical_hash, json_write, load_extension_source,
)

ROOT = Path(__file__).resolve().parents[2]


class SmokeContinuationGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='smoke_continuation_guard_')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def source(self, marked):
        """Freeze a completed step-3 verifier fixture and real training sources."""
        root = self.directory
        (root / 'checkpoints').mkdir()
        config = {'steps': 3}
        binding = {'split': {'train': ['train'], 'val': ['val']},
                   'stats': {'fit_split': 'train'}}
        fields = {'model_source_sha256': 'act_model.py', 'reader_source_sha256': 'pack_data.py',
                  'normalizer_source_sha256': 'data.py', 'trainer_source_sha256': 'train.py',
                  'engine_source_sha256': 'engine.py'}
        provenance = {'config': config, 'variant': 'gesture', 'pack_binding': binding,
                      'engineering_overfit': False, 'validation_scope': 'heldout_sessions',
                      **{key: hashlib.sha256((ROOT/'training/papera'/name).read_bytes()).hexdigest()
                         for key, name in fields.items()}}
        signature = canonical_hash(provenance)
        runtime = {'engineering_smoke': marked in ('all', 'runtime')}
        completion = {'completed': True, 'step': 3, 'target_steps': 3,
                      'engineering_smoke': marked in ('all', 'completion')}
        doc = {'signature': signature, 'provenance': provenance, 'runtime': runtime}
        for name, value in (('run', doc), ('config', config), ('pack_binding', binding),
                            ('split', binding['split']), ('stats', binding['stats']),
                            ('completion', completion)):
            json_write(root/f'{name}.json', value)
        checkpoint = {'version': CHECKPOINT_VERSION, 'signature': signature,
                      'provenance': provenance, 'step': 3, 'best_prior_action_l1': .1,
                      'engineering_smoke': marked in ('all', 'checkpoint'),
                      'optimizer': {'param_groups': [{'lr': 1e-5}, {'lr': 1e-6}]}}
        torch.save(checkpoint, root/'checkpoints/last.pt')
        torch.save(checkpoint, root/'checkpoints/best.pt')
        return root/'checkpoints/last.pt'

    def test_full_data_control_passes_actual_source_verification(self):
        result = load_extension_source(self.source('none'), 60000)
        self.assertEqual(result['checkpoint']['step'], 3)
        self.assertEqual(result['run_doc']['provenance']['variant'], 'gesture')

    def test_rejects_completed_smoke_with_all_three_frozen_markers(self):
        with self.assertRaisesRegex(ValueError, 'smoke'):
            load_extension_source(self.source('all'), 60000)

    def test_rejects_smoke_marker_in_runtime(self):
        with self.assertRaisesRegex(ValueError, 'smoke'):
            load_extension_source(self.source('runtime'), 60000)

    def test_rejects_smoke_marker_in_completion(self):
        with self.assertRaisesRegex(ValueError, 'smoke'):
            load_extension_source(self.source('completion'), 60000)

    def test_rejects_smoke_marker_in_checkpoint(self):
        with self.assertRaisesRegex(ValueError, 'smoke'):
            load_extension_source(self.source('checkpoint'), 60000)


if __name__ == '__main__':
    unittest.main()
