"""A reviewed engine extraction cannot change a frozen Panda experiment."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.papera import source_compatibility as compatibility


def digest(value):
    return hashlib.sha256(value).hexdigest()


class ReferenceSourceCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        files = {'act_model.py': b'reviewed model with explicit reference arguments',
                 'train.py': b'reviewed trainer using extracted update engine',
                 'engine.py': b'exact reviewed shared update engine',
                 'pack_data.py': b'unchanged causal task reader',
                 'data.py': b'unchanged training-only normalization'}
        for name, value in files.items():
            (self.directory / name).write_bytes(value)
        self.current = {'model_source_sha256': digest(files['act_model.py']),
                        'trainer_source_sha256': digest(files['train.py']),
                        'reader_source_sha256': digest(files['pack_data.py']),
                        'normalizer_source_sha256': digest(files['data.py']),
                        'engine_source_sha256': digest(files['engine.py']),
                        'pack': 'fixed/task/pack', 'variant': 'baseline',
                        'config': {'model': {'state_dim': 16, 'chunk_size': 30},
                                   'learning_rate': 1e-5, 'steps': 40000},
                        'pack_binding': {'split': {'train': ['a'], 'val': ['b']},
                                         'stats': {'state_mean': [0.]}}}
        self.previous = copy.deepcopy(self.current)
        self.previous.pop('engine_source_sha256')
        self.previous['model_source_sha256'] = digest(b'old same-Panda graph')
        self.previous['trainer_source_sha256'] = digest(b'old inline same-Panda updates')
        reason = 'reviewed-reference-profile-shared-engine-fixture'
        pairs = {(name, self.previous[field], self.current[field]): reason
                 for name, field in (('act_model.py', 'model_source_sha256'),
                                     ('train.py', 'trainer_source_sha256'))}
        self.bundle = {
            'old_model_sha256': self.previous['model_source_sha256'],
            'new_model_sha256': self.current['model_source_sha256'],
            'engine_sha256': self.current['engine_source_sha256'], 'reason': reason}
        trainer_pair = (self.previous['trainer_source_sha256'], self.current['trainer_source_sha256'])
        for mock in (
            patch.object(compatibility, '__file__', str(self.directory / 'source_compatibility.py')),
            patch.dict(compatibility.REVIEWED_SOURCE_PAIRS, pairs, clear=True),
            patch.dict(compatibility.REVIEWED_ENGINE_MIGRATIONS, {trainer_pair: self.bundle}, clear=True),
        ):
            mock.start()
            self.addCleanup(mock.stop)

    def test_exact_new_sources_and_reviewed_old_sources_preserve_frozen_identity(self):
        self.assertEqual(compatibility.verify_training_sources(self.current)['compatibility'],
                         'exact-current-sources')
        report = compatibility.verify_training_sources(self.previous)
        self.assertEqual(report['compatibility'], 'reviewed-training-interface-only')
        self.assertEqual(report['sha256']['engine.py'], self.current['engine_source_sha256'])
        before_current, before_previous = copy.deepcopy(self.current), copy.deepcopy(self.previous)
        restored = compatibility.compatible_resume_provenance(self.current, self.previous)
        self.assertEqual(restored, self.previous)
        self.assertIsNot(restored, self.previous)
        self.assertEqual(self.current, before_current)
        self.assertEqual(self.previous, before_previous)

    def test_source_pairs_without_the_exact_engine_bundle_are_insufficient(self):
        with patch.dict(compatibility.REVIEWED_ENGINE_MIGRATIONS, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Frozen training source changed'):
                compatibility.verify_training_sources(self.previous)
            with self.assertRaisesRegex(ValueError, 'Checkpoint provenance changed'):
                compatibility.compatible_resume_provenance(self.current, self.previous)
        with patch.dict(compatibility.REVIEWED_SOURCE_PAIRS, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Frozen training source changed'):
                compatibility.verify_training_sources(self.previous)

    def test_a_future_engine_change_is_not_approved_by_this_migration(self):
        (self.directory / 'engine.py').write_bytes(b'a later unreviewed math change')
        for frozen in (self.previous, self.current):
            with self.subTest(frozen_has_engine='engine_source_sha256' in frozen):
                with self.assertRaisesRegex(ValueError, 'Frozen training source changed: engine.py'):
                    compatibility.verify_training_sources(frozen)
        with self.assertRaisesRegex(ValueError, 'engine.py'):
            compatibility.compatible_resume_provenance(self.current, self.previous)

    def test_new_metadata_cannot_omit_or_forge_its_engine(self):
        missing = copy.deepcopy(self.current)
        missing.pop('engine_source_sha256')
        wrong = {**self.current, 'engine_source_sha256': digest(b'another engine')}
        for bad in (missing, wrong):
            with self.subTest(has_engine='engine_source_sha256' in bad):
                with self.assertRaisesRegex(ValueError, 'Frozen training source changed: engine.py'):
                    compatibility.verify_training_sources(bad)
        (self.directory / 'engine.py').unlink()
        with self.assertRaisesRegex(ValueError, 'engine.py'):
            compatibility.verify_training_sources(self.current)

    def test_reviewed_source_change_cannot_change_data_model_stats_or_variant(self):
        changes = [
            {'pack': 'another/pack'}, {'variant': 'gesture'},
            {'config': {**self.current['config'], 'steps': 40001}},
            {'config': {**self.current['config'], 'learning_rate': 1e-4}},
            {'config': {**self.current['config'], 'model': {'state_dim': 14, 'chunk_size': 100}}},
            {'pack_binding': {'split': {'train': ['b'], 'val': ['a']}, 'stats': {'state_mean': [0.]}}},
            {'pack_binding': {'split': {'train': ['a'], 'val': ['b']}, 'stats': {'state_mean': [1.]}}},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'Checkpoint provenance changed'):
                compatibility.compatible_resume_provenance({**self.current, **change}, self.previous)

    def test_existing_checkpoint_bytes_and_signature_need_no_rewrite(self):
        import torch
        from training.papera.train import CHECKPOINT_VERSION, canonical_hash, load_checkpoint
        path = self.directory / 'last.pt'
        signature = canonical_hash(self.previous)
        torch.save({'version': CHECKPOINT_VERSION, 'signature': signature,
                    'provenance': self.previous, 'step': 300,
                    'model': {'example': torch.ones(2)}}, path)
        before = path.read_bytes()
        migrated = compatibility.compatible_resume_provenance(self.current, self.previous)
        checkpoint = load_checkpoint(path, canonical_hash(migrated))
        self.assertEqual(checkpoint['step'], 300)
        self.assertEqual(checkpoint['signature'], signature)
        self.assertEqual(checkpoint['provenance'], self.previous)
        self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
