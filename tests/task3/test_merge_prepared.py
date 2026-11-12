"""A Prepared union preserves original sources and never restores exclusions."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from dataset.task3.merge_prepared import merge_prepared, validate_manifest
from tests.task3.merge_prepared_fixture import source_prepared_fixture


class MergePreparedTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='prepared_union_')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.fixture = source_prepared_fixture(self.root / 'sources')
        self.inputs = self.fixture.__enter__()
        self.addCleanup(self.fixture.__exit__, None, None, None)

    def merge(self, **kwargs):
        return merge_prepared(self.inputs['roots'], self.root / 'merged',
                              split_file=self.inputs['split'], **kwargs)

    def change_source(self, number, callback):
        root = self.inputs['roots'][number]
        path = root / 'manifest.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        callback(value)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

    def change_curation_config(self, number, callback):
        from dataset.task3.prepare_paper_dataset import digest
        from dataset.task3.curation_config import effective_config

        def update(manifest):
            callback(manifest['config'])
            manifest['config_sha256'] = digest(manifest['config'])
            manifest['effective_config_sha256'] = digest(effective_config(manifest['config']))
            (self.inputs['roots'][number] / 'parameters.json').write_text(
                json.dumps(manifest['config']), encoding='utf-8')
        self.change_source(number, update)

    def test_union_keeps_frozen_files_original_bindings_and_global_split(self):
        with patch('dataset.task3.training_pack.cache_video', side_effect=AssertionError('must not decode RGB')):
            result = self.merge()
        self.assertEqual(len(result['episodes']), 4)
        self.assertEqual(result['task_binding'], self.inputs['canonical_binding'])
        self.assertEqual(result['source_task_bindings'],
                         [self.inputs['canonical_binding'], self.inputs['historical_binding']])
        self.assertEqual(len(result['dataset_collection_compatibility']['split']['train']), 2)
        self.assertEqual(len(result['dataset_collection_compatibility']['split']['val']), 2)
        for episode in result['episodes']:
            source = Path(episode['source_prepared'])
            original = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(episode['source_task_binding'], original['task_binding'])
            old = next(ep for ep in original['episodes'] if ep['key'] == episode['key'])
            for field in ('arrays_path', 'index_path', 'sequence_path'):
                self.assertEqual((self.root / 'merged' / episode[field]).read_bytes(),
                                 (source / old[field]).read_bytes())
            self.assertEqual(episode['video_sources'], old['video_sources'])
        self.assertEqual(validate_manifest(result), result)

    def test_one_source_can_be_normalized_into_a_learning_domain(self):
        result = merge_prepared(self.inputs['roots'][:1], self.root / 'single')
        self.assertEqual(len(result['episodes']), 2)
        self.assertEqual(len(result['source_task_bindings']), 1)

    def test_duplicate_keys_are_rejected_before_output_is_created(self):
        root = self.inputs['roots'][0]
        with self.assertRaisesRegex(ValueError, 'distinct'):
            merge_prepared([root, root], self.root / 'merged')
        self.assertFalse((self.root / 'merged').exists())

    def test_duplicate_episode_keys_in_distinct_prepared_directories_are_rejected(self):
        duplicate = self.root / 'duplicate'
        shutil.copytree(self.inputs['roots'][0], duplicate)
        with self.assertRaisesRegex(ValueError, 'Duplicate episode keys'):
            merge_prepared([self.inputs['roots'][0], duplicate], self.root / 'merged')
        self.assertFalse((self.root / 'merged').exists())

    def test_scene_or_camera_or_schema_binding_changes_are_rejected(self):
        for field, value in (('scene_id', 'FOREIGN'), ('schema', {}), ('task_version', 'different')):
            original = (self.inputs['roots'][1] / 'manifest.json').read_text(encoding='utf-8')
            self.change_source(1, lambda manifest: manifest['task_binding'].update({field: value}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'domain differs'):
                self.merge()
            (self.inputs['roots'][1] / 'manifest.json').write_text(original, encoding='utf-8')
        self.change_source(1, lambda manifest: manifest.update(cameras=['front']))
        with self.assertRaisesRegex(ValueError, 'domain differs'):
            self.merge()

    def test_real_metadata_must_match_the_prepared_binding(self):
        source = json.loads((self.inputs['roots'][1] / 'manifest.json').read_text(encoding='utf-8'))
        path = Path(source['episodes'][0]['source_ref']['metadata_path'])
        value = json.loads(path.read_text(encoding='utf-8'))
        value['task_binding']['scene_id'] = 'tampered'
        path.write_text(json.dumps(value), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'metadata changed'):
            self.merge()
        self.assertFalse((self.root / 'merged').exists())

    def test_fabricated_historical_proof_cannot_authorize_other_configuration(self):
        self.change_source(1, lambda manifest: manifest['recorded_collection_compatibility'].update(
            recorded_collection_sha256='f'*64))
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            self.merge()

    def test_fingerprint_consistent_camera_change_is_still_rejected(self):
        def update(manifest):
            entry = manifest['episodes'][0]
            path = Path(entry['source_ref']['metadata_path'])
            metadata = json.loads(path.read_text(encoding='utf-8'))
            metadata['config']['training_camera']['distance'] = 9.
            path.write_text(json.dumps(metadata), encoding='utf-8')
            fingerprint = next(f for f in entry['source_fingerprints'] if Path(f['path']) == path)
            fingerprint.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=path.stat().st_size)
        self.change_source(1, update)
        with self.assertRaisesRegex(ValueError, 'inconsistent frozen|scene/camera/physical'):
            self.merge()

    def test_safe_stitch_parameters_cannot_differ(self):
        self.change_curation_config(1, lambda config: config['sa']['thresholds'].update(
            action_position_m=config['sa']['thresholds']['action_position_m']*2))
        with self.assertRaisesRegex(ValueError, 'Safe Stitch/curation'):
            self.merge()

    def test_configured_exclusion_cannot_return_as_an_included_episode(self):
        self.change_curation_config(0, lambda config: config['quality']['excluded_episodes'].append(
            {'episode_key': 'current_train_session/episode_000000', 'reason': 'explicit user rejection'}))
        with self.assertRaisesRegex(ValueError, 'excluded episode cannot re-enter'):
            self.merge()

    def test_master_split_cannot_overlap_or_omit_sessions(self):
        path = self.inputs['split']
        path.write_text(json.dumps({'train_sessions': ['current_train_session'],
                                   'val_sessions': ['current_train_session']}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            self.merge()

    def test_episode_source_mapping_tampering_is_rejected_offline(self):
        result = self.merge()
        result['episodes'][0]['source_task_binding'] = self.inputs['historical_binding']
        with self.assertRaisesRegex(ValueError, 'association differs'):
            validate_manifest(result)

    def test_offline_union_validation_does_not_open_raw_or_prepared_files(self):
        result = self.merge()
        with patch.object(Path, 'open', side_effect=AssertionError('offline metadata check opened a file')):
            self.assertEqual(validate_manifest(result), result)

    def test_historical_only_input_uses_frozen_registered_recording_reference(self):
        with patch('collection.config.recording_config', return_value=self.inputs['reference_collection_config']):
            result = merge_prepared(self.inputs['roots'][1:], self.root / 'historical_only')
        self.assertEqual(result['task_binding'], self.inputs['canonical_binding'])
        self.assertEqual(result['source_task_bindings'], [self.inputs['historical_binding']])
        self.assertEqual(result['dataset_collection_compatibility']['reference_collection_config'],
                         self.inputs['reference_collection_config'])

    def test_exclusion_ledger_cannot_be_removed_from_union(self):
        result = self.merge()
        source = result['source_prepared_manifests'][0]
        source['selection'].append({'key': 'rejected/episode_000002', 'included': False,
                                    'reason': 'explicit synthetic quality exclusion', 'quality': 'rejected'})
        with self.assertRaisesRegex(ValueError, 'lost a source episode exclusion'):
            validate_manifest(result)


class MergePreparedExclusionTests(unittest.TestCase):
    def test_source_exclusion_and_selection_survive_without_copying_rejected_episode(self):
        excluded = 'current_train_session/episode_000000'
        with tempfile.TemporaryDirectory(prefix='prepared_union_exclusion_') as temporary:
            root = Path(temporary)
            with source_prepared_fixture(root / 'sources', excluded=[excluded]) as inputs:
                result = merge_prepared(inputs['roots'], root / 'merged', split_file=inputs['split'])
            self.assertEqual(len(result['episodes']), 3)
            self.assertNotIn(excluded, {ep['key'] for ep in result['episodes']})
            self.assertTrue(any(x['key'] == excluded and x['included'] is False for x in result['selection']))
            self.assertIn(excluded, {x['episode_key'] for x in result['config']['quality']['excluded_episodes']})
            self.assertFalse((root / 'merged/episodes' / excluded).exists())


if __name__ == '__main__':
    unittest.main()
