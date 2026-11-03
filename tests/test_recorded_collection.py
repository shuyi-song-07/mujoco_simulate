"""Historical collection thresholds require an explicit, bounded data scope."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tasks.artifacts import fields, source_contract
from tasks.recorded_collection import (
    PROOF_FIELD, SOURCE, recorded_collection_provenance, recorded_collection_scope,
)
from tasks.registry import validate_binding


class RecordedCollectionTests(unittest.TestCase):
    def setUp(self):
        self.current = {
            'version': 'papera-task-binding-v1',
            'task_id': 'tray_to_platform', 'task_version': '1.2.1',
            'scene_id': 'HIGHWALL_PLATFORM', 'manifest_sha256': 'a' * 64,
            'schema': {'state_dim': 16, 'action_dim': 14},
            'hooks': {'dataset_adapter': 'tasks.adapters.tray:dataset_adapter'},
            'sources': {SOURCE: '1' * 64, 'tasks/tray_to_platform/scene.py': '2' * 64},
        }
        self.recorded = copy.deepcopy(self.current)
        self.recorded['sources'][SOURCE] = '3' * 64
        self.directory = tempfile.TemporaryDirectory(prefix='recorded_collection_')
        self.addCleanup(self.directory.cleanup)
        self.metadata_path = Path(self.directory.name) / 'episode_metadata.json'
        self.provenance = {
            'config_differences': [
                {'path': 'gesture_control.close_score_min', 'current': .8, 'recorded': .6},
                {'path': 'gesture_control.open_score_min', 'current': .55, 'recorded': .5},
                {'path': 'gesture_control.left_close_score_min', 'current': None, 'recorded': .5},
            ],
            'metadata_paths': [str(self.metadata_path)],
        }
        self.task = SimpleNamespace(id='tray_to_platform', binding=lambda: copy.deepcopy(self.current))
        self.get_patch = patch('tasks.registry.get', return_value=self.task)
        self.get_patch.start()
        self.addCleanup(self.get_patch.stop)

    def write_metadata(self, **updates):
        metadata = {
            'task_id': self.recorded['task_id'],
            'task_binding': self.recorded,
            'schema_version': 'task3-papera-v2',
            'config': {'task_binding': self.recorded},
        }
        metadata.update(updates)
        self.metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
        return {'metadata_path': self.metadata_path, 'engineering': False}

    def test_historical_binding_is_rejected_by_default_and_after_scope(self):
        with self.assertRaisesRegex(ValueError, 'Frozen task/assets differ'):
            validate_binding(self.recorded)
        with recorded_collection_scope(self.recorded, self.provenance) as proof:
            self.assertIs(validate_binding(self.recorded), self.task)
            self.assertIs(validate_binding(self.current), self.task)
            self.assertEqual(proof['recorded_collection_sha256'], '3' * 64)
            self.assertEqual(proof['current_collection_sha256'], '1' * 64)
        self.assertIsNone(recorded_collection_provenance(self.recorded))
        with self.assertRaisesRegex(ValueError, 'Frozen task/assets differ'):
            validate_binding(self.recorded)

    def test_only_one_exact_recorded_binding_is_accepted(self):
        other = copy.deepcopy(self.recorded)
        other['sources'][SOURCE] = '4' * 64
        with recorded_collection_scope(self.recorded, self.provenance):
            with self.assertRaisesRegex(ValueError, 'Frozen task/assets differ'):
                validate_binding(other)
            with self.assertRaisesRegex(ValueError, 'Cross-task input rejected'):
                validate_binding(self.recorded, 'tray_transfer')

    def test_other_sources_versions_and_schemas_cannot_enter_scope(self):
        for changed in ('source', 'version', 'schema', 'task', 'extra_source', 'missing_source'):
            with self.subTest(changed=changed):
                binding = copy.deepcopy(self.recorded)
                if changed == 'source':
                    binding['sources']['tasks/tray_to_platform/scene.py'] = '4' * 64
                elif changed == 'version':
                    binding['task_version'] = '1.2.2'
                elif changed == 'schema':
                    binding['schema']['action_dim'] = 15
                elif changed == 'task':
                    binding['task_id'] = 'tray_transfer'
                elif changed == 'extra_source':
                    binding['sources']['unregistered.py'] = '5' * 64
                else:
                    del binding['sources']['tasks/tray_to_platform/scene.py']
                with self.assertRaisesRegex(ValueError, 'only config/collection.json'):
                    with recorded_collection_scope(binding, self.provenance):
                        self.fail('invalid scope entered')

    def test_current_source_changes_during_scope_are_rejected(self):
        with recorded_collection_scope(self.recorded, self.provenance):
            self.current['sources']['tasks/tray_to_platform/scene.py'] = '4' * 64
            with self.assertRaisesRegex(ValueError, 'Frozen task/assets differ'):
                validate_binding(self.recorded)

    def test_scope_freezes_inputs_and_resets_after_exception(self):
        binding = copy.deepcopy(self.recorded)
        provenance = copy.deepcopy(self.provenance)
        with self.assertRaisesRegex(RuntimeError, 'scope exit'):
            with recorded_collection_scope(binding, provenance) as proof:
                binding['sources'][SOURCE] = '4' * 64
                provenance['config_differences'].clear()
                proof['metadata_paths'].clear()
                self.assertIs(validate_binding(self.recorded), self.task)
                self.assertTrue(recorded_collection_provenance(self.recorded)['metadata_paths'])
                raise RuntimeError('scope exit')
        with self.assertRaises(ValueError):
            validate_binding(self.recorded)

    def test_compatibility_does_not_leak_into_other_threads(self):
        with recorded_collection_scope(self.recorded, self.provenance):
            with ThreadPoolExecutor(max_workers=1) as pool:
                with self.assertRaisesRegex(ValueError, 'Frozen task/assets differ'):
                    pool.submit(validate_binding, self.recorded).result()
            self.assertIs(validate_binding(self.recorded), self.task)

    def test_current_binding_scope_is_a_noop(self):
        with recorded_collection_scope(self.current, None) as proof:
            self.assertIsNone(proof)
            self.assertIsNone(recorded_collection_provenance(self.current))
            self.assertIs(validate_binding(self.current), self.task)
            with self.assertRaises(ValueError):
                validate_binding(self.recorded)

    def test_provenance_requires_allowed_finite_thresholds_and_record_paths(self):
        invalid = [
            {'config_differences': [], 'metadata_paths': [str(self.metadata_path)]},
            {'config_differences': [{'path': 'training_camera.distance', 'current': 1.2, 'recorded': 1.3}],
             'metadata_paths': [str(self.metadata_path)]},
        ]
        for value in (-.1, 1.1, True, float('nan'), None):
            proof = copy.deepcopy(self.provenance)
            proof['config_differences'][0]['recorded'] = value
            invalid.append(proof)
        proof = copy.deepcopy(self.provenance)
        proof['metadata_paths'] = ['relative/episode_metadata.json']
        invalid.append(proof)
        for proof in invalid:
            with self.subTest(proof=proof):
                with self.assertRaises(ValueError):
                    with recorded_collection_scope(self.recorded, proof):
                        self.fail('invalid provenance entered scope')

    def test_source_contract_preserves_metadata_identity_and_adapter_checks(self):
        ref = self.write_metadata()
        with recorded_collection_scope(self.recorded, self.provenance):
            contract = source_contract([ref])
            self.assertEqual(contract['task_binding'], self.recorded)
            self.assertEqual(contract[PROOF_FIELD]['metadata_paths'], [str(self.metadata_path)])
            self.write_metadata(task_id='tray_transfer')
            with self.assertRaisesRegex(ValueError, 'Task identity mismatch'):
                source_contract([ref])
            self.write_metadata(config={'task_binding': self.current})
            with self.assertRaisesRegex(ValueError, 'Task identity mismatch'):
                source_contract([ref])
            self.write_metadata(schema_version='invalid')
            with self.assertRaisesRegex(ValueError, 'requires task3-papera-v2'):
                source_contract([ref])

    def test_source_contract_still_rejects_mixed_bindings(self):
        first = self.write_metadata()
        second_path = self.metadata_path.with_name('current_metadata.json')
        metadata = json.loads(self.metadata_path.read_text(encoding='utf-8'))
        metadata['task_binding'] = self.current
        metadata['config']['task_binding'] = self.current
        second_path.write_text(json.dumps(metadata), encoding='utf-8')
        with recorded_collection_scope(self.recorded, self.provenance):
            with self.assertRaisesRegex(ValueError, 'one exact task version'):
                source_contract([first, {'metadata_path': second_path, 'engineering': False}])

    def test_artifact_proof_is_copied_but_never_grants_default_acceptance(self):
        ref = self.write_metadata()
        with recorded_collection_scope(self.recorded, self.provenance):
            manifest = source_contract([ref])
            copied = fields(manifest)
            self.assertEqual(copied[PROOF_FIELD], manifest[PROOF_FIELD])
            copied[PROOF_FIELD]['metadata_paths'].clear()
            self.assertTrue(manifest[PROOF_FIELD]['metadata_paths'])
        with self.assertRaisesRegex(ValueError, 'Frozen task/assets differ'):
            fields(manifest)


class RecordedInputGroupingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.current = {'task_id': 'tray_to_platform', 'task_version': '1',
                        'schema': {'state_dim': 16, 'action_dim': 14},
                        'sources': {'config/collection.json': 'a'*64, 'scene.xml': 'b'*64}}
        self.old = copy.deepcopy(self.current)
        self.old['sources']['config/collection.json'] = 'c'*64
        self.config = {'gesture_control': {'close_score_min': .8, 'open_score_min': .55,
                                          'motion_score_min': .75},
                       'scene': {'table_top_z': .35}, 'training_camera': {'name': 'oblique'}}
        self.workflow = SimpleNamespace(source=self.root, engineering=False,
            task=SimpleNamespace(id='tray_to_platform', binding=lambda: copy.deepcopy(self.current)))
        self.refs = []
        self.add_record('current', self.current, self.config)
        old_config = copy.deepcopy(self.config)
        old_config['gesture_control'].update(close_score_min=.6, open_score_min=.5, left_close_score_min=.5)
        self.add_record('recorded', self.old, old_config)
        self.discovery = patch('dataset.recorded_collection.discover_episodes', return_value=self.refs)
        self.discovery.start()
        self.addCleanup(self.discovery.stop)

    def add_record(self, name, binding, config):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        path = directory / 'episode_metadata.json'
        value = {'task_id': 'tray_to_platform', 'task_binding': copy.deepcopy(binding),
                 'config': {**copy.deepcopy(config), 'task_binding': copy.deepcopy(binding)}}
        path.write_text(json.dumps(value), encoding='utf-8')
        self.refs.append({'key': name+'/episode_000000', 'dataset_root': directory,
                          'metadata_path': path})

    def groups(self):
        from dataset.recorded_collection import recorded_groups
        return recorded_groups(self.workflow)

    def change_recorded(self, callback):
        path = self.refs[1]['metadata_path']
        value = json.loads(path.read_text(encoding='utf-8'))
        callback(value)
        path.write_text(json.dumps(value), encoding='utf-8')

    def test_threshold_changes_are_separate_groups_with_original_source_paths(self):
        groups = self.groups()
        self.assertEqual([group['current'] for group in groups], [True, False])
        self.assertEqual([len(group['refs']) for group in groups], [1, 1])
        self.assertEqual(len(groups[1]['provenance']['config_differences']), 3)
        self.assertEqual(groups[1]['sessions']['recorded'], str((self.root/'recorded').resolve()))
        self.assertEqual(groups[1]['binding'], self.old)

    def test_camera_or_physics_changes_are_not_collection_compatibility(self):
        self.change_recorded(lambda value: value['config']['scene'].update(table_top_z=.4))
        with self.assertRaisesRegex(ValueError, 'Unapproved frozen setting'):
            self.groups()

    def test_another_asset_change_is_rejected(self):
        def change(value):
            for binding in (value['task_binding'], value['config']['task_binding']):
                binding['sources']['scene.xml'] = 'd'*64
        self.change_recorded(change)
        with self.assertRaisesRegex(ValueError, 'task/assets differ'):
            self.groups()

    def test_invalid_thresholds_are_not_accepted(self):
        self.change_recorded(lambda value: value['config']['gesture_control'].update(close_score_min=True))
        with self.assertRaisesRegex(ValueError, 'Invalid gesture threshold'):
            self.groups()

    def test_identity_mismatch_is_not_fixed_or_ignored(self):
        self.change_recorded(lambda value: value.update(task_id='tray_transfer'))
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.groups()


class RecordedCollectionCliTests(unittest.TestCase):
    def test_external_native_source_uses_the_same_formal_process(self):
        from dataset.manage_dataset import main

        with patch('dataset.manage_dataset.DatasetWorkflow') as factory, \
                patch('dataset.recorded_collection.process_recorded') as recorded, \
                patch('builtins.print'):
            self.assertEqual(main(['--task','tray_to_platform','process',
                                   '--source','incoming-native','--recorded-collection']),0)
        recorded.assert_called_once_with(factory.return_value,config=None,fresh=False,
                                         workers=2,source=Path('incoming-native'))

    def test_single_prepare_keeps_the_existing_public_dispatch(self):
        from dataset.manage_dataset import main

        with patch('dataset.manage_dataset.DatasetWorkflow') as factory, \
                patch('dataset.recorded_collection.prepare_recorded') as joined, \
                patch('builtins.print'):
            self.assertEqual(main(['--task','tray_to_platform','prepare','--run','curation-a']),0)
        factory.return_value.prepare.assert_called_once_with(Path('curation-a'),'auto',output=None)
        joined.assert_not_called()

    def test_multiple_curations_use_one_join_with_an_explicit_split(self):
        from dataset.manage_dataset import main

        with patch('dataset.manage_dataset.DatasetWorkflow') as factory, \
                patch('dataset.recorded_collection.prepare_recorded') as joined, \
                patch('builtins.print'):
            self.assertEqual(main(['--task','tray_to_platform','prepare','--run',
                'curation-a','curation-b','--recorded-collection','--split-file','split.json',
                '--workers','2']),0)
        joined.assert_called_once_with(factory.return_value,[Path('curation-a'),Path('curation-b')],
            review_mode='auto',output=None,split_file=Path('split.json'),allow_recorded=True,workers=2)
        factory.return_value.prepare.assert_not_called()

    def test_default_process_retains_the_strict_workflow_dispatch(self):
        from dataset.manage_dataset import main

        with patch('dataset.manage_dataset.DatasetWorkflow') as factory, \
                patch('dataset.recorded_collection.process_recorded') as recorded, \
                patch('builtins.print'):
            self.assertEqual(main(['--task', 'tray_to_platform', 'process']), 0)
        factory.assert_called_once_with('tray_to_platform', engineering=False)
        factory.return_value.process.assert_called_once_with(config=None, fresh=False, workers=2)
        recorded.assert_not_called()

    def test_recorded_collection_flag_dispatches_explicit_grouping(self):
        from dataset.manage_dataset import main

        with patch('dataset.manage_dataset.DatasetWorkflow') as factory, \
                patch('dataset.recorded_collection.process_recorded') as recorded, \
                patch('builtins.print'):
            self.assertEqual(main(['--task', 'tray_to_platform', 'process',
                                   '--recorded-collection', '--fresh', '--workers', '3',
                                   '--config', 'frozen-curation.json']), 0)
        factory.assert_called_once_with('tray_to_platform', engineering=False)
        recorded.assert_called_once_with(factory.return_value,
            config=Path('frozen-curation.json'), fresh=True, workers=3)
        factory.return_value.process.assert_not_called()


class ConfigurationDifferenceTests(unittest.TestCase):
    def test_integer_float_equivalence_preserves_the_old_public_import(self):
        from dataset.recorded_collection import config_differences
        self.assertEqual(config_differences({'value': [1, 0.]}, {'value': [1., 0]}), [])

    def test_boolean_values_cannot_masquerade_as_numbers_even_inside_lists(self):
        from tasks.recorded_collection import config_differences
        for current, recorded in ((True, 1), (False, 0), ([True], [1]),
                                  ({'flag': [False]}, {'flag': [0.]})):
            with self.subTest(current=current, recorded=recorded):
                self.assertTrue(config_differences(current, recorded))


if __name__ == '__main__':
    unittest.main()
