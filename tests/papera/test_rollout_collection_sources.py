"""Rollout uses the same physical domain and original collection provenance."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dataset.task3 import prepare_paper_dataset as prepared
from dataset.task3 import training_pack
from dataset.task3.merge_prepared import merge_prepared
from rollout.protocol import (CONFIG_VALIDATION_RENDER_COMPATIBILITY, _digest,
                              load_scene_provenance, verify_oblique_camera)
from simulation.mujoco.dual_arm.action_contract import STATE_NAMES, ACTION_NAMES_V2
from simulation.mujoco.dual_arm.config import config_hash
from simulation.mujoco.dual_arm.render_identity import render_identity
from tests.task3.merge_prepared_fixture import source_prepared_fixture


class RolloutCollectionSourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='rollout_collection_sources_')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw_root = self.root / 'sources'
        context = source_prepared_fixture(self.raw_root)
        self.sources = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        seed = 100
        for path in self.sources['roots']:
            manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
            for episode in manifest['episodes']:
                metadata_path = Path(episode['source_ref']['metadata_path'])
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                seed += 1
                metadata.update(seed=seed, outcome='success',
                    state_names=STATE_NAMES, action_names=ACTION_NAMES_V2,
                    action_units=['m','m','m','rad','rad','rad','m']*2,
                    config_hash=config_hash(metadata['config']),
                    render_identity=render_identity(metadata['config']))
                prepared.write_json(metadata_path, metadata)
                for record in episode['source_fingerprints']:
                    if Path(record['path']) == metadata_path:
                        record.update(sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                            size=metadata_path.stat().st_size, mtime_ns=metadata_path.stat().st_mtime_ns)
            prepared.write_json(path/'manifest.json', manifest)
        self.prepared = self.root/'merged'
        merge_prepared(self.sources['roots'], self.prepared, split_file=self.sources['split'])
        self.pack = self.root/'pack'
        self.manifest = training_pack.build(self.prepared, self.pack,
            split_file=self.sources['split'], width=64, height=48)

    def test_verified_collection_threshold_groups_load_one_physical_domain(self):
        source = load_scene_provenance(self.pack, raw_root=self.raw_root)
        self.assertEqual(len(source['sources']), 4)
        self.assertEqual(source['demonstration_seeds'], [101,102,103,104])
        self.assertEqual(source['config']['task_binding'], self.manifest['task_binding'])
        self.assertEqual(source['training_cameras'], ['oblique'])
        self.assertEqual(source['source_task_bindings'], self.manifest['source_task_bindings'])

    def test_raw_source_binding_cannot_be_reassigned(self):
        altered = copy.deepcopy(self.manifest)
        altered['episodes'][-1]['source_task_binding'] = self.sources['canonical_binding']
        prepared.write_json(self.pack/'manifest.json', altered)
        with self.assertRaises(ValueError):
            load_scene_provenance(self.pack, raw_root=self.raw_root)

    def test_forged_same_domain_ledger_cannot_change_the_physical_scene(self):
        for name in ('manifest.json', 'prepared.manifest.json'):
            path = self.pack/name
            value = json.loads(path.read_text(encoding='utf-8'))
            value['dataset_collection_compatibility']['groups'][-1][
                'frozen_collection_config']['scene']['table_top_z'] += .01
            prepared.write_json(path, value)
        with self.assertRaises(ValueError):
            load_scene_provenance(self.pack, raw_root=self.raw_root)

    def test_changed_raw_metadata_is_rejected(self):
        path = Path(self.manifest['episodes'][-1]['source_ref']['metadata_path'])
        value = json.loads(path.read_text(encoding='utf-8'))
        value['config']['training_camera']['distance'] += .01
        prepared.write_json(path, value)
        with self.assertRaises(ValueError):
            load_scene_provenance(self.pack, raw_root=self.raw_root)

    def audited_camera_source(self):
        episode = self.manifest['episodes'][-1]
        metadata = json.loads(Path(episode['source_ref']['metadata_path']).read_text(encoding='utf-8'))
        identity = metadata['render_identity']
        rule = CONFIG_VALIDATION_RENDER_COMPATIBILITY
        identity['source_files_sha256'][rule['path']] = rule['recorded_sha256']
        identity['source_files_sha256']['config/collection.json'] = metadata['task_binding']['sources']['config/collection.json']
        self.update_identity_digest(identity)
        return episode, metadata

    @staticmethod
    def update_identity_digest(identity):
        identity['identity_sha256'] = _digest({key:value for key,value in identity.items() if key != 'identity_sha256'})

    def test_only_audited_validation_change_has_explicit_camera_proof(self):
        episode, metadata = self.audited_camera_source()
        proof = verify_oblique_camera(episode, metadata,
            collection_union=self.manifest['dataset_collection_compatibility'])
        self.assertTrue(proof['compatibility']['verified'])
        self.assertIn(CONFIG_VALIDATION_RENDER_COMPATIBILITY,
            proof['compatibility']['source_compatibility'])
        self.assertEqual(proof['render_identity'], metadata['render_identity'])
        with self.assertRaises(ValueError):
            verify_oblique_camera(episode, metadata)

    def test_unreviewed_source_change_does_not_enter_render_compatibility(self):
        episode, metadata = self.audited_camera_source()
        metadata['render_identity']['source_files_sha256'][CONFIG_VALIDATION_RENDER_COMPATIBILITY['path']] = 'f'*64
        self.update_identity_digest(metadata['render_identity'])
        with self.assertRaisesRegex(ValueError, 'Unreviewed'):
            verify_oblique_camera(episode, metadata,
                collection_union=self.manifest['dataset_collection_compatibility'])

    def test_camera_change_cannot_hide_behind_audited_source_compatibility(self):
        episode, metadata = self.audited_camera_source()
        metadata['render_identity']['actual_fovy_degrees'] += 1
        self.update_identity_digest(metadata['render_identity'])
        with self.assertRaisesRegex(ValueError, 'camera/physics/assets'):
            verify_oblique_camera(episode, metadata,
                collection_union=self.manifest['dataset_collection_compatibility'])

    def audited_tray_controller_source(self):
        episode, metadata = self.audited_camera_source()
        metadata['render_identity']['source_files_sha256'][
            'simulation/mujoco/dual_arm/task_controller.py'] = (
            '23bad8440fc843f110619e50b1737fd1bc4c370a5500f85b7fdb996e3de2bd91')
        self.update_identity_digest(metadata['render_identity'])
        return episode, metadata

    def test_recorded_open_tray_controller_has_audited_render_proof(self):
        episode, metadata = self.audited_tray_controller_source()
        proof = verify_oblique_camera(episode, metadata,
            collection_union=self.manifest['dataset_collection_compatibility'])
        rules = proof['compatibility']['source_compatibility']
        self.assertTrue(any(r['rule'] == 'tray-open-reset-and-push-capability-metadata-only' for r in rules))
        self.assertEqual(proof['render_identity'], metadata['render_identity'])

    def test_controller_compatibility_rejects_closed_initial_gripper(self):
        episode, metadata = self.audited_tray_controller_source()
        metadata['config']['scene']['initial_gripper'] = 'closed'
        identity = render_identity(metadata['config'])
        identity['source_files_sha256']['config/collection.json'] = metadata['task_binding']['sources']['config/collection.json']
        identity['source_files_sha256']['simulation/mujoco/dual_arm/task_controller.py'] = (
            '23bad8440fc843f110619e50b1737fd1bc4c370a5500f85b7fdb996e3de2bd91')
        self.update_identity_digest(identity)
        metadata['render_identity'] = identity
        with self.assertRaisesRegex(ValueError, 'Unreviewed rendering controller'):
            verify_oblique_camera(episode, metadata,
                collection_union=self.manifest['dataset_collection_compatibility'])

    def test_controller_compatibility_rejects_unknown_current_source(self):
        episode, metadata = self.audited_tray_controller_source()
        current = render_identity(metadata['config'])
        current['source_files_sha256']['simulation/mujoco/dual_arm/task_controller.py'] = 'f'*64
        with patch('simulation.mujoco.dual_arm.render_identity.render_identity', return_value=current):
            with self.assertRaises(ValueError):
                verify_oblique_camera(episode, metadata,
                    collection_union=self.manifest['dataset_collection_compatibility'])

    def test_controller_compatibility_does_not_hide_camera_change(self):
        episode, metadata = self.audited_tray_controller_source()
        metadata['render_identity']['actual_fovy_degrees'] += 1
        self.update_identity_digest(metadata['render_identity'])
        with self.assertRaisesRegex(ValueError, 'camera/physics/assets'):
            verify_oblique_camera(episode, metadata,
                collection_union=self.manifest['dataset_collection_compatibility'])


if __name__ == '__main__':
    unittest.main()
