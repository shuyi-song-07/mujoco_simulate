"""Registry integration: no source-project imports or fixed task enumeration."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tasks.registry import ROOT, Task, discover, get, hook, validate_binding

class TaskRegistryTests(unittest.TestCase):
    def test_frozen_protocol_roundtrip_reuse_and_cross_task_rejection(self):
        """Protocol persistence needs no historical Pack or downloaded model."""
        from rollout import rollout
        task = get('tray_to_platform')
        settings = json.loads((ROOT / task.manifest['evaluation_config']).read_text(encoding='utf-8'))
        settings.update(task_id=task.id, task_binding=task.binding(), pack='engineering-fixture')
        source = {'config': task.config(), 'demonstration_seeds': []}
        with tempfile.TemporaryDirectory() as directory, \
                patch('rollout.protocol.load_scene_provenance', return_value=source) as loader, \
                patch('tasks.registry.hook', return_value=lambda *a, **kw: {'summary': {'cases': 50}}), \
                patch.object(rollout, 'implementation_hashes', return_value={'fixture': 'current'}), \
                patch.object(rollout, 'git_commit', return_value='fixture'), \
                patch('builtins.print'):
            path = Path(directory) / 'protocol.json'
            created = rollout.prepare_protocol(settings, path)
            self.assertEqual(len(created['seeds']), 50)
            self.assertEqual(rollout.verify_protocol(path, settings), created)
            self.assertEqual(rollout.prepare_protocol(settings, path), created)
            loader.assert_called_once()
            wrong = copy.deepcopy(settings)
            wrong['task_id'] = 'h_frame_push_align'
            with self.assertRaisesRegex(ValueError, 'Settings/seed'):
                rollout.verify_protocol(path, wrong)
            with patch.object(rollout, 'implementation_hashes', return_value={'fixture': 'changed'}):
                with self.assertRaisesRegex(ValueError, 'code/assets changed'):
                    rollout.verify_protocol(path, settings)
            damaged = json.loads(path.read_text(encoding='utf-8'))
            damaged['seeds'][0] += 1
            path.write_text(json.dumps(damaged), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'signature changed'):
                rollout.verify_protocol(path, settings)

    def test_registered_scenes_and_shared_reset(self):
        from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
        from rollout.protocol import sample_initial_state
        for name in ('tray_to_platform', 'h_frame_push_align'):
            task=get(name);config=task.config();c=DualArmTaskController(config)
            c.reset(823)
            initial=sample_initial_state(config,823)
            np.testing.assert_array_equal(c.initial_state['object_position'],initial['object_position'])
            np.testing.assert_array_equal(c.initial_state['target_position'],initial['target_position'])
            for _ in range(30):c.step(1/60)
            self.assertTrue(np.isfinite(c.data.qpos).all())
            self.assertNotEqual(c.get_status()['task_state'],'FAIL')

    def test_unknown_task_and_foreign_binding_fail(self):
        with self.assertRaisesRegex(ValueError, 'Unknown task'):
            get('missing_task')
        with self.assertRaisesRegex(ValueError, 'Cross-task input rejected'):
            validate_binding(get('tray_to_platform').binding(), 'h_frame_push_align')
        b=get('tray_to_platform').binding();b['task_version']='bad'
        with self.assertRaises(ValueError):validate_binding(b)

    def test_new_manifest_is_discovered_without_fixed_enum(self):
        # Temporary directory's parent is the real discovery root. Automatic cleanup is scoped by TemporaryDirectory.
        with tempfile.TemporaryDirectory(prefix='new_task_',dir=ROOT/'tasks') as directory:
            folder=Path(directory);m=copy.deepcopy(get('tray_to_platform').manifest)
            m.update(task_id=folder.name,display_name='engineering registration fixture',scene_id='CUSTOM')
            (folder/'task.json').write_text(json.dumps(m),encoding='utf-8')
            task=discover()[folder.name]
            from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
            c=DualArmTaskController(task.config());c.step(1/60)
            self.assertEqual(c.config['scene']['scenario'],'CUSTOM')
            self.assertEqual(c.config['task_binding']['task_id'],folder.name)
            m['status'] = 'development'
            (folder/'task.json').write_text(json.dumps(m), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'is development'):
                get(folder.name)
            del m['schema'];(folder/'task.json').write_text(json.dumps(m),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'missing fields'):discover()

if __name__=='__main__':unittest.main()
