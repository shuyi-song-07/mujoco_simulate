"""Registry integration: no source-project imports or fixed task enumeration."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tasks.registry import ROOT, Task, discover, get, hook, validate_binding

class TaskRegistryTests(unittest.TestCase):
    def test_registered_scenes_and_shared_reset(self):
        from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
        from rollout.protocol import sample_initial_state
        for name in ('tray_transfer','tray_push_then_transfer','tray_raised_placement'):
            task=get(name);config=task.config();c=DualArmTaskController(config)
            c.reset(823)
            initial=sample_initial_state(config,823)
            np.testing.assert_array_equal(c.initial_state['object_position'],initial['object_position'])
            np.testing.assert_array_equal(c.initial_state['target_position'],initial['target_position'])
            for _ in range(30):c.step(1/60)
            self.assertTrue(np.isfinite(c.data.qpos).all())
            self.assertNotEqual(c.get_status()['task_state'],'FAIL')

    def test_unknown_development_and_foreign_binding_fail(self):
        for task in ('push','lift','missing_task'):
            with self.assertRaises(ValueError):get(task)
        with self.assertRaises(ValueError):validate_binding(get('tray_transfer').binding(),'tray_raised_placement')
        b=get('tray_transfer').binding();b['task_version']='bad'
        with self.assertRaises(ValueError):validate_binding(b)

    def test_new_manifest_is_discovered_without_fixed_enum(self):
        # Temporary directory's parent is the real discovery root. Automatic cleanup is scoped by TemporaryDirectory.
        with tempfile.TemporaryDirectory(prefix='new_task_',dir=ROOT/'tasks') as directory:
            folder=Path(directory);m=copy.deepcopy(get('tray_transfer').manifest)
            m.update(task_id=folder.name,display_name='engineering registration fixture',scene_id='CUSTOM')
            (folder/'task.json').write_text(json.dumps(m),encoding='utf-8')
            task=discover()[folder.name]
            from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
            c=DualArmTaskController(task.config());c.step(1/60)
            self.assertEqual(c.config['scene']['scenario'],'CUSTOM')
            self.assertEqual(c.config['task_binding']['task_id'],folder.name)
            del m['schema'];(folder/'task.json').write_text(json.dumps(m),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'missing fields'):discover()

if __name__=='__main__':unittest.main()
