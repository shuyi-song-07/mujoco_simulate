"""Routing/legacy-API checks for the task-directory-only refactor."""
import copy
import importlib
import json
from pathlib import Path
import unittest

from tasks.registry import get
from tasks.stages import stage_adapter

ROOT = Path(__file__).resolve().parents[2]


class LearningLayoutTests(unittest.TestCase):
    def test_legacy_imports_are_same_module_objects(self):
        for old, new in (
            ('training.papera.act_model', 'training.act.act_model'),
            ('training.papera.train', 'training.act.train'),
            ('training.papera.data', 'training.act.data'),
            ('rollout.executor', 'rollout.common.executor'),
            ('rollout.rollout', 'rollout.common.rollout'),
        ):
            self.assertIs(importlib.import_module(old), importlib.import_module(new))

    def test_each_task_routes_training_dimensions_and_success_rules(self):
        for task_id in ('push_t', 'tray_to_platform', 'h_frame_push_align', 'push_r'):
            task = get(task_id)
            trainer = stage_adapter(task, 'training')
            self.assertEqual(trainer.__name__, 'training.' + task_id + '.adapter')
            config = {'model': {'hidden_dim': 512}}
            trainer.configure_defaults(config, task, 'act')
            schema = task.manifest['schema']
            self.assertEqual(config['model'], dict(hidden_dim=512,
                state_dim=schema['state_dim'], action_dim=schema['action_dim']))
            evaluator = stage_adapter(task, 'rollout')
            self.assertEqual(evaluator.__name__, 'rollout.' + task_id + '.adapter')
            settings = json.loads((ROOT / task.manifest['evaluation_config']).read_text(encoding='utf-8'))
            settings['task_id'] = task_id
            evaluator.validate_success(settings)
            invalid = copy.deepcopy(settings)
            invalid['version'] = 'unregistered-evaluation-rule'
            with self.assertRaises(ValueError):
                evaluator.validate_success(invalid)

    def test_single_arm_diffusion_stays_unsupported(self):
        with self.assertRaises(ValueError):
            stage_adapter('push_t', 'training').validate_model('diffusion')

    def test_new_task_can_inherit_controller_without_common_task_id_edit(self):
        config = {'task_id': 'future_task', 'controller': 'dual_panda_v1'}
        self.assertEqual(stage_adapter(config, 'training').SUPPORTED_MODELS, ('act', 'diffusion'))
        self.assertTrue(callable(stage_adapter(config, 'rollout').create_executor))


if __name__ == '__main__':
    unittest.main()
