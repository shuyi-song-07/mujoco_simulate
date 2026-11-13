"""Task-local stage routing must preserve the pre-move dataset decisions."""
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests/task3'))


class DatasetLayoutTests(unittest.TestCase):
    def test_old_import_and_new_import_are_one_module_for_instrumentation(self):
        old = importlib.import_module('dataset.task3.curation_rules')
        new = importlib.import_module('dataset.common.curation_rules')
        self.assertIs(old, new)
        with patch('dataset.task3.curation_rules.load_config', return_value={'sentinel': True}):
            self.assertEqual(new.load_config(), {'sentinel': True})

    def test_task_evidence_classes_are_owned_by_their_dataset_folder(self):
        from tasks.stages import stage_adapter
        for task in ('push_t', 'tray_to_platform', 'h_frame_push_align', 'push_r'):
            with self.subTest(task=task):
                stage = stage_adapter(task, 'dataset')
                self.assertEqual(stage.__name__, f'dataset.{task}.adapter')
                self.assertEqual(type(stage.adapter_factory()).__module__, f'dataset.{task}.evidence')
                for name in ('audit', 'reclip', 'prepare', 'pack', 'verify_pack'):
                    self.assertTrue(callable(getattr(stage, name)))

    def test_new_tasks_can_inherit_existing_schema_without_a_central_task_switch(self):
        from tasks.stages import stage_adapter
        for controller, schema in (('dual_panda_v1', 'task3-papera-v2'),
                                   ('single_panda_push_v1', 'single-panda-push-v1')):
            with self.subTest(controller=controller):
                selected = stage_adapter({'task_id': 'future_dataset_task', 'controller': controller}, 'dataset')
                self.assertEqual(selected.__name__, f'dataset.common.adapters.{controller}')
                evidence = selected.adapter_factory()
                evidence.validate({'schema_version': schema})
                with self.assertRaises(ValueError):
                    evidence.validate({'schema_version': 'incompatible'})

    def test_workflow_uses_task_selected_reclip_instead_of_a_central_task_switch(self):
        from dataset.manage_dataset import DatasetWorkflow
        from tasks.stages import stage_adapter
        root = ROOT/'.local/checks/task_layout'; root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='dataset-routing-', dir=root) as temporary:
            workflow = DatasetWorkflow.__new__(DatasetWorkflow)
            workflow.task = SimpleNamespace(id='push_t')
            workflow.output = Path(temporary)
            selected = stage_adapter('push_t', 'dataset')
            source, destination = Path(temporary)/'audit', Path(temporary)/'curation'
            with patch.object(selected, 'reclip', return_value=(destination, {})) as called:
                result = workflow.reclip(source, candidate=True)
            self.assertEqual(result, [destination])
            self.assertEqual(called.call_args.args[0], source)

    def test_relocation_preserves_complete_wait_and_release_decisions(self):
        from dataset.common.curation_rules import select_rows, load_config
        from dataset.common.prepare_paper_dataset import serial
        from tests.task3.test_three_frame_curation import wait_rows
        from tests.task3.test_terminal_tail import release_rows
        from tests.task3.test_single_arm_data_schema import single
        # Frozen before any moves. Includes every reason, event, seam and label,
        # not just the number of frames left in the resulting video.
        cases = (
            ('dual_push', wait_rows(), 'f493e267d5f40fef508797ee36d2374350f9e160076567570e1ca5e96786dd89'),
            ('dual_pick', release_rows(), 'b66e8ccdbd8c382a3024e5f20cfe49723bb1321fa1ed265b5a74802844ef9181'),
            ('single_push', single(wait_rows()), 'e21f909859c418e0ec9340c50687151121882ab045f59bb8cd9ca701c2aa49bd'),
        )
        for name, rows, expected in cases:
            with self.subTest(task_family=name):
                result = select_rows(rows, load_config())
                content = json.dumps(result, sort_keys=True, separators=(',', ':'), default=serial).encode()
                self.assertEqual(hashlib.sha256(content).hexdigest(), expected)


if __name__ == '__main__':
    unittest.main()
