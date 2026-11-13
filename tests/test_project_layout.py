"""Structural migration contracts; no robot/camera behavior is mocked as acceptance."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from common import layout
from tasks.stages import stage_adapter


class LayoutTests(unittest.TestCase):
    def test_old_and_new_modules_share_private_state(self):
        pairs = [('dataset.task3.curation_rules', 'dataset.common.curation_rules'),
                 ('rollout.action_chunks', 'rollout.common.action_chunks'),
                 ('simulation.mujoco.dual_arm.command_protocol', 'simulation.mujoco.common.command_protocol')]
        for old, new in pairs:
            a,b=importlib.import_module(old),importlib.import_module(new)
            self.assertIs(a,b)
            with patch.object(a,'_layout_probe',object(),create=True):
                self.assertIs(a._layout_probe,b._layout_probe)
            self.assertIsNotNone(a.__spec__.origin)

    def test_all_existing_tasks_have_real_stage_adapters(self):
        for name in ('push_t','push_r','h_frame_push_align','tray_to_platform'):
            for stage in ('collection','dataset','training','rollout'):
                self.assertEqual(stage_adapter(name,stage).__name__,f'{stage}.{name}.adapter')

    def test_additional_task_directory_needs_no_parent_switch(self):
        root=layout.ROOT/'collection'
        with tempfile.TemporaryDirectory(prefix='layout_task_',dir=root) as directory:
            folder=Path(directory)
            (folder/'adapter.py').write_text('MARKER = "new task adapter"\n',encoding='utf-8')
            importlib.invalidate_caches()
            self.assertEqual(stage_adapter(folder.name,'collection').MARKER,'new task adapter')

    def test_fallback_does_not_silently_reuse_push_t_scene(self):
        value={'task_id':'another_single_task','controller':'single_panda_push_v1'}
        for stage in ('collection','rollout'):
            with self.assertRaisesRegex(ValueError,'task-specific'):
                stage_adapter(value,stage)

    def test_internal_adapter_import_failure_is_not_hidden(self):
        with patch('tasks.stages.importlib.import_module',side_effect=ModuleNotFoundError(name='missing_dependency')):
            with self.assertRaises(ModuleNotFoundError) as found:
                stage_adapter('push_t','collection')
            self.assertEqual(found.exception.name,'missing_dependency')

    def test_alias_import_then_runpy_and_reload(self):
        script = "import importlib,runpy; m=importlib.import_module('dataset.task3.prepare_paper_dataset'); importlib.reload(m); result=runpy.run_module('dataset.task3.prepare_paper_dataset',run_name='__main__'); print(callable(result['implementation_fingerprints']))"
        result=subprocess.run([sys.executable,'-B','-c',script],cwd=layout.ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('True',result.stdout)

    def test_dataset_evidence_uses_explicit_task_stage(self):
        from tasks.registry import get,hook
        task=get('push_t')
        adapter=stage_adapter(task,'dataset')
        with patch.object(adapter,'adapter_factory',return_value='custom evidence') as factory:
            self.assertEqual(hook(task.config(),'dataset_adapter')(),'custom evidence')
            factory.assert_called_once()

    def test_new_task_binding_covers_stage_routing_dependencies(self):
        from tasks import registry
        binding=registry.get('push_t').binding()
        path='collection/push_t/adapter.py'
        self.assertIn(path,binding['sources'])
        real_sha=registry.sha
        with patch.object(registry,'sha',side_effect=lambda p: '0'*64
                          if Path(p).resolve()==(layout.ROOT/path).resolve() else real_sha(p)):
            with self.assertRaisesRegex(ValueError,'Frozen task/assets differ'):
                registry.validate_binding(binding)

    def test_migration_rejects_tampering_and_incomplete_catalogs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            actual=root/'new.py';actual.write_bytes(b'new import wiring')
            sha=hashlib.sha256(actual.read_bytes()).hexdigest()
            oldsha=hashlib.sha256(b'old import wiring').hexdigest()
            before={'old.py':oldsha};after={'new.py':sha}
            manifest={'files':{'old.py':{'path':'new.py','before_sha256':oldsha,'after_sha256':sha}},
                      'bundle':after,'catalogs':{'code':{'before':before,'after':after}}}
            path=root/'migration.json';path.write_text(json.dumps(manifest),encoding='utf-8')
            with patch.object(layout,'ROOT',root),patch.object(layout,'MANIFEST',path):
                self.assertTrue(layout.source_mapping_matches(before,after))
                self.assertTrue(layout.source_hash_matches('old.py',oldsha))
                self.assertFalse(layout.source_mapping_matches({},after))
                self.assertFalse(layout.source_mapping_matches({},{}))
                self.assertFalse(layout.source_mapping_matches({'old.py':'0'*64},after))
                self.assertFalse(layout.source_mapping_matches(before,{**after,'extra.py':sha}))
                binding={'task_id':'fixture','sources':before,'camera':'fixed'}
                current={**binding,'sources':{'old.py':sha}}
                self.assertTrue(layout.equivalent_binding(binding,current))
                self.assertFalse(layout.equivalent_binding(binding,{**current,'camera':'changed'}))
                actual.write_bytes(b'unreviewed algorithm change')
                self.assertFalse(layout.source_mapping_matches(before,after))
                self.assertFalse(layout.source_hash_matches('old.py',oldsha))
                self.assertFalse(layout.equivalent_binding(binding,current))


if __name__=='__main__':unittest.main()
