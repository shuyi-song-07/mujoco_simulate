"""Navigation only: Chinese Windows paths survive Python and both PowerShells."""
import base64
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dataset.manage_dataset import DatasetWorkflow, write_navigation


class DatasetNavigationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='navigation_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / '轻量化数据采集' / 'paperA 空格[测试]'
        self.root.mkdir(parents=True)
        self.workflow = DatasetWorkflow.__new__(DatasetWorkflow)
        self.workflow.source = self.root / 'datasets' / 'tray_to_platform'
        self.workflow.output = self.root / 'outputs' / 'dataset' / 'tray_to_platform'
        self.workflow.output.mkdir(parents=True)
        self.workflow.task = SimpleNamespace(id='tray_to_platform')
        self.workflow.engineering = True

    def generate(self):
        w = self.workflow
        curation = self.root / '裁剪结果'
        prepared = self.root / '准备结果'
        pack = self.root / '训练数据包'
        with patch.object(w, 'catalog'):
            w._remember([curation])
        with patch.object(w, '_version', return_value=(curation, True)), \
                patch('dataset.manage_dataset.pipeline.prepare'):
            w.prepare(curation, output=prepared, review_mode='auto')
        with patch('tasks.artifacts.check_artifact'), \
                patch('dataset.task3.training_pack.build', return_value={'complete': True}):
            w.pack(prepared, output=pack, split_file=self.root / '划分.json')
        return [('CURATION_CURRENT.json', 'selected_version', curation),
                ('PREPARED_CURRENT.json', 'path', prepared),
                ('TRAINING_PACK_CURRENT.json', 'path', pack)]

    def test_three_workflow_writers_use_utf8_standard_json_and_forward_paths(self):
        for filename, key, expected in self.generate():
            with self.subTest(filename=filename):
                raw = (self.workflow.output / filename).read_bytes()
                parsed = json.loads(raw.decode('utf-8'))
                self.assertEqual(parsed[key], expected.resolve().as_posix())
                self.assertTrue(raw.isascii(), 'Unicode escapes avoid legacy ANSI decoding corruption')
                self.assertFalse(parsed['training_started'] if 'training_started' in parsed else parsed['training_ready'])

    def test_navigation_roundtrips_through_available_powershells(self):
        shells = [shutil.which(name) for name in ('powershell.exe', 'pwsh.exe')]
        shells = [name for name in shells if name]
        if not shells:
            self.skipTest('PowerShell is unavailable')
        for filename, key, expected in self.generate():
            path = self.workflow.output / filename
            for shell in shells:
                for encoding in ('', '-Encoding UTF8'):
                    with self.subTest(filename=filename, shell=shell, encoding=encoding):
                        # EncodedCommand avoids shell/console encoding corrupting the test itself.
                        script = ("$ErrorActionPreference='Stop'; "
                            f"$doc=Get-Content -LiteralPath '{str(path).replace(chr(39), chr(39)*2)}' -Raw {encoding} | ConvertFrom-Json; "
                            f"[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes([string]$doc.{key}))")
                        result = subprocess.run([shell, '-NoProfile', '-NonInteractive', '-EncodedCommand',
                            base64.b64encode(script.encode('utf-16le')).decode('ascii')],
                            capture_output=True, timeout=20)
                        self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8', errors='replace'))
                        value = base64.b64decode(result.stdout.strip()).decode('utf-8')
                        self.assertEqual(Path(value), expected.resolve())

    def test_candidates_and_incomplete_builds_do_not_select_new_outputs(self):
        rows = self.generate()
        before = {name: (self.workflow.output / name).read_bytes() for name, _, _ in rows}
        with patch.object(self.workflow, '_version', return_value=(self.root / '裁剪结果', True)), \
                patch('dataset.manage_dataset.pipeline.prepare'):
            self.workflow.prepare(self.root, output=self.root/'新候选', candidate=True)
        with patch('tasks.artifacts.check_artifact'), \
                patch('dataset.task3.training_pack.build', return_value={'complete': False}):
            self.workflow.pack(self.root, output=self.root/'未完成', split_file=self.root/'split.json')
        with patch('tasks.artifacts.check_artifact'), \
                patch('dataset.task3.training_pack.build', return_value={'complete': True}):
            self.workflow.pack(self.root, output=self.root/'候选包', split_file=self.root/'split.json', candidate=True)
        for name, raw in before.items():
            self.assertEqual((self.workflow.output/name).read_bytes(), raw)

    def test_standard_json_escaping_and_failed_write_preserve_previous_pointer(self):
        path = self.workflow.output / 'CURATION_CURRENT.json'
        # Backslashes, quote characters and non-ASCII also round-trip if a caller
        # supplies native Windows spelling instead of the workflow's forward slashes.
        value = {'selected_version': 'D:\\轻量化数据采集\\paperA\\new\\test',
                 'note': '中文 "引号"\n换行'}
        write_navigation(path, value)
        original = path.read_bytes()
        self.assertEqual(json.loads(original.decode('utf-8')), value)
        with self.assertRaises(ValueError):
            write_navigation(path, {'invalid': float('nan')})
        self.assertEqual(path.read_bytes(), original)
        with patch.object(Path, 'replace', side_effect=PermissionError('busy')):
            with self.assertRaises(PermissionError):
                write_navigation(path, {'selected_version': 'new'})
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(list(self.workflow.output.glob('*.partial')))

    def test_frozen_preview_validates_task_and_index_without_reselecting_current(self):
        import hashlib
        from dataset.manage_dataset import POLICY, pipeline
        run = self.root/'历史裁剪'
        run.mkdir()
        index = run/'frames.jsonl'
        index.write_text('{}\n', encoding='utf-8')
        manifest = {'task_id':self.workflow.task.id, 'task_binding':{'task_id':self.workflow.task.id, 'old':True},
            'engineering_only':True, 'version':pipeline.VERSION, 'pipeline_stage':'curation',
            'selection_policy':POLICY, 'episodes':[{'index_path':'frames.jsonl',
                'index_sha256':hashlib.sha256(index.read_bytes()).hexdigest()}]}
        path=run/'manifest.json'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        pointer=self.workflow.output/'CURATION_CURRENT.json'
        write_navigation(pointer, {'selected_version':run.as_posix()})
        before=pointer.read_bytes()
        with patch('dataset.manage_dataset.serve_review') as serve:
            self.workflow.preview(open_browser=False)
            self.assertTrue(serve.call_args.kwargs['read_only'])
            self.assertTrue(serve.call_args.kwargs['frozen_preview'])
            self.assertEqual(pointer.read_bytes(),before)
        manifest['task_id']='other'
        path.write_text(json.dumps(manifest),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Cross-task'):
            self.workflow.preview(open_browser=False)
        manifest['task_id']=self.workflow.task.id
        path.write_text(json.dumps(manifest),encoding='utf-8')
        index.write_text('changed',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'指纹'):
            self.workflow.preview(open_browser=False)

    def test_preview_menu_occupied_port_does_not_close_old_service_or_process_data(self):
        import socket
        from dataset.manage_dataset import preview_menu
        with socket.socket() as busy:
            if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
                busy.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
            busy.bind(('127.0.0.1',0));busy.listen()
            port=busy.getsockname()[1]
            if port>65515:self.skipTest('No room for adjacent test ports')
            with patch('dataset.manage_dataset.DatasetWorkflow',return_value=self.workflow), \
                 patch('builtins.input',return_value='1'), \
                 patch.object(self.workflow,'preview') as preview, \
                 patch.object(self.workflow,'process') as process:
                preview_menu('tray_to_platform',port,open_browser=False)
                self.assertGreater(preview.call_args.args[1],port)
                self.assertFalse(preview.call_args.kwargs['open_browser'])
                process.assert_not_called()
            self.assertEqual(busy.getsockname()[1],port)


if __name__ == '__main__':
    unittest.main()
