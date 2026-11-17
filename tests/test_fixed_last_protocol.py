"""All-training membership, immutable conversion and completed-Last contract."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from dataset.common import training_pack
from tests.papera.test_safe_stitch_data import make_pack
from training.papera.pack_data import prepare_pack
from training.protocol import VERSION, validate_training_data, validate_last

ROOT = Path(__file__).resolve().parents[1]


class FixedLastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = ROOT / '.local/checks/all_train_last_20261116'
        folder.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=folder)
        cls.root = Path(cls.temp.name)
        cls.old = make_pack(cls.root)
        cls.original = (cls.old/'manifest.json').read_bytes()
        cls.pack = cls.root/'all_train'
        cls.manifest = training_pack.repartition_all_train(cls.old, cls.pack)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_conversion_preserves_observations_actions_and_certificates(self):
        self.assertEqual(self.original, (self.old/'manifest.json').read_bytes())
        checked = training_pack.verify(self.pack)
        self.assertEqual(checked['episodes'], 2)
        self.assertEqual(self.manifest['split']['val'], [])
        for entry in self.manifest['episodes']:
            a = training_pack.read_episode(self.old, entry['key'])
            b = training_pack.read_episode(self.pack, entry['key'], checked=checked)
            try:
                for field in a['arrays']:
                    np.testing.assert_array_equal(a['arrays'][field], b['arrays'][field])
                np.testing.assert_array_equal(a['images']['oblique'], b['images']['oblique'])
                self.assertEqual(a['diagnostics'], b['diagnostics'])
            finally:
                a['images']['oblique']._mmap.close(); b['images']['oblique']._mmap.close()
        with self.assertRaises((ValueError, FileExistsError)):
            training_pack.repartition_all_train(self.old, self.old)

    def test_all_episodes_contribute_to_statistics(self):
        binding = prepare_pack(self.pack, image_size=(64,48))
        validate_training_data({'experiment_protocol': VERSION}, binding)
        self.assertEqual(len(binding['train_keys']), 2)
        self.assertFalse(binding['validation_keys'])
        # The fixture's two sessions have joint values 0 and 100.
        self.assertAlmostEqual(binding['stats']['state_mean'][0], 50.)
        wrong = copy.deepcopy(binding); wrong['split']['val'] = ['anything']
        with self.assertRaises(ValueError):
            validate_training_data({'experiment_protocol': VERSION}, wrong)

    def test_interrupted_last_and_best_are_not_final_models(self):
        config = {'experiment_protocol': VERSION, 'steps': 100000}
        completed = {'status': 'complete', 'step':100000, 'budget_steps':100000}
        validate_last(config, completed, 100000, 'last')
        for status, step, name in [('interrupted', 2, 'last'), ('complete',99999,'last'),
                                    ('complete',100000,'best'), ('complete',100000,'step_100000')]:
            with self.subTest(status=status, step=step, name=name), self.assertRaises(ValueError):
                validate_last(config, dict(completed,status=status), step, name)

    def test_frozen_pack_survives_current_source_changes_without_changing_task(self):
        from tasks.artifacts import check_artifact, check_training_pack
        from tasks.registry import get
        from training.papera.pack_data import restore_binding
        task = get('tray_to_platform')
        current = copy.deepcopy(task.binding())
        current['sources']['tasks/tray_to_platform/evaluation.json'] = '0'*64
        updated = Mock()
        updated.binding.return_value = current
        with patch('tasks.registry.get', return_value=updated):
            with self.assertRaises(ValueError):
                check_artifact(self.pack, 'tray_to_platform', engineering=True)
            frozen = check_training_pack(self.pack, 'tray_to_platform', engineering=True)
            self.assertEqual(frozen['task_binding'], self.manifest['task_binding'])
            binding = prepare_pack(self.pack, image_size=(64,48), check_hashes=False)
            self.assertFalse(binding['check_hashes'])
            self.assertEqual(dict(restore_binding(binding)), dict(binding))
            with self.assertRaises(ValueError):
                check_training_pack(self.pack, 'h_frame_push_align', engineering=True)
            with self.assertRaises(ValueError):
                check_training_pack(self.pack, 'tray_to_platform', engineering=False)
            current['schema']['action_dim'] = 7
            with self.assertRaises(ValueError):
                check_training_pack(self.pack, 'tray_to_platform', engineering=True)

    def test_dp_no_validation_and_exact_resume(self):
        from training.diffusion.train import run
        cfg = json.loads((ROOT/'config/training_diffusion.json').read_text(encoding='utf-8'))
        cfg.update(device='cpu', steps=4, batch_size=2, num_workers=0, cpu_threads=2,
                   warmup_steps=1, save_every=2, log_every=1, image_size=[64,48], candidate_save_every=2)
        cfg['model'].update(feature_dim=16,time_dim=16,down_dims=[16,32],diffusion_steps=8,
                            inference_steps=2,backbone_weights=None,crop_shape=[42,56])
        full = run(cfg,self.pack,'tray_to_platform',output=self.root/'full',engineering=True)
        resumed = run(cfg,self.pack,'tray_to_platform',output=self.root/'resumed',engineering=True,stop_after=2)
        run(cfg,self.pack,'tray_to_platform',resume=resumed/'checkpoints/last.pt',engineering=True)
        a = torch.load(full/'checkpoints/last.pt',weights_only=False,map_location='cpu')
        b = torch.load(resumed/'checkpoints/last.pt',weights_only=False,map_location='cpu')
        for field in ('model','ema'):
            for key in a[field]:
                torch.testing.assert_close(a[field][key],b[field][key],rtol=0,atol=0)
        self.assertEqual(a['ema_state'],b['ema_state'])
        self.assertEqual(a['scheduler'],b['scheduler'])
        self.assertEqual(a['step'],4)
        self.assertFalse((resumed/'checkpoints/best.pt').exists())
        rows=[json.loads(r) for r in (resumed/'metrics.jsonl').read_text().splitlines()]
        self.assertFalse(any(r['event']=='validation' for r in rows))
        done=json.loads((resumed/'completion.json').read_text())
        validate_last(cfg,done,b['step'],'last')


if __name__ == '__main__':
    unittest.main()
