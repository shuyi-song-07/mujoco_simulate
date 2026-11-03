"""Complete updates, checkpoints and exact CPU continuation on real Pack tensors."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from tests.papera.test_safe_stitch_data import make_pack
from training.diffusion.train import ROOT, EMA, run, validate_config
from training.diffusion.inference import load_policy


class DiffusionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / '.local/checks/diffusion').mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT / '.local/checks/diffusion')
        cls.root = Path(cls.temp.name)
        cls.pack = make_pack(cls.root)
        cls.config = json.loads((ROOT / 'config/training_diffusion.json').read_text(encoding='utf-8'))
        cls.config.update(device='cpu', precision='fp32', steps=4, batch_size=2,
            num_workers=0, cpu_threads=2, warmup_steps=1, validation_every=2,
            validation_batches=1, validation_sample_batches=1, save_every=2,
            log_every=1, image_size=[64,48])
        cls.config['model'].update(feature_dim=16, time_dim=16, down_dims=[16,32],
            diffusion_steps=8, inference_steps=2, backbone_weights=None, observation_horizon=1)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_update_ema_checkpoint_and_cpu_resume_equal_uninterrupted(self):
        a = run(self.config, self.pack, 'tray_transfer', output=self.root / 'uninterrupted', engineering=True)
        b = run(self.config, self.pack, 'tray_transfer', output=self.root / 'resumed', stop_after=2, engineering=True)
        initial = torch.load(b / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
        self.assertEqual(initial['step'], 2)
        self.assertEqual(initial['ema_updates'], 2)
        self.assertTrue(any(not torch.equal(initial['model'][key], value)
                            for key, value in initial['ema'].items()))
        run(self.config, self.pack, 'tray_transfer', resume=b / 'checkpoints/last.pt', engineering=True)
        full = torch.load(a / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
        resumed = torch.load(b / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
        self.assertEqual(resumed['step'], 4)
        self.assertEqual(resumed['sampler']['cursor'], full['sampler']['cursor'])
        for section in ('model', 'ema'):
            for key in full[section]:
                torch.testing.assert_close(full[section][key], resumed[section][key], rtol=0, atol=0)
        self.assertEqual(resumed['scheduler'], full['scheduler'])
        self.assertEqual(json.loads((b / 'completion.json').read_text())['status'], 'complete')
        self.assertTrue((b / 'tensorboard').is_dir())
        self.assertTrue((b / 'checkpoints/best.pt').is_file())
        policy, normalizer, metadata = load_policy(b / 'checkpoints/best.pt')
        self.assertEqual(metadata['task_id'], 'tray_transfer')
        with torch.no_grad():
            pred = policy.predict_action(torch.zeros(1,1,1,3,48,64), torch.zeros(1,1,16),
                                         generator=torch.Generator().manual_seed(12))
        self.assertEqual(normalizer.decode_action(pred.numpy()).shape, (1,30,14))

    def test_check_only_creates_no_training_run(self):
        output = self.root / 'check_only_must_not_exist'
        result = run(self.config, self.pack, 'tray_transfer', output=output, check_only=True, engineering=True)
        self.assertEqual(result['state'], [1,16])
        self.assertEqual(result['action'], [30,14])
        self.assertFalse(output.exists())

    def test_config_rejects_invalid_budget_and_nonfinite_optimizer_values(self):
        for key, value in (('steps', 0), ('num_workers', -1), ('learning_rate', float('nan')),
                           ('ema_decay', 1.), ('betas', [.9, 1.])):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(config)


if __name__ == '__main__':
    unittest.main()
