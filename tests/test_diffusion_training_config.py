"""Exercise the DP entrypoints through LeRobot's real configuration decoder."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from training.Diffusion.train_diffusion_variation_baseline import build_cli, make_parser


class DiffusionTrainingConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import draccus
            import torch
            from lerobot.configs.train import TrainPipelineConfig
            from lerobot.optim.optimizers import AdamWConfig
            from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
        except ImportError as error:
            raise unittest.SkipTest(f"LeRobot training dependencies unavailable: {error}") from error

        # Install the same temporal compatibility support as a training run,
        # including registration of DiffusionConfig with the CLI decoder.
        from training.Diffusion.diffusion_runtime import install_temporal_padding

        install_temporal_padding()
        cls.draccus = draccus
        cls.torch = torch
        cls.config_class = TrainPipelineConfig
        cls.optimizer_class = AdamWConfig
        cls.scheduler_class = CosineDecayWithWarmupSchedulerConfig

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def decode_config(self, *, auxiliary=False, motion=False, extra=()):
        args = make_parser(auxiliary=auxiliary, motion=motion).parse_args([
            "--dataset-root", str(self.root / "dataset"),
            "--output-dir", str(self.root / "output"),
            "--device", "cpu",
            *extra,
        ])
        cli = build_cli(args, auxiliary=auxiliary)
        # validate() also consults sys.argv for resume/pretrained overrides.
        with patch.object(sys, "argv", ["train_diffusion.py", *cli]):
            config = self.draccus.parse(self.config_class, args=cli)
            optimizer_before_validation = config.optimizer
            scheduler_before_validation = config.scheduler
            config.validate()
        # A valid-looking CLI can otherwise silently turn AdamW into Adam when
        # LeRobot reapplies its default policy training preset.
        self.assertIs(config.optimizer, optimizer_before_validation)
        self.assertIs(config.scheduler, scheduler_before_validation)
        return config

    def test_all_three_modes_decode_and_preserve_the_shared_training_settings(self):
        modes = (
            (False, False, "diffusion_variation_baseline"),
            (True, False, "diffusion_variation_gesture"),
            (True, True, "diffusion_variation_gesture_hand_delta"),
        )
        for auxiliary, motion, job_name in modes:
            with self.subTest(mode=job_name):
                config = self.decode_config(auxiliary=auxiliary, motion=motion)
                self.assertEqual(config.job_name, job_name)
                self.assertEqual(config.policy.horizon, 30)
                self.assertEqual(config.policy.n_obs_steps, 2)
                self.assertEqual(config.policy.n_action_steps, 8)
                self.assertEqual(config.policy.resize_shape, (480, 640))
                self.assertEqual(config.policy.down_dims, (256, 512, 1024))
                self.assertTrue(config.policy.use_group_norm)
                self.assertIsNone(config.policy.pretrained_backbone_weights)
                self.assertTrue(config.policy.use_film_scale_modulation)
                self.assertEqual(config.policy.noise_scheduler_type, "DDIM")
                self.assertEqual(config.policy.num_train_timesteps, 100)
                self.assertEqual(config.policy.num_inference_steps, 10)
                self.assertEqual(config.policy.prediction_type, "epsilon")
                self.assertTrue(config.policy.do_mask_loss_for_padding)
                self.assertFalse(config.use_policy_training_preset)
                self.assertIsInstance(config.optimizer, self.optimizer_class)
                self.assertEqual(config.optimizer.betas, (0.95, 0.999))
                self.assertEqual(config.optimizer.weight_decay, 1e-6)
                self.assertIsInstance(config.scheduler, self.scheduler_class)
                self.assertEqual(config.scheduler.num_warmup_steps, 500)
                self.assertEqual(config.scheduler.num_decay_steps, config.steps)
                self.assertEqual(config.steps, 40_000)
                self.assertEqual(config.batch_size, 4)
                self.assertEqual((config.eval_steps, config.save_freq), (1000, 1000))
                self.assertTrue(config.ema.enable)

    def test_real_adamw_and_schedule_warm_up_then_reach_and_keep_ten_percent(self):
        config = self.decode_config()
        parameter = self.torch.nn.Parameter(self.torch.tensor(1.0))
        optimizer = config.optimizer.build([parameter])
        self.assertIsInstance(optimizer, self.torch.optim.AdamW)
        self.assertEqual(optimizer.defaults["lr"], 1e-4)
        scheduler = config.scheduler.build(optimizer, config.steps)
        observed = {0: scheduler.get_last_lr()[0]}
        checkpoints = {1, 250, 499, 500, 1000, 20_000, 40_000, 40_001}
        for step in range(1, config.steps + 2):
            optimizer.step()
            scheduler.step()
            if step in checkpoints:
                observed[step] = scheduler.get_last_lr()[0]

        self.assertLess(observed[0], observed[1])
        self.assertLess(observed[1], observed[250])
        self.assertLess(observed[250], observed[499])
        self.assertGreater(observed[500], 0.99 * 1e-4)
        self.assertGreater(observed[500], observed[1000])
        self.assertGreater(observed[1000], observed[20_000])
        self.assertGreater(observed[20_000], observed[40_000])
        self.assertAlmostEqual(observed[40_000], 1e-5)
        self.assertAlmostEqual(observed[40_001], 1e-5)

    def test_learning_rate_override_also_changes_the_schedule_floor(self):
        config = self.decode_config(extra=("--learning-rate", "0.0002", "--steps", "2000"))
        parameter = self.torch.nn.Parameter(self.torch.tensor(1.0))
        optimizer = config.optimizer.build([parameter])
        self.assertEqual(optimizer.defaults["lr"], 2e-4)
        scheduler = config.scheduler.build(optimizer, config.steps)
        for _ in range(config.steps):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 2e-5)

    def test_smoke_run_remains_a_small_valid_configuration(self):
        config = self.decode_config(auxiliary=True, motion=True, extra=("--smoke-test",))
        self.assertEqual(config.steps, 1)
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.num_workers, 0)
        self.assertEqual(config.eval_steps, 0)
        self.assertEqual(config.save_freq, 1)
        self.assertEqual(config.policy.resize_shape, (96, 96))
        self.assertEqual(config.policy.horizon, 8)
        self.assertEqual(config.policy.down_dims, (32, 64))


if __name__ == "__main__":
    unittest.main()
