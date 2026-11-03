"""Exercise the real temporal U-Net at a 30-action horizon on CPU."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d

from training.Diffusion.diffusion_runtime import install_temporal_padding


class DiffusionTemporalPaddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        install_temporal_padding()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_config(self, horizon=30, **overrides):
        kwargs = dict(
            device="cpu",
            horizon=horizon,
            n_obs_steps=2,
            n_action_steps=8,
            down_dims=(8, 16, 32),
            kernel_size=3,
            n_groups=4,
            diffusion_step_embed_dim=8,
            pretrained_backbone_weights=None,
            input_features={
                "observation.state": PolicyFeature(FeatureType.STATE, (9,)),
                "observation.environment_state": PolicyFeature(FeatureType.ENV, (1,)),
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        )
        kwargs.update(overrides)
        return DiffusionConfig(**kwargs)

    def test_thirty_actions_forward_backward_and_conditioning_hook(self):
        model = DiffusionConditionalUnet1d(self.make_config(), global_cond_dim=18)
        actions = torch.randn(2, 30, 7, requires_grad=True)
        condition = torch.randn(2, 18, requires_grad=True)
        timestep = torch.tensor([0, 99])
        captured = []

        def capture(_module, args, kwargs):
            captured.append((args[0].shape, kwargs["global_cond"]))

        handle = model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            prediction = model(actions, timestep, global_cond=condition)
        finally:
            handle.remove()
        self.assertEqual(prediction.shape, actions.shape)
        self.assertEqual(captured[0][0], actions.shape)
        self.assertIs(captured[0][1], condition)

        # Only the thirty real action positions become noise-MSE targets.
        target = torch.randn_like(actions)
        torch.nn.functional.mse_loss(prediction, target).backward()
        for gradient in (actions.grad, condition.grad, model.final_conv[-1].weight.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_padding_matches_replicated_boundary_and_cropped_output(self):
        model = DiffusionConditionalUnet1d(self.make_config(), global_cond_dim=18).eval()
        actions = torch.randn(1, 30, 7)
        condition = torch.randn(1, 18)
        timestep = torch.tensor([10])
        extended = torch.cat((actions, actions[:, -1:].expand(-1, 2, -1)), dim=1)
        original_forward = DiffusionConditionalUnet1d.forward.__wrapped__
        with torch.no_grad():
            expected = original_forward(model, extended, timestep, global_cond=condition)[:, :30]
            actual = model(actions, timestep, global_cond=condition)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_aligned_lengths_preserve_upstream_results(self):
        original_forward = DiffusionConditionalUnet1d.forward.__wrapped__
        for horizon in (8, 16, 32):
            with self.subTest(horizon=horizon):
                model = DiffusionConditionalUnet1d(
                    self.make_config(horizon), global_cond_dim=18
                ).eval()
                actions = torch.randn(1, horizon, 7)
                condition = torch.randn(1, 18)
                timestep = torch.tensor([50])
                with torch.no_grad():
                    expected = original_forward(model, actions, timestep, global_cond=condition)
                    actual = model(actions, timestep, global_cond=condition)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_serialization_retains_thirty_action_horizon(self):
        config = self.make_config()
        self.assertEqual(config.horizon, 30)
        self.assertEqual(len(config.action_delta_indices), 30)
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            saved = json.loads((Path(directory) / "config.json").read_text())
            self.assertEqual(saved["horizon"], 30)
            restored = DiffusionConfig.from_pretrained(directory, local_files_only=True)
        self.assertEqual(restored.horizon, 30)
        self.assertEqual(restored.n_action_steps, 8)

    def test_state_dict_remains_compatible_with_upstream_unet(self):
        aligned = DiffusionConditionalUnet1d(self.make_config(32), global_cond_dim=18)
        unaligned = DiffusionConditionalUnet1d(self.make_config(30), global_cond_dim=18)
        self.assertEqual(aligned.state_dict().keys(), unaligned.state_dict().keys())
        unaligned.load_state_dict(aligned.state_dict(), strict=True)

    def test_installation_is_idempotent(self):
        validation = DiffusionConfig.__post_init__
        forward = DiffusionConditionalUnet1d.forward
        install_temporal_padding()
        self.assertIs(DiffusionConfig.__post_init__, validation)
        self.assertIs(DiffusionConditionalUnet1d.forward, forward)

    def test_nonpositive_horizon_is_rejected(self):
        for horizon in (0, -1):
            with self.subTest(horizon=horizon), self.assertRaisesRegex(ValueError, "positive"):
                self.make_config(horizon)

    def test_other_upstream_validation_still_runs(self):
        with self.assertRaisesRegex(ValueError, "ResNet"):
            self.make_config(vision_backbone="not_a_resnet")


if __name__ == "__main__":
    unittest.main()
