"""Small CPU checks for DP equations, causal inputs, masks and K=30 alignment."""

from dataclasses import replace
import inspect
import math
import unittest

import torch
from torch import nn

from training.diffusion.model import DiffusionConfig, DiffusionPolicy, SpatialSoftmax
from training.diffusion.schedule import CosineDiffusionSchedule


class DiffusionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(31)
        self.config = DiffusionConfig(observation_horizon=2, feature_dim=16, time_dim=16, down_dims=(16, 32),
                                      prediction_horizon=7, action_horizon=3,
                                      diffusion_steps=12, inference_steps=3)
        self.images = torch.rand(2, 2, 1, 3, 64, 64)
        self.state = torch.randn(2, 2, 16)
        self.actions = torch.randn(2, 7, 14)
        self.is_pad = torch.tensor([[False, False, False, False, True, True, True],
                                    [False, False, False, False, False, False, True]])

    @staticmethod
    def generator(seed=57):
        return torch.Generator().manual_seed(seed)

    def test_padding_values_cannot_change_valid_encoding_loss_or_gradients(self):
        model = DiffusionPolicy(self.config).train()
        altered = self.actions.clone()
        altered[self.is_pad] = float("nan")
        seen = []
        hook = model.denoiser.register_forward_pre_hook(lambda _module, inputs: seen.append(inputs[0].detach().clone()))
        first = model.compute_loss(self.images, self.state, self.actions, self.is_pad, self.generator())
        first["loss"].backward()
        first_gradient = model.denoiser.output[-1].weight.grad.clone()
        model.zero_grad(set_to_none=True)
        second = model.compute_loss(self.images, self.state, altered, self.is_pad, self.generator())
        second["loss"].backward()
        hook.remove()
        torch.testing.assert_close(first["loss"], second["loss"], rtol=0, atol=0)
        torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(seen[0][self.is_pad]).item(), 0)
        torch.testing.assert_close(first_gradient, model.denoiser.output[-1].weight.grad, rtol=0, atol=0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_loss_is_mean_over_valid_coordinates_and_backprop_updates_network(self):
        model = DiffusionPolicy(self.config).train()
        seen = []
        hook = model.denoiser.register_forward_hook(lambda _module, _inputs, output: seen.append(output.detach()))
        losses = model.compute_loss(self.images, self.state, self.actions, self.is_pad, self.generator())
        hook.remove()
        noise = torch.randn(self.actions.shape, generator=self.generator())
        expected = (seen[0] - noise).square()[~self.is_pad].mean()
        torch.testing.assert_close(losses["loss"], expected)
        torch.testing.assert_close(losses["noise_mse"], expected)
        before = model.denoiser.output[-1].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        losses["loss"].backward()
        self.assertIsNotNone(model.backbone[0].weight.grad)
        self.assertGreater(model.backbone[0].weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.denoiser.output[-1].weight))

    def test_all_padding_is_differentiable_zero_even_for_nan_targets(self):
        model = DiffusionPolicy(self.config).train()
        losses = model.compute_loss(self.images, self.state, torch.full_like(self.actions, float("nan")),
                                    torch.ones_like(self.is_pad), self.generator())
        self.assertEqual(losses["loss"].item(), 0)
        losses["loss"].backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() and torch.count_nonzero(g) == 0 for g in gradients))

    def test_seeded_sampling_is_reproducible_varies_by_seed_and_restores_train_mode(self):
        model = DiffusionPolicy(self.config).train()
        first = model.predict_action(self.images, self.state, self.generator(3))
        second = model.predict_action(self.images, self.state, self.generator(3))
        third = model.predict_action(self.images, self.state, self.generator(4))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(torch.equal(first, third))
        self.assertEqual(first.shape, (2, 7, 14))
        self.assertFalse(first.requires_grad)
        self.assertTrue(torch.isfinite(first).all())
        self.assertLessEqual(first.abs().max().item(), 1)
        self.assertTrue(model.training)

    def test_single_camera_shapes_match_and_observation_encoding_is_per_sample(self):
        model = DiffusionPolicy(self.config).eval()
        with torch.no_grad():
            explicit_camera = model.encode_observations(self.images, self.state)
            implicit_camera = model.encode_observations(self.images[:, :, 0], self.state)
            singleton = model.encode_observations(self.images[:1], self.state[:1])
        torch.testing.assert_close(explicit_camera, implicit_camera, rtol=0, atol=0)
        torch.testing.assert_close(explicit_camera[:1], singleton, rtol=1e-4, atol=1e-5)
        self.assertFalse(any(isinstance(module, nn.BatchNorm2d) for module in model.modules()))
        parameters = tuple(inspect.signature(model.predict_action).parameters)
        self.assertEqual(parameters, ("images", "state", "generator", "num_inference_steps"))

    def test_full_k30_plan_aligns_unet_and_does_not_shift_by_observation_horizon(self):
        # Three scales require internal K30 -> K32 padding, then exact cropping.
        model = DiffusionPolicy(replace(self.config, prediction_horizon=30, action_horizon=8,
                                       down_dims=(16, 32, 64))).train()
        actions = torch.randn(2, 30, 14)
        loss = model.compute_loss(self.images, self.state, actions, None, self.generator())["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        result = model.predict_action(self.images, self.state, self.generator(), num_inference_steps=2)
        self.assertEqual(result.shape, (2, 30, 14))
        self.assertTrue(torch.isfinite(result).all())
        # Supplying a perfect epsilon predictor makes a known clean trajectory
        # appear in full: index 0 must remain the next legal tick, with no To-1.
        expected = torch.linspace(-.9, .9, 30).reshape(1, 30, 1).expand(2, 30, 14)
        schedule = model.noise_scheduler

        class Oracle(nn.Module):
            def forward(self, sample, timesteps, _observations):
                alpha = schedule.alphas_cumprod[timesteps].reshape(-1, 1, 1)
                return (sample - alpha.sqrt() * expected) / (1 - alpha).sqrt()

        model.denoiser = Oracle()
        oracle_result = model.predict_action(self.images, self.state, self.generator(), num_inference_steps=3)
        torch.testing.assert_close(oracle_result, expected, rtol=1e-4, atol=2e-4)

    def test_one_step_inference_uses_terminal_training_timestep(self):
        model = DiffusionPolicy(self.config).eval()
        seen = []
        hook = model.denoiser.register_forward_pre_hook(lambda _module, inputs: seen.append(inputs[1].clone()))
        result = model.predict_action(self.images, self.state, self.generator(), num_inference_steps=1)
        hook.remove()
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.equal(seen[0], torch.full((2,), self.config.diffusion_steps - 1)))
        self.assertTrue(torch.isfinite(result).all())

    def test_observation_history_order_is_preserved(self):
        model = DiffusionPolicy(self.config).eval()
        with torch.no_grad():
            normal = model.encode_observations(self.images, self.state)
            reversed_history = model.encode_observations(self.images.flip(1), self.state.flip(1))
        width = self.config.feature_dim + self.config.state_dim
        torch.testing.assert_close(normal.reshape(2, 2, width).flip(1), reversed_history.reshape(2, 2, width))

    def test_invalid_config_observations_targets_and_inference_steps_fail_explicitly(self):
        for updates in ({"state_dim": 8}, {"action_dim": 7}, {"time_dim": 3}, {"down_dims": ()},
                        {"observation_horizon": 0}, {"action_horizon": 8}, {"inference_steps": 13},
                        {"diffusion_steps": True}, {"clip_sample": 1}, {"spatial_keypoints": 0},
                        {"spatial_temperature": 0}, {"spatial_temperature": float("nan")},
                        {"spatial_temperature": True}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                replace(self.config, **updates)
        with self.assertRaises(FileNotFoundError):
            DiffusionPolicy(replace(self.config, backbone_weights="not-a-local-weight-file.pth"))
        model = DiffusionPolicy(self.config)
        with self.assertRaisesRegex(ValueError, "one oblique"):
            model.encode_observations(self.images.expand(-1, -1, 2, -1, -1, -1), self.state)
        with self.assertRaisesRegex(ValueError, "state"):
            model.encode_observations(self.images, self.state[:, -1])
        with self.assertRaisesRegex(ValueError, "RGB"):
            model.encode_observations(self.images + 2, self.state)
        invalid = self.actions.clone()
        invalid[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Unpadded"):
            model.compute_loss(self.images, self.state, invalid, self.is_pad)
        with self.assertRaisesRegex(ValueError, "is_pad"):
            model.compute_loss(self.images, self.state, self.actions, self.is_pad.float())
        for steps in (0, -1, 13, True, 1.5):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                model.predict_action(self.images, self.state, num_inference_steps=steps)


class SpatialSoftmaxTests(unittest.TestCase):
    def test_uniform_maps_have_centered_coordinates_on_rectangular_grid(self):
        pool = SpatialSoftmax(1, 2)
        with torch.no_grad():
            pool.projection.weight.zero_()
            pool.projection.bias.zero_()
        coordinates = pool(torch.ones(3, 1, 3, 5))
        self.assertEqual(coordinates.shape, (3, 4))
        torch.testing.assert_close(coordinates, torch.zeros_like(coordinates), atol=1e-7, rtol=0)

    def test_equal_average_features_keep_different_locations_and_finite_gradients(self):
        pool = SpatialSoftmax(1, 1)
        with torch.no_grad():
            pool.projection.weight.fill_(1)
            pool.projection.bias.zero_()
        features = torch.zeros(2, 1, 3, 5)
        features[0, 0, 1, 0] = 6
        features[1, 0, 1, -1] = 6
        features.requires_grad_()
        torch.testing.assert_close(features[0].mean(), features[1].mean())
        coordinates = pool(features)
        self.assertLess(coordinates[0, 0].item(), -.9)
        self.assertGreater(coordinates[1, 0].item(), .9)
        torch.testing.assert_close(coordinates[:, 1], torch.zeros(2), atol=1e-7, rtol=0)
        coordinates.sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(pool.projection.weight.grad).all())


class DiffusionScheduleTests(unittest.TestCase):
    def test_cosine_coefficients_and_forward_process_match_reference_equation(self):
        schedule = CosineDiffusionSchedule(100, clip_sample=False)
        first_alpha = math.cos((.01 + .008) / 1.008 * math.pi / 2) ** 2
        initial_alpha = math.cos(.008 / 1.008 * math.pi / 2) ** 2
        self.assertAlmostEqual(schedule.betas[0].item(), 1 - first_alpha / initial_alpha, places=7)
        self.assertAlmostEqual(schedule.betas[-1].item(), .999, places=7)
        self.assertTrue(torch.all(schedule.alphas_cumprod[:-1] > schedule.alphas_cumprod[1:]))
        clean = torch.full((2, 3, 14), 2.)
        noise = torch.full_like(clean, -.5)
        timesteps = torch.tensor([0, 99])
        alpha = schedule.alphas_cumprod[timesteps].reshape(2, 1, 1)
        expected = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        torch.testing.assert_close(schedule.add_noise(clean, noise, timesteps), expected)

    def test_ddim_oracle_reconstructs_clean_at_actual_previous_timestep(self):
        schedule = CosineDiffusionSchedule(100, clip_sample=False)
        clean = torch.randn(2, 30, 14)
        noise = torch.randn_like(clean)
        sample = schedule.add_noise(clean, noise, torch.tensor([99, 99]))
        for previous in (85, 49, 0, -1):
            result = schedule.ddim_step(noise, 99, previous, sample)
            if previous < 0:
                expected = clean
            else:
                expected = schedule.add_noise(clean, noise, torch.tensor([previous, previous]))
            torch.testing.assert_close(result, expected, rtol=1e-3, atol=2e-4)

    def test_ddim_clips_predicted_clean_actions_by_default(self):
        schedule = CosineDiffusionSchedule(100)
        sample = torch.tensor([[[-30., 0., 30.]]])
        result = schedule.ddim_step(torch.zeros_like(sample), 99, -1, sample)
        torch.testing.assert_close(result, torch.tensor([[[-1., 0., 1.]]]))

    def test_trailing_grid_is_strict_and_uses_requested_step_count(self):
        schedule = CosineDiffusionSchedule(100)
        for steps in (1, 3, 7, 10, 99, 100):
            timesteps = schedule.inference_timesteps(steps)
            self.assertEqual(len(timesteps), steps)
            self.assertEqual(timesteps[0], 99)
            self.assertTrue(all(a > b for a, b in zip(timesteps, timesteps[1:])))
            self.assertGreaterEqual(timesteps[-1], 0)


if __name__ == "__main__":
    unittest.main()
