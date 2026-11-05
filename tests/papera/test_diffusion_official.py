"""Differential checks for the vendored official DP core and task adapters.

These compare actual production calls against unwrapped upstream operators.
They require no downloaded benchmark data, checkpoint or temporary directory.
"""
import inspect
from pathlib import Path
import unittest

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop
from diffusers import DDPMScheduler
from robomimic.models.base_nets import VisualCore, CropRandomizer as RobomimicCropRandomizer
from robomimic.models.obs_nets import ObservationEncoder

from training.diffusion.model import DiffusionConfig, DiffusionPolicy
from training.diffusion.official.conditional_unet1d import ConditionalUnet1D
from training.diffusion.official.crop_randomizer import CropRandomizer


def tiny_config(**overrides):
    fields = dict(observation_horizon=2, feature_dim=16, time_dim=16,
                  down_dims=(16, 32, 64), prediction_horizon=30,
                  action_horizon=8, diffusion_steps=12, inference_steps=3,
                  image_size=(64, 64), crop_shape=None, backbone_weights=None)
    fields.update(overrides)
    return DiffusionConfig(**fields)


class OfficialDiffusionCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(104)

    def reference_unet(self, model):
        c = model.config
        reference = ConditionalUnet1D(
            input_dim=c.action_dim, local_cond_dim=None,
            global_cond_dim=c.observation_horizon * (c.feature_dim + c.state_dim),
            diffusion_step_embed_dim=c.time_dim, down_dims=list(c.down_dims),
            kernel_size=c.kernel_size, n_groups=c.n_groups,
            cond_predict_scale=c.cond_predict_scale,
        )
        reference.load_state_dict(model.denoiser.network.state_dict(), strict=True)
        return reference.eval()

    def test_aligned_sequence_matches_unmodified_official_unet_with_same_weights(self):
        policy = DiffusionPolicy(tiny_config(prediction_horizon=32)).eval()
        self.assertIsInstance(policy.denoiser.network, ConditionalUnet1D)
        source = Path(inspect.getfile(type(policy.denoiser.network))).resolve()
        self.assertEqual(source.parent.name, "official")
        reference = self.reference_unet(policy)
        sample, timesteps = torch.randn(2, 32, 14), torch.tensor([0, 11])
        condition = torch.randn(2, 64)
        with torch.no_grad():
            actual = policy.denoiser(sample, timesteps, condition)
            expected = reference(sample, timesteps, global_cond=condition)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(set(reference.state_dict()), set(policy.denoiser.network.state_dict()))

    def test_k30_is_only_zero_length_alignment_and_exact_output_crop(self):
        policy = DiffusionPolicy(tiny_config()).eval()
        reference = self.reference_unet(policy)
        sample, timesteps = torch.randn(2, 30, 14), torch.tensor([1, 10])
        condition = torch.randn(2, 64)
        with self.assertRaises(RuntimeError):
            reference(sample, timesteps, global_cond=condition)
        padded = F.pad(sample.transpose(1, 2), (0, 2)).transpose(1, 2)
        with torch.no_grad():
            expected = reference(padded, timesteps, global_cond=condition)[:, :30]
            actual = policy.denoiser(sample, timesteps, condition)
        self.assertEqual(actual.shape, (2, 30, 14))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_policy_ddpm_chain_matches_upstream_every_step_and_rng(self):
        policy = DiffusionPolicy(tiny_config()).eval()
        self.assertIs(type(policy.noise_scheduler), DDPMScheduler)
        images = torch.rand(1, 2, 1, 3, 64, 64)
        state = torch.randn(1, 2, 16)
        model_steps = []
        hook = policy.denoiser.register_forward_pre_hook(
            lambda _module, args: model_steps.append(args[0].detach().clone()))
        generator = torch.Generator().manual_seed(92)
        actual = policy.predict_action(images, state, generator=generator)
        hook.remove()
        actual_rng_state = generator.get_state()
        reference = DDPMScheduler.from_config(policy.noise_scheduler.config)
        reference.set_timesteps(policy.config.inference_steps)
        ref_generator = torch.Generator().manual_seed(92)
        sample = torch.randn((1, 30, 14), generator=ref_generator)
        reference_steps = []
        with torch.no_grad():
            condition = policy.encode_observations(images, state)
            for step in reference.timesteps:
                reference_steps.append(sample.clone())
                prediction = policy.denoiser(sample, step.expand(1), condition)
                sample = reference.step(prediction, step, sample, generator=ref_generator).prev_sample
        self.assertEqual(len(model_steps), policy.config.inference_steps)
        for actual_step, expected_step in zip(model_steps, reference_steps):
            torch.testing.assert_close(actual_step, expected_step, rtol=0, atol=0)
        torch.testing.assert_close(actual, sample, rtol=0, atol=0)
        self.assertTrue(torch.equal(actual_rng_state, ref_generator.get_state()))

    def test_padded_targets_are_sanitized_then_receive_real_forward_noise(self):
        policy = DiffusionPolicy(tiny_config()).train()
        images = torch.rand(1, 2, 1, 3, 64, 64)
        state = torch.randn(1, 2, 16)
        action = torch.rand(1, 30, 14)
        mask = torch.arange(30).unsqueeze(0) >= 19
        action[mask] = float("nan")
        observed = []
        hook = policy.denoiser.register_forward_pre_hook(
            lambda _module, args: observed.append((args[0].detach().clone(), args[1].detach().clone())))
        loss = policy.compute_loss(images, state, action, mask,
                                   generator=torch.Generator().manual_seed(93))["loss"]
        hook.remove()
        self.assertTrue(torch.isfinite(loss))
        sample, timesteps = observed[0]
        noise = torch.randn(action.shape, generator=torch.Generator().manual_seed(93))
        expected = policy.noise_scheduler.add_noise(action.masked_fill(mask.unsqueeze(-1), 0),
                                                    noise, timesteps)
        torch.testing.assert_close(sample, expected, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(sample).all())
        self.assertGreater(torch.count_nonzero(sample[mask]).item(), 0)


class OfficialVisualInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make_model(self):
        torch.manual_seed(501)
        return DiffusionPolicy(tiny_config(crop_shape=(56, 56)))

    def test_uses_robomimic_visual_core_and_official_group_rule(self):
        policy = self.make_model()
        self.assertIsInstance(policy.visual_encoder, VisualCore)
        groups = [module for module in policy.visual_encoder.backbone.modules()
                  if isinstance(module, nn.GroupNorm)]
        self.assertTrue(groups)
        self.assertFalse(any(isinstance(module, nn.BatchNorm2d)
                             for module in policy.visual_encoder.modules()))
        for module in groups:
            self.assertEqual(module.num_groups, module.num_channels // 16)
        self.assertEqual(policy.visual_encoder.output_shape([3, 56, 56]), [16])

    def test_eval_encoder_receives_official_rgb_range_and_center_crop(self):
        policy = self.make_model().eval()
        images = torch.rand(2, 2, 1, 3, 64, 64)
        state = torch.randn(2, 2, 16)
        seen = []
        hook = policy.visual_encoder.register_forward_pre_hook(
            lambda _module, args: seen.append(args[0].detach().clone()))
        with torch.no_grad():
            encoded = policy.encode_observations(images, state)
        hook.remove()
        expected = center_crop(images.reshape(4, 3, 64, 64) * 2 - 1, [56, 56])
        torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
        with torch.no_grad():
            features = F.relu(policy.visual_encoder(expected)).reshape(2, 2, 16)
        expected_encoding = torch.cat((state, features), -1).flatten(1)
        torch.testing.assert_close(encoded, expected_encoding, rtol=0, atol=0)

    def test_full_condition_matches_original_observation_encoder_in_train_and_eval(self):
        policy = self.make_model()
        # Build the original robomimic observation group separately. Merely
        # calling VisualCore twice would miss its surrounding ReLU and key order.
        visual = VisualCore(
            input_shape=[3, 56, 56], backbone_class="ResNet18Conv",
            backbone_kwargs={"pretrained": False, "input_coord_conv": False},
            pool_class="SpatialSoftmax", pool_kwargs={
                "num_kp": 32, "temperature": 1., "learnable_temperature": False,
                "noise_std": 0., "output_variance": False},
            flatten=True, feature_dimension=16)

        def official_group_norm(module):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.BatchNorm2d):
                    setattr(module, name, nn.GroupNorm(child.num_features // 16, child.num_features))
                else:
                    official_group_norm(child)

        official_group_norm(visual)
        visual.load_state_dict(policy.visual_encoder.state_dict(), strict=True)
        reference = ObservationEncoder(feature_activation=nn.ReLU)
        reference.register_obs_key("agent_pos", [16])
        reference.register_obs_key("image", [3, 64, 64], net=visual,
            randomizer=RobomimicCropRandomizer(input_shape=[3, 64, 64],
                crop_height=56, crop_width=56, num_crops=1, pos_enc=False))
        reference.make()
        # The upstream hybrid replaces robomimic's randomizer after building
        # the observation encoder to obtain center-crop evaluation semantics.
        reference.obs_randomizers["image"] = CropRandomizer(
            input_shape=[3, 64, 64], crop_height=56, crop_width=56, num_crops=1, pos_enc=False)
        self.assertEqual(list(reference.obs_shapes), ["agent_pos", "image"])
        images, state = torch.rand(2, 2, 1, 3, 64, 64), torch.randn(2, 2, 16)
        for train in (False, True):
            with self.subTest(training=train), torch.no_grad():
                policy.train(train)
                reference.train(train)
                torch.manual_seed(992)
                actual = policy.encode_observations(images, state)
                actual_rng = torch.get_rng_state()
                torch.manual_seed(992)
                expected = reference({"agent_pos": state.reshape(4, 16),
                                      "image": images.reshape(4, 3, 64, 64) * 2 - 1})
                expected = expected.reshape(2, 2, 32).flatten(1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(torch.equal(actual_rng, torch.get_rng_state()))

    def test_train_crop_matches_official_operator_with_same_global_rng(self):
        policy = self.make_model().train()
        self.assertIsInstance(policy.crop_randomizer, CropRandomizer)
        reference = CropRandomizer(input_shape=[3, 64, 64], crop_height=56,
                                   crop_width=56, num_crops=1, pos_enc=False).train()
        image = torch.rand(3, 3, 64, 64)
        torch.manual_seed(991)
        expected = reference(image)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(991)
        actual = policy.crop_randomizer(image)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        torch.manual_seed(992)
        self.assertFalse(torch.equal(actual, policy.crop_randomizer(image)))


if __name__ == "__main__":
    unittest.main()
