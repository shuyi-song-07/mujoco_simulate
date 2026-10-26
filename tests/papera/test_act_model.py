"""Fast CPU tests for action boundaries, missing labels and posterior leakage."""
from dataclasses import replace
import unittest

import torch

from training.papera.act_model import ACTConfig, ACTPolicy, compute_loss


class ACTModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(29)
        self.config = ACTConfig(hidden_dim=32, heads=4, feedforward_dim=64,
                                encoder_layers=1, decoder_layers=1, posterior_layers=1,
                                latent_dim=4, chunk_size=5, dropout=0.0)
        self.images = torch.rand(2, 3, 3, 64, 64)
        self.state = torch.randn(2, 16)
        self.actions = torch.randn(2, 5, 14)
        self.is_pad = torch.tensor([[False, False, False, True, True],
                                    [False, False, False, False, True]])

    def test_padding_is_removed_before_posterior_projection_and_loss(self):
        model = ACTPolicy(self.config).train()
        altered = self.actions.clone()
        altered[self.is_pad] = float("nan")
        torch.manual_seed(8)
        normal = model(self.images, self.state, self.actions, self.is_pad)
        torch.manual_seed(8)
        padded_nan = model(self.images, self.state, altered, self.is_pad)
        for key in ("mu", "logvar", "action"):
            torch.testing.assert_close(normal[key], padded_nan[key], rtol=0, atol=0)
        masks = {"action": ~self.is_pad}
        normal_losses = compute_loss(normal, {"action": self.actions}, masks)
        padded_losses = compute_loss(padded_nan, {"action": altered}, masks)
        for key in ("action", "kl", "total"):
            torch.testing.assert_close(normal_losses[key], padded_losses[key], rtol=0, atol=0)
        self.assertEqual(normal_losses["action_count"].item(), 7 * 14)

    def test_single_camera_preserves_padding_invariance_and_updates_baseline(self):
        model = ACTPolicy(replace(self.config, num_cameras=1)).train()
        images = self.images[:, :1]
        altered = self.actions.clone()
        altered[self.is_pad] = float("nan")
        torch.manual_seed(8)
        normal = model(images, self.state, self.actions, self.is_pad)
        torch.manual_seed(8)
        padded = model(images, self.state, altered, self.is_pad)
        for key in ("mu", "logvar", "action"):
            torch.testing.assert_close(normal[key], padded[key], rtol=0, atol=0)
        loss = compute_loss(padded, {"action": altered}, {"action": ~self.is_pad})
        self.assertEqual(loss["action_count"].item(), 7 * 14)
        before = model.action_head.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss["total"].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.action_head.weight))
        self.assertIsNone(model.gesture_head.weight.grad)
        model.eval()
        output = model(images, self.state, altered)
        self.assertEqual(output["action"].shape, (2, 5, 14))
        self.assertTrue(torch.equal(output["mu"], torch.zeros_like(output["mu"])))
        with self.assertRaisesRegex(ValueError, "1 cameras"):
            model(self.images, self.state)

    def test_auxiliary_heads_cannot_read_future_actions_or_posterior(self):
        model = ACTPolicy(replace(self.config, variant="gesture_skeleton")).train()
        first = model(self.images, self.state, self.actions, self.is_pad)
        second = model(self.images, self.state, self.actions + 500, self.is_pad)
        for key in ("gesture_logits", "skeleton", "stage_logits"):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(first["mu"], second["mu"]))

    def test_predict_uses_only_robot_observations_and_restores_training(self):
        model = ACTPolicy(replace(self.config, dropout=.2)).train()
        first = model.predict(self.images, self.state)
        second = model.predict(self.images, self.state)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(model.training)
        self.assertEqual(first.shape, (2, 5, 14))
        self.assertFalse(first.requires_grad)
        model.eval()
        observed_only = model(self.images, self.state)
        # Even unusable labels cannot affect eval: they are not model inputs.
        labels_supplied = model(self.images, self.state, torch.full_like(self.actions, float("nan")))
        torch.testing.assert_close(observed_only["action"], labels_supplied["action"], rtol=0, atol=0)
        self.assertFalse(observed_only["posterior_mask"].any())

    def test_four_variants_have_finite_backward_and_identical_initial_weights(self):
        previous_state = None
        for variant in ("baseline", "gesture", "gesture_skeleton", "robot_stage"):
            with self.subTest(variant=variant):
                torch.manual_seed(37)
                model = ACTPolicy(replace(self.config, variant=variant)).train()
                if previous_state is not None:
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(previous_state[name], value, rtol=0, atol=0)
                else:
                    previous_state = {k: v.clone() for k, v in model.state_dict().items()}
                skeleton = torch.randn(2, 2, 21, 3)
                skeleton[0, 1] = float("nan")
                targets = {"action": self.actions, "gesture": torch.tensor([[0, -1], [1, 2]]),
                           "skeleton": skeleton, "stage": torch.tensor([2, 7])}
                masks = {"action": ~self.is_pad,
                         "gesture": torch.tensor([[True, False], [True, True]]),
                         "skeleton": torch.tensor([[True, False], [True, True]]),
                         "stage": torch.tensor([True, True])}
                output = model(self.images, self.state, self.actions, self.is_pad)
                losses = compute_loss(output, targets, masks)
                self.assertTrue(torch.isfinite(losses["total"]))
                losses["total"].backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertIsNotNone(model.backbone[0].weight.grad)
                if variant == "baseline":
                    self.assertEqual(losses["gesture_count"].item(), 0)
                    self.assertIsNone(model.gesture_head.weight.grad)
                if variant == "gesture_skeleton":
                    self.assertEqual(losses["gesture_count"].item(), 3)
                    self.assertEqual(losses["skeleton_count"].item(), 3 * 21 * 3)
                    self.assertIsNotNone(model.skeleton_head.weight.grad)

    def test_missing_labels_and_all_padding_have_zero_loss_healthy_gradient(self):
        model = ACTPolicy(replace(self.config, variant="gesture_skeleton")).train()
        pad = torch.ones(2, 5, dtype=torch.bool)
        actions = torch.full_like(self.actions, float("nan"))
        output = model(self.images, self.state, actions, pad)
        targets = {"action": actions, "gesture": torch.full((2, 2), -999, dtype=torch.long),
                   "skeleton": torch.full((2, 2, 21, 3), float("nan"))}
        masks = {"action": ~pad, "gesture": torch.zeros(2, 2, dtype=torch.bool),
                 "skeleton": torch.zeros(2, 2, dtype=torch.bool)}
        losses = compute_loss(output, targets, masks)
        self.assertEqual(losses["total"].item(), 0)
        self.assertEqual(losses["kl_count"].item(), 0)
        losses["total"].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_masked_coordinate_normalization_is_not_batch_or_padding_dependent(self):
        model = ACTPolicy(replace(self.config, variant="gesture_skeleton")).eval()
        output = model(self.images, self.state)
        target_action = output["action"].detach() + 2
        target_skeleton = output["skeleton"].detach() + 2
        target_skeleton[0, 1] = float("nan")
        losses = compute_loss(output, {"action": target_action, "skeleton": target_skeleton},
                              {"action": ~self.is_pad, "skeleton": torch.tensor([[True, False], [True, True]])})
        self.assertAlmostEqual(losses["action"].item(), 2, places=6)
        self.assertAlmostEqual(losses["skeleton"].item(), 1.5, places=6)

    def test_invalid_contract_or_targets_fail_explicitly(self):
        for updates in ({"hidden_dim": 30}, {"action_dim": 7}, {"num_cameras": 2},
                        {"dropout": float("nan")}, {"variant": "mystery"}, {"chunk_size": 0}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                replace(self.config, **updates)
        with self.assertRaises(FileNotFoundError):
            ACTPolicy(replace(self.config, backbone_weights="definitely-not-a-local-weight-file.pt"))
        model = ACTPolicy(self.config).train()
        bad = self.actions.clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Unpadded"):
            model(self.images, self.state, bad, self.is_pad)
        with self.assertRaisesRegex(ValueError, "is_pad"):
            model(self.images, self.state, self.actions, self.is_pad.float())
        with self.assertRaisesRegex(ValueError, "state"):
            model(self.images, self.state[:, :7])
        with self.assertRaisesRegex(ValueError, "RGB"):
            model(self.images + 2, self.state)
        output = model(self.images, self.state, self.actions, self.is_pad)
        with self.assertRaisesRegex(ValueError, "weights"):
            compute_loss(output, {"action": self.actions}, weights={"surprise": 1})
        with self.assertRaisesRegex(ValueError, "weights"):
            compute_loss(output, {"action": self.actions}, weights={"kl": -1})


if __name__ == "__main__":
    unittest.main()
