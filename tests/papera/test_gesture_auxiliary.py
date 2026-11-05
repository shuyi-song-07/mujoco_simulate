"""CPU checks for training-only labels and unchanged diffusion inference."""
from dataclasses import replace
import unittest

import torch

from training.gesture_auxiliary import masked_gesture_loss, summarize_gesture_confusion
from training.papera.act_model import _classification
from training.diffusion.model import DiffusionConfig, DiffusionPolicy


class GestureAuxiliaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_loss_matches_act_mask_and_ignores_unknown(self):
        logits = torch.tensor([[[1., 3., 2.], [float("nan")] * 3],
                               [[2., 1., 3.], [3., 2., 1.]]], requires_grad=True)
        labels = torch.tensor([[1, -999], [2, 1]])
        mask = torch.tensor([[True, False], [True, True]])
        actual = masked_gesture_loss(logits, labels, mask)
        expected, count = _classification(logits, labels, mask, "gesture")
        torch.testing.assert_close(actual["loss"], expected, rtol=0, atol=0)
        self.assertEqual(actual["count"].item(), count.item())
        self.assertEqual(actual["confusion"].sum().item(), 3)
        actual["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 1].abs().sum().item(), 0)

    def test_all_missing_returns_zero_and_false_unknown_required(self):
        logits = torch.full((2, 2, 3), float("nan"), requires_grad=True)
        labels = torch.full((2, 2), -1, dtype=torch.long)
        mask = torch.zeros((2, 2), dtype=torch.bool)
        result = masked_gesture_loss(logits, labels, mask)
        self.assertEqual(result["loss"].item(), 0)
        self.assertEqual(result["count"].item(), 0)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            masked_gesture_loss(torch.zeros_like(logits), labels, ~mask)

    def test_macro_metrics_expose_minority_class_failure(self):
        confusion = torch.tensor([[[62, 0, 0], [23, 0, 0], [15, 0, 0]],
                                  [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        result = summarize_gesture_confusion(confusion)
        self.assertAlmostEqual(result["left"]["macro_recall"], 1 / 3)
        self.assertEqual(result["left"]["recall"], [1., 0., 0.])
        self.assertIsNone(result["right"]["macro_recall"])

    def models(self):
        config = DiffusionConfig(observation_horizon=2, feature_dim=16, time_dim=16,
                                 down_dims=(16, 32), prediction_horizon=7,
                                 action_horizon=3, diffusion_steps=12, inference_steps=3,
                                 image_size=(64,64), crop_shape=(56,56))
        torch.manual_seed(19)
        baseline = DiffusionPolicy(config)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(19)
        gesture = DiffusionPolicy(replace(config, variant="gesture"))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, gesture.state_dict()[key], rtol=0, atol=0)
        return baseline, gesture

    def inputs(self):
        return (torch.rand(2, 2, 1, 3, 64, 64), torch.rand(2, 2, 16), torch.rand(2, 7, 14),
                torch.tensor([[False] * 4 + [True] * 3, [False] * 6 + [True]]))

    def test_shared_initialization_rng_noise_loss_and_sampling_unchanged(self):
        baseline, gesture = self.models()
        images, state, action, pad = self.inputs()
        crop_rng = torch.get_rng_state()
        first = baseline.compute_loss(images, state, action, pad, torch.Generator().manual_seed(13))
        torch.set_rng_state(crop_rng)
        second = gesture.compute_loss(images, state, action, pad, torch.Generator().manual_seed(13),
            gesture=torch.tensor([[0, 1], [2, -1]]),
            gesture_mask=torch.tensor([[True, True], [True, False]]))
        torch.testing.assert_close(first["noise_mse"], second["noise_mse"], rtol=0, atol=0)
        torch.testing.assert_close(second["loss"], first["loss"] + .1 * second["gesture_loss"])
        predicted = baseline.predict_action(images, state, torch.Generator().manual_seed(41))
        changed_head = gesture.predict_action(images, state, torch.Generator().manual_seed(41))
        torch.testing.assert_close(predicted, changed_head, rtol=0, atol=0)
        with torch.no_grad():
            gesture.gesture_head.weight.fill_(float("nan"))
        torch.testing.assert_close(predicted, gesture.predict_action(
            images, state, torch.Generator().manual_seed(41)), rtol=0, atol=0)

    def test_auxiliary_cannot_read_targets_or_noise_and_reaches_only_encoder(self):
        _, model = self.models()
        images, state, action, pad = self.inputs()
        kwargs = dict(gesture=torch.tensor([[0, 1], [2, -1]]),
                      gesture_mask=torch.tensor([[True, True], [True, False]]))
        crop_rng = torch.get_rng_state()
        first = model.compute_loss(images, state, action, pad, torch.Generator().manual_seed(13), **kwargs)
        torch.set_rng_state(crop_rng)
        second = model.compute_loss(images, state, action + 500, pad, torch.Generator().manual_seed(26), **kwargs)
        torch.testing.assert_close(first["gesture_logits"], second["gesture_logits"], rtol=0, atol=0)
        first["gesture_loss"].backward()
        self.assertGreater(model.backbone[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.gesture_head.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.denoiser.parameters()))

    def test_gesture_train_requires_masks_and_eval_can_measure_original_noise(self):
        _, model = self.models()
        images, state, action, pad = self.inputs()
        with self.assertRaisesRegex(ValueError, "explicit"):
            model.compute_loss(images, state, action, pad)
        model.eval()
        result = model.compute_loss(images, state, action, pad)
        torch.testing.assert_close(result["loss"], result["noise_mse"])


if __name__ == "__main__":
    unittest.main()
