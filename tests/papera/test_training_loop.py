"""Resume and validation regressions independent of GPU/data-pack availability."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.papera.act_model import ACTConfig, ACTPolicy, compute_loss
from training.papera.train import (
    CHECKPOINT_VERSION, StatefulBatchSampler, atomic_checkpoint, load_checkpoint,
    lr_multiplier, restore_rng, rng_state, seed_everything, validate, validate_config,
)


class IdentityNormalizer:
    def decode_action(self, values):
        return np.asarray(values)


class PredictionFixture(nn.Module):
    """Prediction values are indexed by state, never by the supplied labels."""
    def __init__(self, predictions, variant="gesture_skeleton"):
        super().__init__()
        self.config = SimpleNamespace(variant=variant)
        self.register_buffer("predictions", predictions)
        self.calls = 0

    def forward(self, images, state, *args, **kwargs):
        if args or kwargs:
            raise AssertionError("Validation leaked future action targets to model")
        if self.training:
            raise AssertionError("Validation must disable dropout/posterior training")
        self.calls += 1
        batch = len(state)
        return {"action": self.predictions[state[:, 0].long()],
                "mu": state.new_zeros(batch, 4), "logvar": state.new_zeros(batch, 4),
                "posterior_mask": torch.zeros(batch, dtype=torch.bool),
                "gesture_logits": torch.tensor([[[8., 0., 0.], [0., 8., 0.]]]).expand(batch, -1, -1),
                "skeleton": state.new_zeros(batch, 2, 21, 3),
                "stage_logits": torch.tensor([[0., 0., 8., 0.]]).expand(batch, -1),
                "variant": self.config.variant}


class TrainingLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_sampler_resume_inside_epoch_and_at_epoch_boundary(self):
        for count in (2, 3, 7):
            with self.subTest(consumed=count):
                sampler = StatefulBatchSampler(11, 4, 8)
                iterator = iter(sampler)
                prefix = [next(iterator) for _ in range(count)]
                saved = copy.deepcopy(sampler.state_dict())
                expected = [next(iterator) for _ in range(15)]
                resumed = StatefulBatchSampler(11, 4, 987)
                resumed.load_state_dict(saved)
                restored_iterator = iter(resumed)
                actual = [next(restored_iterator) for _ in range(15)]
                self.assertEqual(expected, actual)
                if count == 3:
                    self.assertEqual(sorted(sum(prefix, [])), list(range(11)))
                    self.assertEqual(saved["cursor"], 11)
        sampler = StatefulBatchSampler(11, 4, 8)
        state = copy.deepcopy(sampler.state_dict())
        state["order"][0] = state["order"][1]
        with self.assertRaisesRegex(ValueError, "permutation"):
            sampler.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, "count/batch"):
            StatefulBatchSampler(11, 3, 8).load_state_dict(sampler.state_dict())

    def test_rng_checkpoint_is_weights_only_compatible_and_round_trips(self):
        seed_everything(123)
        state = rng_state()
        expected = (random.random(), np.random.rand(5), torch.rand(5))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            atomic_checkpoint(path, {"version": CHECKPOINT_VERSION,
                                     "signature": "test-contract", "rng": state})
            loaded = load_checkpoint(path, "test-contract")
            restore_rng(loaded["rng"])
            actual = (random.random(), np.random.rand(5), torch.rand(5))
            self.assertEqual(expected[0], actual[0])
            np.testing.assert_array_equal(expected[1], actual[1])
            torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
            self.assertFalse(path.with_suffix(".writing").exists())
            with self.assertRaisesRegex(ValueError, "Resume rejected"):
                load_checkpoint(path, "different-contract")
            atomic_checkpoint(path, {"version": "legacy-task3", "signature": "test-contract"})
            with self.assertRaisesRegex(ValueError, "legacy"):
                load_checkpoint(path, "test-contract")

    @staticmethod
    def _new_tiny_training():
        seed_everything(91)
        model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Dropout(.35), nn.Linear(8, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
        schedule = {"warmup_steps": 2, "steps": 7, "minimum_lr_fraction": .1}
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, schedule))
        sampler = StatefulBatchSampler(11, 4, 61)
        # Same loader design as the production trainer: private generator keeps
        # iterator construction from consuming the model's global Torch RNG.
        generator = torch.Generator().manual_seed(62)
        loader = DataLoader(list(range(11)), batch_sampler=sampler, num_workers=0, generator=generator)
        return model, optimizer, scheduler, sampler, loader

    @staticmethod
    def _tiny_step(model, optimizer, scheduler, ids):
        x = torch.stack((ids.float(), ids.float() / 10, torch.ones_like(ids).float()), dim=1)
        target = torch.stack((ids.float() / 11, ids.float() / 7), dim=1)
        optimizer.zero_grad(set_to_none=True)
        # Exercise all three RNG streams used in checkpoint restore.
        prediction = model(x) + torch.randn(len(ids), 2) * .01
        loss = (prediction - target).square().mean() * (1 + random.random() * .01 + np.random.rand() * .01)
        loss.backward(); optimizer.step(); scheduler.step()
        return float(loss.detach())

    def test_resume_reproduces_optimizer_scheduler_rng_and_sample_order(self):
        for cut in (2, 3):
            with self.subTest(cut=cut), tempfile.TemporaryDirectory() as directory:
                model, opt, schedule, sampler, loader = self._new_tiny_training()
                iterator = iter(loader)
                for _ in range(cut):
                    self._tiny_step(model, opt, schedule, next(iterator))
                path = Path(directory) / "checkpoint.pt"
                atomic_checkpoint(path, {"version": CHECKPOINT_VERSION, "signature": "tiny",
                    "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": schedule.state_dict(), "sampler": sampler.state_dict(), "rng": rng_state()})
                expected_losses, expected_ids = [], []
                for _ in range(cut, 7):
                    ids = next(iterator); expected_ids.append(ids.tolist())
                    expected_losses.append(self._tiny_step(model, opt, schedule, ids))
                expected_model = copy.deepcopy(model.state_dict())
                expected_lr = opt.param_groups[0]["lr"]
                # New process equivalent: construct everything, then restore.
                model, opt, schedule, sampler, loader = self._new_tiny_training()
                cp = load_checkpoint(path, "tiny")
                model.load_state_dict(cp["model"]); opt.load_state_dict(cp["optimizer"])
                schedule.load_state_dict(cp["scheduler"]); sampler.load_state_dict(cp["sampler"])
                restore_rng(cp["rng"])
                iterator = iter(loader)
                actual_losses, actual_ids = [], []
                for _ in range(cut, 7):
                    ids = next(iterator); actual_ids.append(ids.tolist())
                    actual_losses.append(self._tiny_step(model, opt, schedule, ids))
                self.assertEqual(expected_ids, actual_ids)
                self.assertEqual(expected_losses, actual_losses)
                self.assertEqual(expected_lr, opt.param_groups[0]["lr"])
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, expected_model[name], rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get("PAPERA_GPU_TESTS") == "1" and torch.cuda.is_available(),
                         "Set PAPERA_GPU_TESTS=1 on a CUDA machine for ACT resume regression")
    def test_gpu_bf16_act_convolution_attention_resume_is_bitwise_equal(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("The regression specifically targets BF16 CUDA execution")
        fixture_generator = torch.Generator().manual_seed(48)
        samples = []
        for index in range(7):
            pad = torch.arange(5) >= (2 if index % 2 == 0 else 5)
            action = torch.randn(5, 14, generator=fixture_generator)
            action[pad] = float("nan")
            samples.append({"images": torch.rand(3, 3, 64, 64, generator=fixture_generator),
                            "state": torch.randn(16, generator=fixture_generator),
                            "action": action, "is_pad": pad})

        def construct():
            seed_everything(2016)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertIn(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), (":4096:8", ":16:8"))
            model = ACTPolicy(ACTConfig(hidden_dim=64, heads=4, feedforward_dim=128,
                encoder_layers=1, decoder_layers=1, posterior_layers=1, latent_dim=8,
                chunk_size=5, dropout=.1)).cuda().train()
            backbone = {id(p) for p in model.backbone.parameters()}
            optimizer = torch.optim.AdamW([
                {"params": [p for p in model.parameters() if id(p) not in backbone], "lr": 1e-4},
                {"params": list(model.backbone.parameters()), "lr": 1e-5}])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .9 ** step)
            sampler = StatefulBatchSampler(len(samples), 2, 2016)
            loader = DataLoader(samples, batch_sampler=sampler, num_workers=0,
                                generator=torch.Generator().manual_seed(2017))
            return model, optimizer, scheduler, sampler, loader

        def step(model, optimizer, scheduler, batch):
            b = {k: v.cuda() for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(b["images"], b["state"], b["action"], b["is_pad"])
                losses = compute_loss(prediction, {"action": b["action"]}, {"action": ~b["is_pad"]})
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); torch.cuda.synchronize()
            return float(losses["total"].detach())

        def payload(model, optimizer, scheduler, sampler):
            return {"version": CHECKPOINT_VERSION, "signature": "small-gpu-act",
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "sampler": sampler.state_dict(), "rng": rng_state()}

        with tempfile.TemporaryDirectory() as directory:
            partial_path, full_path = (Path(directory) / name for name in ("partial.pt", "full.pt"))
            model, optimizer, scheduler, sampler, loader = construct()
            iterator = iter(loader)
            step(model, optimizer, scheduler, next(iterator))
            atomic_checkpoint(partial_path, payload(model, optimizer, scheduler, sampler))
            expected_losses = [step(model, optimizer, scheduler, next(iterator)) for _ in range(2)]
            atomic_checkpoint(full_path, payload(model, optimizer, scheduler, sampler))
            expected = load_checkpoint(full_path, "small-gpu-act")
            del model, optimizer, scheduler, sampler, loader, iterator
            torch.cuda.empty_cache()
            model, optimizer, scheduler, sampler, loader = construct()
            checkpoint = load_checkpoint(partial_path, "small-gpu-act")
            model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"]); sampler.load_state_dict(checkpoint["sampler"])
            restore_rng(checkpoint["rng"])
            iterator = iter(loader)
            actual_losses = [step(model, optimizer, scheduler, next(iterator)) for _ in range(2)]
            self.assertEqual(actual_losses, expected_losses)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), expected["model"][name], rtol=0, atol=0)
            actual_optimizer = optimizer.state_dict()
            self.assertEqual(actual_optimizer["param_groups"], expected["optimizer"]["param_groups"])
            for index, fields in actual_optimizer["state"].items():
                for name, value in fields.items():
                    previous = expected["optimizer"]["state"][index][name]
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value.cpu(), previous.cpu(), rtol=0, atol=0)
                    else:
                        self.assertEqual(value, previous)
            self.assertEqual(scheduler.state_dict(), expected["scheduler"])
            self.assertEqual(sampler.cursor, expected["sampler"]["cursor"])
            current_rng = rng_state()
            torch.testing.assert_close(current_rng["torch"], expected["rng"]["torch"], rtol=0, atol=0)
            for actual, previous in zip(current_rng["cuda"], expected["rng"]["cuda"]):
                torch.testing.assert_close(actual, previous, rtol=0, atol=0)

    def _validation_fixture(self):
        target = torch.zeros(3, 4, 14)
        predicted = torch.zeros_like(target)
        for index in range(3):
            for offset in (0, 7):
                predicted[index, :, offset:offset + 3] = torch.tensor([.003, .004, 0]) * (index + 1)
                predicted[index, :, offset + 6] = .002 * (index + 1)
                target[index, :, offset + 5] = -np.pi + .1
                predicted[index, :, offset + 5] = np.pi - .1
        pad = torch.tensor([[False, False, False, True], [False, True, True, True],
                            [False, False, True, True]])
        # Padding errors must not corrupt either model losses or metrics.
        target[pad] = float("nan")
        predicted[pad] = 100000.
        common = {"images": torch.zeros(3, 3, 3, 32, 32), "state": torch.zeros(3, 16),
                  "action": target, "is_pad": pad,
                  "gesture": torch.tensor([[0, -1], [2, 1], [-1, -1]]),
                  "gesture_mask": torch.tensor([[True, False], [True, True], [False, False]]),
                  "skeleton": torch.ones(3, 2, 21, 3),
                  "skeleton_mask": torch.tensor([[True, False], [True, True], [False, False]]),
                  "stage": torch.tensor([2, 1, 2])}
        common["state"][:, 0] = torch.arange(3)
        common["skeleton"][~common["skeleton_mask"]] = float("nan")
        loader = [{k: v[indices] for k, v in common.items()} for indices in (slice(0, 2), slice(2, 3))]
        return predicted, common, loader

    def test_validation_prior_only_padding_and_weighted_physical_metrics(self):
        predicted, data, loader = self._validation_fixture()
        model = PredictionFixture(predicted).train()
        result = validate(model, loader, IdentityNormalizer(), torch.device("cpu"), "fp32")
        self.assertTrue(model.training)
        self.assertEqual(model.calls, 2)
        self.assertTrue(result["prior_only"])
        self.assertIsNone(result["success_rate"])
        self.assertEqual(result["valid_action_coordinates"], 6 * 14)
        # Batch sizes and valid lengths differ: this catches mean-of-means bugs.
        self.assertAlmostEqual(result["xyz_error_mm"], (5 * 3 + 10 * 1 + 15 * 2) / 6, places=5)
        self.assertAlmostEqual(result["gripper_error_mm"], (2 * 3 + 4 * 1 + 6 * 2) / 6, places=5)
        self.assertAlmostEqual(result["rpy_mae_degrees"], .2 / 3 * 180 / np.pi, places=4)
        direct_l1 = (predicted - data["action"]).abs()[~data["is_pad"]].mean().item()
        self.assertAlmostEqual(result["prior_action_l1"], direct_l1, places=6)
        self.assertEqual(result["gesture_support"], 3)
        self.assertAlmostEqual(result["gesture_accuracy"], 2 / 3)
        self.assertEqual(result["skeleton_coordinate_count"], 3 * 21 * 3)
        self.assertAlmostEqual(result["skeleton_huber"], .5)
        stage_model = PredictionFixture(predicted, "robot_stage").eval()
        stage = validate(stage_model, loader, IdentityNormalizer(), torch.device("cpu"), "fp32")
        self.assertAlmostEqual(stage["stage_accuracy"], 2 / 3)
        self.assertFalse(stage_model.training)

    def test_validation_restores_train_mode_after_error(self):
        predicted, _, loader = self._validation_fixture()
        loader[0]["action"][0, 0, 0] = float("nan")
        model = PredictionFixture(predicted).train()
        with self.assertRaisesRegex(ValueError, "finite"):
            validate(model, loader, IdentityNormalizer(), torch.device("cpu"), "fp32")
        self.assertTrue(model.training)

    def test_validation_without_valid_actions_cannot_report_perfect_score(self):
        # An empty loader must never create a false best checkpoint.
        model = PredictionFixture(torch.zeros(1, 4, 14))
        with self.assertRaises((ValueError, RuntimeError)):
            validate(model, [], IdentityNormalizer(), torch.device("cpu"), "fp32")

    def test_schedule_and_invalid_config(self):
        config = json.loads((Path(__file__).resolve().parents[2] / "config/training_baseline.json").read_text())
        validate_config(config)
        self.assertAlmostEqual(lr_multiplier(0, config), 1 / config["warmup_steps"])
        self.assertAlmostEqual(lr_multiplier(config["warmup_steps"], config), 1.)
        self.assertAlmostEqual(lr_multiplier(config["steps"], config), config["minimum_lr_fraction"])
        for key, value in (("num_workers", 9), ("steps", 0), ("batch_size", 0),
                           ("precision", "fp16"), ("allow_exploratory", False), ("seed", -1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config({**config, key: value})


if __name__ == "__main__":
    unittest.main()
