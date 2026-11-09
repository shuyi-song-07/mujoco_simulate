"""Real ACT-loss updates, iterator ordering and exact-resume engine regressions."""
import ast
import copy
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.papera.act_model import compute_loss
from training.papera.engine import (
    move_batch, targets_masks, precision_context, optimize_batch, run_loop,
)
from training.papera.train import (
    StatefulBatchSampler, rng_state, restore_rng, seed_everything,
)


class TinyPolicy(nn.Module):
    """Small differentiable policy; the production ACT Loss remains real."""
    def __init__(self, nonfinite=False):
        super().__init__()
        self.dropout = nn.Dropout(.35)
        self.action_head = nn.Linear(3, 14)
        self.latent_head = nn.Linear(3, 2)
        self.nonfinite = nonfinite

    def forward(self, images, state, actions, is_pad):
        action = self.action_head(self.dropout(state))[:, None].expand_as(actions)
        mu = self.latent_head(state)
        return {"action": action, "mu": mu,
                "logvar": torch.full_like(mu, 1e5) if self.nonfinite else torch.zeros_like(mu),
                "posterior_mask": ~is_pad.all(1), "variant": "baseline"}


def sample(index):
    state = torch.tensor([index / 10., .2, 1.])
    return {"images": torch.zeros(1, 3, 2, 2), "state": state,
            "action": torch.full((2, 14), index / 50.),
            "is_pad": torch.tensor([False, bool(index % 2)]),
            "gesture": torch.zeros(2, dtype=torch.long),
            "gesture_mask": torch.zeros(2, dtype=torch.bool),
            "skeleton": torch.zeros(2, 21, 3),
            "skeleton_mask": torch.zeros(2, 21, dtype=torch.bool),
            "stage": torch.tensor(0, dtype=torch.long)}


def configuration(steps=7):
    return {"steps": steps, "precision": "fp32", "gradient_clip_norm": 1.,
            "loss_weights": {"action": 1., "kl": 10.}}


def new_training(seed=103):
    seed_everything(seed)
    model = TinyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda tick: 1. / (1 + tick / 10))
    sampler = StatefulBatchSampler(7, 3, 107, prefetch_safe=True)
    generator = torch.Generator().manual_seed(109)
    loader = DataLoader([sample(i) for i in range(7)], batch_sampler=sampler,
                        generator=generator, num_workers=0)
    return model, optimizer, scheduler, sampler, generator, loader


class TrainingEngineTests(unittest.TestCase):
    def test_finite_iterator_restarts_after_validation_and_can_stop(self):
        model, optimizer, scheduler, _, _, _ = new_training()
        batch = next(iter(DataLoader([sample(0), sample(1)], batch_size=2)))
        trace = []

        class Loader:
            def __iter__(self):
                trace.append("iterator")
                yield batch
                yield batch

        def after(step, losses, norm):
            trace.append(f"step{step}")
            self.assertTrue(torch.isfinite(losses["total"]))
            self.assertTrue(torch.isfinite(norm))
            if step == 2:
                trace.append("validation")
            return step < 3

        before = model.action_head.weight.detach().clone()
        step = run_loop(model, Loader(), optimizer, scheduler, device=torch.device("cpu"),
                        config=configuration(8), variant="baseline",
                        before_iterator=lambda: trace.append("initial-validation"), after_step=after)
        self.assertEqual(step, 3)
        self.assertEqual(trace, ["initial-validation", "iterator", "step1", "step2",
                                 "validation", "iterator", "step3"])
        self.assertFalse(torch.equal(before, model.action_head.weight))
        self.assertTrue(optimizer.state)

    def test_checkpoint_resume_preserves_samples_rng_and_optimizer_updates(self):
        for cut in (2, 3):
            with self.subTest(cut=cut):
                model, optimizer, scheduler, sampler, generator, loader = new_training()
                original_losses = []
                checkpoint = {}

                def after(step, losses, _norm):
                    sampler.mark_consumed()
                    original_losses.append(float(losses["total"].detach()))
                    if step == cut:
                        checkpoint.update(copy.deepcopy({"model": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "sampler": sampler.state_dict(), "rng": rng_state(),
                            "loader_generator": generator.get_state()}))

                run_loop(model, loader, optimizer, scheduler, device=torch.device("cpu"),
                         config=configuration(), variant="baseline", after_step=after)
                expected_model = copy.deepcopy(model.state_dict())
                expected_optimizer = copy.deepcopy(optimizer.state_dict())
                expected_rng = torch.get_rng_state().clone()

                resumed, opt, sched, samples, gen, batches = new_training(seed=999)
                resumed.load_state_dict(checkpoint["model"])
                opt.load_state_dict(checkpoint["optimizer"])
                sched.load_state_dict(checkpoint["scheduler"])
                samples.load_state_dict(checkpoint["sampler"])
                restore_rng(checkpoint["rng"])
                iterator = iter(batches)
                # This is the production caller's existing exact-resume order.
                gen.set_state(checkpoint["loader_generator"])
                resumed_losses = []

                def after_resume(_step, losses, _norm):
                    samples.mark_consumed()
                    resumed_losses.append(float(losses["total"].detach()))

                self.assertEqual(run_loop(resumed, batches, opt, sched,
                    device=torch.device("cpu"), config=configuration(), variant="baseline",
                    start_step=cut, iterator=iterator, after_step=after_resume), 7)
                self.assertEqual(original_losses[cut:], resumed_losses)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                for name, value in expected_model.items():
                    torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
                for index, state in expected_optimizer["state"].items():
                    for name, value in state.items():
                        torch.testing.assert_close(opt.state_dict()["state"][index][name], value, rtol=0, atol=0)

    def test_nonfinite_objective_does_not_update_or_commit(self):
        model = TinyPolicy(nonfinite=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        batch = next(iter(DataLoader([sample(1)], batch_size=1)))
        previous = copy.deepcopy(model.state_dict())
        callbacks = []
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite loss"):
            run_loop(model, [batch], optimizer, None, device=torch.device("cpu"),
                config=configuration(), variant="baseline", after_step=lambda *args: callbacks.append(args))
        self.assertEqual(callbacks, [])
        self.assertEqual(len(optimizer.state), 0)
        for name, value in previous.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

    def test_empty_loader_and_precreated_iterator_order_are_explicit(self):
        model, optimizer, scheduler, _, _, _ = new_training()
        common = dict(device=torch.device("cpu"), config=configuration(), variant="baseline")
        with self.assertRaisesRegex(ValueError, "no batches"):
            run_loop(model, [], optimizer, scheduler, **common)
        with self.assertRaisesRegex(ValueError, "already created"):
            run_loop(model, [], optimizer, scheduler, iterator=iter([]), before_iterator=lambda: None, **common)

    def test_pre_extraction_update_matches_actual_archived_source(self):
        # Optional independent source oracle captured before this refactor. It
        # validates the real old statements, not a test rewriting our engine.
        before = Path(__file__).resolve().parents[2] / ".local/checks/act_reference_B_20261106/before/train.py"
        if not before.is_file():
            self.skipTest("Optional pre-extraction source snapshot is unavailable")
        tree = ast.parse(before.read_text(encoding="utf-8"))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        loop = next(node for node in ast.walk(run) if isinstance(node, ast.While))
        end = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Expr)
                   and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                   and node.value.func.attr == "mark_consumed")
        args = ast.parse("def original(model, iterator, optimizer, scheduler, device, config, variant): pass").body[0].args
        body = ast.parse("step = 0").body + copy.deepcopy(loop.body[:end])
        body += ast.parse("return losses, norm").body
        node = ast.FunctionDef(name="original", args=args, body=body, decorator_list=[])
        scope = dict(torch=torch, move_batch=move_batch, precision_context=precision_context,
                     targets_masks=targets_masks, compute_loss=compute_loss)
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(module, str(before), "exec"), scope)
        first, opt1, sched1, _, _, _ = new_training()
        second, opt2, sched2, _, _, _ = new_training()
        batch = next(iter(DataLoader([sample(0), sample(1)], batch_size=2)))
        state = rng_state()
        expected, expected_norm = scope["original"](first, iter([batch]), opt1, sched1,
            torch.device("cpu"), configuration(), "baseline")
        expected_rng = torch.get_rng_state().clone()
        restore_rng(state)
        actual, actual_norm = optimize_batch(second, batch, opt2, sched2,
            device=torch.device("cpu"), config=configuration(), variant="baseline")
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        torch.testing.assert_close(actual_norm, expected_norm, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for name, value in first.state_dict().items():
            torch.testing.assert_close(second.state_dict()[name], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
