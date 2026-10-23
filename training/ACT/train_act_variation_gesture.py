#!/usr/bin/env python3
"""Train ACT with masked gesture and optional wrist-displacement losses.

Labels 1..7 correspond to recognized MediaPipe gestures. Label 0 is unknown
and never contributes to the gesture loss. The policy still learns actions on
those frames. A standard ACT checkpoint is exported for ordinary inference;
the auxiliary heads are saved beside it as separate safetensors files.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .train_act_variation_baseline import (
        GESTURE_KEY, HAND_DELTA_KEY, HAND_VALID_KEY, make_parser, run,
    )
except ImportError:  # Direct execution: python training/ACT/<script>.py
    from train_act_variation_baseline import (
        GESTURE_KEY, HAND_DELTA_KEY, HAND_VALID_KEY, make_parser, run,
    )


NUM_GESTURES = 7


def masked_gesture_loss(logits: torch.Tensor, labels: torch.Tensor,
                        action_is_pad: torch.Tensor, gesture_is_pad: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Cross entropy over known, real frames only; return a graph-connected zero otherwise."""
    labels = labels.reshape(logits.shape[:2]).long()
    valid = ((labels >= 1) & (labels <= NUM_GESTURES) &
             ~action_is_pad.bool() & ~gesture_is_pad.bool())
    target = (labels.clamp(1, NUM_GESTURES) - 1).reshape(-1)
    per_frame = F.cross_entropy(logits.reshape(-1, NUM_GESTURES), target, reduction="none")
    valid_count = int(valid.sum().item())
    loss = (per_frame * valid.reshape(-1)).sum() / max(valid_count, 1)
    return loss, valid_count


def masked_hand_delta_loss(prediction: torch.Tensor, target: torch.Tensor,
                           valid: torch.Tensor, action_is_pad: torch.Tensor,
                           delta_is_pad: torch.Tensor, valid_is_pad: torch.Tensor,
                           scale: float) -> tuple[torch.Tensor, int]:
    """Smooth L1 over finite, valid, real wrist-displacement frames only."""
    target = target.reshape_as(prediction)
    valid = valid.reshape(prediction.shape[:2]).bool()
    mask = (valid & ~action_is_pad.bool() & ~delta_is_pad.bool() &
            ~valid_is_pad.bool() & torch.isfinite(target).all(dim=-1))
    safe_target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
    per_axis = F.smooth_l1_loss(prediction, safe_target / scale, reduction="none")
    count = int(mask.sum().item())
    loss = (per_axis * mask.unsqueeze(-1)).sum() / max(count * target.shape[-1], 1)
    return loss, count


def install_auxiliary(train_module, gesture_weight: float,
                      hand_delta_weight: float = 0.0,
                      hand_delta_scale: float = 0.03) -> None:
    from lerobot.datasets import factory as dataset_factory
    from lerobot.policies.act.modeling_act import ACTPolicy
    from safetensors.torch import save_file

    original_datasets = train_module.make_train_eval_datasets
    original_policy = train_module.make_policy

    def make_gesture_datasets(cfg):
        original_resolve = dataset_factory.resolve_delta_timestamps

        def resolve_with_gesture(policy_cfg, ds_meta):
            offsets = original_resolve(policy_cfg, ds_meta) or {}
            if GESTURE_KEY not in ds_meta.features or "action" not in offsets:
                raise ValueError("Dataset must contain gesture labels and action chunk timestamps")
            # The same offsets give one gesture target per predicted action frame.
            result = {**offsets, GESTURE_KEY: list(offsets["action"])}
            if hand_delta_weight > 0:
                missing = {HAND_DELTA_KEY, HAND_VALID_KEY} - set(ds_meta.features)
                if missing:
                    raise ValueError(f"Dataset is missing hand displacement fields: {sorted(missing)}")
                result[HAND_DELTA_KEY] = list(offsets["action"])
                result[HAND_VALID_KEY] = list(offsets["action"])
            return result

        dataset_factory.resolve_delta_timestamps = resolve_with_gesture
        try:
            return original_datasets(cfg)
        finally:
            dataset_factory.resolve_delta_timestamps = original_resolve

    class GestureACTPolicy(ACTPolicy):
        def forward(self, batch):
            self._capture_gesture = True
            self._gesture_hidden = None
            try:
                action_loss, metrics = super().forward(batch)
            finally:
                self._capture_gesture = False
            hidden = self._gesture_hidden
            self._gesture_hidden = None
            if hidden is None:
                raise RuntimeError("ACT decoder output was not captured for gesture supervision")
            logits = self.gesture_head(hidden)
            gesture_loss, count = masked_gesture_loss(
                logits, batch[GESTURE_KEY], batch["action_is_pad"], batch[f"{GESTURE_KEY}_is_pad"]
            )
            metrics["gesture_loss"] = gesture_loss.item()
            metrics["gesture_valid_frames"] = count
            total = action_loss + self.gesture_weight * gesture_loss
            if self.hand_delta_head is not None:
                prediction = self.hand_delta_head(hidden)
                motion_loss, motion_count = masked_hand_delta_loss(
                    prediction, batch[HAND_DELTA_KEY], batch[HAND_VALID_KEY],
                    batch["action_is_pad"], batch[f"{HAND_DELTA_KEY}_is_pad"],
                    batch[f"{HAND_VALID_KEY}_is_pad"], self.hand_delta_scale,
                )
                metrics["hand_delta_loss"] = motion_loss.item()
                metrics["hand_delta_valid_frames"] = motion_count
                total = total + self.hand_delta_weight * motion_loss
            return total, metrics

        def _save_pretrained(self, save_directory: Path, state_dict=None):
            full_state = self.state_dict() if state_dict is None else state_dict
            base_state = {key: value for key, value in full_state.items()
                          if not key.startswith(("gesture_head.", "hand_delta_head."))}
            head_state = {key.removeprefix("gesture_head."): value.detach().cpu().contiguous()
                          for key, value in full_state.items() if key.startswith("gesture_head.")}
            motion_state = {key.removeprefix("hand_delta_head."): value.detach().cpu().contiguous()
                            for key, value in full_state.items()
                            if key.startswith("hand_delta_head.")}
            if set(head_state) != {"weight", "bias"}:
                raise RuntimeError("Auxiliary gesture head is missing from checkpoint state")
            super()._save_pretrained(save_directory, state_dict=base_state)
            save_file(head_state, str(Path(save_directory) / "gesture_head.safetensors"))
            if self.hand_delta_head is not None:
                if set(motion_state) != {"weight", "bias"}:
                    raise RuntimeError("Auxiliary hand-delta head is missing from checkpoint state")
                save_file(motion_state, str(Path(save_directory) / "hand_delta_head.safetensors"))

    def make_gesture_policy(*args, **kwargs):
        policy = original_policy(*args, **kwargs)
        if not isinstance(policy, ACTPolicy):
            raise TypeError("Gesture supervision requires LeRobot ACTPolicy")
        policy.__class__ = GestureACTPolicy
        policy.gesture_weight = gesture_weight
        policy.hand_delta_weight = hand_delta_weight
        policy.hand_delta_scale = hand_delta_scale
        # Keep subsequent ACT/VAE randomness comparable with the baseline run.
        rng_state = torch.random.get_rng_state()
        policy.gesture_head = nn.Linear(policy.config.dim_model, NUM_GESTURES).to(policy.config.device)
        policy.hand_delta_head = (
            nn.Linear(policy.config.dim_model, 2).to(policy.config.device)
            if hand_delta_weight > 0 else None
        )
        torch.random.set_rng_state(rng_state)
        policy._capture_gesture = False
        policy._gesture_hidden = None

        def capture_decoder(_module, _inputs, output):
            if policy._capture_gesture:
                policy._gesture_hidden = output.transpose(0, 1)

        policy.model.decoder.register_forward_hook(capture_decoder)
        return policy

    train_module.make_train_eval_datasets = make_gesture_datasets
    train_module.make_policy = make_gesture_policy


if __name__ == "__main__":
    run(
        make_parser(auxiliary=True, motion=True).parse_args(),
        auxiliary=True,
        install_auxiliary=install_auxiliary,
    )
