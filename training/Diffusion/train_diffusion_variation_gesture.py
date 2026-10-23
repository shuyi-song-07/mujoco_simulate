#!/usr/bin/env python3
"""Train LeRobot Diffusion Policy with gesture-code auxiliary classification.

The gesture head sees only the policy's image/state conditioning features. It
does not receive the clean action trajectory or the noisy trajectory. Code 0
and padded labels are excluded from its cross-entropy loss. At inference, load
the ordinary Diffusion Policy checkpoint without the auxiliary head.
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_diffusion_variation_baseline import (
    GESTURE_KEY, HAND_DELTA_KEY, HAND_VALID_KEY, make_parser, run,
)


NUM_GESTURES = 7


def masked_gesture_loss(logits: torch.Tensor, labels: torch.Tensor,
                        gesture_is_pad: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Cross entropy on recognized, unpadded current-frame gesture labels."""
    labels = labels.reshape(-1).long()
    gesture_is_pad = gesture_is_pad.reshape(-1).bool()
    valid = (labels >= 1) & (labels <= NUM_GESTURES) & ~gesture_is_pad
    targets = labels.clamp(1, NUM_GESTURES) - 1
    per_frame = F.cross_entropy(logits, targets, reduction="none")
    count = int(valid.sum().item())
    loss = (per_frame * valid).sum() / max(count, 1)
    return loss, count


def masked_hand_delta_loss(prediction: torch.Tensor, target: torch.Tensor,
                           valid: torch.Tensor, is_pad: torch.Tensor,
                           scale: float) -> tuple[torch.Tensor, int]:
    """Smooth L1 on real wrist deltas only, in scaled webcam coordinates."""
    valid = valid.reshape(-1).bool() & ~is_pad.reshape(-1).bool()
    valid &= torch.isfinite(target).all(dim=-1)
    safe_target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
    per_axis = F.smooth_l1_loss(prediction, safe_target / scale, reduction="none")
    count = int(valid.sum().item())
    loss = (per_axis * valid.unsqueeze(-1)).sum() / max(count * target.shape[-1], 1)
    return loss, count


def install_auxiliary(train_module, gesture_weight: float,
                      hand_delta_weight: float, hand_delta_scale: float) -> None:
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from safetensors.torch import save_file

    original_policy = train_module.make_policy

    class GestureDiffusionPolicy(DiffusionPolicy):
        def get_optim_params(self):
            # LeRobot's DiffusionPolicy returns only diffusion.parameters().
            # Include the new head so its weights actually train.
            if self.hand_delta_head is not None:
                return chain(self.diffusion.parameters(), self.gesture_head.parameters(),
                             self.hand_delta_head.parameters())
            return chain(self.diffusion.parameters(), self.gesture_head.parameters())

        def forward(self, batch):
            self._capture_gesture = True
            self._gesture_condition = None
            try:
                diffusion_loss, _ = super().forward(batch)
            finally:
                self._capture_gesture = False
            condition = self._gesture_condition
            self._gesture_condition = None
            if condition is None:
                raise RuntimeError("Diffusion image/state conditioning was not captured")
            logits = self.gesture_head(condition)
            # Observation history has n_obs_steps elements. The last one is
            # the current observation; its gesture label matches this sample.
            labels = batch[GESTURE_KEY][:, -1]
            is_pad = batch[f"{GESTURE_KEY}_is_pad"][:, -1]
            gesture_loss, count = masked_gesture_loss(logits, labels, is_pad)
            metrics = {"diffusion_loss": diffusion_loss.item(),
                       "gesture_loss": gesture_loss.item(),
                       "gesture_valid_frames": count}
            total = diffusion_loss + self.gesture_weight * gesture_loss
            if self.hand_delta_head is not None:
                prediction = self.hand_delta_head(condition)
                motion_loss, motion_count = masked_hand_delta_loss(
                    prediction, batch[HAND_DELTA_KEY][:, -1],
                    batch[HAND_VALID_KEY][:, -1],
                    batch[f"{HAND_DELTA_KEY}_is_pad"][:, -1],
                    self.hand_delta_scale,
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
                            for key, value in full_state.items() if key.startswith("hand_delta_head.")}
            if set(head_state) != {"weight", "bias"}:
                raise RuntimeError("Auxiliary gesture head is missing from checkpoint state")
            if self.hand_delta_head is not None and set(motion_state) != {"weight", "bias"}:
                raise RuntimeError("Auxiliary hand-delta head is missing from checkpoint state")
            super()._save_pretrained(save_directory, state_dict=base_state)
            save_file(head_state, str(Path(save_directory) / "gesture_head.safetensors"))
            if motion_state:
                save_file(motion_state, str(Path(save_directory) / "hand_delta_head.safetensors"))

    def make_gesture_policy(*args, **kwargs):
        policy = original_policy(*args, **kwargs)
        if not isinstance(policy, DiffusionPolicy):
            raise TypeError("Gesture supervision requires LeRobot DiffusionPolicy")
        policy.__class__ = GestureDiffusionPolicy
        policy.gesture_weight = gesture_weight
        policy.hand_delta_weight = hand_delta_weight
        policy.hand_delta_scale = hand_delta_scale
        state_dim = policy.config.robot_state_feature.shape[0]
        num_cameras = len(policy.config.image_features)
        encoder = policy.diffusion.rgb_encoder
        rgb_dim = encoder[0].feature_dim if isinstance(encoder, nn.ModuleList) else encoder.feature_dim
        condition_dim = (state_dim + num_cameras * rgb_dim) * policy.config.n_obs_steps
        # Keep subsequent diffusion-noise draws comparable to the baseline.
        rng_state = torch.random.get_rng_state()
        policy.gesture_head = nn.Linear(condition_dim, NUM_GESTURES).to(policy.config.device)
        policy.hand_delta_head = (
            nn.Linear(condition_dim, 2).to(policy.config.device)
            if hand_delta_weight > 0 else None
        )
        torch.random.set_rng_state(rng_state)
        policy._capture_gesture = False
        policy._gesture_condition = None

        def capture_condition(_module, _inputs, kwargs):
            if policy._capture_gesture:
                policy._gesture_condition = kwargs.get("global_cond")

        policy.diffusion.unet.register_forward_pre_hook(capture_condition, with_kwargs=True)
        return policy

    train_module.make_policy = make_gesture_policy


if __name__ == "__main__":
    run(make_parser(auxiliary=True).parse_args(), auxiliary=True, install_auxiliary=install_auxiliary)
