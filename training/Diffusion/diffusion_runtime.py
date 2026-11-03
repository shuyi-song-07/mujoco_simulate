"""Allow DP horizons such as 30 without changing LeRobot checkpoint weights.

LeRobot's temporal U-Net requires lengths aligned with its downsampling stages.
Only the U-Net input is extended: diffusion noise, dataset targets, scheduler
updates, and losses retain the configured number of actions.
"""

from __future__ import annotations

from functools import wraps


def install_temporal_padding() -> None:
    """Install once per process, before constructing or loading a DP config.

    Replicating the final noisy action supplies a consistent convolution boundary
    during training and inference. Cropping the prediction immediately afterward
    keeps the extra positions out of the diffusion process and training loss.
    """
    import torch
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d

    if getattr(DiffusionConfig, "_mujoco_temporal_padding_installed", False):
        return

    original_post_init = DiffusionConfig.__post_init__
    original_forward = DiffusionConditionalUnet1d.forward

    @wraps(original_post_init)
    def validate_padded_horizon(self):
        horizon = self.horizon
        if horizon < 1:
            raise ValueError(f"Diffusion horizon must be positive, got {horizon}")
        factor = 2 ** len(self.down_dims)
        # Preserve all upstream validation while validating the length actually
        # presented to the U-Net. Serialized configs retain the real horizon.
        self.horizon = ((horizon + factor - 1) // factor) * factor
        try:
            original_post_init(self)
        finally:
            self.horizon = horizon

    @wraps(original_forward)
    def forward_with_temporal_padding(self, x, timestep, global_cond=None):
        length = x.shape[1]
        if length < 1:
            raise ValueError("Diffusion U-Net needs at least one action timestep")
        factor = 2 ** len(self.config.down_dims)
        padding = (-length) % factor
        if padding:
            x = torch.cat((x, x[:, -1:].expand(-1, padding, -1)), dim=1)
        prediction = original_forward(self, x, timestep, global_cond=global_cond)
        return prediction[:, :length]

    DiffusionConfig.__post_init__ = validate_padded_horizon
    DiffusionConditionalUnet1d.forward = forward_with_temporal_padding
    DiffusionConfig._mujoco_temporal_padding_installed = True
