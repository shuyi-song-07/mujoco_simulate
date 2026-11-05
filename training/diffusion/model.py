"""Official DP components with only the Paper A observation/action adapter.

Pinned upstream U-Net and crop code retain the original network mathematics.
Vision is robomimic's VisualCore. Task adaptations are S16/A14, future-only
Pack labels, masked invalid targets, K30 alignment and configurable image sizes.
Old local-variant checkpoints are intentionally incompatible.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from robomimic.models.base_nets import VisualCore, SpatialSoftmax
from .official.conditional_unet1d import ConditionalUnet1D
from .official.crop_randomizer import CropRandomizer

from .schedule import make_noise_scheduler
from training.gesture_auxiliary import masked_gesture_loss


IMPLEMENTATION = "official-dp-unet-hybrid-papera-v1"


@dataclass(frozen=True)
class DiffusionConfig:
    state_dim: int = 16
    action_dim: int = 14
    observation_horizon: int = 2
    prediction_horizon: int = 30
    action_horizon: int = 8
    feature_dim: int = 64
    spatial_keypoints: int = 32
    spatial_temperature: float = 1.0
    time_dim: int = 128
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    image_size: tuple[int, int] = (640, 480)  # width, height
    crop_shape: tuple[int, int] | None = None  # height, width
    diffusion_steps: int = 100
    inference_steps: int = 100
    clip_sample: bool = True
    backbone_weights: str | None = None
    variant: str = "baseline"
    gesture_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.variant not in ("baseline", "gesture"):
            raise ValueError("Diffusion variant must be baseline or gesture")
        if (isinstance(self.gesture_weight, bool) or not isinstance(self.gesture_weight, (int, float))
                or not math.isfinite(self.gesture_weight) or self.gesture_weight < 0):
            raise ValueError("gesture_weight must be finite and nonnegative")
        for name in ("state_dim", "action_dim", "observation_horizon", "prediction_horizon",
                     "action_horizon", "feature_dim", "spatial_keypoints", "time_dim", "diffusion_steps", "inference_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (self.state_dim, self.action_dim) != (16, 14):
            raise ValueError("Paper A requires state_dim=16 and action_dim=14")
        if self.action_horizon > self.prediction_horizon:
            raise ValueError("action_horizon cannot exceed prediction_horizon")
        if self.inference_steps > self.diffusion_steps:
            raise ValueError("inference_steps cannot exceed diffusion_steps")
        if self.time_dim < 4 or self.time_dim % 2:
            raise ValueError("time_dim must be even and at least 4")
        if not isinstance(self.clip_sample, bool):
            raise ValueError("clip_sample must be boolean")
        if (isinstance(self.spatial_temperature, bool)
                or not isinstance(self.spatial_temperature, (int, float))
                or not math.isfinite(self.spatial_temperature) or self.spatial_temperature <= 0):
            raise ValueError("spatial_temperature must be finite and positive")
        if not isinstance(self.down_dims, (tuple, list)) or not self.down_dims:
            raise ValueError("down_dims must be a nonempty tuple of channel widths")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in self.down_dims):
            raise ValueError("down_dims must contain integer channel widths >= 2")
        object.__setattr__(self, "down_dims", tuple(self.down_dims))
        if type(self.kernel_size) is not int or self.kernel_size < 1 or not self.kernel_size % 2:
            raise ValueError("kernel_size must be a positive odd integer")
        if type(self.n_groups) is not int or self.n_groups < 1:
            raise ValueError("n_groups must be a positive integer")
        if any(v < 8 or v % self.n_groups for v in self.down_dims) or self.down_dims[0] % 8:
            raise ValueError("down_dims must divide n_groups; first width must also divide 8")
        if type(self.cond_predict_scale) is not bool:
            raise ValueError("cond_predict_scale must be boolean")
        if (not isinstance(self.image_size, (tuple, list)) or len(self.image_size) != 2
                or any(type(v) is not int or v < 32 for v in self.image_size)):
            raise ValueError("image_size must be (width,height), each >=32")
        object.__setattr__(self, "image_size", tuple(self.image_size))
        if self.crop_shape is not None:
            if (not isinstance(self.crop_shape, (list, tuple)) or len(self.crop_shape) != 2
                    or any(type(v) is not int or v < 32 for v in self.crop_shape)
                    or any(a >= b for a,b in zip(self.crop_shape, reversed(self.image_size)))):
                raise ValueError("crop_shape must be (height,width), >=32 and smaller than image, or None")
            object.__setattr__(self, "crop_shape", tuple(self.crop_shape))
        if self.backbone_weights is not None and not isinstance(self.backbone_weights, str):
            raise ValueError("backbone_weights must be a local path string or None")


def _replace_batch_norm(module):
    """Original hybrid image policy GN rule, with no local group heuristic."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(child.num_features // 16, child.num_features))
        else:
            _replace_batch_norm(child)


class ConditionalTemporalUNet(nn.Module):
    """Length adapter around the unchanged upstream graph."""
    def __init__(self, config):
        super().__init__()
        self.alignment = 2 ** (len(config.down_dims) - 1)
        self.network = ConditionalUnet1D(
            input_dim=config.action_dim, local_cond_dim=None,
            global_cond_dim=config.observation_horizon * (config.feature_dim + config.state_dim),
            diffusion_step_embed_dim=config.time_dim, down_dims=config.down_dims,
            kernel_size=config.kernel_size, n_groups=config.n_groups,
            cond_predict_scale=config.cond_predict_scale)

    @property
    def output(self):
        return self.network.final_conv

    def forward(self, sample, timesteps, observations):
        horizon = sample.shape[1]
        padding = (-horizon) % self.alignment
        if padding:
            sample = F.pad(sample, (0, 0, 0, padding))
        return self.network(sample, timesteps, global_cond=observations)[:, :horizon]


class DiffusionPolicy(nn.Module):
    """Observation-only policy producing an entire future K-step action plan.

    action_horizon is the execution/replanning interval, not an output offset
    or an output-length reduction. The rollout executes plan[:, :action_horizon].
    """

    def __init__(self, config: DiffusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or DiffusionConfig()
        c = self.config
        shape = (3, *reversed(c.image_size))
        if c.crop_shape is None:
            self.crop_randomizer = None
            visual_shape = shape
        else:
            self.crop_randomizer = CropRandomizer(shape, *c.crop_shape, num_crops=1, pos_enc=False)
            visual_shape = (3, *c.crop_shape)
        self.visual_encoder = VisualCore(
            input_shape=visual_shape, backbone_class="ResNet18Conv",
            backbone_kwargs={"pretrained": False, "input_coord_conv": False},
            pool_class="SpatialSoftmax", pool_kwargs={
                "num_kp": c.spatial_keypoints, "learnable_temperature": False,
                "temperature": c.spatial_temperature, "noise_std": 0.0, "output_variance": False},
            flatten=True, feature_dimension=c.feature_dim)
        if c.backbone_weights is not None:
            from torchvision.models import resnet18
            path = Path(c.backbone_weights)
            if not path.is_file():
                raise FileNotFoundError(f"Local ResNet18 weights not found: {path}")
            backbone = resnet18(weights=None)
            backbone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            self.visual_encoder.backbone.nets = nn.Sequential(*list(backbone.children())[:-2])
        _replace_batch_norm(self.visual_encoder)
        self.denoiser = ConditionalTemporalUNet(c)
        self.noise_scheduler = make_noise_scheduler(c.diffusion_steps, c.clip_sample)
        if c.variant == "gesture":
            # Auxiliary classification reads observations only; preserve baseline RNG.
            with torch.random.fork_rng(devices=[]):
                self.gesture_head = nn.Linear(c.observation_horizon * (c.feature_dim + c.state_dim), 6)

    @property
    def backbone(self):
        return self.visual_encoder.backbone.nets

    @property
    def spatial_pool(self):
        return self.visual_encoder.pool

    @property
    def image_projection(self):
        return self.visual_encoder.nets[-1]

    def encode_observations(self, images: Tensor, state: Tensor) -> Tensor:
        """Encode only chronological RGB and state; no target/label argument."""
        c = self.config
        if images.ndim == 6:
            if images.shape[2] != 1:
                raise ValueError("images require exactly one oblique camera")
            images = images[:, :, 0]
        if images.ndim != 5 or images.shape[1:3] != (c.observation_horizon, 3):
            raise ValueError(f"images must be [B, {c.observation_horizon}, 1, 3, H, W] or [B, To, 3, H, W]")
        batch = images.shape[0]
        if batch < 1 or images.shape[-2:] != tuple(reversed(c.image_size)):
            raise ValueError("images must match configured image_size")
        if state.shape != (batch, c.observation_horizon, c.state_dim):
            raise ValueError(f"state must be [B, {c.observation_horizon}, {c.state_dim}]")
        if not images.is_floating_point() or not state.is_floating_point():
            raise ValueError("images and state must be floating-point tensors")
        if images.device != state.device:
            raise ValueError("images and state must share a device")
        if not torch.isfinite(images).all() or not torch.isfinite(state).all():
            raise ValueError("Robot observations must be finite")
        if (images < 0).any() or (images > 1).any():
            raise ValueError("images must be RGB in [0, 1]")
        frames = images.reshape(batch * c.observation_horizon, 3, *images.shape[-2:])
        # Upstream image-range normalization and original crop/VisualCore.
        frames = frames * 2 - 1
        if self.crop_randomizer is not None:
            frames = self.crop_randomizer.forward_in(frames)
        # ObservationEncoder in the official hybrid applies ReLU after VisualCore.
        features = F.relu(self.visual_encoder(frames))
        if self.crop_randomizer is not None:
            features = self.crop_randomizer.forward_out(features)
        features = features.reshape(batch, c.observation_horizon, c.feature_dim)
        # Canonical official obs_shapes order is low-dimensional state, then RGB.
        return torch.cat((state, features), dim=-1).flatten(1)

    def compute_loss(self, images: Tensor, state: Tensor, action: Tensor,
                     is_pad: Tensor | None, generator: torch.Generator | None = None, *,
                     gesture: Tensor | None = None,
                     gesture_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Official epsilon MSE with Pack-invalid coordinates excluded.

        Padding may contain NaNs. Clear it before any arithmetic, and exclude
        padded coordinates from both numerator and denominator. All-padding
        batches have a differentiable zero loss with finite zero gradients.
        """
        c = self.config
        batch = images.shape[0]
        if action.shape != (batch, c.prediction_horizon, c.action_dim):
            raise ValueError(f"action must be [B, {c.prediction_horizon}, {c.action_dim}]")
        if not action.is_floating_point() or action.device != state.device:
            raise ValueError("action must be floating-point on the observation device")
        if is_pad is None:
            is_pad = torch.zeros(action.shape[:2], device=action.device, dtype=torch.bool)
        if is_pad.shape != action.shape[:2] or is_pad.dtype != torch.bool or is_pad.device != action.device:
            raise ValueError("is_pad must be boolean [B, prediction_horizon] on the observation device")
        clean = action.masked_fill(is_pad.unsqueeze(-1), 0)
        if not torch.isfinite(clean).all():
            raise ValueError("Unpadded action targets must be finite")
        condition = self.encode_observations(images, state)
        noise = torch.randn(action.shape, dtype=action.dtype, device=action.device, generator=generator)
        timesteps = torch.randint(c.diffusion_steps, (batch,), device=action.device, generator=generator)
        noisy = self.noise_scheduler.add_noise(clean, noise, timesteps)
        # Keep noise at invalid positions; only the loss excludes their labels.
        prediction = self.denoiser(noisy, timesteps, condition)
        squared_error = (prediction.float() - noise.float()).square().masked_fill(is_pad.unsqueeze(-1), 0)
        count = (~is_pad).sum() * c.action_dim
        loss = squared_error.sum() / count.clamp_min(1)
        result = {"loss": loss, "noise_mse": loss.detach()}
        if c.variant == "gesture" and (gesture is not None or gesture_mask is not None or self.training):
            if gesture is None or gesture_mask is None:
                raise ValueError("Gesture training requires explicit current-anchor labels and masks")
            # Reads RGB/state only: no noisy actions, epsilon, timestep or future.
            # The six independent rows are two separate three-way classifiers.
            logits = self.gesture_head(condition).reshape(batch, 2, 3)
            auxiliary = masked_gesture_loss(logits, gesture, gesture_mask)
            result.update(loss=loss + c.gesture_weight * auxiliary["loss"],
                          gesture_loss=auxiliary["loss"], gesture_count=auxiliary["count"],
                          gesture_sum=auxiliary["sum"], gesture_confusion=auxiliary["confusion"],
                          gesture_logits=logits)
        return result

    @torch.no_grad()
    def predict_action(self, images: Tensor, state: Tensor,
                       generator: torch.Generator | None = None,
                       num_inference_steps: int | None = None) -> Tensor:
        """Official diffusers DDPM reverse process; all draws use generator."""
        steps = self.config.inference_steps if num_inference_steps is None else num_inference_steps
        if type(steps) is not int or not 1 <= steps <= self.config.diffusion_steps:
            raise ValueError("num_inference_steps must be within the trained diffusion schedule")
        was_training = self.training
        self.eval()
        try:
            condition = self.encode_observations(images, state)
            sample = torch.randn((images.shape[0], self.config.prediction_horizon, self.config.action_dim),
                                 device=state.device, dtype=torch.float32, generator=generator)
            self.noise_scheduler.set_timesteps(steps, device=state.device)
            for timestep in self.noise_scheduler.timesteps:
                prediction = self.denoiser(sample, timestep, condition)
                sample = self.noise_scheduler.step(prediction, timestep, sample, generator=generator).prev_sample
            return sample
        finally:
            self.train(was_training)
