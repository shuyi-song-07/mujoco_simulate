"""Paper A Diffusion Policy baseline, adapted to its causal action contract.

This is a local adaptation, not an exact reproduction of the paper's network,
training setup, or reported results. Reference: https://diffusion-policy.cs.columbia.edu/
and the authors' conditional_unet1d.py in https://github.com/real-stanford/diffusion_policy.

One oblique RGB history and 16D robot-state history are the only inference
inputs. ResNet18 with GroupNorm feeds the standard spatial-softmax pooling
used by the authors' robomimic image encoder: a 1x1 convolution creates 32
keypoint maps, then each spatial distribution yields an expected (x, y) in
[-1, 1]. The flattened coordinates project to the per-frame feature dimension.
Full-frame RGB follows the local dataset contract. ImageNet normalization is
internal; state/action normalization belongs to the dataset. No weights are
downloaded. Spatial-softmax reference and default configuration:
https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/models/base_nets.py
https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/generate_paper_configs.py

The temporal U-Net receives diffusion-time and observation features through
FiLM. Training predicts epsilon using valid-coordinate MSE. Sampling uses the
cosine forward schedule and deterministic DDIM (eta=0). Output index zero is
the next legal control tick after the newest observation: there is no To-1
offset, because the Pack already defines future actions with that alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import resnet18

from .schedule import CosineDiffusionSchedule
from training.gesture_auxiliary import masked_gesture_loss


@dataclass(frozen=True)
class DiffusionConfig:
    state_dim: int = 16
    action_dim: int = 14
    observation_horizon: int = 1
    prediction_horizon: int = 30
    action_horizon: int = 8
    feature_dim: int = 128
    spatial_keypoints: int = 32
    spatial_temperature: float = 1.0
    time_dim: int = 128
    down_dims: tuple[int, ...] = (128, 256, 512)
    diffusion_steps: int = 100
    inference_steps: int = 10
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
        if self.backbone_weights is not None and not isinstance(self.backbone_weights, str):
            raise ValueError("backbone_weights must be a local path string or None")


def _groups(channels: int) -> int:
    groups = math.gcd(channels, 8)
    # Keep at least two channels per group even at a one-tick bottleneck.
    return groups // 2 if groups == channels and groups > 1 else groups


def _replace_batch_norm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            replacement = nn.GroupNorm(_groups(child.num_features), child.num_features)
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            _replace_batch_norm(child)


class SpatialSoftmax(nn.Module):
    """Learned keypoint maps -> flattened expected (x, y) coordinates.

    This is standard robomimic spatial softmax with fixed temperature, zero
    keypoint noise, and no covariance output. The grid follows the feature map
    resolution, preserving full-frame inputs without a fixed-size crop.
    """

    def __init__(self, input_dim: int, keypoints: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.projection = nn.Conv2d(input_dim, keypoints, kernel_size=1)
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, features: Tensor) -> Tensor:
        logits = self.projection(features)
        height, width = logits.shape[-2:]
        # FP32 probabilities and expectations remain stable under CUDA AMP.
        probabilities = F.softmax(logits.float().flatten(2) / self.temperature, dim=-1)
        y, x = torch.meshgrid(torch.linspace(-1, 1, height, device=features.device),
                              torch.linspace(-1, 1, width, device=features.device), indexing="ij")
        expected_x = (probabilities * x.flatten()).sum(-1)
        expected_y = (probabilities * y.flatten()).sum(-1)
        return torch.stack((expected_x, expected_y), dim=-1).flatten(1)


class _TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.exp(
            -math.log(10000) * torch.arange(dim // 2, dtype=torch.float32) / (dim // 2 - 1)))
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.Mish(), nn.Linear(4 * dim, dim))

    def forward(self, timesteps: Tensor) -> Tensor:
        phases = timesteps.float()[:, None] * self.frequencies[None]
        return self.mlp(torch.cat((phases.sin(), phases.cos()), dim=-1))


class _ConditionalResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.first = nn.Sequential(nn.Conv1d(input_dim, output_dim, 3, padding=1),
                                   nn.GroupNorm(_groups(output_dim), output_dim), nn.Mish())
        self.film = nn.Sequential(nn.Mish(), nn.Linear(condition_dim, 2 * output_dim))
        self.second = nn.Sequential(nn.Conv1d(output_dim, output_dim, 3, padding=1),
                                    nn.GroupNorm(_groups(output_dim), output_dim), nn.Mish())
        self.residual = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else nn.Identity()

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        scale, bias = self.film(condition).unsqueeze(-1).chunk(2, dim=1)
        hidden = self.first(value) * (1 + scale) + bias
        return self.second(hidden) + self.residual(value)


class ConditionalTemporalUNet(nn.Module):
    """Temporal U-Net with explicit padding so K=30 also aligns all skips."""

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.alignment = 2 ** (len(config.down_dims) - 1)
        self.time_embedding = _TimeEmbedding(config.time_dim)
        condition_dim = config.time_dim + config.observation_horizon * (config.feature_dim + config.state_dim)
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        input_dim = config.action_dim
        for index, width in enumerate(config.down_dims):
            self.down_blocks.append(nn.ModuleList([
                _ConditionalResidualBlock(input_dim, width, condition_dim),
                _ConditionalResidualBlock(width, width, condition_dim),
            ]))
            if index < len(config.down_dims) - 1:
                self.downsample.append(nn.Conv1d(width, width, 3, stride=2, padding=1))
            input_dim = width
        self.middle = nn.ModuleList([
            _ConditionalResidualBlock(input_dim, input_dim, condition_dim),
            _ConditionalResidualBlock(input_dim, input_dim, condition_dim),
        ])
        self.upsample = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for width in reversed(config.down_dims[:-1]):
            self.upsample.append(nn.ConvTranspose1d(input_dim, width, 4, stride=2, padding=1))
            self.up_blocks.append(nn.ModuleList([
                _ConditionalResidualBlock(2 * width, width, condition_dim),
                _ConditionalResidualBlock(width, width, condition_dim),
            ]))
            input_dim = width
        self.output = nn.Sequential(nn.Conv1d(input_dim, input_dim, 3, padding=1),
                                    nn.GroupNorm(_groups(input_dim), input_dim), nn.Mish(),
                                    nn.Conv1d(input_dim, config.action_dim, 1))

    def forward(self, sample: Tensor, timesteps: Tensor, observations: Tensor) -> Tensor:
        horizon = sample.shape[1]
        hidden = sample.transpose(1, 2)
        hidden = F.pad(hidden, (0, (-horizon) % self.alignment))
        condition = torch.cat((self.time_embedding(timesteps), observations), dim=-1)
        skips = []
        for index, blocks in enumerate(self.down_blocks):
            for block in blocks:
                hidden = block(hidden, condition)
            skips.append(hidden)
            if index < len(self.downsample):
                hidden = self.downsample[index](hidden)
        for block in self.middle:
            hidden = block(hidden, condition)
        for upsample, blocks, skip in zip(self.upsample, self.up_blocks, reversed(skips[:-1])):
            hidden = torch.cat((upsample(hidden), skip), dim=1)
            for block in blocks:
                hidden = block(hidden, condition)
        return self.output(hidden)[..., :horizon].transpose(1, 2)


class DiffusionPolicy(nn.Module):
    """Observation-only policy producing an entire future K-step action plan.

    action_horizon is the execution/replanning interval, not an output offset
    or an output-length reduction. The rollout executes plan[:, :action_horizon].
    """

    def __init__(self, config: DiffusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or DiffusionConfig()
        c = self.config
        backbone = resnet18(weights=None)
        if c.backbone_weights is not None:
            path = Path(c.backbone_weights)
            if not path.is_file():
                raise FileNotFoundError(f"Local ResNet18 weights not found: {path}")
            backbone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        # Load standard torchvision BN weights first, then preserve the affine
        # weights while removing batch statistics for small policy batches.
        _replace_batch_norm(backbone)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.spatial_pool = SpatialSoftmax(512, c.spatial_keypoints, c.spatial_temperature)
        self.image_projection = nn.Linear(2 * c.spatial_keypoints, c.feature_dim)
        self.denoiser = ConditionalTemporalUNet(c)
        self.noise_scheduler = CosineDiffusionSchedule(c.diffusion_steps, c.clip_sample)
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))
        if c.variant == "gesture":
            # Append only after every shared parameter has been initialized.
            # Preserve global RNG so the same seed also yields identical later
            # diffusion-noise draws in baseline and auxiliary experiments.
            with torch.random.fork_rng(devices=[]):
                self.gesture_head = nn.Linear(c.observation_horizon * (c.feature_dim + c.state_dim), 6)

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
        if batch < 1 or min(images.shape[-2:]) < 32:
            raise ValueError("images require a nonempty batch and height/width >= 32")
        if state.shape != (batch, c.observation_horizon, c.state_dim):
            raise ValueError(f"state must be [B, {c.observation_horizon}, {c.state_dim}]")
        if not images.is_floating_point() or not state.is_floating_point():
            raise ValueError("images and state must be floating-point tensors")
        if images.device != state.device:
            raise ValueError("images and state must share a device")
        if not torch.isfinite(images).all() or not torch.isfinite(state).all():
            raise ValueError("Robot observations must be finite")
        if (images < 0).any() or (images > 1).any():
            raise ValueError("images must be RGB in [0, 1]; ImageNet normalization is internal")
        frames = images.reshape(batch * c.observation_horizon, 3, *images.shape[-2:])
        feature_maps = self.backbone((frames - self.image_mean) / self.image_std)
        features = self.spatial_pool(feature_maps)
        features = self.image_projection(features).reshape(batch, c.observation_horizon, c.feature_dim)
        return torch.cat((features, state), dim=-1).flatten(1)

    def compute_loss(self, images: Tensor, state: Tensor, action: Tensor,
                     is_pad: Tensor | None, generator: torch.Generator | None = None, *,
                     gesture: Tensor | None = None,
                     gesture_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Masked epsilon MSE; padded target values never enter the denoiser.

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
        noisy = noisy.masked_fill(is_pad.unsqueeze(-1), 0)
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
        """Return normalized [B, K, 14] from the next legal control tick.

        A supplied generator controls the initial Gaussian sample. DDIM eta=0
        draws no further noise. The caller owns action denormalization and
        executes only the first action_horizon steps before observing again.
        """
        steps = self.config.inference_steps if num_inference_steps is None else num_inference_steps
        timesteps = self.noise_scheduler.inference_timesteps(steps)
        was_training = self.training
        self.eval()
        try:
            condition = self.encode_observations(images, state)
            # Keep reverse diffusion arithmetic in FP32 even under autocast.
            sample = torch.randn((images.shape[0], self.config.prediction_horizon, self.config.action_dim),
                                 device=state.device, dtype=torch.float32, generator=generator)
            for index, timestep in enumerate(timesteps):
                batch_timesteps = torch.full((images.shape[0],), timestep, device=state.device, dtype=torch.long)
                epsilon = self.denoiser(sample, batch_timesteps, condition)
                previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
                sample = self.noise_scheduler.ddim_step(epsilon, timestep, previous, sample)
            return sample
        finally:
            self.train(was_training)
