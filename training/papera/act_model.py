"""Paper A's ACT-style CVAE, implemented for the Task3 data contract.

Reference: Zhao et al., Learning Fine-Grained Bimanual Manipulation with
Low-Cost Hardware (ACT), https://tonyzhaozh.github.io/aloha/ . This is a local
adaptation, not an exact reproduction of the authors' architecture or results.

One oblique image (or three historical camera images) and 16 robot measurements
are the *only* inference inputs. Camera count is frozen in the model config.
An observation-only transformer feeds the auxiliary heads. The posterior reads
training action chunks separately; its latent enters only the action decoder.
This separation prevents auxiliary supervision from seeing its future actions.
RGB tensors are float [0, 1]; state/action normalization belongs to the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.ops.misc import FrozenBatchNorm2d


VARIANTS = ("baseline", "gesture", "gesture_skeleton", "robot_stage")


@dataclass(frozen=True)
class ACTConfig:
    variant: str = "baseline"
    state_dim: int = 16
    action_dim: int = 14
    num_cameras: int = 3
    chunk_size: int = 30
    hidden_dim: int = 256
    heads: int = 8
    feedforward_dim: int = 1024
    encoder_layers: int = 4
    decoder_layers: int = 4
    posterior_layers: int = 4
    latent_dim: int = 32
    dropout: float = 0.1
    stage_classes: int = 10
    backbone_weights: str | None = None

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant {self.variant!r}; expected {VARIANTS}")
        for name in ("state_dim", "action_dim", "num_cameras", "chunk_size", "hidden_dim",
                     "heads", "feedforward_dim", "encoder_layers", "decoder_layers",
                     "posterior_layers", "latent_dim", "stage_classes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (self.state_dim, self.action_dim) != (16, 14) or self.num_cameras not in (1, 3):
            raise ValueError("Task3 requires state=16, action=14, and 1 oblique or 3 historical cameras")
        if self.hidden_dim % self.heads or self.hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by heads and by 4")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")


@dataclass(frozen=True)
class LossWeights:
    action: float = 1.0
    kl: float = 10.0
    gesture: float = 1.0
    skeleton: float = 1.0
    stage: float = 1.0


def _sine_2d(height: int, width: int, dim: int, device: torch.device,
             dtype: torch.dtype) -> Tensor:
    """Normalized spatial positions, without assuming a particular resolution."""
    y = torch.linspace(0, 2 * math.pi, height, device=device, dtype=torch.float32)
    x = torch.linspace(0, 2 * math.pi, width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    frequencies = 10000 ** (-torch.arange(dim // 4, device=device).float() / (dim // 4))
    phases_x = xx.flatten()[:, None] * frequencies[None, :]
    phases_y = yy.flatten()[:, None] * frequencies[None, :]
    return torch.cat((phases_y.sin(), phases_y.cos(), phases_x.sin(), phases_x.cos()), -1).to(dtype)


def _encoder(config: ACTConfig, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        config.hidden_dim, config.heads, config.feedforward_dim, config.dropout,
        activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(config.hidden_dim),
                                 enable_nested_tensor=False)


class ACTPolicy(nn.Module):
    """Action chunk policy with optional training-only hand/stage losses.

    The variant changes loss selection, not initialization or inference inputs.
    All heads exist in every variant so an identical random seed gives the same
    common starting weights. No pretrained weight is downloaded implicitly.
    """

    def __init__(self, config: ACTConfig | None = None) -> None:
        super().__init__()
        self.config = config or ACTConfig()
        c = self.config
        backbone = resnet18(weights=None, norm_layer=FrozenBatchNorm2d)
        if c.backbone_weights is not None:
            weights_path = Path(c.backbone_weights)
            if not weights_path.is_file():
                raise FileNotFoundError(f"Local ResNet18 weights not found: {weights_path}")
            weights = torch.load(weights_path, map_location="cpu", weights_only=True)
            # Official torchvision ResNet18 checkpoints are plain state dictionaries.
            backbone.load_state_dict(weights, strict=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.image_projection = nn.Conv2d(512, c.hidden_dim, kernel_size=1)
        self.camera_embedding = nn.Embedding(c.num_cameras, c.hidden_dim)
        self.state_projection = nn.Linear(c.state_dim, c.hidden_dim)
        self.state_position = nn.Parameter(torch.zeros(1, 1, c.hidden_dim))
        self.observation_encoder = _encoder(c, c.encoder_layers)

        self.posterior_encoder = _encoder(c, c.posterior_layers)
        self.posterior_cls = nn.Parameter(torch.zeros(1, 1, c.hidden_dim))
        self.posterior_state = nn.Linear(c.state_dim, c.hidden_dim)
        self.posterior_action = nn.Linear(c.action_dim, c.hidden_dim)
        self.posterior_position = nn.Embedding(c.chunk_size + 2, c.hidden_dim)
        self.posterior_distribution = nn.Linear(c.hidden_dim, 2 * c.latent_dim)
        self.latent_projection = nn.Linear(c.latent_dim, c.hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            c.hidden_dim, c.heads, c.feedforward_dim, c.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer, c.decoder_layers, norm=nn.LayerNorm(c.hidden_dim))
        self.action_queries = nn.Embedding(c.chunk_size, c.hidden_dim)
        self.action_head = nn.Linear(c.hidden_dim, c.action_dim)
        self.gesture_head = nn.Linear(c.hidden_dim, 2 * 3)
        self.skeleton_head = nn.Linear(c.hidden_dim, 2 * 21 * 3)
        self.stage_head = nn.Linear(c.hidden_dim, c.stage_classes)
        # PyTorch clones transformer layers with identical initial values.
        # Give the independent layers independent draws, without touching CNN
        # pretrained weights or changing the seed across the four variants.
        for transformer in (self.observation_encoder, self.posterior_encoder, self.action_decoder):
            for parameter in transformer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))

    def _observations(self, images: Tensor, state: Tensor) -> Tensor:
        c = self.config
        if images.ndim != 5 or images.shape[1:3] != (c.num_cameras, 3):
            raise ValueError(f"images must have shape [batch, {c.num_cameras} cameras, 3 channels, height, width]")
        batch = images.shape[0]
        if batch < 1 or min(images.shape[-2:]) < 32:
            raise ValueError("images require a nonempty batch and height/width >= 32")
        if state.shape != (batch, c.state_dim):
            raise ValueError(f"state must have shape [{batch}, {c.state_dim}]")
        if not images.is_floating_point() or not state.is_floating_point():
            raise ValueError("images and state must be floating-point tensors")
        if images.device != state.device:
            raise ValueError("images and state must share a device")
        if not torch.isfinite(images).all() or not torch.isfinite(state).all():
            raise ValueError("Robot observations must be finite")
        if (images < 0).any() or (images > 1).any():
            raise ValueError("images must be RGB in [0, 1]; ImageNet normalization is internal")
        flat_images = images.reshape(batch * c.num_cameras, 3, *images.shape[-2:])
        features = self.image_projection(self.backbone((flat_images - self.image_mean) / self.image_std))
        _, dim, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2).reshape(batch, c.num_cameras, height * width, dim)
        positions = _sine_2d(height, width, dim, images.device, tokens.dtype)
        tokens = tokens + positions[None, None] + self.camera_embedding.weight[None, :, None]
        state_token = self.state_projection(state).unsqueeze(1) + self.state_position
        return self.observation_encoder(torch.cat((state_token, tokens.flatten(1, 2)), dim=1))

    def _posterior(self, state: Tensor, actions: Tensor, is_pad: Tensor | None
                   ) -> tuple[Tensor, Tensor, Tensor]:
        c = self.config
        batch = state.shape[0]
        if actions.shape != (batch, c.chunk_size, c.action_dim):
            raise ValueError(f"actions must have shape [{batch}, {c.chunk_size}, {c.action_dim}]")
        if actions.device != state.device or not actions.is_floating_point():
            raise ValueError("actions must be floating-point and on the observation device")
        if is_pad is None:
            is_pad = torch.zeros((batch, c.chunk_size), device=state.device, dtype=torch.bool)
        if is_pad.shape != actions.shape[:2] or is_pad.dtype != torch.bool or is_pad.device != state.device:
            raise ValueError("is_pad must be boolean [batch, chunk_size] on the observation device")
        # Clear BEFORE projection: NaN * zero would otherwise poison attention.
        cleaned = torch.where(is_pad.unsqueeze(-1), torch.zeros_like(actions), actions)
        if not torch.isfinite(cleaned).all():
            raise ValueError("Unpadded action targets must be finite")
        tokens = torch.cat((self.posterior_cls.expand(batch, -1, -1),
                            self.posterior_state(state).unsqueeze(1),
                            self.posterior_action(cleaned)), dim=1)
        tokens = tokens + self.posterior_position.weight.unsqueeze(0)
        mask = torch.cat((torch.zeros((batch, 2), dtype=torch.bool, device=state.device), is_pad), dim=1)
        encoded = self.posterior_encoder(tokens, src_key_padding_mask=mask)[:, 0]
        mu, logvar = self.posterior_distribution(encoded).chunk(2, dim=-1)
        # Bound exponentiation for stable mixed precision while preserving the CVAE.
        return mu, logvar.clamp(-10, 10), ~is_pad.all(dim=1)

    def forward(self, images: Tensor, state: Tensor, actions: Tensor | None = None,
                is_pad: Tensor | None = None) -> dict[str, Any]:
        """At eval time targets are ignored and the latent is exactly zero."""
        observation = self._observations(images, state)
        batch = state.shape[0]
        # Heads cannot read z, posterior tokens, action targets, or future images.
        summary = observation.mean(dim=1)
        if self.training and actions is not None:
            mu, logvar, posterior_mask = self._posterior(state, actions, is_pad)
            latent = mu + (.5 * logvar).exp() * torch.randn_like(mu)
        else:
            mu = state.new_zeros((batch, self.config.latent_dim))
            logvar = torch.zeros_like(mu)
            latent = torch.zeros_like(mu)
            posterior_mask = torch.zeros(batch, device=state.device, dtype=torch.bool)
        memory = torch.cat((self.latent_projection(latent).unsqueeze(1), observation), dim=1)
        queries = self.action_queries.weight.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.action_decoder(queries, memory)
        return {
            "action": self.action_head(decoded), "mu": mu, "logvar": logvar,
            "posterior_mask": posterior_mask,
            "gesture_logits": self.gesture_head(summary).reshape(batch, 2, 3),
            "skeleton": self.skeleton_head(summary).reshape(batch, 2, 21, 3),
            "stage_logits": self.stage_head(summary), "variant": self.config.variant,
        }

    @torch.no_grad()
    def predict(self, images: Tensor, state: Tensor) -> Tensor:
        """Deterministic robot-only prediction; preserves the caller's train mode."""
        was_training = self.training
        try:
            self.eval()
            return self(images, state)["action"]
        finally:
            self.train(was_training)


def _valid_mask(mask: Tensor | None, shape: torch.Size | tuple[int, ...], reference: Tensor,
                name: str) -> Tensor:
    if mask is None:
        return torch.ones(shape, device=reference.device, dtype=torch.bool)
    if mask.dtype != torch.bool or mask.device != reference.device:
        raise ValueError(f"{name} mask must be boolean and on the prediction device")
    while mask.ndim < len(shape):
        mask = mask.unsqueeze(-1)
    try:
        return mask.expand(shape)
    except RuntimeError as exc:
        raise ValueError(f"{name} mask cannot broadcast to {tuple(shape)}") from exc


def _regression(prediction: Tensor, target: Tensor, mask: Tensor | None, name: str,
                smooth: bool = False) -> tuple[Tensor, Tensor]:
    if prediction.shape != target.shape or prediction.device != target.device:
        raise ValueError(f"{name} target must match prediction shape and device")
    valid = _valid_mask(mask, target.shape, prediction, name)
    clean_target = torch.where(valid, target, torch.zeros_like(target))
    clean_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    if not torch.isfinite(clean_target).all() or not torch.isfinite(clean_prediction).all():
        raise ValueError(f"Valid {name} predictions/targets must be finite")
    error = (F.smooth_l1_loss(clean_prediction, clean_target, reduction="none") if smooth
             else F.l1_loss(clean_prediction, clean_target, reduction="none"))
    count = valid.sum()
    return error.sum() / count.clamp_min(1), count


def _classification(logits: Tensor, target: Tensor, mask: Tensor | None,
                    name: str) -> tuple[Tensor, Tensor]:
    if target.shape != logits.shape[:-1] or target.device != logits.device:
        raise ValueError(f"{name} labels must match logits without the class dimension")
    if target.dtype != torch.long:
        raise ValueError(f"{name} labels must be torch.long")
    valid = _valid_mask(mask, target.shape, logits, name)
    clean_target = torch.where(valid, target, torch.zeros_like(target))
    if ((clean_target < 0) | (clean_target >= logits.shape[-1])).any():
        raise ValueError(f"Valid {name} labels are out of range")
    clean_logits = torch.where(valid.unsqueeze(-1), logits, torch.zeros_like(logits))
    if not torch.isfinite(clean_logits).all():
        raise ValueError(f"Valid {name} logits must be finite")
    error = F.cross_entropy(clean_logits.reshape(-1, logits.shape[-1]), clean_target.flatten(),
                            reduction="none").reshape(target.shape)
    count = valid.sum()
    return torch.where(valid, error, torch.zeros_like(error)).sum() / count.clamp_min(1), count


def compute_loss(output: Mapping[str, Any], targets: Mapping[str, Tensor],
                 masks: Mapping[str, Tensor] | None = None,
                 weights: Mapping[str, float] | LossWeights | None = None,
                 variant: str | None = None) -> dict[str, Tensor]:
    """Masked losses, with no training credit assigned to padding/missing labels.

    masks use True=valid (the opposite of forward's is_pad). Action L1 averages
    over valid coordinates; skeleton SmoothL1 over valid coordinates; gesture
    CE over valid hand instances. KL sums latent dimensions then averages over
    examples with at least one real action, matching ACT's usual KL convention.
    Each missing auxiliary task has exactly zero loss and zero valid count.
    """
    selected = variant or output.get("variant", "baseline")
    if selected not in VARIANTS:
        raise ValueError(f"Unknown loss variant {selected!r}")
    masks = masks or {}
    if weights is None:
        weight_values = vars(LossWeights())
    elif isinstance(weights, LossWeights):
        weight_values = vars(weights)
    else:
        unknown = set(weights) - set(vars(LossWeights()))
        if unknown:
            raise ValueError(f"Unknown loss weights: {sorted(unknown)}")
        weight_values = {**vars(LossWeights()), **weights}
    if any(not math.isfinite(float(v)) or v < 0 for v in weight_values.values()):
        raise ValueError("Loss weights must be finite and nonnegative")
    action, action_count = _regression(output["action"], targets["action"], masks.get("action"), "action")
    zero = action * 0
    zero_count = action_count.new_zeros(())
    result: dict[str, Tensor] = {"action": action, "action_count": action_count}
    mu, logvar = output["mu"].float(), output["logvar"].float()
    valid_kl = output["posterior_mask"] & _valid_mask(masks.get("action"), targets["action"].shape,
                                                     output["action"], "action").flatten(1).any(1)
    if mu.shape != logvar.shape or mu.ndim != 2 or valid_kl.shape != mu.shape[:1]:
        raise ValueError("Invalid posterior tensors")
    clean_mu = torch.where(valid_kl[:, None], mu, torch.zeros_like(mu))
    clean_logvar = torch.where(valid_kl[:, None], logvar, torch.zeros_like(logvar))
    if not torch.isfinite(clean_mu).all() or not torch.isfinite(clean_logvar).all():
        raise ValueError("Valid posterior statistics must be finite")
    kl = -.5 * (1 + clean_logvar - clean_mu.square() - clean_logvar.exp()).sum(dim=-1)
    result["kl_count"] = valid_kl.sum()
    result["kl"] = kl.sum() / result["kl_count"].clamp_min(1)
    for name in ("gesture", "skeleton", "stage"):
        result[name], result[f"{name}_count"] = zero, zero_count
    if selected in ("gesture", "gesture_skeleton") and "gesture" in targets:
        result["gesture"], result["gesture_count"] = _classification(
            output["gesture_logits"], targets["gesture"], masks.get("gesture"), "gesture")
    if selected == "gesture_skeleton" and "skeleton" in targets:
        result["skeleton"], result["skeleton_count"] = _regression(
            output["skeleton"], targets["skeleton"], masks.get("skeleton"), "skeleton", smooth=True)
    if selected == "robot_stage" and "stage" in targets:
        result["stage"], result["stage_count"] = _classification(
            output["stage_logits"], targets["stage"], masks.get("stage"), "stage")
    result["total"] = sum(result[name] * weight_values[name]
                           for name in ("action", "kl", "gesture", "skeleton", "stage"))
    return result
