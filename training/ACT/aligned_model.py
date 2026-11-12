"""GJB's pinned original ACT core, adapted to the existing single-arm schema.

The original DETRVAE, Post-LN/ReLU transformer, positional encodings, image
backbone graph and latent fusion are imported directly. Task-specific changes
are S9 input projections, A7 targets, K30/overview configuration, finite padded
actions, and observation-only auxiliary heads outside the baseline graph.
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
from .official.backbone import FrozenBatchNorm2d, BackboneBase, Joiner
from .official.position_encoding import PositionEmbeddingSine
from .official.transformer import Transformer, TransformerEncoder, TransformerEncoderLayer
from .official.detr_vae import DETRVAE

IMPLEMENTATION = "official-act-detrvae-single-arm-v1"
VARIANTS = ("baseline", "gesture", "gesture_skeleton", "robot_stage")


@dataclass(frozen=True)
class ACTConfig:
    variant: str = "baseline"
    state_dim: int = 9
    action_dim: int = 7
    num_cameras: int = 1
    chunk_size: int = 30
    hidden_dim: int = 512
    heads: int = 8
    feedforward_dim: int = 3200
    encoder_layers: int = 4
    decoder_layers: int = 7
    posterior_layers: int = 4
    latent_dim: int = 32
    dropout: float = 0.1
    stage_classes: int = 10
    backbone_weights: str | None = None

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant {self.variant!r}; expected {VARIANTS}")
        for name in ("state_dim", "action_dim", "num_cameras", "chunk_size", "hidden_dim",
                     "heads", "feedforward_dim", "encoder_layers", "decoder_layers",
                     "posterior_layers", "latent_dim", "stage_classes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (self.state_dim, self.action_dim) not in ((9, 7),) or self.num_cameras != 1:
            raise ValueError("Single-arm adapter requires state=9, action=7 and one overview camera")
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


def build_official_model(c):
    """Assemble original components with local-only weight loading.

    Official Backbone() implicitly downloads pretrained weights. Constructing
    its original BackboneBase with the same ResNet/FrozenBN avoids network I/O.
    No early ResNet layer is frozen: upstream's freeze loop is commented out.
    """
    resnet = resnet18(weights=None, norm_layer=FrozenBatchNorm2d)
    if c.backbone_weights is not None:
        path = Path(c.backbone_weights)
        if not path.is_file():
            raise FileNotFoundError(f"Local ResNet18 weights not found: {path}")
        resnet.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    backbone = BackboneBase(resnet, train_backbone=True, num_channels=512, return_interm_layers=False)
    joined = Joiner(backbone, PositionEmbeddingSine(c.hidden_dim // 2, normalize=True))
    joined.num_channels = 512
    transformer = Transformer(d_model=c.hidden_dim, nhead=c.heads,
        num_encoder_layers=c.encoder_layers, num_decoder_layers=c.decoder_layers,
        dim_feedforward=c.feedforward_dim, dropout=c.dropout, activation="relu",
        normalize_before=False, return_intermediate_dec=True)
    posterior = TransformerEncoder(TransformerEncoderLayer(c.hidden_dim, c.heads,
        c.feedforward_dim, c.dropout, activation="relu", normalize_before=False),
        c.posterior_layers, norm=None)
    # Upstream state_dim names the output action width. Replace its two
    # hardcoded qpos projections with S9 and its posterior action input with A7.
    model = DETRVAE([joined], transformer, posterior, state_dim=c.action_dim,
        num_queries=c.chunk_size, camera_names=[f"camera_{i}" for i in range(c.num_cameras)])
    if c.state_dim != 14:
        model.input_proj_robot_state = nn.Linear(c.state_dim, c.hidden_dim)
        model.encoder_joint_proj = nn.Linear(c.state_dim, c.hidden_dim)
    if c.action_dim != 14:
        model.encoder_action_proj = nn.Linear(c.action_dim, c.hidden_dim)
    if c.latent_dim != model.latent_dim:
        model.latent_dim = c.latent_dim
        model.latent_proj = nn.Linear(c.hidden_dim, 2 * c.latent_dim)
        model.latent_out_proj = nn.Linear(c.latent_dim, c.hidden_dim)
    return model


class ACTPolicy(nn.Module):
    """Original ACT graph plus noninvasive observation auxiliary heads."""
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ACTConfig()
        c = self.config
        self.model = build_official_model(c)
        # These heads never enter the original ACT forward graph, and their
        # initialization does not advance the baseline model's random stream.
        with torch.random.fork_rng(devices=[]):
            self.gesture_head = nn.Linear(c.hidden_dim + c.state_dim, 6)
            self.skeleton_head = nn.Linear(c.hidden_dim + c.state_dim, 2*21*3)
            self.stage_head = nn.Linear(c.hidden_dim + c.state_dim, c.stage_classes)
        self.register_buffer("image_mean", torch.tensor([.485,.456,.406]).reshape(1,1,3,1,1))
        self.register_buffer("image_std", torch.tensor([.229,.224,.225]).reshape(1,1,3,1,1))

    @property
    def backbone(self):
        return self.model.backbones[0][0].body

    @property
    def action_head(self):
        return self.model.action_head

    @property
    def image_projection(self):
        return self.model.input_proj

    @property
    def posterior_encoder(self):
        return self.model.encoder

    def _validate_observations(self, images, state):
        c = self.config
        if images.ndim != 5 or images.shape[1:3] != (c.num_cameras,3):
            raise ValueError(f"images must have shape [batch, {c.num_cameras} cameras, 3 channels, height, width]")
        batch = images.shape[0]
        if batch < 1 or min(images.shape[-2:]) < 32:
            raise ValueError("images require a nonempty batch and height/width >= 32")
        if state.shape != (batch,c.state_dim):
            raise ValueError(f"state must have shape [{batch}, {c.state_dim}]")
        if not images.is_floating_point() or not state.is_floating_point():
            raise ValueError("images and state must be floating-point tensors")
        if images.device != state.device:
            raise ValueError("images and state must share a device")
        if not torch.isfinite(images).all() or not torch.isfinite(state).all():
            raise ValueError("Robot observations must be finite")
        if (images < 0).any() or (images > 1).any():
            raise ValueError("images must be RGB in [0, 1]; ImageNet normalization is internal")

    def forward(self, images, state, actions=None, is_pad=None, *, posterior_in_eval=False):
        """Task evaluation defaults to z=0; the reference can request its posterior."""
        self._validate_observations(images,state)
        c,batch = self.config,state.shape[0]
        cleaned = None
        posterior_mask = torch.zeros(batch,device=state.device,dtype=torch.bool)
        if type(posterior_in_eval) is not bool:
            raise ValueError("posterior_in_eval must be an explicit boolean")
        if actions is not None and (self.training or posterior_in_eval):
            if actions.shape != (batch,c.chunk_size,c.action_dim):
                raise ValueError(f"actions must have shape [{batch}, {c.chunk_size}, {c.action_dim}]")
            if actions.device != state.device or not actions.is_floating_point():
                raise ValueError("actions must be floating-point and on the observation device")
            if is_pad is None:
                is_pad = torch.zeros(actions.shape[:2],device=state.device,dtype=torch.bool)
            if is_pad.shape != actions.shape[:2] or is_pad.dtype != torch.bool or is_pad.device != state.device:
                raise ValueError("is_pad must be boolean [batch, chunk_size] on the observation device")
            cleaned = actions.masked_fill(is_pad.unsqueeze(-1),0)
            if not torch.isfinite(cleaned).all():
                raise ValueError("Unpadded action targets must be finite")
            posterior_mask = ~is_pad.all(1)
        pooled = []
        # Capture projected image features before any latent or proprio token
        # enters the original transformer. Auxiliaries cannot read actions/z.
        handle = self.model.input_proj.register_forward_hook(
            lambda _module,_inputs,value: pooled.append(value.mean(dim=(-2,-1))))
        try:
            action,pad_logits,(mu,logvar) = self.model(state,
                (images-self.image_mean)/self.image_std, None, cleaned, is_pad if cleaned is not None else None)
        finally:
            handle.remove()
        if mu is None:
            mu = state.new_zeros((batch,c.latent_dim))
            logvar = torch.zeros_like(mu)
        summary = torch.cat((torch.stack(pooled,dim=1).mean(dim=1),state),dim=-1)
        return {"action":action,"mu":mu,"logvar":logvar,"posterior_mask":posterior_mask,
            "is_pad_logits":pad_logits,
            "gesture_logits":self.gesture_head(summary).reshape(batch,2,3),
            "skeleton":self.skeleton_head(summary).reshape(batch,2,21,3),
            "stage_logits":self.stage_head(summary),"variant":c.variant}

    @torch.no_grad()
    def predict(self,images,state):
        was_training = self.training
        try:
            self.eval()
            return self(images,state)["action"]
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

    masks use True=valid (the opposite of forward's is_pad). Official Action L1
    masks invalid errors then averages over all B*K*A coordinates; skeleton SmoothL1
    over valid coordinates; gesture
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
    action_valid, action_count = _regression(output["action"], targets["action"], masks.get("action"), "action")
    valid_action = _valid_mask(masks.get("action"), targets["action"].shape, output["action"], "action")
    clean_target = torch.where(valid_action, targets["action"], torch.zeros_like(targets["action"]))
    clean_prediction = torch.where(valid_action, output["action"], torch.zeros_like(output["action"]))
    # Exactly the upstream policy.py reduction: invalid errors zero, full mean.
    action = F.l1_loss(clean_prediction, clean_target, reduction="none").mean()
    zero = action * 0
    zero_count = action_count.new_zeros(())
    result: dict[str, Tensor] = {"action": action, "action_count": action_count,
        "action_valid_l1": action_valid, "action_full_count": action_count.new_tensor(targets["action"].numel())}
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
