"""Training-only three-category supervision, independent of either action head.

The labels are the Pack's causal recorded recognizer categories, not accepted
control modes. Unknown/missing categories require a false mask. This module
does not infer labels from robot actions, skeletons, or task phase.
"""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

CATEGORIES = ("ILoveYou", "Open_Palm", "Closed_Fist")
VERSION = "recorded-category-masked-ce-v1"


def summarize_gesture_confusion(confusion) -> dict:
    """Per-hand class metrics; absent true classes are excluded from macro recall."""
    values = torch.as_tensor(confusion, dtype=torch.float64, device="cpu")
    if values.shape != (2, 3, 3) or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Gesture confusion must be finite nonnegative [2, 3, 3]")
    result = {"categories": list(CATEGORIES)}
    for side, matrix in zip(("left", "right"), values):
        support, predicted = matrix.sum(1), matrix.sum(0)
        true_positive = matrix.diagonal()
        valid = support > 0
        recall = true_positive / support.clamp_min(1)
        precision = true_positive / predicted.clamp_min(1)
        f1 = 2 * true_positive / (support + predicted).clamp_min(1)
        result[side] = {"support": support.long().tolist(),
            "recall": [float(value) if present else None for value, present in zip(recall, valid)],
            "precision": precision.tolist(), "f1": f1.tolist(),
            "macro_recall": float(recall[valid].mean()) if valid.any() else None,
            "macro_f1": float(f1[valid].mean()) if valid.any() else None}
    return result


def masked_gesture_loss(logits: Tensor, labels: Tensor, valid: Tensor) -> dict[str, Tensor]:
    """Unweighted CE averaged over valid hand instances, as in ACT.

    The final dimension has three classes and the previous dimension is
    independent left/right hands. Missing labels contribute neither loss nor
    denominator. Returning sufficient statistics avoids averaging unequal
    minibatch coverage in validation.
    """
    if logits.ndim != 3 or logits.shape[1:] != (2, 3):
        raise ValueError("gesture logits must have shape [B, 2, 3]")
    if labels.shape != logits.shape[:2] or valid.shape != labels.shape:
        raise ValueError("gesture labels and masks must have shape [B, 2]")
    if labels.dtype != torch.long or valid.dtype != torch.bool:
        raise ValueError("gesture labels must be long and masks must be boolean")
    if labels.device != logits.device or valid.device != logits.device:
        raise ValueError("gesture tensors must share a device")
    clean_labels = torch.where(valid, labels, torch.zeros_like(labels))
    if ((clean_labels < 0) | (clean_labels >= 3)).any():
        raise ValueError("Valid gesture labels must be in [0, 2]; Unknown must be masked")
    clean_logits = torch.where(valid.unsqueeze(-1), logits, torch.zeros_like(logits))
    if not torch.isfinite(clean_logits).all():
        raise ValueError("Valid gesture logits must be finite")
    per_hand = F.cross_entropy(clean_logits.float().reshape(-1, 3), clean_labels.flatten(),
                              reduction="none").reshape(labels.shape)
    numerator = per_hand.masked_fill(~valid, 0).sum()
    count = valid.sum()
    confusion = torch.stack([torch.bincount(
        (clean_labels[:, side][valid[:, side]] * 3
         + clean_logits[:, side].argmax(-1)[valid[:, side]]), minlength=9).reshape(3, 3)
        for side in range(2)])
    return {"loss": numerator / count.clamp_min(1), "count": count,
            "sum": numerator.detach(), "confusion": confusion.detach()}
