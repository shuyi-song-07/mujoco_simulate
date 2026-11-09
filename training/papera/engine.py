"""Shared ACT parameter-update loop for Task Packs and reference experiments.

Data selection, validation, logging and checkpoint policy belong to the caller.
Both entry profiles execute these same forward/loss/backward/update operations.
"""
from contextlib import nullcontext

import torch

from training.papera.act_model import compute_loss


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def targets_masks(batch):
    targets = {key: batch[key] for key in ("action", "gesture", "skeleton", "stage")}
    masks = {"action": ~batch["is_pad"], "gesture": batch["gesture_mask"],
             "skeleton": batch["skeleton_mask"], "stage": torch.ones_like(batch["stage"], dtype=torch.bool)}
    return targets, masks


def precision_context(device, precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and precision == "bf16" else nullcontext()


def optimize_batch(model, cpu_batch, optimizer, scheduler, *, device, config, variant):
    """One completed update; preserve the Task trainer's exact operation order."""
    batch = move_batch(cpu_batch, device)
    optimizer.zero_grad(set_to_none=True)
    with precision_context(device, config["precision"]):
        output = model(batch["images"], batch["state"], batch["action"], batch["is_pad"])
        targets, masks = targets_masks(batch)
        losses = compute_loss(output, targets, masks, config["loss_weights"], variant)
    loss = losses["total"]
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss; stopping before optimizer update")
    loss.backward()
    if config["gradient_clip_norm"] is None:
        norm = torch.nn.utils.get_total_norm(
            [p.grad for p in model.parameters() if p.grad is not None], error_if_nonfinite=True)
    else:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return losses, norm


def run_loop(model, loader, optimizer, scheduler, *, device, config, variant,
             start_step=0, iterator=None, before_iterator=None, after_step=None):
    """Fit to a cumulative step budget, with caller-owned protocol callbacks.

    A supplied iterator is used without reconstruction, preserving the Task
    trainer's exact-resume DataLoader/RNG behavior. For finite loaders, the next
    epoch iterator is created only after the preceding after_step callback, so
    the reference profile can validate before sampling the next epoch.

    before_iterator runs once before an initial iterator is created; it is not
    repeated at epoch boundaries. after_step(step, losses, norm) may return
    False to stop after a completed update. It must commit sampler state before
    saving a checkpoint, as the Task trainer did before this extraction.
    """
    step = start_step
    if step >= config["steps"]:
        return step
    if iterator is None:
        if before_iterator is not None:
            before_iterator()
        iterator = iter(loader)
    elif before_iterator is not None:
        raise ValueError("before_iterator cannot precede an already created iterator")
    while step < config["steps"]:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise ValueError("Training loader produced no batches") from exc
        losses, norm = optimize_batch(model, batch, optimizer, scheduler,
            device=device, config=config, variant=variant)
        step += 1
        if after_step is not None and after_step(step, losses, norm) is False:
            break
    return step
