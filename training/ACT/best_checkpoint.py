"""Track the trainer's validation loss and point ``best`` at a saved checkpoint."""

from functools import wraps
import inspect
import json
import logging
import math
import os
from pathlib import Path


def install_best_checkpoint(train_module) -> None:
    """Support both the 0.6.1 direct-forward and newer module-call eval loops.

    Each validation step must also be a regular checkpoint step. The observer
    measures the same mean of batch losses as LeRobot, without rounding the
    value printed to the console or running another validation pass.
    """
    import torch

    original_policy = train_module.make_policy
    original_save = train_module.save_checkpoint
    save_signature = inspect.signature(original_save)
    state = {"loss": None, "best": math.inf}

    @wraps(original_policy)
    def make_policy(*args, **kwargs):
        policy = original_policy(*args, **kwargs)
        original_forward = policy.forward
        original_train = policy.train
        totals = {"sum": 0.0, "count": 0, "device": None}

        @wraps(original_forward)
        def forward(*forward_args, **forward_kwargs):
            result = original_forward(*forward_args, **forward_kwargs)
            if not policy.training:
                loss = result[0].detach()
                totals["sum"] += loss.item()
                totals["count"] += 1
                totals["device"] = loss.device
            return result

        @wraps(original_train)
        def train(mode=True):
            was_training = policy.training
            result = original_train(mode)
            if not mode:
                totals.update(sum=0.0, count=0, device=None)
                state["loss"] = None
            elif not was_training and totals["count"]:
                # The trainer averages batch losses per rank, then across ranks.
                loss = torch.tensor(
                    totals["sum"] / totals["count"], device=totals["device"],
                )
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(loss)
                    loss /= torch.distributed.get_world_size()
                state["loss"] = loss.item()
            return result

        policy.forward = forward
        policy.train = train
        return policy

    @wraps(original_save)
    def save_checkpoint(*args, **kwargs):
        bound = save_signature.bind(*args, **kwargs)
        result = original_save(*args, **kwargs)
        arguments = bound.arguments
        cfg, step = arguments["cfg"], arguments["step"]
        accelerator = arguments.get("accelerator")
        if accelerator is not None and not accelerator.is_main_process:
            return result
        if cfg.eval_steps <= 0 or step % cfg.eval_steps:
            return result
        loss = state["loss"]
        if loss is None or not math.isfinite(loss):
            return result
        directory = Path(arguments["checkpoint_dir"])
        record = {"step": step, "eval_loss": loss, "checkpoint": directory.name}
        with (directory.parent / "validation_losses.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if loss < state["best"]:
            # Publish the pointer only after the complete checkpoint is written.
            pending = directory.parent / ".best.tmp"
            pending.symlink_to(directory.name)
            os.replace(pending, directory.parent / "best")
            metadata = directory.parent / ".best.json.tmp"
            metadata.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            os.replace(metadata, directory.parent / "best.json")
            state["best"] = loss
            logging.info("Best validation checkpoint: step %s, eval_loss=%.8g", step, loss)
        return result

    train_module.make_policy = make_policy
    train_module.save_checkpoint = save_checkpoint
