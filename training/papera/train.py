"""Standard ACT: task-bound single oblique Safe Stitch baseline and gesture auxiliary supervision.

config/training_baseline.json is the current entry. Dataset owns all curation
and sequence certificates; this trainer only fits normalization and K targets.
Historical three-view reproduction stays in the original project.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from training.papera.act_model import ACTConfig, ACTPolicy, compute_loss
from training.papera.data import file_sha256

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_VERSION = "papera-act-offline-checkpoint-v1"
EXPERIMENTS = ("baseline", "gesture")


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def resolve_path(path):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def seed_everything(seed):
    # cuDNN alone is insufficient: BF16 attention/other CUDA reductions can
    # diverge after restore despite identical samples and RNG. Configure this
    # before the first CUDA matmul and reject nondeterministic implementations.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # Fixed seed and recorded numerical settings support comparison; different
    # GPU libraries/machines are not promised bitwise reproducible.
    torch.backends.cudnn.deterministic = True


class StatefulBatchSampler(Sampler):
    """Checkpoint consumed batches, not batches merely prefetched by workers."""
    def __init__(self, size, batch_size, seed, prefetch_safe=False):
        if size < 1 or batch_size < 1:
            raise ValueError("Positive sample and batch counts required")
        self.size, self.batch_size = int(size), int(batch_size)
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)
        self.cursor, self.epoch = 0, 0
        self.prefetch_safe = prefetch_safe
        self._pending = deque()
        self._committed = self._issued_state()

    def __iter__(self):
        while True:
            if self.cursor == self.size:
                self.order = torch.randperm(self.size, generator=self.generator)
                self.cursor = 0; self.epoch += 1
            end = min(self.size, self.cursor + self.batch_size)
            batch = self.order[self.cursor:end].tolist()
            self.cursor = end
            if self.prefetch_safe:
                self._pending.append(self._issued_state())
            yield batch

    def _issued_state(self):
        return {"size": self.size, "batch_size": self.batch_size, "order": self.order,
                "cursor": self.cursor, "epoch": self.epoch, "generator": self.generator.get_state()}

    def mark_consumed(self):
        if self.prefetch_safe:
            if not self._pending:
                raise RuntimeError("No issued batch to commit")
            self._committed = self._pending.popleft()

    def state_dict(self):
        return self._committed if self.prefetch_safe else self._issued_state()

    def load_state_dict(self, state):
        if (state["size"], state["batch_size"]) != (self.size, self.batch_size):
            raise ValueError("Resume sample count/batch size changed")
        order = state["order"].cpu()
        if not torch.equal(torch.sort(order).values, torch.arange(self.size)):
            raise ValueError("Checkpoint sampler order is not a permutation")
        if not 0 <= state["cursor"] <= self.size:
            raise ValueError("Checkpoint sampler cursor invalid")
        self.order, self.cursor, self.epoch = order, state["cursor"], state["epoch"]
        self.generator.set_state(state["generator"].cpu())
        self._pending.clear()
        self._committed = self._issued_state()


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def lr_multiplier(step, config):
    continuation = config.get("lr_schedule")
    if continuation is not None:
        if continuation["kind"] == "constant_checkpoint_lr":
            # Continue from the exact LR stored in the source checkpoint.  The
            # 20k schedule has already reached its floor; do not jump back up
            # when the cumulative budget is extended.
            return 1.0
        raise ValueError(f"Unsupported learning-rate schedule: {continuation['kind']}")
    warmup = config["warmup_steps"]
    if warmup and step < warmup:
        return (step + 1) / warmup
    fraction = (step - warmup) / max(1, config["steps"] - warmup)
    floor = config["minimum_lr_fraction"]
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * min(1., max(0., fraction))))


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


@torch.no_grad()
def validate(model, loader, normalizer, device, precision="bf16", max_batches=None):
    """Always use prior z=0; never choose a checkpoint by future-informed loss."""
    was_training = model.training
    model.eval()
    sums, counts = {}, {}
    xyz_sum = grip_sum = angle_sum = 0.; physical_count = 0
    horizons = {}
    confusion = np.zeros((2, 3, 3), dtype=np.int64)
    stage_correct = stage_count = 0
    try:
        for number, cpu_batch in enumerate(loader):
            if max_batches is not None and number >= max_batches:
                break
            b = move_batch(cpu_batch, device)
            with precision_context(device, precision):
                out = model(b["images"], b["state"])  # No future target enters the model.
                targets, masks = targets_masks(b)
                losses = compute_loss(out, targets, masks, variant=model.config.variant)
            for key in ("action", "gesture", "skeleton", "stage"):
                count = float(losses[key + "_count"].detach().cpu())
                value = float(losses[key].detach().cpu())
                if not math.isfinite(value):
                    raise FloatingPointError(f"Validation {key} is not finite")
                sums[key] = sums.get(key, 0.) + value * count
                counts[key] = counts.get(key, 0.) + count
            pred = normalizer.decode_action(out["action"].float().cpu().numpy())
            actual = normalizer.decode_action(b["action"].float().cpu().numpy())
            valid = (~b["is_pad"]).cpu().numpy()
            for h in (0, 4, 9, 29):
                if h >= pred.shape[1]:
                    continue
                row = horizons.setdefault(h, {"samples": 0, "xyz": 0., "grip": 0., "rpy": 0.})
                supported = valid[:, h]
                row["samples"] += int(supported.sum())
                for offset in (0, 7):
                    estimate, target = pred[supported, h], actual[supported, h]
                    row["xyz"] += float(np.linalg.norm(
                        estimate[:, offset:offset + 3] - target[:, offset:offset + 3], axis=-1).sum())
                    row["grip"] += float(np.abs(estimate[:, offset + 6] - target[:, offset + 6]).sum())
                    delta = estimate[:, offset + 3:offset + 6] - target[:, offset + 3:offset + 6]
                    row["rpy"] += float(np.abs((delta + np.pi) % (2 * np.pi) - np.pi).sum())
            for offset in (0, 7):
                xyz_sum += float(np.linalg.norm(pred[..., offset:offset+3] - actual[..., offset:offset+3], axis=-1)[valid].sum())
                grip_sum += float(np.abs(pred[..., offset+6] - actual[..., offset+6])[valid].sum())
                delta = pred[..., offset+3:offset+6] - actual[..., offset+3:offset+6]
                wrapped = (delta + np.pi) % (2 * np.pi) - np.pi
                angle_sum += float(np.abs(wrapped)[valid].sum())
            physical_count += int(valid.sum()) * 2
            predicted_class = out["gesture_logits"].argmax(-1).cpu().numpy()
            labels, mask = b["gesture"].cpu().numpy(), b["gesture_mask"].cpu().numpy()
            for side in range(2):
                for true, estimated in zip(labels[mask[:, side], side], predicted_class[mask[:, side], side]):
                    confusion[side, int(true), int(estimated)] += 1
            stage_correct += int((out["stage_logits"].argmax(-1) == b["stage"]).sum().cpu())
            stage_count += len(b["stage"])
        if counts.get("action", 0.) <= 0:
            raise ValueError("Validation has no valid action targets; cannot select a checkpoint")
        result = {"prior_action_l1": sums.get("action", 0.) / counts["action"],
            "xyz_error_mm": xyz_sum / max(1, physical_count) * 1000,
            "gripper_error_mm": grip_sum / max(1, physical_count) * 1000,
            "rpy_mae_degrees": angle_sum / max(1, physical_count * 3) * 180 / np.pi,
            "valid_action_coordinates": counts.get("action", 0.), "prior_only": True,
            "success_rate": None}
        for h, row in horizons.items():
            arms = row["samples"] * 2
            result.update({
                f"h{h}_valid_samples": row["samples"],
                f"h{h}_xyz_error_mm": row["xyz"] / arms * 1000 if arms else None,
                f"h{h}_gripper_error_mm": row["grip"] / arms * 1000 if arms else None,
                f"h{h}_rpy_mae_degrees": row["rpy"] / (arms * 3) * 180 / np.pi if arms else None,
            })
        if model.config.variant in ("gesture", "gesture_skeleton"):
            result["gesture_confusion"] = confusion.tolist()
            support = confusion.sum()
            result["gesture_accuracy"] = float(np.trace(confusion.sum(0)) / support) if support else None
            result["gesture_loss"] = sums["gesture"] / max(1., counts["gesture"])
            result["gesture_support"] = int(support)
        if model.config.variant == "gesture_skeleton":
            result["skeleton_huber"] = sums["skeleton"] / max(1., counts["skeleton"])
            result["skeleton_coordinate_count"] = counts["skeleton"]
        if model.config.variant == "robot_stage":
            result["stage_accuracy"] = stage_correct / max(1, stage_count)
        return result
    finally:
        model.train(was_training)


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix(".writing")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, expected_signature):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Not a compatible Paper A checkpoint; legacy Task3 checkpoints cannot resume")
    if checkpoint.get("signature") != expected_signature:
        raise ValueError("Resume rejected: data/split/normalization/model/config/variant differs")
    return checkpoint


def checkpoint_elapsed(checkpoint, run_dir):
    """Committed elapsed time; older checkpoints fall back to logs at/before step.

    The fallback is approximate by at most the logging interval, not a reset to
    zero. Ignore log entries produced after the checkpoint being restored.
    """
    value = checkpoint.get("elapsed_s")
    if value is not None:
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid checkpoint elapsed_s")
        return float(value)
    elapsed = 0.0
    metrics = Path(run_dir) / "metrics.jsonl"
    if metrics.is_file():
        for line in metrics.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            value = row.get("elapsed_s", 0.)
            if row.get("step", 0) <= checkpoint["step"] and math.isfinite(value):
                elapsed = max(elapsed, value)
    return elapsed


def load_extension_source(checkpoint_path, target_steps):
    """Read a completed baseline last.pt for a new, lineage-tracked run."""
    checkpoint_path = Path(checkpoint_path).resolve()
    if checkpoint_path.name != "last.pt" or checkpoint_path.parent.name != "checkpoints":
        raise ValueError("--continue-from must name a run's checkpoints/last.pt")
    source_run = checkpoint_path.parent.parent
    required = (source_run / "run.json", source_run / "config.json", source_run / "completion.json",
                source_run / "pack_binding.json", source_run / "split.json", source_run / "stats.json",
                source_run / "checkpoints/best.pt")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Cannot extend an incomplete or unfrozen source run; missing: {missing}")
    run_doc = json.loads((source_run / "run.json").read_text(encoding="utf-8"))
    provenance = run_doc.get("provenance", {})
    signature = run_doc.get("signature")
    if not signature or canonical_hash(provenance) != signature:
        raise ValueError("Source run provenance signature is invalid")
    from training.papera.source_compatibility import verify_training_sources
    verify_training_sources(provenance)
    frozen_config = provenance.get("config")
    if not isinstance(frozen_config, dict) or frozen_config.get("steps") is None:
        raise ValueError("Source run has no frozen training configuration")
    if provenance.get("variant") != "baseline" or provenance.get("engineering_overfit"):
        raise ValueError("Only a completed full-data ACT baseline can be extended")
    if file_sha256(source_run / "config.json") != json_file_hash(frozen_config):
        raise ValueError("Source config.json differs from the run's frozen provenance")
    for filename, value in (("pack_binding.json", provenance.get("pack_binding")),
                            ("split.json", provenance.get("pack_binding", {}).get("split")),
                            ("stats.json", provenance.get("pack_binding", {}).get("stats"))):
        if value is None or file_sha256(source_run / filename) != json_file_hash(value):
            raise ValueError(f"Source run frozen file differs: {filename}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint.get("version") != CHECKPOINT_VERSION
            or checkpoint.get("signature") != signature
            or checkpoint.get("provenance") != provenance):
        raise ValueError("Source last.pt does not match its frozen run provenance")
    completion = json.loads((source_run / "completion.json").read_text(encoding="utf-8"))
    if (completion.get("completed") is not True
            or completion.get("step") != checkpoint.get("step")
            or completion.get("step") != frozen_config["steps"]
            or completion.get("target_steps") != frozen_config["steps"]):
        raise ValueError("Source run must be fully completed at its frozen target step")
    if type(target_steps) is not int or target_steps <= checkpoint["step"]:
        raise ValueError("The new cumulative target must be greater than the source checkpoint step")
    if type(checkpoint.get("best_prior_action_l1")) not in (int, float) or not math.isfinite(checkpoint["best_prior_action_l1"]):
        raise ValueError("Source checkpoint has no finite validation-selected best metric")
    param_groups = checkpoint.get("optimizer", {}).get("param_groups", [])
    start_lrs = [float(group["lr"]) for group in param_groups]
    if len(start_lrs) != 2 or any(not math.isfinite(lr) or lr <= 0 for lr in start_lrs):
        raise ValueError("Source checkpoint does not contain the expected two finite optimizer learning rates")
    best_checkpoint = torch.load(source_run / "checkpoints/best.pt", map_location="cpu", weights_only=True)
    if (best_checkpoint.get("version") != CHECKPOINT_VERSION
            or best_checkpoint.get("signature") != signature
            or best_checkpoint.get("provenance") != provenance
            or best_checkpoint.get("step", target_steps + 1) > checkpoint["step"]):
        raise ValueError("Source best.pt does not match the completed source run")
    return {"checkpoint": checkpoint, "best_checkpoint": best_checkpoint, "run": source_run,
            "run_doc": run_doc, "completion": completion, "signature": signature,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "best_checkpoint_sha256": file_sha256(source_run / "checkpoints/best.pt"),
            "start_lrs": start_lrs}


def validate_extension_config(config, source, target_steps):
    """Permit only a cumulative step extension and its explicit LR hold phase."""
    expected = copy.deepcopy(source["run_doc"]["provenance"]["config"])
    expected["steps"] = target_steps
    expected["lr_schedule"] = {"kind": "constant_checkpoint_lr",
                               "start_step": source["checkpoint"]["step"],
                               "start_lrs": source["start_lrs"]}
    expected.setdefault("_help", {})["steps"] = (
        f"从已完成的{source['checkpoint']['step']}步 last.pt 延续，累计目标{target_steps}步；"
        "续训阶段保持checkpoint中的最低学习率，不重启升高。")
    if config != expected:
        raise ValueError("Continuation may change only total steps and the explicit hold-at-checkpoint-LR schedule")


def validate_config(config):
    for key in ("steps", "batch_size", "validation_every", "validation_stride", "save_every", "log_every", "cpu_threads"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    workers = config.get("num_workers")
    if type(workers) is not int or not 0 <= workers <= 8:
        raise ValueError("num_workers must be an integer in [0,8]")
    if workers and "pack_path" not in config:
        raise ValueError("Historical three-view exact resume requires num_workers=0")
    if "pack_path" in config:
        fields = config["model"]
        if (fields["state_dim"], fields["action_dim"], fields["num_cameras"], fields["chunk_size"]) != (16, 14, 1, 30):
            raise ValueError("Current baseline requires one camera, S16, A14 and K30")
        if any(key in config for key in ("dataset", "data_preparation", "sequence_data")):
            raise ValueError("Safe Stitch configuration cannot contain legacy data-processing options")
        image_size = config.get("image_size")
        if (not isinstance(image_size, list) or len(image_size) != 2
                or any(type(x) is not int or not 32 <= x <= 2048 for x in image_size)):
            raise ValueError("image_size must be [width,height], integer pixels in [32,2048]")
    if config.get("precision") not in ("bf16", "fp32"):
        raise ValueError("precision must be bf16 or fp32")
    if type(config.get("seed")) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not 0 <= config["warmup_steps"] < config["steps"]:
        raise ValueError("warmup_steps must be nonnegative and less than steps")
    for key in ("learning_rate", "backbone_learning_rate", "weight_decay", "gradient_clip_norm"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not 0 < config["minimum_lr_fraction"] <= 1:
        raise ValueError("minimum_lr_fraction must be in (0,1]")
    lr_schedule = config.get("lr_schedule")
    if lr_schedule is not None:
        if not isinstance(lr_schedule, dict) or lr_schedule.get("kind") != "constant_checkpoint_lr":
            raise ValueError("lr_schedule.kind must be constant_checkpoint_lr")
        if type(lr_schedule.get("start_step")) is not int or not 0 < lr_schedule["start_step"] < config["steps"]:
            raise ValueError("lr_schedule.start_step must be a positive step below the target budget")
        start_lrs = lr_schedule.get("start_lrs")
        if (not isinstance(start_lrs, list) or len(start_lrs) != 2
                or any(type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0 for lr in start_lrs)):
            raise ValueError("lr_schedule.start_lrs must contain the two positive checkpoint learning rates")
    if config.get("allow_exploratory") is not True:
        raise ValueError("This run requires the explicit exploratory-data setting authorized by the user")


def json_file_hash(value):
    """Hash the exact UTF-8 bytes written by json_write, including the newline."""
    return hashlib.sha256((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")).hexdigest()


def reconcile_resume_logs(output, step):
    """Preserve abandoned post-checkpoint logs outside the active metric stream."""
    output = Path(output)
    metric_path = output / "metrics.jsonl"
    kept, abandoned = [], []
    if metric_path.is_file():
        for line in metric_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                (kept if json.loads(line)["step"] <= step else abandoned).append(line)
    completion = output / "completion.json"
    if abandoned or completion.exists():
        history = output / "resume_history" / f"from_step{step}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        history.mkdir(parents=True, exist_ok=False)
        if abandoned:
            (history / "uncommitted_metrics.jsonl").write_text("\n".join(abandoned)+"\n", encoding="utf-8", newline="\n")
            temporary = metric_path.with_suffix(".writing")
            temporary.write_text("\n".join(kept)+("\n" if kept else ""), encoding="utf-8", newline="\n")
            os.replace(temporary, metric_path)
        if completion.exists():
            os.replace(completion, history / "previous_completion.json")


def prepare_safe_data(config, count=None, resume=None):
    """Verify the explicit candidate; CURRENT is discovery information only."""
    from dataset.task3.training_pack import verify
    from training.papera.pack_data import prepare_pack, restore_binding
    pack = resolve_path(config["pack_path"])
    config["pack_path"] = str(pack)
    expected = config.get("pack_manifest_sha256")
    actual = file_sha256(pack / "manifest.json")
    if expected is not None and expected != actual:
        raise ValueError("Explicit Pack manifest SHA differs from training configuration")
    pointer = ROOT / "outputs/dataset/TRAINING_PACK_CURRENT.json"
    if pointer.is_file():
        navigation = json.loads(pointer.read_text(encoding="utf-8"))
        print(f"Dataset navigation CURRENT: {navigation.get('path', navigation.get('selected_version'))}", flush=True)
    print(f"Verifying explicit Safe Stitch Pack: {pack}\nmanifest SHA256: {actual}", flush=True)
    if resume:
        frozen = Path(resume).resolve().parent.parent / "pack_binding.json"
        binding = restore_binding(json.loads(frozen.read_text(encoding="utf-8")))
        if Path(binding["pack_path"]) != pack:
            raise ValueError("Resume cannot change the frozen Pack path")
        keys = binding["train_keys"] if count is not None else None
        if count is not None and len(keys) != count:
            raise ValueError("Resume engineering subset differs")
    else:
        checked = verify(pack)
        split = checked["manifest"]["split"]
        keys = None
        if count is not None:
            if type(count) is not int or not 1 <= count <= 2:
                raise ValueError("Single-view engineering overfit permits only 1 or 2 episodes")
            keys = config.get("overfit_episode_keys", split["train"][:count])
            if len(keys) != count or len(set(keys)) != count or not set(keys) <= set(split["train"]):
                raise ValueError("Engineering subset must contain N distinct frozen train episode keys")
        binding = prepare_pack(pack, selected_train_keys=keys, chunk_size=config["model"]["chunk_size"],
                               checked=checked, image_size=config["image_size"])
    if (list(binding["image_size"]) != config["image_size"]
            or binding["chunk_size"] != config["model"]["chunk_size"]):
        raise ValueError("Resume image size or chunk size differs from frozen binding")
    return pack, binding, keys


def run(config, variant="baseline", output=None, resume=None, smoke=False, stop_after=None,
        overfit_episodes=None, continue_from=None, extension_source=None):
    config = json.loads(json.dumps(config))
    if variant not in EXPERIMENTS:
        raise ValueError(f"本轮只提供 ACT baseline 和 ACT + gesture，收到：{variant}")
    if resume is not None and continue_from is not None:
        raise ValueError("Use either exact --resume or new-run --continue-from, not both")
    if continue_from is not None:
        if resume is not None or smoke or overfit_episodes is not None or variant != "baseline":
            raise ValueError("Checkpoint extension is only supported for a full-data single-view baseline")
        extension_source = extension_source or load_extension_source(continue_from, config.get("steps"))
        validate_extension_config(config, extension_source, config.get("steps"))
    is_overfit = overfit_episodes is not None
    safe_stitch = "pack_path" in config
    if not safe_stitch or not config.get('pack_path'):
        raise ValueError('Explicit --pack pointing to the official Safe Stitch Training Pack is required')
    from tasks.artifacts import check_artifact
    artifact = check_artifact(resolve_path(config['pack_path']), config['task_id'],
                              engineering=bool(config.get('engineering_only', False)))
    if artifact['engineering_only'] and not (smoke or is_overfit):
        raise ValueError('Engineering Pack cannot start formal training')
    if artifact['task_binding'] != config['task_binding']:
        raise ValueError('Training/task binding mismatch')
    if is_overfit and smoke:
        raise ValueError("--overfit-episodes and --smoke-test are mutually exclusive")
    if is_overfit and (type(config.get("steps")) is not int or not 1 <= config["steps"] <= 3000):
        raise ValueError("Overfit diagnostics require an explicit step budget from 1 to 3000")
    if smoke:
        config.update(steps=3, warmup_steps=0, validation_every=3, save_every=3, log_every=1)
    validate_config(config)
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    binding = None
    pack, binding, keys = prepare_safe_data(config, overfit_episodes, resume or continue_from)
    sequence_provenance = None
    if is_overfit and output is not None:
        if (ROOT / ".local/checks").resolve() not in resolve_path(output).parents:
            raise ValueError("Overfit diagnostic output must be inside .local/checks")
    torch.set_num_threads(config["cpu_threads"])
    seed_everything(config["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select D:/Conda/envs/paperA_train/python.exe in VS Code")
    device = torch.device("cuda")
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable; explicitly use fp32 rather than silently changing experiment settings")

    model_fields = dict(config["model"])
    weights = resolve_path(model_fields["backbone_weights"]) if model_fields.get("backbone_weights") else None
    if weights is not None and not weights.is_file():
        raise FileNotFoundError(f"Pretrained ResNet18 weights missing; run setup first: {weights}")
    model_fields.update(variant=variant, backbone_weights=str(weights) if weights else None)
    model_config = ACTConfig(**model_fields)
    provenance = {"pack": str(pack), "pack_manifest_sha256": file_sha256(pack / "manifest.json"),
        "split_sha256": json_file_hash(binding["split"]) if binding else file_sha256(pack / "split.json"),
        "stats_sha256": json_file_hash(binding["stats"]) if binding else file_sha256(pack / "stats.json"),
        "pretrained_weights_sha256": file_sha256(weights) if weights else None,
        "model_source_sha256": file_sha256(Path(__file__).with_name("act_model.py")),
        "reader_source_sha256": file_sha256(Path(__file__).with_name("pack_data.py" if binding else "data.py")),
        "trainer_source_sha256": file_sha256(Path(__file__)),
        "task_id": config["task_id"], "task_binding": config["task_binding"],
        "config": config, "variant": variant, "model": asdict(model_config),
        "exploratory": True, "human_quality_review": "pending", "offline_only": True,
        "action_alignment": "next_control_tick_at_30hz_anchors_v1",
        "sequence_data": sequence_provenance, "engineering_overfit": is_overfit,
        "overfit_episode_keys": keys,
        "validation_scope": "overfit_reconstruction" if is_overfit else "heldout_sessions"}
    if binding:
        provenance.update(data_contract=binding["version"], pack_binding=dict(binding),
                          sequence_contract=binding["sequence_contract"], cameras=binding["cameras"],
                          action_alignment=binding["action_alignment"],
                          normalizer_source_sha256=file_sha256(Path(__file__).with_name("data.py")),
                          pack_reader_source_sha256=file_sha256(ROOT / "dataset/task3/training_pack.py"))
    if resume:
        resume_run_doc = json.loads((Path(resume).resolve().parent.parent / "run.json").read_text(encoding="utf-8"))
        prior_lineage = resume_run_doc.get("provenance", {}).get("continued_from")
        if prior_lineage is not None:
            provenance["continued_from"] = prior_lineage
    if extension_source is not None:
        provenance["continued_from"] = {
            "run": str(extension_source["run"]),
            "checkpoint": str(Path(continue_from).resolve()),
            "checkpoint_sha256": extension_source["checkpoint_sha256"],
            "best_checkpoint_sha256": extension_source["best_checkpoint_sha256"],
            "parent_signature": extension_source["signature"],
            "parent_step": extension_source["checkpoint"]["step"],
            "parent_best_prior_action_l1": extension_source["checkpoint"]["best_prior_action_l1"],
            "optimizer_learning_rates": extension_source["start_lrs"],
            "extension_lr_policy": "hold checkpoint learning rates through the cumulative target",
        }
    if resume:
        from training.papera.source_compatibility import compatible_resume_provenance
        provenance = compatible_resume_provenance(provenance, resume_run_doc["provenance"])
    signature = canonical_hash(provenance)
    if safe_stitch and resume and Path(resume).name != "last.pt":
        raise ValueError("Resume current baseline from checkpoints/last.pt; best.pt is for inference, not an in-place training rollback")
    checkpoint = (load_checkpoint(resume, signature) if resume else
                  extension_source["checkpoint"] if extension_source is not None else None)
    if resume:
        output = Path(resume).resolve().parent.parent
        if not (output / "run.json").is_file():
            raise ValueError("Resume checkpoint must remain in its run/checkpoints directory")
        if binding:
            frozen_run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            if frozen_run.get("signature") != signature or frozen_run.get("provenance") != provenance:
                raise ValueError("Resume run metadata differs from the complete checkpoint")
            for filename, value in (("stats.json", binding["stats"]), ("split.json", binding["split"]),
                                    ("pack_binding.json", binding), ("config.json", config)):
                if file_sha256(output / filename) != json_file_hash(value):
                    raise ValueError(f"Resume frozen file changed: {filename}")
            best_path = output / "checkpoints/best.pt"
            if best_path.exists():
                best_checkpoint = load_checkpoint(best_path, signature)
                if best_checkpoint["step"] > checkpoint["step"]:
                    raise ValueError("best.pt is newer than last.pt: interrupted checkpoint pair requires explicit recovery, not silent rollback")
                del best_checkpoint
        if is_overfit and (ROOT / ".local/checks").resolve() not in output.parents:
            raise ValueError("Overfit diagnostic resume must remain inside .local/checks")
    else:
        root = (ROOT / ".local/checks/papera-overfit" if is_overfit else
                ROOT / ".local/checks/papera-training-20261016" if smoke else resolve_path(config["output_root"]))
        if output is not None:
            output = resolve_path(output)
        elif extension_source is not None:
            output = root / (f"{variant}_continued_from{extension_source['checkpoint']['step']}"
                             f"_to{config['steps']}_seed{config['seed']}_{datetime.now():%Y%m%d_%H%M%S_%f}")
        else:
            output = root / (
                f"{'overfit_' if is_overfit else 'smoke_' if smoke else ''}{variant}_seed{config['seed']}_{datetime.now():%Y%m%d_%H%M%S_%f}")
        if output.exists():
            raise ValueError("New run output already exists; use --resume or a new run directory")
        if extension_source is not None and output == extension_source["run"]:
            raise ValueError("A continuation must use a new output directory; the completed source run is immutable")
    from training.papera.pack_data import SafeStitchDataset
    train = SafeStitchDataset(binding, split="train", chunk_size=model_config.chunk_size)
    heldout = None
    writer = None
    try:
        heldout = SafeStitchDataset(binding, split="train" if is_overfit else "val",
                                   chunk_size=model_config.chunk_size, stride=config["validation_stride"])
        if variant == 'gesture':
            from dataset.task3.training_pack import read_episode
            usable = sum(int(read_episode(pack,key,checked=binding.checked,include_evidence=False)['arrays']['category_mask'].sum())
                         for key in binding['train_keys']) if hasattr(binding,'checked') else None
            if usable == 0: raise ValueError('Gesture experiment requires valid auxiliary labels in the training split')
        model = ACTPolicy(model_config).to(device)
        backbone_ids = {id(p) for p in model.backbone.parameters()}
        optimizer = torch.optim.AdamW([
            {"params": [p for p in model.parameters() if id(p) not in backbone_ids], "lr": config["learning_rate"]},
            {"params": list(model.backbone.parameters()), "lr": config["backbone_learning_rate"]}],
            weight_decay=config["weight_decay"])
        sampler = StatefulBatchSampler(len(train), config["batch_size"], config["seed"], prefetch_safe=safe_stitch)
        loader_generator = torch.Generator().manual_seed(config["seed"] + 1)
        worker_options = {"num_workers": config["num_workers"]}
        if config["num_workers"]:
            worker_options.update(multiprocessing_context="spawn", prefetch_factor=2, persistent_workers=True)
        train_loader = DataLoader(train, batch_sampler=sampler, pin_memory=True,
                                  generator=loader_generator, **worker_options)
        val_loader = DataLoader(heldout, batch_size=config["batch_size"], shuffle=False,
                                num_workers=0, pin_memory=True, generator=loader_generator)
        step, best = 0, math.inf
        resume_loader_generator = None
        resume_elapsed = 0.0
        if resume:
            model.load_state_dict(checkpoint["model"], strict=True)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda tick: lr_multiplier(tick, config))
            optimizer.load_state_dict(checkpoint["optimizer"]); scheduler.load_state_dict(checkpoint["scheduler"])
            sampler.load_state_dict(checkpoint["sampler"])
            if "loader_generator" in checkpoint:
                resume_loader_generator = checkpoint["loader_generator"]
            step, best = checkpoint["step"], checkpoint["best_prior_action_l1"]
            resume_elapsed = checkpoint_elapsed(checkpoint, output)
            restore_rng(checkpoint["rng"])
            if binding:
                reconcile_resume_logs(output, step)
            # Keep frozen run/checkpoint signatures intact; record actual code
            # used for this restoration separately, including reviewed migrations.
            from training.papera.source_compatibility import verify_training_sources
            with (output / "resume_sources.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, **verify_training_sources(provenance)}) + "\n")
        elif extension_source is not None:
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            for group, expected_lr in zip(optimizer.param_groups, config["lr_schedule"]["start_lrs"]):
                if not math.isclose(float(group["lr"]), float(expected_lr), rel_tol=1e-9, abs_tol=1e-15):
                    raise ValueError("Continuation LR config does not match the source optimizer state")
                group["initial_lr"] = float(expected_lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda tick: lr_multiplier(tick, config))
            sampler.load_state_dict(checkpoint["sampler"])
            if "loader_generator" in checkpoint:
                resume_loader_generator = checkpoint["loader_generator"]
            step, best = checkpoint["step"], checkpoint["best_prior_action_l1"]
            restore_rng(checkpoint["rng"])
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda tick: lr_multiplier(tick, config))
        if resume:
            checkpoint = None
        elif extension_source is not None:
            del extension_source["checkpoint"]
            checkpoint = None
        if not resume:
            output.mkdir(parents=True, exist_ok=False)
            (output / "checkpoints").mkdir()
            runtime = {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
                "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                "train_windows": len(train), "validation_windows": len(heldout),
                "parameters": sum(p.numel() for p in model.parameters()), "engineering_smoke": smoke,
                "engineering_overfit": is_overfit, "overfit_episode_keys": keys,
                "validation_scope": provenance["validation_scope"],
                "precision": config["precision"], "num_workers": config["num_workers"],
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}
            if extension_source is not None:
                runtime.update(continued_from=str(extension_source["run"]),
                               continued_checkpoint=str(Path(continue_from).resolve()),
                               continued_checkpoint_sha256=extension_source["checkpoint_sha256"],
                               starting_step=step, starting_learning_rates=extension_source["start_lrs"])
            try:
                runtime["git_commit"] = subprocess.check_output([r"C:\Program Files\Git\cmd\git.exe", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                runtime["git_commit"] = None
            json_write(output / "run.json", {"signature": signature, "provenance": provenance, "runtime": runtime})
            if binding:
                json_write(output / "pack_binding.json", binding)
                json_write(output / "stats.json", binding["stats"])
                json_write(output / "split.json", binding["split"])
                json_write(output / "config.json", config)
            if extension_source is not None:
                source_metrics = extension_source["run"] / "metrics.jsonl"
                if source_metrics.is_file():
                    shutil.copy2(source_metrics, output / "metrics.jsonl")
                source_tensorboard = extension_source["run"] / "tensorboard"
                if source_tensorboard.is_dir():
                    shutil.copytree(source_tensorboard, output / "tensorboard", dirs_exist_ok=True)
                best_payload = dict(extension_source["best_checkpoint"])
                best_payload.update(signature=signature, provenance=provenance)
                atomic_checkpoint(output / "checkpoints" / "best.pt", best_payload)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output / "tensorboard"), purge_step=step if (resume or extension_source is not None) else None)
        print(f"{variant} | train windows={len(train)}, validation={len(heldout)} | output={output}", flush=True)
        steps_per_epoch = math.ceil(len(train) / config["batch_size"])
        print(f"累计训练预算：{config['steps']} 次参数更新；每轮约 {steps_per_epoch} 步，"
              f"预算约 {config['steps'] / steps_per_epoch:.2f} 轮。"
              "使用固定预算和验证集最优 checkpoint；未启用提前停止。", flush=True)
        if extension_source is not None:
            print(f"从 step {step} 的 last.pt 延续到 {config['steps']}；续训保持 checkpoint 学习率："
                  f"head={optimizer.param_groups[0]['lr']:.3g}, backbone={optimizer.param_groups[1]['lr']:.3g}。", flush=True)
        print("Exploratory data; human quality remains pending. Offline loss is not task success rate.", flush=True)
        if is_overfit:
            print(f"Engineering overfit: {len(keys)} training episodes; validation reuses these same episodes. "
                  "This is reconstruction evidence, not generalization or formal evaluation.", flush=True)
        model.train(); iterator = iter(train_loader); started = time.perf_counter(); starting_step = step
        elapsed_offset = (extension_source["completion"].get("elapsed_s", 0.0)
                          if extension_source is not None else resume_elapsed)
        if resume_loader_generator is not None:
            # Reopening an iterator consumes a worker base-seed. There are no
            # stochastic transforms; retain the saved generator stream for
            # subsequent validation/loader checkpoints, just as uninterrupted.
            loader_generator.set_state(resume_loader_generator.cpu())

        def save(name):
            payload = {"version": CHECKPOINT_VERSION, "signature": signature, "provenance": provenance,
                "step": step, "best_prior_action_l1": best, "model": model.state_dict(),
                "elapsed_s": elapsed_offset + time.perf_counter() - started,
                "executing_trainer_sha256": file_sha256(Path(__file__)),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "sampler": sampler.state_dict(), "rng": rng_state(), "engineering_smoke": smoke,
                "loader_generator": loader_generator.get_state(),
                "engineering_overfit": is_overfit,
                "deployable": False, "closed_loop_verified": False}
            atomic_checkpoint(output / "checkpoints" / name, payload)

        if extension_source is not None:
            save("last.pt")
            del extension_source["best_checkpoint"]

        try:
            while step < config["steps"]:
                b = move_batch(next(iterator), device)
                optimizer.zero_grad(set_to_none=True)
                with precision_context(device, config["precision"]):
                    out = model(b["images"], b["state"], b["action"], b["is_pad"])
                    targets, masks = targets_masks(b)
                    losses = compute_loss(out, targets, masks, config["loss_weights"], variant)
                loss = losses["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss; stopping before optimizer update")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); step += 1
                sampler.mark_consumed()
                if step % config["log_every"] == 0 or step == 1:
                    record = {"step": step, "epoch": sampler.state_dict()["epoch"], "event": "train",
                        **{k: float(v.detach().cpu()) for k, v in losses.items()},
                        "gradient_norm": float(norm), "lr": optimizer.param_groups[0]["lr"],
                        "elapsed_s": elapsed_offset + time.perf_counter() - started}
                    with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                    for key, value in record.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar("train/" + key, value, step)
                    print(f"step {step}/{config['steps']} loss={record['total']:.5f} action={record['action']:.5f} "
                          f"KL={record['kl']:.5f} elapsed={record['elapsed_s']:.1f}s", flush=True)
                if step % config["validation_every"] == 0 or step == config["steps"]:
                    metrics = validate(model, val_loader, train.normalizer, device, config["precision"],
                                       max_batches=2 if smoke else None)
                    metrics.update(step=step, event="validation", smoke_subset=smoke,
                                   validation_scope=provenance["validation_scope"],
                                   engineering_overfit=is_overfit)
                    with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)) and value is not None:
                            writer.add_scalar("validation/" + key, value, step)
                    print(f"validation prior L1={metrics['prior_action_l1']:.5f}, XYZ={metrics['xyz_error_mm']:.3f}mm", flush=True)
                    horizon_text = ", ".join(
                        f"h{h}={metrics[f'h{h}_xyz_error_mm']:.2f}mm "
                        f"(n={metrics[f'h{h}_valid_samples']})"
                        for h in (0, 4, 9, 29) if metrics.get(f"h{h}_xyz_error_mm") is not None)
                    print(f"prior horizon XYZ: {horizon_text}", flush=True)
                    if metrics["prior_action_l1"] < best:
                        best = metrics["prior_action_l1"]; save("best.pt")
                if step % config["save_every"] == 0 or step == config["steps"]:
                    save("last.pt")
                if stop_after is not None and step - starting_step >= stop_after:
                    save("last.pt"); break
        except KeyboardInterrupt:
            # A mid-step interrupt may have consumed a batch; only completed
            # scheduled checkpoints are exact-resume points. Do not overwrite
            # them with partially executed optimizer/sampler state.
            print("Interrupted. Resume the latest completed checkpoint; an unfinished step is not saved.", flush=True)
            return output
        torch.cuda.synchronize()
        json_write(output / "completion.json", {"step": step, "target_steps": config["steps"],
            "completed": step == config["steps"], "elapsed_s": elapsed_offset + time.perf_counter() - started,
            "engineering_smoke": smoke, "best_prior_action_l1": best if math.isfinite(best) else None,
            "engineering_overfit": is_overfit, "overfit_episode_keys": keys,
            "validation_scope": provenance["validation_scope"],
            "continued_from_step": provenance.get("continued_from", {}).get("parent_step"),
            "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "deployable": False, "human_quality_review": "pending"})
        return output
    finally:
        if writer:
            writer.close()
        train.close()
        if heldout is not None:
            heldout.close()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--config", type=Path,
                        help="Default: config/training_baseline.json; resume defaults to its frozen run config")
    parser.add_argument("--variant", choices=EXPERIMENTS, default="baseline",
                        help="唯一实验差别：是否启用训练期手势分类辅助损失")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--continue-from", type=Path,
                        help="从已完成run的last.pt派生新运行；配合--steps给出累计总步数")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int, help="最大参数更新次数；不是完整数据集遍历轮数")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overfit-episodes", type=int, choices=(1, 2),
                        help="工程诊断：只拟合 N 条训练示范并在同一子集重建验证；须指定 --steps 1..3000")
    parser.add_argument("--stop-after", type=int, help="Engineering controlled checkpoint test; retain configured schedule")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.resume is not None and args.continue_from is not None:
        parser.error("--resume 与 --continue-from 不能同时使用")
    extension_source = None
    if args.continue_from is not None:
        if args.config is not None:
            parser.error("--continue-from 自动读取来源run的冻结配置；不接受外部--config")
        if args.steps is None:
            parser.error("--continue-from 必须指定累计目标，例如 --steps 50000")
        extension_source = load_extension_source(args.continue_from, args.steps)
        config = copy.deepcopy(extension_source["run_doc"]["provenance"]["config"])
        config["steps"] = args.steps
        config["lr_schedule"] = {"kind": "constant_checkpoint_lr",
                                 "start_step": extension_source["checkpoint"]["step"],
                                 "start_lrs": extension_source["start_lrs"]}
        config.setdefault("_help", {})["steps"] = (
            f"从已完成的{extension_source['checkpoint']['step']}步 last.pt 延续，"
            f"累计目标{args.steps}步；续训阶段保持checkpoint中的最低学习率，不重启升高。")
        args.variant = extension_source["run_doc"]["provenance"]["variant"]
        if args.variant != "baseline":
            parser.error("当前只允许延续单视角Baseline")
    elif args.config is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    elif args.resume is not None:
        previous = json.loads((args.resume.resolve().parent.parent / "run.json").read_text(encoding="utf-8"))
        config = previous["provenance"]["config"]
        if previous["provenance"].get("engineering_overfit"):
            args.overfit_episodes = len(previous["provenance"]["overfit_episode_keys"])
        args.variant = previous["provenance"]["variant"]
        args.smoke_test = previous["runtime"].get("engineering_smoke", False)
    else:
        config = json.loads((ROOT / "config/training_baseline.json").read_text(encoding="utf-8"))
    from tasks.registry import get
    task = get(args.task)
    if args.resume or args.continue_from:
        if config.get('task_id') != task.id or config.get('task_binding') != task.binding():
            raise ValueError('Resume task or sources differ from frozen training configuration')
        if args.pack and args.pack.resolve() != resolve_path(config['pack_path']):
            raise ValueError('Resume cannot switch Pack')
    else:
        if args.pack is None: parser.error('--pack must be explicit; CURRENT is navigation only')
        config.update(task_id=task.id, task_binding=task.binding(), pack_path=str(args.pack.resolve()),
                      output_root=f'outputs/training/{task.id}/{args.variant}')
        manifest=json.loads((args.pack/'manifest.json').read_text(encoding='utf-8'))
        config['engineering_only']=bool(manifest.get('engineering_only', False))
        config['pack_manifest_sha256']=file_sha256(args.pack/'manifest.json')
    if args.seed is not None:
        config["seed"] = args.seed
    if args.steps is not None:
        if args.resume is not None:
            parser.error("--resume 保持原冻结步数；如需增加总步数，请改用--continue-from")
        config["steps"] = args.steps
    if args.overfit_episodes is not None and args.steps is None and args.resume is None:
        parser.error("--overfit-episodes requires --steps (1..3000)")
    output = run(config, args.variant, args.output, args.resume, args.smoke_test, args.stop_after,
                 overfit_episodes=args.overfit_episodes, continue_from=args.continue_from,
                 extension_source=extension_source)
    print(f"Run files: {output}")


if __name__ == "__main__":
    main()
