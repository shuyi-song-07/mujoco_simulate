"""Official-data profile of the local ACT Trainer, for an isolated A/B check.

Only the HDF5 batch adapter and the authors' sampling/validation protocol are
specific to this profile. Actual optimization and iteration use the same
engine as the formal Panda Pack Trainer. This is not a second temporary loop.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import shutil
from types import SimpleNamespace
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.papera.act_model import ACTConfig, ACTPolicy, compute_loss
from training.papera.engine import move_batch, run_loop, targets_masks

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "742c753c0d4a5d87076c8f69e5628c79a8cc5488"
DEFAULT_UPSTREAM = ROOT / ".local/checks/act_reference_20261106/upstream"
CACHE = ROOT / ".local/checks/official_reproduction_20261105/act/torchhub"
SCHEMA = "official-aloha-joint-s14-a14-top-k100"


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def adapt_batch(batch, chunk_size=100):
    """Change the official four-tensor tuple into the local batch dictionary.

    The original Reader already owns alignment, normalization and RGB. Do not
    shift actions, add state fields, or alter its random observation selection.
    Auxiliary placeholders are masked; Baseline does not learn from them.
    """
    images, state, action, is_pad = batch
    count = len(state)
    if (images.ndim != 5 or images.shape[1:3] != (1, 3)
            or state.shape != (count, 14) or action.ndim != 3 or action.shape[-1] != 14
            or is_pad.shape != action.shape[:2] or action.shape[1] < chunk_size):
        raise ValueError("Official reference expects one RGB, S14, A14 and a full K100 mask")
    return {"images": images, "state": state, "action": action[:, :chunk_size],
        "is_pad": is_pad[:, :chunk_size],
        "gesture": torch.zeros(count, 2, dtype=torch.long),
        "gesture_mask": torch.zeros(count, 2, dtype=torch.bool),
        "skeleton": torch.zeros(count, 2, 21, 3),
        "skeleton_mask": torch.zeros(count, 2, dtype=torch.bool),
        "stage": torch.zeros(count, dtype=torch.long)}


class BatchAdapter:
    """Keep the original Dataset/DataLoader, including its RNG consumption."""
    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield adapt_batch(batch)


@contextmanager
def _upstream_modules(upstream):
    paths = [str(upstream), str(upstream / "detr")]
    original_path = sys.path[:]
    sys.path[:0] = paths
    try:
        modules = {}
        for name in ("utils", "detr.models"):
            module = importlib.import_module(name)
            if not Path(module.__file__).resolve().is_relative_to(upstream):
                raise ValueError(f"Unexpected module already loaded for {name}: {module.__file__}")
            modules[name] = module
        yield modules
    finally:
        sys.path[:] = original_path


def _initial_model(config, original_models, utils, device):
    """Control initialization independently of the local builder's order."""
    utils.set_seed(0)
    args = SimpleNamespace(backbone="resnet18", dilation=False, lr_backbone=1e-5,
        masks=False, position_embedding="sine", hidden_dim=512, dropout=.1,
        nheads=8, pre_norm=False, enc_layers=4, dec_layers=7, dim_feedforward=3200,
        num_queries=100, camera_names=["top"])
    original = original_models.build_ACT_model(args).to(device)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
    local = ACTPolicy(config).to(device)
    local.model.load_state_dict(original.state_dict(), strict=True)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    del original
    return local


def posterior_validation(model, loader, device, mean_dict):
    """Use A's held-out posterior objective, not the Task's z=0 selector."""
    was_training = model.training
    model.eval()
    rows = []
    try:
        with torch.inference_mode():
            for cpu_batch in loader:
                batch = move_batch(cpu_batch, device)
                output = model(batch["images"], batch["state"], batch["action"], batch["is_pad"],
                               posterior_in_eval=True)
                targets, masks = targets_masks(batch)
                losses = compute_loss(output, targets, masks, {"action": 1., "kl": 10.}, "baseline")
                rows.append({"l1": losses["action"], "kl": losses["kl"], "loss": losses["total"]})
            if not rows:
                raise ValueError("Official validation has no batches")
            # Preserve the authors' equal-batch averaging, including batch8+2.
            summary = mean_dict(rows)
            result = {key: float(value.detach().cpu()) for key, value in summary.items()}
            if not all(math.isfinite(value) for value in result.values()):
                raise FloatingPointError("Reference posterior validation is nonfinite")
            return result
    finally:
        model.train(was_training)


def _atomic_save(path, value):
    staging = path.with_suffix(".writing")
    torch.save(value, staging)
    staging.replace(path)


def _save_curves(output, train_rows, val_rows):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    for key in ("l1", "kl", "loss"):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([row["step"] / 5 for row in train_rows], [row[key] for row in train_rows], label="train", lw=.7)
        ax.plot([row["epoch"] for row in val_rows], [row[key] for row in val_rows], label="validation", lw=.9)
        ax.set(xlabel="Epoch", ylabel=key, title="Local ACT Trainer / official data")
        ax.legend(); fig.tight_layout(); fig.savefig(output / f"train_val_{key}_seed_0.png"); plt.close(fig)


def run_reference(data, upstream, output, *, epochs=2000):
    """Complete B using local ACT, Loss, AdamW and the shared production loop."""
    data = Path(data).resolve()
    upstream = Path(upstream).resolve() if upstream is not None else DEFAULT_UPSTREAM.resolve()
    output = Path(output).resolve()
    checks = (ROOT / ".local/checks").resolve()
    if not output.is_relative_to(checks) or output == checks or output.is_relative_to(upstream) or output.is_relative_to(data):
        raise ValueError("Reference output must be a new isolated directory below .local/checks")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite B: {output}")
    if type(epochs) is not int or not 1 <= epochs <= 2000:
        raise ValueError("Reference epochs must be from 1 to 2000; the full comparison uses 2000")
    if not (upstream / "utils.py").is_file() or not (upstream / "detr/models/detr_vae.py").is_file():
        raise FileNotFoundError("The existing fixed official ACT checkout is required")
    if not all((data / f"episode_{index}.hdf5").is_file() for index in range(50)):
        raise FileNotFoundError("Reference requires all 50 original HDF5 episodes")
    cached_weight = CACHE / "checkpoints/resnet18-f37072fd.pth"
    if not cached_weight.is_file():
        raise FileNotFoundError("The existing ResNet18 cache is required; reference never downloads")
    if not torch.cuda.is_available():
        raise RuntimeError("Reference B requires CUDA, as did A")
    from torch.utils.tensorboard import SummaryWriter

    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()
    writer = SummaryWriter(str(output / "tensorboard"))
    started = time.monotonic()
    model_config = ACTConfig(state_dim=14, action_dim=14, num_cameras=1, chunk_size=100,
        hidden_dim=512, heads=8, feedforward_dim=3200, encoder_layers=4, decoder_layers=7,
        posterior_layers=4, latent_dim=32, dropout=.1, backbone_weights=str(cached_weight))
    config = {"profile": SCHEMA, "steps": epochs * 5, "epochs": epochs,
        "batch_size": 8, "precision": "fp32", "gradient_clip_norm": None,
        "learning_rate": 1e-5, "backbone_learning_rate": 1e-5, "weight_decay": 1e-4,
        "loss_weights": {"action": 1., "kl": 10.}, "seed": 0, "cpu_threads": 4,
        "num_workers": 0, "model": asdict(model_config)}
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.hub.set_dir(str(CACHE))
    device = torch.device("cuda")
    train_rows, val_rows = [], []
    best, best_epoch, step = math.inf, None, 0
    complete = False
    status = {"complete": False, "scope": "Full local Trainer official-data comparison, not a Panda task run",
        "data": str(data), "upstream": str(upstream), "output": str(output), "official_commit": COMMIT,
        "config": config, "python": sys.executable, "gpu": torch.cuda.get_device_name(0),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "h5py")},
        "precision_flags": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32},
        "initialization": "original factory seed0; exact core state and post-construction CPU/CUDA RNG copied",
        "sampling": "original episode-uniform/random anchor, original shuffle=True loaders",
        "normalization": "original all50 statistics, deliberately matches A only",
        "validation": "original posterior L1+KL, equal-batch mean; epochs0..1999 before training",
        "engine": str(Path(__file__).with_name("engine.py")),
        "production_task_defaults_changed": False, "training_started": False}
    _json(output / "run.json", status)
    _json(output / "config.json", config)

    def no_download(*_args, **_kwargs):
        raise RuntimeError("Reference B network weight download is prohibited")

    try:
        with _upstream_modules(upstream) as modules, patch.object(torch.hub, "download_url_to_file", no_download):
            utils = modules["utils"]
            utils.set_seed(1)
            def windows_loader(*args, **kwargs):
                kwargs["num_workers"] = 0
                kwargs.pop("prefetch_factor", None)
                return DataLoader(*args, **kwargs)
            with patch.object(utils, "DataLoader", windows_loader):
                raw_train, raw_val, stats, is_sim = utils.load_data(str(data), 50, ["top"], 8, 8)
            if not is_sim or len(raw_train) != 5 or len(raw_val) != 2:
                raise ValueError("Reference must keep the original 40/10 split and batch8")
            split = {"train": [int(x) for x in raw_train.dataset.episode_ids],
                     "val": [int(x) for x in raw_val.dataset.episode_ids]}
            _json(output / "split.json", split)
            with (output / "dataset_stats.pkl").open("wb") as stream:
                pickle.dump(stats, stream)
            train_loader, val_loader = BatchAdapter(raw_train), BatchAdapter(raw_val)
            model = _initial_model(model_config, modules["detr.models"], utils, device)
            backbone_ids = {id(p) for p in model.backbone.parameters()}
            optimizer = torch.optim.AdamW([
                {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids], "lr": 1e-5},
                {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 1e-5}], weight_decay=1e-4)
            for source in (Path(__file__), Path(__file__).with_name("engine.py"),
                    Path(__file__).with_name("act_model.py"), Path(__file__).with_name("train.py")):
                target = output / "source_snapshot" / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
            status.update(training_started=True, split=split, model_parameters=sum(p.numel() for p in model.parameters()))
            _json(output / "run.json", status)

            def save(name):
                from training.papera.train import rng_state
                rng = rng_state()
                payload = {"version": "papera-act-official-data-reference-v1", "step": step,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": None,
                    "best_posterior_loss": best, "best_epoch": best_epoch, "config": config,
                    "rng": rng, "split": split, "data": str(data), "official_commit": COMMIT,
                    "schema": SCHEMA, "wall_seconds": time.monotonic()-started}
                _atomic_save(output / "checkpoints" / name, payload)

            def record(row):
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                for key in ("l1", "kl", "loss"):
                    writer.add_scalar(f"{row['event']}/{key}", row[key], row["step"])

            def validate_epoch(epoch):
                nonlocal best, best_epoch
                print(f"\nEpoch {epoch}", flush=True)
                metric = posterior_validation(model, val_loader, device, utils.compute_dict_mean)
                row = {"event": "validation", "epoch": epoch, "step": step, **metric}
                val_rows.append(row); record(row)
                print(f"Val loss: {metric['loss']:.6f} | L1={metric['l1']:.6f} KL={metric['kl']:.6f}", flush=True)
                if metric["loss"] < best:
                    best, best_epoch = metric["loss"], epoch
                    save("best.pt")
                    _atomic_save(output / "policy_best.ckpt", {"model."+k: v for k,v in model.model.state_dict().items()})

            def after_step(current, losses, norm):
                nonlocal step
                step = current
                metric = {"l1": float(losses["action"].detach().cpu()),
                          "kl": float(losses["kl"].detach().cpu()), "loss": float(losses["total"].detach().cpu())}
                row = {"event": "train", "epoch": (step-1)//5, "step": step,
                       **metric, "gradient_norm": float(norm.detach().cpu())}
                train_rows.append(row); record(row)
                if step % 5 == 0:
                    recent = train_rows[-5:]
                    print(f"Train mean loss: {sum(x['loss'] for x in recent)/5:.6f} | step {step}/{config['steps']}", flush=True)
                    # Save after the same completed epochs as A; I/O cannot
                    # alter the next epoch's NumPy/Torch random streams.
                    if (step//5-1) % 100 == 0 or step == config["steps"]:
                        save("last.pt")
                    if step < config["steps"]:
                        validate_epoch(step//5)

            model.train()
            step = run_loop(model, train_loader, optimizer, None, device=device, config=config,
                variant="baseline", before_iterator=lambda: validate_epoch(0), after_step=after_step)
            if step != config["steps"] or len(val_rows) != epochs or len(train_rows) != epochs*5:
                raise RuntimeError("Local engine did not complete the frozen official budget")
            save("last.pt")
            _save_curves(output, train_rows, val_rows)
            complete = True
    finally:
        writer.close()
        status.update(complete=complete, completed_steps=step, best_epoch=best_epoch,
                      best_posterior_loss=best if math.isfinite(best) else None, wall_seconds=time.monotonic()-started)
        _json(output / "completion.json", status)
    print(f"B complete: {step} updates, best posterior={best:.6f} at epoch {best_epoch}", flush=True)
    return output
