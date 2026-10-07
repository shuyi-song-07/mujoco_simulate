#!/usr/bin/env python
"""Task 3 ACT launcher with strict preflight, episode holdout, and provenance.

LeRobot 0.6.1 is used through its installed Python API. Numeric normalization
is recomputed using training episodes only; validation episodes never fit it.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from dataset.task3.validate_task3_dataset import read_jsonl, validate_dataset


def numeric_training_stats(root: Path, episode_ids: list[int]) -> dict:
    """Compute population statistics without reading validation examples."""
    chosen = set(episode_ids)
    buffers = {"observation.state": [], "action": []}
    for path in sorted((root / "data").rglob("*.parquet")):
        values = pq.read_table(path, columns=["episode_index", *buffers]).to_pydict()
        mask = np.asarray([int(value) in chosen for value in values["episode_index"]])
        for key in buffers:
            if mask.any():
                buffers[key].append(np.asarray(values[key], dtype=np.float64)[mask])
    stats = {}
    for key, pieces in buffers.items():
        if not pieces:
            raise ValueError("Train/eval split contains no training frames")
        array = np.concatenate(pieces)
        stats[key] = {"mean": array.mean(0).astype(np.float32), "std": array.std(0).astype(np.float32),
                      "min": array.min(0).astype(np.float32), "max": array.max(0).astype(np.float32),
                      "count": np.asarray([len(array)], dtype=np.int64)}
    return stats


def dataset_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in ("meta/info.json", "meta/task3.json", "research/episodes.jsonl"):
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def install_training_hooks(train_module, root: Path, output: Path, provenance: dict, validation: dict):
    """Extend only this entry point; never edit LeRobot or Task 1/2 code."""
    make_datasets = train_module.make_train_eval_datasets
    save_checkpoint = train_module.save_checkpoint
    update_last = train_module.update_last_checkpoint

    def write_provenance(directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "task3_metadata.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def make_task3_datasets(cfg):
        train, heldout = make_datasets(cfg)
        train_ids = list(train.episodes) if train.episodes is not None else list(range(train.num_episodes))
        eval_ids = list(heldout.episodes) if heldout is not None else []
        if set(train_ids).intersection(eval_ids):
            raise RuntimeError("Training and validation episodes overlap")
        statistics = numeric_training_stats(root, train_ids)
        # The factory loads global metadata stats even after episode splitting.
        # Replace numeric stats before ACT processors are constructed.
        train.meta.stats.update(copy.deepcopy(statistics))
        if heldout is not None:
            heldout.meta.stats.update(copy.deepcopy(statistics))
        provenance["train_episode_indices"] = train_ids
        provenance["validation_episode_indices"] = eval_ids
        provenance["normalization"] = "state/action fitted on training episodes only; RGB uses fixed ImageNet statistics"
        output.mkdir(parents=True, exist_ok=True)
        write_provenance(output)
        (output / "dataset_validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return train, heldout

    def save_task3_checkpoint(*args, **kwargs):
        save_checkpoint(*args, **kwargs)
        directory = kwargs.get("checkpoint_dir", args[0] if args else None)
        write_provenance(Path(directory) / "pretrained_model")

    def update_task3_last(directory):
        try:
            return update_last(directory)
        except OSError as error:
            if os.name != "nt" or getattr(error, "winerror", None) != 1314:
                raise
            # Windows without symlink privileges still has a valid saved checkpoint.
            pointer = Path(directory).parent / "last-checkpoint.txt"
            pointer.write_text(Path(directory).name + "\n", encoding="utf-8")
            print(f"Saved checkpoint: {directory} (Windows pointer: {pointer})")
            return pointer

    train_module.make_train_eval_datasets = make_task3_datasets
    train_module.save_checkpoint = save_task3_checkpoint
    train_module.update_last_checkpoint = update_task3_last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/task3_dual_arm")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", type=Path, help="Saved pretrained_model directory or its train_config.json")
    parser.add_argument("--pretrained-path", type=Path, help="Fine-tune an existing Task 3 ACT checkpoint with a fresh optimizer")
    parser.add_argument("--print-command", action="store_true", help="Run preflight and print the LeRobot arguments without training")
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    report = validate_dataset(root)
    if not report["valid"]:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit("Task 3 training preflight failed; do not train on empty, failed, or malformed data")
    if args.resume and args.pretrained_path:
        parser.error("Choose either --resume or --pretrained-path")
    if args.smoke_test and args.resume:
        parser.error("--smoke-test cannot be combined with --resume")
    fingerprint = dataset_fingerprint(root)
    source = args.resume or args.pretrained_path
    resume_config = {}
    if source:
        source = source.expanduser().resolve()
        source_dir = source.parent if source.is_file() else source
        metadata_path = source_dir / "task3_metadata.json"
        if not metadata_path.exists():
            raise SystemExit("Source checkpoint is missing task3_metadata.json; a single-arm checkpoint is not compatible")
        source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if source_metadata["task3"]["config_hash"] != json.loads((root / "meta/task3.json").read_text(encoding="utf-8"))["config_hash"]:
            raise SystemExit("Source checkpoint and dataset use different Task 3 robot/camera configurations")
        if args.resume:
            if source_metadata.get("dataset_fingerprint") != fingerprint:
                raise SystemExit("Resume dataset changed since checkpoint; use a fresh fine-tuning run instead")
            source = source_dir / "train_config.json"
            resume_config = json.loads(source.read_text(encoding="utf-8"))
    steps = args.steps if args.steps is not None else (1 if args.smoke_test else resume_config.get("steps", 100000))
    batch = args.batch_size if args.batch_size is not None else (1 if args.smoke_test else resume_config.get("batch_size", 8))
    eval_split = 0.0 if args.smoke_test else (resume_config.get("dataset", {}).get("eval_split", args.eval_split) if args.resume else args.eval_split)
    if not 0 <= eval_split < 1 or steps < 1 or batch < 1 or args.chunk_size < 1:
        parser.error("Invalid steps, batch size, chunk size, or episode validation fraction")
    if eval_split > 0 and report["episode_count"] < 2:
        raise SystemExit("Episode holdout needs at least 2 successful episodes. Use --smoke-test for a one-episode pipeline check")
    output = (args.output_dir or (Path(resume_config["output_dir"]) if args.resume else
              PROJECT_ROOT / "outputs" / f"task3_act_seed{args.seed}")).expanduser().resolve()
    if output.exists() and not args.resume:
        raise SystemExit("Output directory exists; use a new directory or --resume")
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
    if args.smoke_test:
        os.environ["HF_HUB_OFFLINE"] = "1"
    cli = [f"--dataset.repo_id={args.repo_id}", f"--dataset.root={root}", "--dataset.video_backend=pyav",
           "--dataset.use_imagenet_stats=true", f"--dataset.eval_split={eval_split}",
           f"--output_dir={output}", "--job_name=task3_dual_arm_act", f"--policy.device={args.device}",
           "--policy.push_to_hub=false", f"--steps={steps}", f"--batch_size={batch}",
           f"--num_workers={args.num_workers}", "--persistent_workers=false", "--wandb.enable=false",
           "--env_eval_freq=0", f"--eval_steps={0 if eval_split == 0 else args.eval_freq}",
           "--save_checkpoint=true", f"--save_freq={1 if args.smoke_test else args.save_freq}",
           f"--log_freq={1 if args.smoke_test else 100}"]
    if args.resume:
        cli += [f"--config_path={source}", "--resume=true"]
    else:
        chunk = 8 if args.smoke_test else args.chunk_size
        cli += ["--policy.type=act", "--policy.vision_backbone=resnet18", f"--seed={args.seed}",
                f"--policy.optimizer_lr={args.learning_rate}", f"--policy.optimizer_lr_backbone={args.learning_rate}",
                "--policy.optimizer_weight_decay=0.0001", f"--policy.chunk_size={chunk}", f"--policy.n_action_steps={chunk}",
                f"--policy.pretrained_backbone_weights={'ResNet18_Weights.IMAGENET1K_V1' if args.pretrained_backbone and not args.smoke_test else 'null'}"]
        if args.pretrained_path:
            cli += [f"--policy.path={source_dir}"]
    provenance = {"schema_version": "task3-training-v1", "dataset_root": str(root), "dataset_fingerprint": fingerprint,
                  "demonstration_seeds": sorted({int(entry["seed"]) for entry in read_jsonl(root / "research/episodes.jsonl") if entry.get("seed") is not None}),
                  "task3": json.loads((root / "meta/task3.json").read_text(encoding="utf-8")),
                  "train_arguments": cli, "source_checkpoint": str(source) if source else None}
    print(json.dumps({"preflight": "passed", "dataset_episodes": report["episode_count"], "lerobot_arguments": cli}, indent=2))
    if args.print_command:
        return
    from lerobot.scripts import lerobot_train
    install_training_hooks(lerobot_train, root, output, provenance, report)
    sys.argv = [str(Path(__file__)), *cli]
    lerobot_train.main()


if __name__ == "__main__":
    main()
