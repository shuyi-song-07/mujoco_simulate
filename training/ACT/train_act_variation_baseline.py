#!/usr/bin/env python3
"""Train the unmodified LeRobot 0.6.1 ACT on the variation dataset.

The overview RGB camera and robot state are the only policy inputs. Gesture
fields stay in the dataset for the companion auxiliary-supervision experiment,
but are never fed to the policy at inference time. Both entry points use this
exact ACT setup.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GESTURE_KEY = "observation.gesture"
HAND_DELTA_KEY = "observation.hand_delta_uv"
HAND_VALID_KEY = "observation.hand_delta_valid"
IMAGE_KEY = "observation.images.overview"


def check_dataset(root: Path, *, auxiliary: bool = False, motion: bool = False) -> dict:
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise ValueError(f"Missing LeRobot metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    required = {"observation.state", "action"}
    if auxiliary:
        required.add(GESTURE_KEY)
    if motion:
        required.update({HAND_DELTA_KEY, HAND_VALID_KEY})
    if not required.issubset(features):
        raise ValueError(f"Missing dataset features: {sorted(required - features.keys())}")
    if IMAGE_KEY not in features:
        raise ValueError(f"ACT needs the {IMAGE_KEY} camera")
    if info.get("total_frames", 0) < 1 or info.get("total_episodes", 0) < 2:
        raise ValueError("Dataset needs frames and at least two episodes for holdout")
    if not list((root / "data").rglob("*.parquet")):
        raise ValueError(f"Missing Parquet data in {root / 'data'}")
    return info


def policy_input_features(features: dict) -> dict:
    return {key: feature for key, feature in features.items()
            if key in {"observation.state", IMAGE_KEY}}


def install_input_filter(train_module) -> None:
    """Do not normalize or use labels as ACT observations in either experiment."""
    original_make_policy = train_module.make_policy

    def make_policy_without_gesture(*, cfg, ds_meta=None, **kwargs):
        from lerobot.utils.feature_utils import dataset_to_policy_features

        if ds_meta is not None:
            features = dataset_to_policy_features(ds_meta.features)
            cfg.input_features = policy_input_features(features)
        return original_make_policy(cfg=cfg, ds_meta=ds_meta, **kwargs)

    train_module.make_policy = make_policy_without_gesture


def install_training_only_stats(train_module) -> None:
    """Fit state/action normalization without reading held-out episodes."""
    original_datasets = train_module.make_train_eval_datasets

    def make_datasets_with_train_stats(cfg):
        import numpy as np
        import pyarrow.parquet as pq

        train, heldout = original_datasets(cfg)
        root = Path(cfg.dataset.root)
        selected = set(train.episodes if train.episodes is not None else range(train.num_episodes))
        columns = ("observation.state", "action")
        buffers = {key: [] for key in columns}
        for path in sorted((root / "data").rglob("*.parquet")):
            rows = pq.read_table(path, columns=["episode_index", *columns]).to_pydict()
            mask = np.fromiter((int(episode) in selected for episode in rows["episode_index"]),
                               dtype=bool, count=len(rows["episode_index"]))
            if not mask.any():
                continue
            for key in columns:
                buffers[key].append(np.asarray(rows[key], dtype=np.float64)[mask])
        fitted = {}
        for key, pieces in buffers.items():
            if not pieces:
                raise ValueError(f"No training values found for {key}")
            values = np.concatenate(pieces)
            fitted[key] = {
                "mean": values.mean(axis=0).astype(np.float32),
                "std": values.std(axis=0).astype(np.float32),
                "min": values.min(axis=0).astype(np.float32),
                "max": values.max(axis=0).astype(np.float32),
                "count": np.asarray([len(values)], dtype=np.int64),
            }
        train.meta.stats.update(copy.deepcopy(fitted))
        if heldout is not None:
            heldout.meta.stats.update(copy.deepcopy(fitted))
        return train, heldout

    train_module.make_train_eval_datasets = make_datasets_with_train_stats


def make_parser(auxiliary: bool = False, motion: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="LeRobot dataset directory containing meta/ and data/",
    )
    parser.add_argument(
        "--repo-id",
        help="Dataset repository identifier; defaults to local/<dataset-directory-name>",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--eval-steps", type=int, default=5000)
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="One small offline training step")
    parser.add_argument("--print-command", action="store_true", help="Validate and print LeRobot arguments")
    if auxiliary:
        parser.add_argument("--gesture-weight", type=float, default=0.1)
    if motion:
        parser.add_argument(
            "--hand-delta-weight", type=float, default=0.0,
            help="Auxiliary wrist-displacement loss weight; 0 disables it",
        )
        parser.add_argument(
            "--hand-delta-scale", type=float, default=0.03,
            help="Webcam-normalized wrist displacement mapped to one regression unit",
        )
    return parser


def build_cli(args, *, auxiliary: bool) -> list[str]:
    root = args.dataset_root.expanduser().resolve()
    repo_id = args.repo_id or f"local/{root.name}"
    name = ("act_variation_gesture_hand_delta" if auxiliary and
            getattr(args, "hand_delta_weight", 0.0) > 0 else
            "act_variation_gesture" if auxiliary else "act_variation_baseline")
    output = (args.output_dir or PROJECT_ROOT / "outputs" / name).expanduser().resolve()
    steps = 1 if args.smoke_test else args.steps
    batch_size = 1 if args.smoke_test else args.batch_size
    chunk_size = 8 if args.smoke_test else args.chunk_size
    eval_steps = 0 if args.smoke_test else args.eval_steps
    if min(steps, batch_size, chunk_size) < 1 or args.num_workers < 0 or not 0 <= args.eval_split < 1:
        raise ValueError("Invalid steps, batch size, chunk size, workers, or eval split")
    if output.exists() and not args.print_command:
        raise ValueError(f"Output already exists: {output}")
    return [
        f"--dataset.repo_id={repo_id}", f"--dataset.root={root}",
        "--dataset.video_backend=pyav", "--dataset.use_imagenet_stats=true",
        f"--dataset.eval_split={args.eval_split}", "--policy.type=act",
        f"--policy.device={args.device}", "--policy.vision_backbone=resnet18",
        f"--policy.pretrained_backbone_weights={'null' if args.no_pretrained_backbone or args.smoke_test else 'ResNet18_Weights.IMAGENET1K_V1'}",
        f"--policy.chunk_size={chunk_size}", f"--policy.n_action_steps={chunk_size}",
        f"--policy.dim_model={128 if args.smoke_test else 512}",
        f"--policy.n_heads={4 if args.smoke_test else 8}",
        f"--policy.dim_feedforward={512 if args.smoke_test else 3200}",
        f"--policy.n_encoder_layers={1 if args.smoke_test else 4}",
        "--policy.n_decoder_layers=1", "--policy.use_vae=true",
        f"--policy.latent_dim={8 if args.smoke_test else 32}",
        f"--policy.n_vae_encoder_layers={1 if args.smoke_test else 4}",
        "--policy.kl_weight=10", f"--policy.optimizer_lr={args.learning_rate}",
        f"--policy.optimizer_lr_backbone={args.learning_rate}",
        "--policy.optimizer_weight_decay=0.0001", "--policy.push_to_hub=false",
        f"--seed={args.seed}", f"--batch_size={batch_size}",
        f"--num_workers={0 if args.smoke_test else args.num_workers}",
        "--persistent_workers=false", f"--steps={steps}",
        f"--eval_steps={eval_steps}", "--save_checkpoint=true",
        f"--save_freq={1 if args.smoke_test else args.save_freq}",
        f"--log_freq={1 if args.smoke_test else 100}", "--env_eval_freq=0",
        f"--output_dir={output}",
        f"--job_name={name}",
        "--wandb.enable=false",
    ]


def run(args, *, auxiliary: bool = False, install_auxiliary=None) -> None:
    root = args.dataset_root.expanduser().resolve()
    hand_delta_weight = getattr(args, "hand_delta_weight", 0.0)
    hand_delta_scale = getattr(args, "hand_delta_scale", 0.03)
    info = check_dataset(
        root, auxiliary=auxiliary, motion=auxiliary and hand_delta_weight > 0
    )
    if auxiliary and (not 0 < args.gesture_weight or not args.gesture_weight < float("inf")):
        raise ValueError("--gesture-weight must be finite and positive")
    if auxiliary and (not 0 <= hand_delta_weight < float("inf")
                      or not 0 < hand_delta_scale < float("inf")):
        raise ValueError("Hand-delta weight must be nonnegative and scale must be positive")
    cli = build_cli(args, auxiliary=auxiliary)
    print(json.dumps({"dataset": str(root), "episodes": info["total_episodes"],
                      "frames": info["total_frames"], "lerobot_arguments": cli,
                      **({"gesture_weight": args.gesture_weight} if auxiliary else {}),
                      **({"hand_delta_weight": hand_delta_weight,
                          "hand_delta_scale": hand_delta_scale} if hand_delta_weight > 0 else {})},
                     indent=2, ensure_ascii=False))
    if args.print_command:
        return
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache/huggingface"))
    from lerobot.scripts import lerobot_train

    install_input_filter(lerobot_train)
    install_training_only_stats(lerobot_train)
    if install_auxiliary is not None:
        install_auxiliary(lerobot_train, args.gesture_weight,
                          hand_delta_weight, hand_delta_scale)
    sys.argv = [str(Path(__file__)), *cli]
    lerobot_train.main()


if __name__ == "__main__":
    run(make_parser().parse_args())
