#!/usr/bin/env python3
"""Evaluate a trained Diffusion Policy in fresh Panda MuJoCo tasks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
import sys

if "--viewer" in sys.argv:
    os.environ["MUJOCO_GL"] = "glfw"
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from evaluation.act_variation_rollout import (  # noqa: E402
    DEFAULT_MAX_JOINT_STEP_RAD,
    DEFAULT_MAX_XYZ_STEP_M,
    DEFAULT_MAX_YAW_STEP_RAD,
    DEFAULT_SCENE,
    FPS,
    RolloutPanda,
    ViewerClosed,
    checkpoint_camera_names,
    run_episode,
)
from simulation.mujoco.record_mujoco_panda import parse_args as recorder_args  # noqa: E402
from simulation.mujoco.render_workers import _training_cameras  # noqa: E402

DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs/diffusion_variation_baseline/checkpoints/last/pretrained_model"


def load_policy(checkpoint: Path, device: str, action_steps: int | None = None):
    required = ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")
    if not all((checkpoint / name).is_file() for name in required):
        raise ValueError(f"Incomplete LeRobot checkpoint: {checkpoint}")

    config = DiffusionConfig.from_pretrained(str(checkpoint), local_files_only=True)
    config.device = device
    config.pretrained_backbone_weights = None
    camera_names = checkpoint_camera_names(config)
    expected = {
        "observation.state": (9,),
        **{f"observation.images.{name}": (3, 480, 640) for name in camera_names},
    }
    if set(config.input_features) != set(expected) or any(
        tuple(config.input_features[key].shape) != shape for key, shape in expected.items()
    ):
        raise ValueError("Checkpoint inputs must be the Panda 9D state and overview RGB camera")
    if tuple(config.output_features["action"].shape) != (7,):
        raise ValueError("Checkpoint action must be 7D")
    if action_steps is not None:
        if not 1 <= action_steps <= config.horizon - config.n_obs_steps + 1:
            raise ValueError("--action-steps must be between 1 and horizon - n_obs_steps + 1")
        config.n_action_steps = action_steps

    policy = DiffusionPolicy.from_pretrained(
        str(checkpoint), config=config, local_files_only=True, strict=True
    ).eval().to(device)
    pre, post = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {"device": device},
        },
        postprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "unnormalizer_processor": {"device": device},
        },
    )
    return policy, pre, post, config, camera_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--success-dwell-seconds", type=float, default=0.5)
    parser.add_argument("--seed-base", type=int, default=10_000)
    parser.add_argument("--action-steps", type=int, help="Replan after this many actions; defaults to checkpoint config")
    parser.add_argument("--max-xyz-step-m", type=float, default=DEFAULT_MAX_XYZ_STEP_M)
    parser.add_argument("--max-joint-step-rad", type=float, default=DEFAULT_MAX_JOINT_STEP_RAD)
    parser.add_argument("--max-yaw-step-rad", type=float, default=DEFAULT_MAX_YAW_STEP_RAD)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--viewer", action="store_true", help="Show the live MuJoCo viewer")
    parser.add_argument("--realtime", action="store_true", help="Pace the viewer at 30 FPS wall time")
    args = parser.parse_args()

    if args.episodes < 1 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("Episodes and maximum task time must be positive")
    if not math.isfinite(args.success_dwell_seconds) or args.success_dwell_seconds < 0:
        parser.error("Success dwell time must be finite and nonnegative")
    if args.realtime and not args.viewer:
        parser.error("--realtime requires --viewer")
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.max_xyz_step_m, args.max_joint_step_rad, args.max_yaw_step_rad)):
        parser.error("Motion limits must be finite and positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    checkpoint = args.checkpoint.expanduser().resolve()
    scene = args.scene.expanduser().resolve()
    if not scene.is_file():
        parser.error(f"Scene not found: {scene}")
    output = (args.output_dir or PROJECT_ROOT / "outputs" /
              f"diffusion_variation_baseline_rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}").expanduser().resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")

    policy, pre, post, config, camera_names = load_policy(
        checkpoint, args.device, args.action_steps
    )
    rec_args = recorder_args(["--model-path", str(scene), "--random-seed", str(args.seed_base)])
    env = RolloutPanda(rec_args)
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    training_cameras = _training_cameras(rec_args)
    cameras = {name: training_cameras[name] for name in camera_names}
    viewer = None
    if args.viewer:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(env.model, env.data)
        viewer.cam.lookat[:] = (
            rec_args.observation_lookat_x,
            rec_args.observation_lookat_y,
            rec_args.observation_lookat_z,
        )
        viewer.cam.distance = rec_args.observation_distance
        viewer.cam.azimuth = rec_args.observation_azimuth
        viewer.cam.elevation = rec_args.observation_elevation
        viewer.sync()

    output.mkdir(parents=True)
    config_record = {
        "checkpoint": str(checkpoint),
        "scene": str(scene),
        "episodes": args.episodes,
        "fps": FPS,
        "max_task_seconds": args.max_seconds,
        "success_dwell_seconds": args.success_dwell_seconds,
        "seed_base": args.seed_base,
        "device": args.device,
        "viewer": args.viewer,
        "realtime": args.realtime,
        "horizon": config.horizon,
        "n_obs_steps": config.n_obs_steps,
        "n_action_steps": config.n_action_steps,
        "observation_cameras": list(camera_names),
        "resize_shape": config.resize_shape,
        "max_xyz_step_m": args.max_xyz_step_m,
        "max_joint_step_rad": args.max_joint_step_rad,
        "max_yaw_step_rad": args.max_yaw_step_rad,
        "success_rule": "cube within 5 cm XY of plate centre, touching plate, at plate height, stationary, gripper open for dwell interval",
    }
    (output / "config.json").write_text(json.dumps(config_record, indent=2) + "\n", encoding="utf-8")
    rows = []
    try:
        with (output / "rollouts.csv").open("w", newline="", encoding="utf-8") as csv_file, \
             (output / "rollouts.jsonl").open("w", encoding="utf-8") as jsonl_file:
            from evaluation.act_variation_rollout import CSV_FIELDS

            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            try:
                for number in range(1, args.episodes + 1):
                    seed = args.seed_base + number - 1
                    row = run_episode(
                        env, renderer, cameras, policy, pre, post,
                        number=number,
                        seed=seed,
                        max_seconds=args.max_seconds,
                        success_dwell_seconds=args.success_dwell_seconds,
                        video_path=None,
                        viewer=viewer,
                        realtime=args.realtime,
                        max_xyz_step_m=args.max_xyz_step_m,
                        max_joint_step_rad=args.max_joint_step_rad,
                        max_yaw_step_rad=args.max_yaw_step_rad,
                    )
                    writer.writerow(row)
                    csv_file.flush()
                    jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    jsonl_file.flush()
                    rows.append(row)
                    print(f"[{number}/{args.episodes}] {'SUCCESS' if row['success'] else row['failure_reason']} "
                          f"seed={seed} sim={row['sim_time_s']:.1f}s xy={row['cube_target_xy_m']:.3f}m", flush=True)
            except ViewerClosed:
                print("MuJoCo viewer closed; saved completed episodes.", flush=True)
    finally:
        if viewer is not None:
            viewer.close()
        renderer.close()
        by_stack_mode = {}
        for mode in ("table", "red_above", "red_below"):
            selected = [row for row in rows if row["stack_mode"] == mode]
            by_stack_mode[mode] = {
                "episodes": len(selected),
                "successes": sum(row["success"] for row in selected),
                "success_rate": sum(row["success"] for row in selected) / len(selected) if selected else None,
            }
        summary = {
            **config_record,
            "completed_episodes": len(rows),
            "successes": sum(row["success"] for row in rows),
            "success_rate": sum(row["success"] for row in rows) / len(rows) if rows else None,
            "mean_wall_time_s": float(np.mean([row["wall_time_s"] for row in rows])) if rows else None,
            "timeouts": sum(row["failure_reason"] == "timeout" for row in rows),
            "by_stack_mode": by_stack_mode,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Results: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
