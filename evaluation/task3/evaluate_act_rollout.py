#!/usr/bin/env python
"""Run a saved Task 3 ACT policy in real MuJoCo physics on fixed test seeds.

Loads the checkpoint's normalization processors as well as the policy. This
is a closed-loop evaluator, not a replay of demonstration actions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

import av
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CAMERAS = ("top", "front", "side")


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "pretrained_model").is_dir():
        path /= "pretrained_model"
    if not all((path / name).is_file() for name in ("config.json", "model.safetensors", "task3_metadata.json")):
        raise ValueError("Checkpoint must contain ACT weights, config, processors, and task3_metadata.json")
    return path


def safe_policy_action(action, previous, status: dict, max_step_m=0.008) -> tuple[np.ndarray, dict]:
    """Bound motion, hold v1 orientation, and make gripper commands atomic."""
    value = np.asarray(action, dtype=np.float64).reshape(-1)
    previous = np.asarray(previous, dtype=np.float64).reshape(-1)
    if value.shape != (14,) or not np.isfinite(value).all():
        raise ValueError("Policy produced nonfinite or non-14D action")
    filtered = value.copy()
    deltas = [value[offset:offset + 3] - previous[offset:offset + 3] for offset in (0, 7)]
    coupled = status.get("task_state") in {"DUAL_GRASPED", "PLACEMENT"}
    if coupled:
        common = 0.5 * (deltas[0] + deltas[1])
        deltas = [common.copy(), common.copy()]
    clamped = 0
    for offset, delta in zip((0, 7), deltas):
        norm = np.linalg.norm(delta)
        if norm > max_step_m:
            delta = delta * (max_step_m / norm)
            clamped += 1
        filtered[offset:offset + 3] = previous[offset:offset + 3] + delta
        filtered[offset + 3:offset + 6] = previous[offset + 3:offset + 6]
    gripper = 255.0 if float(np.clip(value[[6, 13]], 0, 255).mean()) >= 127.5 else 0.0
    filtered[[6, 13]] = gripper
    return filtered.astype(np.float32), {"motion_clamp_count": clamped, "cooperative_filter": coupled,
                                        "orientation_prediction_error_rad": float(np.linalg.norm(value[[3, 4, 5, 10, 11, 12]] - previous[[3, 4, 5, 10, 11, 12]]))}


class RolloutVideo:
    def __init__(self, path: Path, fps: int):
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width, self.stream.height = 1920, 480
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "24", "preset": "veryfast"}
        self.fps, self.frames = fps, 0

    def append(self, images: dict):
        panel = np.concatenate([images[key] for key in CAMERAS], axis=1)
        frame = av.VideoFrame.from_ndarray(panel, format="rgb24")
        frame.pts = self.frames
        frame.time_base = Fraction(1, self.fps)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def make_observation(controller, images: dict):
    import torch
    state = np.asarray(controller.state_vector(), dtype=np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("Controller produced malformed 16D state")
    batch = {"observation.state": torch.from_numpy(state)}
    for camera in CAMERAS:
        image = np.asarray(images[camera])
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError(f"{camera} must render 640x480 uint8 RGB")
        batch[f"observation.images.{camera}"] = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
    return batch


def load_policy(checkpoint: Path, device: str):
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    config = ACTConfig.from_pretrained(str(checkpoint), local_files_only=True)
    config.device = device
    # We load learned backbone weights from this checkpoint, so no online ImageNet download.
    config.pretrained_backbone_weights = None
    expected_inputs = {"observation.state": (16,), **{f"observation.images.{key}": (3, 480, 640) for key in CAMERAS}}
    for key, shape in expected_inputs.items():
        if key not in config.input_features or tuple(config.input_features[key].shape) != shape:
            raise ValueError(f"ACT checkpoint input {key} does not match Task 3 {shape}")
    if "action" not in config.output_features or tuple(config.output_features["action"].shape) != (14,):
        raise ValueError("ACT checkpoint action dimension must be 14")
    policy = ACTPolicy.from_pretrained(str(checkpoint), config=config, local_files_only=True, strict=True)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}, "normalizer_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}, "unnormalizer_processor": {"device": device}})
    return policy, preprocessor, postprocessor


def run_rollout(controller, renderer, policy, preprocessor, postprocessor, *, seed: int,
                fps: int, max_seconds: float, output: Path, save_video=True, max_step_m=0.008) -> dict:
    import torch
    from simulation.mujoco.dual_arm.dual_arm_render_workers import render_observations
    torch.manual_seed(seed)
    np.random.seed(seed)
    controller.reset(seed=seed)
    simulation_started = float(controller.data.time)
    policy.reset()
    if hasattr(preprocessor, "reset"):
        preprocessor.reset()
    if hasattr(postprocessor, "reset"):
        postprocessor.reset()
    path = output / f"seed_{seed}.pending.mp4"
    video = RolloutVideo(path, fps) if save_video else None
    started = time.perf_counter()
    frame_count = rejected = consecutive_rejections = clamps = 0
    relative_errors, positions, inference_ms = [], [], []
    failure_reason = ""
    status = controller.get_status()
    try:
        with torch.inference_mode():
            for _ in range(int(np.ceil(max_seconds * fps))):
                if status.get("success") or status.get("task_state") in {"DONE", "FAIL"}:
                    break
                images = render_observations(controller, width=640, height=480, renderer=renderer)
                # Shared renderer can return short camera names or policy feature names.
                images = {key: images.get(key, images.get(f"observation.images.{key}")) for key in CAMERAS}
                if video:
                    video.append(images)
                observation = make_observation(controller, images)
                inference_start = time.perf_counter()
                action = postprocessor(policy.select_action(preprocessor(observation))).detach().cpu().numpy()
                inference_ms.append((time.perf_counter() - inference_start) * 1000)
                try:
                    action, diagnostics = safe_policy_action(action, controller.action_vector(), status, max_step_m=max_step_m)
                except ValueError:
                    failure_reason = "invalid_policy_action"
                    break
                clamps += diagnostics["motion_clamp_count"]
                result = controller.apply_policy_action(action)
                accepted = bool(result.get("accepted", False)) if isinstance(result, dict) else bool(result)
                rejected += int(not accepted)
                consecutive_rejections = 0 if accepted else consecutive_rejections + 1
                controller.step(dt=1 / fps)
                frame_count += 1
                status = controller.get_status()
                position = status.get("object_position")
                if position is not None:
                    positions.append(position)
                if status.get("task_state") in {"DUAL_GRASPED", "PLACEMENT"}:
                    relative_errors.append(float(status.get("relative_pose_error_m", status.get("metrics", {}).get("relative_pose_error_m", 0))))
                if status.get("task_state") == "FAIL":
                    break
                if consecutive_rejections >= fps:
                    failure_reason = "action_rejected"
                    break
            if video:
                images = render_observations(controller, width=640, height=480, renderer=renderer)
                video.append({key: images.get(key, images.get(f"observation.images.{key}")) for key in CAMERAS})
    finally:
        if video:
            video.close()
    status = controller.get_status()
    success = bool(status.get("success") or status.get("task_state") == "DONE")
    if not success:
        failure_reason = status.get("failure_reason") or failure_reason or "timeout"
    final_position = np.asarray(status.get("object_position", [np.nan] * 3), dtype=float)
    target = np.asarray(status.get("target_position", [np.nan] * 3), dtype=float)
    final_error = float(np.linalg.norm(final_position - target))
    xy_error = float(np.linalg.norm(final_position[:2] - target[:2]))
    metrics = status.get("metrics", {})
    object_rotation = controller.data.xmat[controller.object_id].reshape(3, 3)
    orientation_error = float(np.arccos(np.clip((np.trace(object_rotation) - 1) / 2, -1, 1)))
    positions = np.asarray(positions, dtype=float)
    return {"seed": seed, "success": success, "failure_reason": failure_reason,
            "completion_time_s": float(controller.data.time) - simulation_started, "evaluation_wall_time_s": time.perf_counter() - started,
            "policy_frames": frame_count, "final_object_pose_error_m": final_error if np.isfinite(final_error) else None,
            "final_object_xy_error_m": xy_error if np.isfinite(xy_error) else None,
            "final_object_orientation_error_rad": orientation_error,
            "collision_count": int(status.get("collision_count", metrics.get("collision_count", 0))),
            "contact_loss_count": int(metrics.get("contact_loss_count", 0)),
            "mean_relative_ee_error_m": float(np.mean(relative_errors)) if relative_errors else None,
            "max_relative_ee_error_m": max(relative_errors, default=None),
            "object_path_length_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()) if len(positions) > 1 else 0.0,
            "rejected_actions": rejected, "motion_clamps": clamps,
            "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else None,
            "video_path": str(path) if video else ""}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10000, 10010)))
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--success-videos", type=int, default=1)
    parser.add_argument("--video-all", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--allow-training-seeds", action="store_true", help="Debug only; such results are not held-out evaluation")
    args = parser.parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    provenance = json.loads((checkpoint / "task3_metadata.json").read_text(encoding="utf-8"))
    config = provenance["task3"]["config"]
    training_seeds = set(provenance.get("demonstration_seeds", []))
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Test seeds must be unique")
    if training_seeds.intersection(args.seeds) and not args.allow_training_seeds:
        parser.error("Test seeds overlap demonstration seeds; choose disjoint held-out seeds")
    duration = args.max_seconds or float(config.get("evaluation", {}).get("max_episode_seconds", 30.0))
    if duration <= 0 or args.success_videos < 0:
        parser.error("Duration must be positive and success video count non-negative")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise SystemExit("Output directory already exists; choose a fresh evaluation directory")
    output.mkdir(parents=True)
    from simulation.mujoco.dual_arm.dual_arm_task_controller import DualArmTaskController
    from simulation.mujoco.dual_arm.dual_arm_render_workers import MultiCameraRenderer
    controller = DualArmTaskController(config=config)
    if controller.config_hash != provenance["task3"]["config_hash"]:
        raise SystemExit("Checkpoint environment configuration hash differs from the effective MuJoCo configuration")
    renderer = MultiCameraRenderer(controller.model, width=640, height=480)
    rows, saved_successes = [], 0
    try:
        policy, preprocessor, postprocessor = load_policy(checkpoint, args.device)
        max_step = float(config.get("control", {}).get("max_ee_step_m", 0.004))
        for seed in args.seeds:
            row = run_rollout(controller, renderer, policy, preprocessor, postprocessor, seed=seed,
                              fps=30, max_seconds=duration, output=output, save_video=not args.no_video, max_step_m=max_step)
            if row["video_path"]:
                path = Path(row["video_path"])
                keep = args.video_all or not row["success"] or saved_successes < args.success_videos
                if keep:
                    final_path = output / f"seed_{seed}_{'success' if row['success'] else 'failure'}.mp4"
                    path.replace(final_path)
                    row["video_path"] = str(final_path)
                    saved_successes += int(row["success"])
                else:
                    path.unlink()
                    row["video_path"] = ""
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    finally:
        renderer.close()
    count = len(rows)
    summary = {"checkpoint": str(checkpoint), "config_hash": provenance["task3"]["config_hash"],
               "seeds": args.seeds, "episodes": count, "success_rate": sum(row["success"] for row in rows) / count,
               "mean_success_completion_time_s": float(np.mean([row["completion_time_s"] for row in rows if row["success"]])) if any(row["success"] for row in rows) else None,
               "mean_final_object_pose_error_m": float(np.mean([row["final_object_pose_error_m"] for row in rows if row["final_object_pose_error_m"] is not None])),
               "mean_final_object_orientation_error_rad": float(np.mean([row["final_object_orientation_error_rad"] for row in rows])),
               "collision_rate": sum(row["collision_count"] > 0 for row in rows) / count,
               "contact_loss_rate": sum(row["contact_loss_count"] > 0 for row in rows) / count,
               "mean_relative_ee_error_m": float(np.mean([row["mean_relative_ee_error_m"] for row in rows if row["mean_relative_ee_error_m"] is not None])) if any(row["mean_relative_ee_error_m"] is not None for row in rows) else None,
               "failure_reason_histogram": dict(Counter(row["failure_reason"] for row in rows if not row["success"])),
               "training_seed_overlap": sorted(training_seeds.intersection(args.seeds)),
               "safety_filter": {"max_step_m": max_step, "orientation": "fixed_v1", "gripper": "atomic", "carry": "common_translation"}}
    with (output / "rollouts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
