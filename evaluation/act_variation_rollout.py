#!/usr/bin/env python3
"""Evaluate the trained single-arm ACT baseline in 50 fresh MuJoCo tasks.

The time limit is simulated task time. Results are saved after each episode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime

if "--viewer" in sys.argv:
    os.environ["MUJOCO_GL"] = "glfw"
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from simulation.mujoco.record_mujoco_panda import MujocoPandaRecorder, parse_args as recorder_args
from simulation.mujoco.render_workers import _training_cameras

DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs/act_variation_baseline/checkpoints/last/pretrained_model"
DEFAULT_SCENE = PROJECT_ROOT / "simulation/mujoco/assets/franka_emika_panda/scene.xml"
FPS = 30
IMAGE_SHAPE = (480, 640, 3)
AVAILABLE_CAMERAS = ("overview", "camera_2", "camera_3")
# Preserve the checkpoint's demonstrated 100-frame action horizon. Shorter
# horizons can be tried explicitly, but changed the successful test trajectory.
DEFAULT_ACTION_STEPS = 100
DEFAULT_MAX_XYZ_STEP_M = 0.01
DEFAULT_MAX_JOINT_STEP_RAD = 0.03
DEFAULT_MAX_YAW_STEP_RAD = 0.02
CSV_FIELDS = (
    "episode", "seed", "success", "failure_reason", "sim_time_s", "wall_time_s",
    "steps", "cube_target_xy_m", "cube_z_m", "plate_top_z_m", "stack_mode", "video_path",
    "screenshot_path",
    "cube_on_plate", "cube_plate_contact", "cube_linear_speed_m_s",
    "cube_angular_speed_rad_s",
    "max_joint_command_step_rad", "max_joint_physical_step_rad",
    "max_yaw_command_step_rad", "max_hand_step_m",
)


class RolloutPanda(MujocoPandaRecorder):
    """Use the recorder's exact scene and layout sampler without recorder workers."""

    def _start_render_workers(self) -> None:
        pass

    def reset_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.reset_simulation()


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def apply_cartesian_action(env: RolloutPanda, action: np.ndarray, *, max_xyz_step_m: float = DEFAULT_MAX_XYZ_STEP_M,
                           max_joint_step_rad: float = DEFAULT_MAX_JOINT_STEP_RAD,
                           max_yaw_step_rad: float = DEFAULT_MAX_YAW_STEP_RAD) -> None:
    """Track the predicted pose with the recorder's orientation-first, rate-limited control."""
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("ACT returned a non-finite or non-7D action")
    model, data, ik = env.model, env.data, env.action_data
    previous_arm_target = data.ctrl[:7].copy()
    previous_yaw_target = float(data.ctrl[env.gripper_yaw_actuator_id])
    ik.qpos[:] = data.qpos
    ik.qpos[:7] = previous_arm_target
    ik.qpos[env.gripper_yaw_qpos_address] = previous_yaw_target
    ik.qvel[:] = 0
    mujoco.mj_forward(model, ik)
    base_rotation = ik.xmat[env.base_id].reshape(3, 3)
    target_position = ik.xpos[env.base_id] + base_rotation @ action[:3]
    position_step = target_position - ik.xpos[env.hand_id]
    position_step *= min(1.0, max_xyz_step_m / max(np.linalg.norm(position_step), 1e-12))

    # Demonstrations use the dedicated yaw joint for Pointing_Up, with the arm
    # holding its initial orientation. Recover that joint target from the hand
    # pose; the rotation matrix makes +pi and -pi roll labels equivalent.
    target_rotation_in_base = rpy_matrix(action[3:6])
    relative_rotation = env.initial_hand_rotation_in_base.T @ target_rotation_in_base
    target_yaw = math.atan2(relative_rotation[0, 1], relative_rotation[0, 0])
    yaw_delta = (target_yaw - previous_yaw_target + math.pi) % (2 * math.pi) - math.pi
    yaw_delta = float(np.clip(yaw_delta, -max_yaw_step_rad, max_yaw_step_rad))
    yaw_joint_limits = model.jnt_range[env.gripper_yaw_joint_id]
    yaw_target = float(np.clip(previous_yaw_target + yaw_delta, *yaw_joint_limits))
    yaw_target = float(np.clip(yaw_target,
                               data.qpos[env.gripper_yaw_qpos_address] - env.args.max_joint_target_lead,
                               data.qpos[env.gripper_yaw_qpos_address] + env.args.max_joint_target_lead))
    yaw_target = float(np.clip(yaw_target, previous_yaw_target - max_yaw_step_rad,
                               previous_yaw_target + max_yaw_step_rad))
    ik.qpos[env.gripper_yaw_qpos_address] = yaw_target
    mujoco.mj_forward(model, ik)

    c, s = math.cos(yaw_target), math.sin(yaw_target)
    local_yaw_rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    desired_rotation = base_rotation @ env.initial_hand_rotation_in_base @ local_yaw_rotation
    current_rotation = ik.xmat[env.hand_id].reshape(3, 3)
    rotation_error = 0.5 * sum(
        np.cross(current_rotation[:, axis], desired_rotation[:, axis]) for axis in range(3)
    )
    mujoco.mj_jacBody(model, ik, env.jacp, env.jacr, env.hand_id)
    rotation_jacobian = env.jacr[:, :7]
    position_jacobian = env.jacp[:, :7]
    rotation_inverse = rotation_jacobian.T @ np.linalg.solve(
        rotation_jacobian @ rotation_jacobian.T + 0.006**2 * np.eye(3), np.eye(3)
    )
    orientation_step = rotation_inverse @ rotation_error
    rotation_nullspace = np.eye(7) - rotation_inverse @ rotation_jacobian
    projected_position_jacobian = position_jacobian @ rotation_nullspace
    remaining_position = position_step - position_jacobian @ orientation_step
    arm_step = orientation_step + rotation_nullspace @ projected_position_jacobian.T @ np.linalg.solve(
        projected_position_jacobian @ projected_position_jacobian.T + 0.03**2 * np.eye(3),
        remaining_position,
    )
    arm_step = np.clip(arm_step, -max_joint_step_rad, max_joint_step_rad)
    arm_target = previous_arm_target + arm_step
    lead = arm_target - data.qpos[:7]
    lead *= min(1.0, env.args.max_joint_target_lead / max(np.max(np.abs(lead)), 1e-12))
    arm_target = data.qpos[:7] + lead
    arm_target = np.clip(arm_target, previous_arm_target - max_joint_step_rad,
                         previous_arm_target + max_joint_step_rad)
    control_limits = model.actuator_ctrlrange[:7]
    data.ctrl[:7] = np.clip(arm_target, control_limits[:, 0], control_limits[:, 1])
    data.ctrl[env.gripper_yaw_actuator_id] = yaw_target
    data.ctrl[7] = 255.0 if action[6] >= 127.5 else 0.0


def render_observation(env: RolloutPanda, renderer: mujoco.Renderer, cameras: dict):
    observation = {
        "observation.state": torch.from_numpy(np.r_[env.data.qpos[:7],
            env.data.qpos[env.gripper_yaw_qpos_address],
            sum(env.data.qpos[address] for address in env.finger_qpos_addresses)].astype(np.float32))
    }
    images = {}
    for name, camera in cameras.items():
        renderer.update_scene(env.data, camera=camera)
        image = renderer.render().copy()
        if image.shape != IMAGE_SHAPE or image.dtype != np.uint8:
            raise ValueError(f"Unexpected {name} image: {image.shape}, {image.dtype}")
        images[name] = image
        observation[f"observation.images.{name}"] = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return observation, images


def save_final_screenshot(env: RolloutPanda, renderer: mujoco.Renderer, camera, path: Path) -> None:
    """Save the episode's final state from the fixed overview camera."""
    renderer.update_scene(env.data, camera=camera)
    image = renderer.render().copy()
    if image.shape != IMAGE_SHAPE or image.dtype != np.uint8:
        raise ValueError(f"Unexpected final screenshot: {image.shape}, {image.dtype}")
    Image.fromarray(image).save(path, format="PNG")


def placement_status(env: RolloutPanda) -> dict:
    """Measure whether the red rigid body is stably supported by the plate."""
    cube = env.data.xpos[env.cube_id]
    half_height = float(env.model.geom_size[env.cube_geom_id][2])
    plate_geom = int(env.model.body_geomadr[env.target_plate_id])
    plate_top = float(env.model.body_pos[env.target_plate_id][2] + env.model.geom_size[plate_geom][1])
    cube_dof = int(env.model.jnt_dofadr[env.cube_joint_id])
    cube_velocity = env.data.qvel[cube_dof:cube_dof + 6]
    touching_plate = any(
        {int(contact.geom1), int(contact.geom2)} == {env.cube_geom_id, plate_geom}
        for contact in env.data.contact[:env.data.ncon]
    )
    height_error = abs(float(cube[2]) - (plate_top + half_height))
    linear_speed = float(np.linalg.norm(cube_velocity[:3]))
    angular_speed = float(np.linalg.norm(cube_velocity[3:]))
    on_plate = bool(
        touching_plate
        and height_error <= 0.025
        and linear_speed <= 0.05
        and angular_speed <= 0.5
    )
    return {
        "on_plate": on_plate,
        "touching_plate": touching_plate,
        "height_error_m": height_error,
        "linear_speed_m_s": linear_speed,
        "angular_speed_rad_s": angular_speed,
    }


def placement_is_stable(env: RolloutPanda) -> bool:
    return placement_status(env)["on_plate"]


def data_gripper_open(env: RolloutPanda) -> bool:
    return bool(env.data.ctrl[7] >= 200 and
                sum(env.data.qpos[address] for address in env.finger_qpos_addresses) >= 0.06)


def checkpoint_camera_names(config) -> tuple[str, ...]:
    """Return supported RGB cameras in the exact order stored by the policy."""
    image_prefix = "observation.images."
    camera_names = tuple(
        key.removeprefix(image_prefix)
        for key in config.input_features
        if key.startswith(image_prefix)
    )
    if not camera_names:
        raise ValueError("Checkpoint has no RGB camera input")
    unsupported = [name for name in camera_names if name not in AVAILABLE_CAMERAS]
    if unsupported:
        raise ValueError(f"Checkpoint uses unsupported cameras: {unsupported}")
    return camera_names


def load_policy(checkpoint: Path, device: str, action_steps: int = DEFAULT_ACTION_STEPS):
    required = ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")
    if not all((checkpoint / name).is_file() for name in required):
        raise ValueError(f"Incomplete LeRobot checkpoint: {checkpoint}")
    config = ACTConfig.from_pretrained(str(checkpoint), local_files_only=True)
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
        raise ValueError(
            "Checkpoint inputs must be the variation dataset's 9D state and supported RGB cameras"
        )
    if tuple(config.output_features["action"].shape) != (7,):
        raise ValueError("Checkpoint action must be 7D")
    if not 1 <= action_steps <= config.chunk_size:
        raise ValueError(f"Action steps must be between 1 and checkpoint chunk size {config.chunk_size}")
    config.n_action_steps = action_steps
    policy = ACTPolicy.from_pretrained(str(checkpoint), config=config, local_files_only=True, strict=True)
    policy = policy.eval().to(device)
    pre, post = make_pre_post_processors(
        config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}, "normalizer_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}, "unnormalizer_processor": {"device": device}},
    )
    return policy, pre, post, camera_names


class ViewerClosed(Exception):
    """The user closed the MuJoCo rollout window."""


def run_episode(env, renderer, cameras, policy, pre, post, *, number, seed, max_seconds,
                success_dwell_seconds, video_path: Path | None, viewer=None, realtime=False,
                max_xyz_step_m=DEFAULT_MAX_XYZ_STEP_M, max_joint_step_rad=DEFAULT_MAX_JOINT_STEP_RAD,
                max_yaw_step_rad=DEFAULT_MAX_YAW_STEP_RAD):
    env.reset_seed(seed)
    if viewer is not None:
        if not viewer.is_running():
            raise ViewerClosed
        viewer.sync()
    policy.reset()
    pre.reset()
    post.reset()
    initial_time = float(env.data.time)
    wall_start = time.perf_counter()
    stable_since = None
    steps = 0
    success = False
    failure_reason = "timeout"
    max_joint_command_step = 0.0
    max_joint_physical_step = 0.0
    max_yaw_command_step = 0.0
    max_hand_step = 0.0
    video = None
    if video_path is not None:
        import av
        video = av.open(str(video_path), "w")
        stream = video.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 1920, 480, "yuv420p"
        stream.options = {"crf": "24", "preset": "veryfast"}
    try:
        with torch.inference_mode():
            for step in range(math.ceil(max_seconds * FPS)):
                wall_frame_started = time.perf_counter()
                if viewer is not None and not viewer.is_running():
                    raise ViewerClosed
                if env.data.time - initial_time >= max_seconds - 1e-9:
                    break
                observation, images = render_observation(env, renderer, cameras)
                if video is not None:
                    import av
                    frame = av.VideoFrame.from_ndarray(images["overview"], format="rgb24")
                    for packet in stream.encode(frame):
                        video.mux(packet)
                try:
                    action = post(policy.select_action(pre(observation)))
                    previous_joint_command = env.data.ctrl[:7].copy()
                    previous_yaw_command = float(env.data.ctrl[env.gripper_yaw_actuator_id])
                    previous_joint_position = env.data.qpos[:7].copy()
                    previous_hand_position = env.data.xpos[env.hand_id].copy()
                    apply_cartesian_action(env, action.detach().cpu().numpy(),
                                           max_xyz_step_m=max_xyz_step_m,
                                           max_joint_step_rad=max_joint_step_rad,
                                           max_yaw_step_rad=max_yaw_step_rad)
                except ValueError:
                    failure_reason = "invalid_action"
                    break
                max_joint_command_step = max(max_joint_command_step, float(np.max(
                    np.abs(env.data.ctrl[:7] - previous_joint_command))))
                max_yaw_command_step = max(max_yaw_command_step, abs(float(
                    env.data.ctrl[env.gripper_yaw_actuator_id] - previous_yaw_command)))
                deadline = min(initial_time + max_seconds, initial_time + (step + 1) / FPS)
                while env.data.time + 1e-9 < deadline:
                    mujoco.mj_step(env.model, env.data)
                max_joint_physical_step = max(max_joint_physical_step, float(np.max(
                    np.abs(env.data.qpos[:7] - previous_joint_position))))
                max_hand_step = max(max_hand_step, float(np.linalg.norm(
                    env.data.xpos[env.hand_id] - previous_hand_position)))
                if viewer is not None:
                    viewer.sync()
                    if realtime:
                        time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - wall_frame_started)))
                steps += 1
                if placement_is_stable(env):
                    stable_since = stable_since or float(env.data.time)
                    if env.data.time - stable_since >= success_dwell_seconds:
                        success, failure_reason = True, ""
                        break
                else:
                    stable_since = None
    finally:
        if video is not None:
            for packet in stream.encode():
                video.mux(packet)
            video.close()
    cube = env.data.xpos[env.cube_id]
    plate = env.data.xpos[env.target_plate_id]
    plate_geom = int(env.model.body_geomadr[env.target_plate_id])
    plate_top = float(env.model.body_pos[env.target_plate_id][2] + env.model.geom_size[plate_geom][1])
    final_placement = placement_status(env)
    red = next(box for box in env.current_layout if box["name"] == "cube")
    if red.get("stacked_on"):
        stack_mode = "red_above"
    elif any(box.get("stacked_on") == "cube" for box in env.current_layout):
        stack_mode = "red_below"
    else:
        stack_mode = "table"
    return {
        "episode": number, "seed": seed, "success": success, "failure_reason": failure_reason,
        "sim_time_s": round(float(env.data.time - initial_time), 3),
        "wall_time_s": round(time.perf_counter() - wall_start, 3), "steps": steps,
        "cube_target_xy_m": round(float(np.linalg.norm(cube[:2] - plate[:2])), 4),
        "cube_z_m": round(float(cube[2]), 4), "plate_top_z_m": round(plate_top, 4),
        "stack_mode": stack_mode, "video_path": str(video_path) if video_path else "",
        "cube_on_plate": final_placement["on_plate"],
        "cube_plate_contact": final_placement["touching_plate"],
        "cube_linear_speed_m_s": round(final_placement["linear_speed_m_s"], 6),
        "cube_angular_speed_rad_s": round(final_placement["angular_speed_rad_s"], 6),
        "max_joint_command_step_rad": round(max_joint_command_step, 6),
        "max_joint_physical_step_rad": round(max_joint_physical_step, 6),
        "max_yaw_command_step_rad": round(max_yaw_command_step, 6),
        "max_hand_step_m": round(max_hand_step, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-seconds", type=float, default=60.0, help="Maximum simulated task time per episode")
    parser.add_argument("--success-dwell-seconds", type=float, default=0.5)
    parser.add_argument("--seed-base", type=int, default=10_000)
    parser.add_argument("--action-steps", type=int, default=DEFAULT_ACTION_STEPS,
                        help="Replan from current observations after this many actions")
    parser.add_argument("--max-xyz-step-m", type=float, default=DEFAULT_MAX_XYZ_STEP_M)
    parser.add_argument("--max-joint-step-rad", type=float, default=DEFAULT_MAX_JOINT_STEP_RAD)
    parser.add_argument("--max-yaw-step-rad", type=float, default=DEFAULT_MAX_YAW_STEP_RAD)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--screenshot-dir", type=Path,
        help="Final-state PNG directory; defaults to a folder on ~/桌面",
    )
    parser.add_argument("--video-every", type=int, default=0, help="Save an overview-camera MP4 every N episodes; 0 disables video")
    parser.add_argument("--viewer", action="store_true", help="Show the live MuJoCo window during rollout")
    parser.add_argument("--realtime", action="store_true", help="Pace the viewer at 30 FPS wall time")
    args = parser.parse_args()
    if args.episodes < 1 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("Episodes and maximum task time must be positive")
    if not math.isfinite(args.success_dwell_seconds) or args.success_dwell_seconds < 0 or args.video_every < 0:
        parser.error("Invalid dwell time or video interval")
    if args.realtime and not args.viewer:
        parser.error("--realtime requires --viewer")
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.max_xyz_step_m, args.max_joint_step_rad, args.max_yaw_step_rad)):
        parser.error("Motion limits must be finite and positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    scene = args.scene.expanduser().resolve()
    if not scene.is_file():
        parser.error(f"Scene not found: {scene}")
    output = (args.output_dir or PROJECT_ROOT / "outputs" /
              f"act_variation_baseline_rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}").expanduser().resolve()
    if output.exists():
        parser.error(f"Output exists: {output}")
    screenshot_dir = (args.screenshot_dir or
                      Path.home() / "桌面" / f"{output.name}_final_states").expanduser().resolve()
    if screenshot_dir.exists():
        parser.error(f"Screenshot directory exists: {screenshot_dir}")
    policy, pre, post, camera_names = load_policy(checkpoint, args.device, args.action_steps)
    rec_args = recorder_args(["--model-path", str(scene), "--random-seed", str(args.seed_base)])
    env = RolloutPanda(rec_args)
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    training_cameras = _training_cameras(rec_args)
    cameras = {name: training_cameras[name] for name in camera_names}
    viewer = None
    if args.viewer:
        from mujoco import viewer as mujoco_viewer
        viewer = mujoco_viewer.launch_passive(env.model, env.data)
        viewer.cam.lookat[:] = rec_args.observation_lookat_x, rec_args.observation_lookat_y, rec_args.observation_lookat_z
        viewer.cam.distance = rec_args.observation_distance
        viewer.cam.azimuth = rec_args.observation_azimuth
        viewer.cam.elevation = rec_args.observation_elevation
        viewer.sync()
    output.mkdir(parents=True)
    screenshot_dir.mkdir(parents=True)
    config = {"checkpoint": str(checkpoint), "scene": str(scene), "episodes": args.episodes,
              "fps": FPS, "max_task_seconds": args.max_seconds, "success_dwell_seconds": args.success_dwell_seconds,
              "seed_base": args.seed_base, "device": args.device, "video_every": args.video_every,
              "viewer": args.viewer, "realtime": args.realtime,
              "screenshot_dir": str(screenshot_dir),
              "observation_cameras": list(camera_names),
              "action_steps": args.action_steps, "max_xyz_step_m": args.max_xyz_step_m,
              "max_joint_step_rad": args.max_joint_step_rad, "max_yaw_step_rad": args.max_yaw_step_rad,
              "success_rule": "red rigid body contacts and is stably supported by the target plate for the dwell interval; centre distance and gripper state are not criteria"}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    rows = []
    try:
        with (output / "rollouts.csv").open("w", newline="", encoding="utf-8") as csv_file, \
             (output / "rollouts.jsonl").open("w", encoding="utf-8") as jsonl_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            try:
                for number in range(1, args.episodes + 1):
                    seed = args.seed_base + number - 1
                    video_path = output / f"episode_{number:03d}.mp4" if args.video_every and number % args.video_every == 0 else None
                    row = run_episode(env, renderer, cameras, policy, pre, post, number=number, seed=seed,
                                      max_seconds=args.max_seconds, success_dwell_seconds=args.success_dwell_seconds,
                                      video_path=video_path, viewer=viewer, realtime=args.realtime,
                                      max_xyz_step_m=args.max_xyz_step_m,
                                      max_joint_step_rad=args.max_joint_step_rad,
                                      max_yaw_step_rad=args.max_yaw_step_rad)
                    outcome = "success" if row["success"] else row["failure_reason"]
                    screenshot_path = screenshot_dir / (
                        f"episode_{number:03d}_seed_{seed}_{outcome}.png"
                    )
                    save_final_screenshot(
                        env, renderer, training_cameras["overview"], screenshot_path
                    )
                    row["screenshot_path"] = str(screenshot_path)
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
            by_stack_mode[mode] = {"episodes": len(selected), "successes": sum(row["success"] for row in selected),
                                   "success_rate": sum(row["success"] for row in selected) / len(selected) if selected else None}
        summary = {**config, "completed_episodes": len(rows), "successes": sum(row["success"] for row in rows),
                   "success_rate": sum(row["success"] for row in rows) / len(rows) if rows else None,
                   "mean_wall_time_s": float(np.mean([row["wall_time_s"] for row in rows])) if rows else None,
                   "timeouts": sum(row["failure_reason"] == "timeout" for row in rows),
                   "by_stack_mode": by_stack_mode}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Results: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
