#!/usr/bin/env python
"""Plot and validate one MuJoCo Panda demonstration action trajectory."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mujoco-action-matplotlib")
)

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pyarrow.parquet as pq


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "assets" / "franka_emika_panda" / "scene.xml"
)


def rotation_matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    """Return roll, pitch, yaw in radians using the standard ZYX convention."""
    horizontal = np.hypot(matrix[0, 0], matrix[1, 0])
    if horizontal > 1e-8:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        pitch = np.arctan2(-matrix[2, 0], horizontal)
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = np.arctan2(-matrix[1, 2], matrix[1, 1])
        pitch = np.arctan2(-matrix[2, 0], horizontal)
        yaw = 0.0
    return np.array([roll, pitch, yaw])


def load_episode(dataset_root: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No data parquet files found under {dataset_root}")
    tables = [
        pq.read_table(path, columns=["episode_index", "observation.state", "action"])
        for path in parquet_files
    ]
    episode_parts = []
    for table in tables:
        indices = np.asarray(table["episode_index"].to_pylist(), dtype=int)
        mask = indices == episode_index
        if np.any(mask):
            states = np.asarray(table["observation.state"].to_pylist(), dtype=float)[mask]
            actions = np.asarray(table["action"].to_pylist(), dtype=float)[mask]
            episode_parts.append((states, actions))
    if not episode_parts:
        raise ValueError(f"Episode {episode_index} is not present in {dataset_root}")
    return (
        np.concatenate([part[0] for part in episode_parts]),
        np.concatenate([part[1] for part in episode_parts]),
    )


def reconstruct_pose(
    states: np.ndarray, model_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    finger_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2"),
    ]
    positions = []
    rotations = []
    for state in states:
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = state[:7]
        for finger_id in finger_ids:
            data.qpos[model.jnt_qposadr[finger_id]] = state[7] / 2.0
        mujoco.mj_forward(model, data)
        base_position = data.xpos[base_id]
        base_rotation = data.xmat[base_id].reshape(3, 3)
        positions.append(base_rotation.T @ (data.xpos[hand_id] - base_position))
        hand_rotation_in_base = base_rotation.T @ data.xmat[hand_id].reshape(3, 3)
        rotations.append(rotation_matrix_to_rpy(hand_rotation_in_base))
    rpy = np.unwrap(np.asarray(rotations), axis=0)
    return np.asarray(positions), np.rad2deg(rpy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--episode", type=int, default=3)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else dataset_root / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    states, actions = load_episode(dataset_root, args.episode)
    if actions.shape[1] != 7:
        raise ValueError("Expected base-frame XYZ, RPY, and gripper action")
    action_xyz = actions[:, :3]
    action_rpy_deg = np.rad2deg(actions[:, 3:6])
    actual_xyz, actual_rpy_deg = reconstruct_pose(states, args.model_path)
    # Euler angles that differ by a full turn describe the same pose. Align the
    # reconstructed branch to the recorded action branch before comparison.
    actual_rpy_deg += 360.0 * np.round(
        (action_rpy_deg[0] - actual_rpy_deg[0]) / 360.0
    )
    time_s = np.arange(len(actions)) / 30.0
    tracking_error = np.linalg.norm(action_xyz - actual_xyz, axis=1)
    action_step = np.linalg.norm(np.diff(action_xyz, axis=0), axis=1)
    action_rpy_step = np.linalg.norm(np.diff(action_rpy_deg, axis=0), axis=1)
    orientation_difference = (
        action_rpy_deg - actual_rpy_deg + 180.0
    ) % 360.0 - 180.0
    orientation_error = np.linalg.norm(orientation_difference, axis=1)

    finite = bool(
        np.isfinite(actions).all()
        and np.isfinite(actual_xyz).all()
        and np.isfinite(actual_rpy_deg).all()
    )
    summary = {
        "dataset": str(dataset_root),
        "episode_index": args.episode,
        "frames": int(len(actions)),
        "duration_s": float(time_s[-1]) if len(time_s) else 0.0,
        "all_values_finite": finite,
        "action_xyz_range_m": np.ptp(action_xyz, axis=0).tolist(),
        "action_max_step_m": float(action_step.max()) if len(action_step) else 0.0,
        "action_p99_step_m": float(np.percentile(action_step, 99)) if len(action_step) else 0.0,
        "tracking_error_median_m": float(np.median(tracking_error)),
        "tracking_error_p95_m": float(np.percentile(tracking_error, 95)),
        "tracking_error_max_m": float(tracking_error.max()),
        "action_rpy_range_deg": np.ptp(action_rpy_deg, axis=0).tolist(),
        "action_rpy_max_step_deg": (
            float(action_rpy_step.max()) if len(action_rpy_step) else 0.0
        ),
        "actual_rpy_range_deg": np.ptp(actual_rpy_deg, axis=0).tolist(),
        "orientation_error_p95_deg": float(np.percentile(orientation_error, 95)),
        "orientation_error_max_deg": float(orientation_error.max()),
    }

    plt.rcParams.update({"font.size": 9})
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax_3d.plot(*action_xyz.T, label="action target", linewidth=2.0)
    ax_3d.plot(*actual_xyz.T, label="actual hand", linewidth=1.2, alpha=0.8)
    ax_3d.scatter(*action_xyz[0], marker="o", s=45, label="start")
    ax_3d.scatter(*action_xyz[-1], marker="X", s=55, label="end")
    ax_3d.set_title("End-effector trajectory in robot-base frame")
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")
    ax_3d.legend(loc="best")

    ax_xyz = fig.add_subplot(grid[0, 1])
    for axis, label in enumerate(("X", "Y", "Z")):
        ax_xyz.plot(time_s, action_xyz[:, axis], label=f"target {label}")
        ax_xyz.plot(
            time_s,
            actual_xyz[:, axis],
            linestyle="--",
            alpha=0.75,
            label=f"actual {label}",
        )
    ax_xyz.set_title("Target and actual XYZ")
    ax_xyz.set_xlabel("Time (s)")
    ax_xyz.set_ylabel("Position (m)")
    ax_xyz.grid(alpha=0.25)
    ax_xyz.legend(ncol=2)

    ax_rpy = fig.add_subplot(grid[1, 0])
    for axis, label in enumerate(("RX / roll", "RY / pitch", "RZ / yaw")):
        ax_rpy.plot(time_s, action_rpy_deg[:, axis], label=f"target {label}")
        ax_rpy.plot(
            time_s,
            actual_rpy_deg[:, axis],
            linestyle="--",
            alpha=0.75,
            label=f"actual {label}",
        )
    ax_rpy.set_title("Target and actual end-effector orientation")
    ax_rpy.set_xlabel("Time (s)")
    ax_rpy.set_ylabel("Angle (deg)")
    ax_rpy.grid(alpha=0.25)
    ax_rpy.legend(ncol=2)

    ax_error = fig.add_subplot(grid[1, 1])
    ax_error.plot(time_s, tracking_error * 1000.0, label="XYZ tracking error")
    ax_error.axhline(
        summary["tracking_error_p95_m"] * 1000.0,
        linestyle="--",
        label="95th percentile",
    )
    ax_error.set_title("Distance between action target and actual hand")
    ax_error.set_xlabel("Time (s)")
    ax_error.set_ylabel("Error (mm)")
    ax_error.grid(alpha=0.25)
    ax_error.legend()

    fig.suptitle(
        f"MuJoCo Panda action check — episode {args.episode} ({len(actions)} frames)",
        fontsize=14,
    )
    image_path = output_dir / f"episode_{args.episode:03d}_action_check.png"
    json_path = output_dir / f"episode_{args.episode:03d}_action_check.json"
    fig.savefig(image_path, dpi=170)
    plt.close(fig)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "image": str(image_path), "report": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
