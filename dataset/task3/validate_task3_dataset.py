#!/usr/bin/env python
"""Audit Task 3 LeRobot v3 shards, video segments, and independent research logs.

This tool is read-only unless --output is supplied. By default it accepts only
nonempty successful demonstrations, suitable for the ACT baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulation.mujoco.dual_arm.recording_policy import physical_change

CAMERAS = ("top", "front", "side")
STATE_NAMES = [f"{side}_joint{i}" for side in ("left", "right") for i in range(1, 8)]
STATE_NAMES = STATE_NAMES[:7] + ["left_finger_width"] + STATE_NAMES[7:] + ["right_finger_width"]
ACTION_NAMES = [f"{side}_{field}" for side in ("left", "right")
                for field in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dataset_path(root: Path, template: str, **values) -> Path:
    """Never follow a dataset-supplied path outside the dataset root."""
    path = (root / template.format(**values)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes its root: {template}")
    return path


def episode_metadata(root: Path) -> list[dict]:
    paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise ValueError("No LeRobot v3 episode index; finish saving and close the recorder first")
    return [row for path in paths for row in pq.read_table(path).to_pylist()]


def inspect_video(path: Path) -> dict:
    times, sizes = [], set()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate or 0)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"Video frame has no timestamp: {path}")
            times.append(float(frame.pts * frame.time_base))
            sizes.add((frame.width, frame.height))
    return {"times": np.asarray(times), "sizes": sizes, "rate": rate}


def metric_max(rows: list[dict], *keys: str) -> float:
    values = []
    for row in rows:
        metrics = row.get("metrics", {})
        for key in keys:
            value = metrics.get(key, row.get(key))
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
    return max(values, default=0.0)


def snapshot_archive(root: Path, episode: int, count: int, expected_hash: str):
    """Read frozen snapshots in their explicit frame order, never infer gaps.

    Legacy datasets may omit the archive. They cannot use physical confirmation
    exceptions without the independent qpos/qvel evidence supplied here.
    """
    directory = dataset_path(root, f"research/episode_{episode:06d}_snapshots")
    if not directory.exists():
        return None
    metadata_path = dataset_path(root, f"research/episode_{episode:06d}_snapshots/task3.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("config_hash") != expected_hash:
        raise ValueError(f"Episode {episode}: snapshot configuration hash differs")
    parts = []
    cursor = 0
    shapes = {"state": (16,), "action": (14,), "qpos": (25,), "qvel": (24,), "time": ()}
    for path in sorted(directory.glob("snapshots_*.npz")):
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Episode {episode}: snapshot path escapes dataset root")
        match = re.fullmatch(r"snapshots_(\d{6,})\.npz", path.name)
        if not match or int(match.group(1)) != cursor:
            raise ValueError(f"Episode {episode}: snapshot block ordering/gap at {path.name}")
        with np.load(resolved, allow_pickle=False) as archive:
            if not all(key in archive for key in shapes):
                raise ValueError(f"Episode {episode}: snapshot fields missing in {path.name}")
            block = {key: np.asarray(archive[key]) for key in shapes}
        size = len(block["time"]) if block["time"].ndim == 1 else 0
        if size <= 0 or any(values.shape != (size, *shapes[key]) or not np.isfinite(values).all()
                            for key, values in block.items()):
            raise ValueError(f"Episode {episode}: malformed/non-finite snapshot block {path.name}")
        parts.append(block)
        cursor += size
    if cursor != count:
        raise ValueError(f"Episode {episode}: snapshot frame count {cursor} differs from {count}")
    result = {key: np.concatenate([part[key] for part in parts]) for key in shapes}
    if np.any(np.diff(result["time"]) < -1e-9):
        raise ValueError(f"Episode {episode}: snapshot simulation time is not monotonic")
    return result


def sampling_cadence(mapped: list[dict], fps: float) -> dict | None:
    """Describe short motion intervals, separating them from omitted pauses."""
    if len(mapped) < 2 or any("sim_time" not in row for row in mapped):
        return None
    times = np.asarray([row["sim_time"] for row in mapped], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) < -1e-9):
        raise ValueError("Research simulation clock is invalid/non-monotonic")
    intervals = []
    for previous, current, delta in zip(mapped, mapped[1:], np.diff(times)):
        if (previous.get("control_mode") not in {"XY", "Z"} or
                current.get("control_mode") != previous.get("control_mode") or
                not 0 < delta <= 4 / fps):
            continue
        changed_target = np.max(np.abs(np.asarray(current["action"])[[0, 1, 2, 7, 8, 9]] -
                                      np.asarray(previous["action"])[[0, 1, 2, 7, 8, 9]])) > 1e-6
        if changed_target:
            intervals.append(float(delta))
    median = float(np.median(intervals)) if intervals else None
    return {"motion_intervals": len(intervals), "nominal_interval_s": 1 / fps,
            "median_motion_interval_s": median,
            "median_playback_speedup": median * fps if median is not None else None,
            "slow_motion_interval_fraction": float(np.mean(np.asarray(intervals) > 1.2 / fps)) if intervals else 0.0,
            "max_retained_simulation_gap_s": float(np.max(np.diff(times))),
            "retained_simulation_span_s": float(times[-1] - times[0])}


def validate_dataset(dataset_root: Path | str, *, allow_failures=False, check_videos=True) -> dict:
    root = Path(dataset_root).expanduser().resolve()
    errors, warnings, episodes_report = [], [], []

    def check(condition, message):
        if not condition:
            errors.append(message)

    try:
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        task3 = json.loads((root / "meta" / "task3.json").read_text(encoding="utf-8"))
        check(str(info.get("codebase_version", "")).startswith("v3"), "Expected LeRobot v3 dataset")
        check(info.get("fps") == 30, "Expected 30 FPS")
        check(info.get("total_episodes", 0) > 0 and info.get("total_frames", 0) > 0,
              "Dataset is empty; directory creation does not constitute a recorded demonstration")
        features = info.get("features", {})
        for key, shape, names in (("observation.state", [16], STATE_NAMES), ("action", [14], ACTION_NAMES)):
            feature = features.get(key, {})
            check(list(feature.get("shape", [])) == shape, f"{key} must have shape {shape}")
            check(feature.get("names") == names, f"{key} field order must be left-first, right-second: {names}")
        expected_images = {f"observation.images.{camera}" for camera in CAMERAS}
        actual_images = {key for key in features if key.startswith("observation.images.")}
        check(actual_images == expected_images, "Expected exactly top/front/side RGB camera features")
        for key in expected_images:
            check(list(features.get(key, {}).get("shape", [])) == [480, 640, 3], f"{key} must be 640x480 RGB")
            check(features.get(key, {}).get("dtype") == "video", f"{key} must be a video feature")
        check(bool(task3.get("config_hash")), "meta/task3.json must contain config_hash")
        check(task3.get("schema_version") == "task3-v1", "meta/task3.json must identify schema_version=task3-v1")
        check(task3.get("field_order") == "left_then_right", "meta/task3.json must identify left_then_right field order")
        check(task3.get("coordinate_frame") == "task_frame", "meta/task3.json must identify common task_frame coordinates")
        transforms = task3.get("transforms", {})
        check(all(key in transforms for key in ("T_WT", "T_WL", "T_WR")), "Task, left base, and right base transforms are required")
        for key in ("T_WT", "T_WL", "T_WR"):
            matrix = np.asarray(transforms.get(key, []), dtype=float)
            check(matrix.shape == (4, 4) and np.isfinite(matrix).all(), f"{key} must be a finite 4x4 transform")
            if matrix.shape == (4, 4):
                check(np.allclose(matrix[3], [0, 0, 0, 1]) and
                      np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5) and
                      abs(np.linalg.det(matrix[:3, :3]) - 1) < 1e-5, f"{key} is not a rigid rotation/translation transform")
        cameras = task3.get("cameras", {})
        check(isinstance(cameras, dict) and all(key in cameras for key in CAMERAS), "Fixed top/front/side camera metadata dictionary is required")
        metadata = episode_metadata(root)
        manifest = read_jsonl(root / "research" / "episodes.jsonl")
        manifests = {int(row["episode_index"]): row for row in manifest}
        check(len(manifests) == len(manifest), "Duplicate research episode manifest entries")
        episode_ids = [int(row["episode_index"]) for row in metadata]
        check(len(episode_ids) == len(set(episode_ids)), "Duplicate LeRobot episode indices")
        check(sorted(episode_ids) == list(range(info.get("total_episodes", 0))), "Episode indices/count mismatch")
        check(set(manifests) == set(episode_ids), "Research manifests must correspond exactly to LeRobot episodes")
        columns = ["index", "episode_index", "frame_index", "timestamp", "observation.state", "action"]
        paths = sorted((root / "data").rglob("*.parquet"))
        if not paths:
            raise ValueError("No numeric data shards")
        # v3 episodes can share shards. Read and group by explicit indices.
        table = pa.concat_tables([pq.read_table(path, columns=columns) for path in paths])
        table = table.sort_by([("index", "ascending")])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=float)
        action = np.asarray(table["action"].to_pylist(), dtype=float)
        check(state.shape == (len(table), 16), "Numeric state rows are not 16D")
        check(action.shape == (len(table), 14), "Numeric action rows are not 14D")
        if state.ndim != 2 or action.ndim != 2 or state.shape[1] != 16 or action.shape[1] != 14:
            raise ValueError("Cannot inspect malformed numeric dimensions")
        check(np.isfinite(state).all() and np.isfinite(action).all(), "NaN/Inf in state or action")
        check(bool(np.all((action[:, [6, 13]] >= 0) & (action[:, [6, 13]] <= 255))), "Gripper action outside [0,255]")
        check(bool(np.all((action[:, 6] >= 127.5) == (action[:, 13] >= 127.5))), "Non-atomic left/right gripper labels")
        check(bool(np.all((state[:, [7, 15]] >= -0.001) & (state[:, [7, 15]] <= 0.09))), "Finger width outside Panda physical range")
        index = np.asarray(table["index"].to_pylist(), dtype=int)
        ep_index = np.asarray(table["episode_index"].to_pylist(), dtype=int)
        frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=int)
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=float)
        check(len(table) == info.get("total_frames"), "info.total_frames does not match numeric rows")
        check(np.array_equal(index, np.arange(len(table))), "Global index is not contiguous/unique")
        check(set(ep_index.tolist()) == set(episode_ids), "Numeric/index episode IDs differ")
        video_cache = {}
        for meta in sorted(metadata, key=lambda row: row["episode_index"]):
            episode = int(meta["episode_index"])
            mask = ep_index == episode
            count = int(mask.sum())
            prefix = f"Episode {episode}: "
            check(count > 0 and count == int(meta["length"]), prefix + "episode length mismatch")
            check(np.array_equal(frame_index[mask], np.arange(count)), prefix + "frame_index not contiguous")
            check(np.allclose(timestamps[mask], np.arange(count) / info["fps"], atol=1e-4), prefix + "effective training clock must be continuous")
            check(np.array_equal(index[mask], np.arange(int(meta["dataset_from_index"]), int(meta["dataset_to_index"]))),
                  prefix + "dataset index interval mismatch")
            data_file = dataset_path(root, info["data_path"], chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"])
            check(data_file in paths, prefix + "indexed numeric shard is missing")
            selected = pq.read_table(data_file, columns=["index", "episode_index"]).to_pydict() if data_file.exists() else {}
            check(sum(int(value) == episode for value in selected.get("episode_index", [])) == count,
                  prefix + "indexed shard does not contain its episode rows")
            camera_counts = {}
            if check_videos:
                for key in sorted(expected_images):
                    start = float(meta[f"videos/{key}/from_timestamp"])
                    end = float(meta[f"videos/{key}/to_timestamp"])
                    path = dataset_path(root, info["video_path"], video_key=key,
                                        chunk_index=meta[f"videos/{key}/chunk_index"],
                                        file_index=meta[f"videos/{key}/file_index"])
                    if path not in video_cache:
                        video_cache[path] = inspect_video(path)
                    video = video_cache[path]
                    # Segment end is exclusive. Encoder padding is outside indexed segments.
                    times = video["times"]
                    selected_times = times[(times >= start - 1e-4) & (times < end - 1e-4)]
                    camera_counts[key] = int(len(selected_times))
                    check(video["sizes"] == {(640, 480)}, prefix + f"{key} image size mismatch")
                    check(abs(video["rate"] - info["fps"]) < 0.1, prefix + f"{key} FPS mismatch")
                    check(len(selected_times) == count, prefix + f"{key} indexed video segment has {len(selected_times)} frames, expected {count}")
                    check(abs(end - start - count / info["fps"]) < 1e-3, prefix + f"{key} segment duration mismatch")
                    if len(selected_times) == count:
                        check(np.allclose(selected_times - start, np.arange(count) / info["fps"], atol=1e-3),
                              prefix + f"{key} frame timestamps are not synchronized")
            entry = manifests.get(episode, {})
            outcome = entry.get("outcome")
            check(outcome in {"success", "failure"}, prefix + "unknown/missing outcome")
            if task3.get("outcome") is not None:
                check(task3["outcome"] == outcome, prefix + "outcome differs from dataset metadata")
            if not allow_failures:
                check(outcome == "success", prefix + "failure episode rejected from baseline training")
            check(entry.get("frames") == count, prefix + "research manifest training frame count differs")
            check(entry.get("config_hash") == task3.get("config_hash"), prefix + "configuration hash differs")
            check(abs(float(entry.get("training_duration_s", -1)) - count / info["fps"]) < 1e-3,
                  prefix + "training_duration_s mismatch")
            rows = read_jsonl(root / "research" / f"episode_{episode:06d}.jsonl")
            check(bool(rows), prefix + "research log is empty")
            if not allow_failures and rows:
                check(rows[-1].get("task_state") == "DONE", prefix + "successful demonstration must end with automatic DONE verification")
                check(not any(row.get("task_state") == "FAIL" for row in rows), prefix + "successful demonstration contains a FAIL state")
            mapped = [row for row in rows if row.get("frame_index") is not None]
            check([row["frame_index"] for row in mapped] == list(range(count)), prefix + "raw-to-training frame mapping mismatch")
            snapshots = snapshot_archive(root, episode, count, task3.get("config_hash"))
            verified_snapshots = {}
            physical_confirmation_frames = []
            for row in mapped:
                frame = int(row["frame_index"])
                check(row.get("training_recordable") is True, prefix + f"training frame {frame} is marked non-recordable")
                local_state = np.asarray(row.get("state", []), dtype=float)
                local_action = np.asarray(row.get("action", []), dtype=float)
                check(local_state.shape == (16,) and local_action.shape == (14,), prefix + "research numeric schema mismatch")
                if local_state.shape == (16,) and local_action.shape == (14,):
                    check(np.allclose(local_state, state[mask][frame], atol=1e-5) and np.allclose(local_action, action[mask][frame], atol=1e-5),
                          prefix + f"research frame {frame} differs from saved training data")
                snapshot = None
                snapshot_ok = False
                if snapshots is not None:
                    snapshot = {key: values[frame] for key, values in snapshots.items()}
                    qpos = snapshot["qpos"]
                    reconstructed_state = np.r_[qpos[:7], qpos[7:9].sum(), qpos[9:16], qpos[16:18].sum()]
                    snapshot_ok = bool(local_state.shape == (16,) and local_action.shape == (14,) and
                                       "sim_time" in row and np.isfinite(row["sim_time"]) and
                                       abs(float(snapshot["time"]) - float(row["sim_time"])) <= 1e-8 and
                                       np.allclose(snapshot["state"], local_state, atol=1e-5, rtol=0) and
                                       np.allclose(snapshot["action"], local_action, atol=1e-5, rtol=0) and
                                       np.allclose(snapshot["state"], reconstructed_state, atol=1e-5, rtol=0))
                    check(snapshot_ok, prefix + f"snapshot frame {frame} differs from mapped time/state/action/qpos")
                    verified_snapshots[frame] = snapshot_ok
                if str(row.get("control_mode", "")).upper() in {"GRASP_CONFIRM", "RELEASE_CONFIRM"}:
                    previous_ok = frame == 0 or verified_snapshots.get(frame - 1, False)
                    previous = {key: values[frame - 1] for key, values in snapshots.items()} if snapshots is not None and frame else None
                    physical = bool(snapshot_ok and previous_ok and
                                    physical_change(snapshot, previous, row.get("task_state")))
                    check(bool(row.get("events")) or physical,
                          prefix + f"confirmation idle row leaked into training at frame {frame}; no verified physical change")
                    if physical and not row.get("events"):
                        physical_confirmation_frames.append(frame)
            wall = [float(row["wall_time_ms"]) for row in rows]
            check(all(a <= b for a, b in zip(wall, wall[1:])), prefix + "research wall clock is not monotonic")
            relative_error = metric_max(rows + [entry], "relative_pose_error_m", "relative_ee_error_m", "relative_position_error_m", "max_relative_pose_error_m")
            collisions = metric_max(rows + [entry], "collision_count", "robot_robot_collision_count")
            contact_loss = metric_max(rows + [entry], "contact_loss_count")
            safety_events = sorted({str(event) for row in rows for event in row.get("events", [])
                                    if isinstance(event, str) and ("collision" in event or "contact_loss" in event)})
            if not allow_failures:
                check(collisions == 0 and not safety_events, prefix + "unsafe collision/contact-loss episode")
                check(contact_loss == 0, prefix + "contact-loss episode rejected")
                limit = task3.get("config", {}).get("cooperation", {}).get("relative_pose_hard_limit_m", 0.04)
                check(relative_error <= float(limit) + 1e-5, prefix + "relative EE error exceeds safety threshold")
                check(not entry.get("failure_reason"), prefix + "successful episode contains a failure_reason")
            xyz = action[mask][:, [0, 1, 2, 7, 8, 9]]
            if outcome == "success":
                closed = action[mask][:, 6] < 127.5
                if not (closed.any() and (~closed).any()):
                    warnings.append(prefix + "no complete close/open sequence; record from RESET/PREGRASP for full-task ACT training")
            max_step = float(np.linalg.norm(np.diff(xyz, axis=0).reshape(-1, 2, 3), axis=2).max()) if count > 1 else 0.0
            if max_step > 0.05:
                warnings.append(prefix + f"large commanded EE jump ({max_step:.3f} m); review before training")
            cadence = sampling_cadence(mapped, float(info["fps"]))
            if cadence and cadence["motion_intervals"] >= 2 and cadence["slow_motion_interval_fraction"] > 0.5:
                warnings.append(prefix +
                                f"motion samples are commonly {cadence['median_motion_interval_s'] * 1000:.1f} ms apart in simulation "
                                f"but stored at {info['fps']} FPS ({cadence['median_playback_speedup']:.2f}x cadence compression); "
                                "review timing quality before training. Existing frames were not resampled")
            episodes_report.append({"episode_index": episode, "frames": count, "outcome": outcome,
                                    "wall_duration_s": entry.get("wall_duration_s"), "training_duration_s": count / info["fps"],
                                    "camera_frames": camera_counts, "max_xyz_step_m": max_step,
                                    "max_relative_ee_error_m": relative_error, "collision_count": collisions,
                                    "contact_loss_count": contact_loss,
                                    "snapshots_verified": snapshots is not None and all(verified_snapshots.values()),
                                    "physical_confirmation_frames": physical_confirmation_frames,
                                    "sampling_cadence": cadence})
    except (OSError, ValueError, TypeError, KeyError, IndexError, pa.ArrowException, av.error.FFmpegError) as error:
        errors.append(f"Dataset cannot be fully audited: {type(error).__name__}: {error}")
    return {"dataset_root": str(root), "valid": not errors, "allow_failures": bool(allow_failures),
            "videos_checked": bool(check_videos), "errors": errors, "warnings": warnings,
            "episodes": episodes_report, "episode_count": len(episodes_report),
            "frame_count": sum(row["frames"] for row in episodes_report)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--allow-failures", action="store_true", help="Audit failure data for analysis; never the baseline default")
    parser.add_argument("--skip-videos", action="store_true", help="Fast numeric/log audit; not a complete training preflight")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_dataset(args.dataset_root, allow_failures=args.allow_failures, check_videos=not args.skip_videos)
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
