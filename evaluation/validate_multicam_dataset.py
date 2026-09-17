#!/usr/bin/env python
"""Validate synchronized numeric and multi-camera MuJoCo demonstrations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    output_dir = root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads((root / "meta" / "info.json").read_text())
    tables = [pq.read_table(path) for path in sorted((root / "data").glob("chunk-*/*.parquet"))]
    table = pa.concat_tables(tables)
    episode_indices = np.asarray(table["episode_index"].to_pylist(), dtype=int)
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=int)
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=float)
    actions = np.asarray(table["action"].to_pylist(), dtype=float)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=float)
    episode_ids = sorted(np.unique(episode_indices).tolist())
    episode_counts = {str(i): int(np.sum(episode_indices == i)) for i in episode_ids}
    final_global_indices = [int(np.flatnonzero(episode_indices == i)[-1]) for i in episode_ids]

    timestamp_errors = []
    frame_index_sequences_valid = True
    episode_action_checks = []
    for episode_id in episode_ids:
        episode_mask = episode_indices == episode_id
        dt = np.diff(timestamps[episode_mask])
        if len(dt):
            timestamp_errors.extend(np.abs(dt - 1.0 / info["fps"]).tolist())
        episode_frame_indices = frame_indices[episode_mask]
        frame_index_sequences_valid &= bool(
            np.array_equal(episode_frame_indices, np.arange(len(episode_frame_indices)))
        )
        episode_actions = actions[episode_mask]
        xyz_steps = np.linalg.norm(np.diff(episode_actions[:, :3], axis=0), axis=1)
        rpy_steps_deg = np.rad2deg(
            np.linalg.norm(np.diff(np.unwrap(episode_actions[:, 3:6], axis=0), axis=0), axis=1)
        )
        gripper = episode_actions[:, 6]
        closed = gripper < 127.5
        gripper_transitions = int(np.count_nonzero(np.diff(closed.astype(np.int8))))
        episode_action_checks.append(
            {
                "episode": int(episode_id),
                "frames": int(len(episode_actions)),
                "duration_s": float((len(episode_actions) - 1) / info["fps"]),
                "max_xyz_step_m": float(xyz_steps.max()) if len(xyz_steps) else 0.0,
                "p99_xyz_step_m": float(np.percentile(xyz_steps, 99)) if len(xyz_steps) else 0.0,
                "max_rpy_step_deg": float(rpy_steps_deg.max()) if len(rpy_steps_deg) else 0.0,
                "gripper_min": float(gripper.min()),
                "gripper_max": float(gripper.max()),
                "gripper_transitions": gripper_transitions,
                "has_close_and_open": bool(closed.any() and (~closed).any()),
            }
        )

    camera_features = sorted(
        key for key in info["features"] if key.startswith("observation.images.")
    )
    camera_results: dict[str, dict] = {}
    final_frames: dict[str, list[Image.Image]] = {}
    for feature in camera_features:
        paths = sorted((root / "videos" / feature).glob("chunk-*/*.mp4"))
        selected_frames: dict[int, Image.Image] = {}
        frame_count = 0
        width = height = None
        for path in paths:
            with av.open(path) as container:
                for frame in container.decode(video=0):
                    image = frame.to_image().convert("RGB")
                    width, height = image.size
                    if frame_count in final_global_indices:
                        selected_frames[frame_count] = image
                    frame_count += 1
        final_frames[feature] = [selected_frames[index] for index in final_global_indices]
        camera_results[feature] = {
            "frames": frame_count,
            "width": width,
            "height": height,
            "matches_data_frames": frame_count == len(table),
        }

    # Compact pages make it practical to visually inspect dozens of episode endings.
    overview_key = next(
        (feature for feature in camera_features if feature.endswith(".overview")),
        camera_features[0],
    )
    overview_page_paths = []
    page_size = 20
    panel_width, panel_height = 320, 240
    columns = 4
    for page_start in range(0, len(episode_ids), page_size):
        page_ids = episode_ids[page_start : page_start + page_size]
        rows = (len(page_ids) + columns - 1) // columns
        sheet = Image.new("RGB", (panel_width * columns, panel_height * rows), color="black")
        draw = ImageDraw.Draw(sheet)
        for page_offset, episode_id in enumerate(page_ids):
            image = final_frames[overview_key][page_start + page_offset].resize(
                (panel_width, panel_height)
            )
            x = (page_offset % columns) * panel_width
            y = (page_offset // columns) * panel_height
            sheet.paste(image, (x, y))
            draw.rectangle((x, y, x + 105, y + 24), fill="black")
            draw.text((x + 7, y + 6), f"episode {episode_id} final", fill="white")
        page_end = page_start + len(page_ids) - 1
        page_path = output_dir / f"overview_finals_{page_start:03d}_{page_end:03d}.png"
        sheet.save(page_path)
        overview_page_paths.append(str(page_path))

    checks_csv_path = output_dir / "episode_action_checks.csv"
    with checks_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=episode_action_checks[0].keys())
        writer.writeheader()
        writer.writerows(episode_action_checks)

    report = {
        "dataset": str(root),
        "episodes": len(episode_ids),
        "frames": len(table),
        "fps": info["fps"],
        "episode_frames": episode_counts,
        "metadata_frame_count_matches": len(table) == info["total_frames"],
        "metadata_episode_count_matches": len(episode_ids) == info["total_episodes"],
        "frame_index_sequences_valid": frame_index_sequences_valid,
        "action_shape": list(actions.shape[1:]),
        "state_shape": list(states.shape[1:]),
        "numeric_values_finite": bool(np.isfinite(actions).all() and np.isfinite(states).all()),
        "max_timestamp_error_s": max(timestamp_errors, default=0.0),
        "camera_streams": camera_results,
        "episode_action_checks": episode_action_checks,
        "all_episodes_have_gripper_close_and_open": all(
            check["has_close_and_open"] for check in episode_action_checks
        ),
        "max_xyz_step_m": max(check["max_xyz_step_m"] for check in episode_action_checks),
        "max_rpy_step_deg": max(check["max_rpy_step_deg"] for check in episode_action_checks),
        "overview_final_frame_pages": overview_page_paths,
        "episode_action_checks_csv": str(checks_csv_path),
    }
    report_path = output_dir / "multicam_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
