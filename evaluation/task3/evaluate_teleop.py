#!/usr/bin/env python
"""Summarize saved Task 3 successes and failures from research manifests.

Wall-clock completion time includes human confirmation waits. Training time
uses only accepted frames; do not confuse these two measures of efficiency.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from dataset.task3.validate_task3_dataset import read_jsonl, metric_max


def finite_mean(values):
    numbers = [float(value) for value in values if isinstance(value, (int, float)) and np.isfinite(value)]
    return float(np.mean(numbers)) if numbers else None


def value_from_log(row: dict, key: str):
    return row.get("metrics", {}).get(key, row.get(key))


def summarize_episode(root: Path, entry: dict) -> dict:
    rows = read_jsonl(root / "research" / f"episode_{int(entry['episode_index']):06d}.jsonl")
    positions = [value_from_log(row, "object_position") for row in rows]
    positions = np.asarray([p for p in positions if isinstance(p, (list, tuple)) and len(p) == 3], dtype=float)
    object_path = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()) if len(positions) > 1 else None
    paused = [str(row.get("control_mode", "")).upper() in {"PAUSE", "CLUTCH", "PAUSED"} for row in rows]
    pause_count = sum(value and (i == 0 or not paused[i - 1]) for i, value in enumerate(paused))
    hand_lost_count = 0
    previously_lost = False
    for row in rows:
        telemetry = row.get("telemetry", {})
        reason = str(row.get("pause_reason", telemetry.get("pause_reason", ""))).lower()
        lost = "hand_lost" in reason or "hand lost" in reason
        hand_lost_count += int(lost and not previously_lost)
        previously_lost = lost
    relative = [value_from_log(row, "relative_pose_error_m") for row in rows
                if row.get("task_state") in {"DUAL_GRASPED", "PLACEMENT"}]
    disagreement = [value_from_log(row, "input_disagreement_m") for row in rows]
    explicitly_annotated_false_triggers = entry.get("metrics", {}).get("false_gripper_trigger_count")
    collisions = metric_max(rows + [entry], "collision_count", "robot_robot_collision_count")
    contact_loss = metric_max(rows + [entry], "contact_loss_count")
    return {"dataset_root": str(root), "episode_index": entry["episode_index"], "seed": entry.get("seed"),
            "config_hash": entry.get("config_hash"), "success": entry.get("outcome") == "success",
            "failure_reason": entry.get("failure_reason") or "", "frames": entry.get("frames", 0),
            "wall_duration_s": entry.get("wall_duration_s"), "training_duration_s": entry.get("training_duration_s"),
            "pause_count": pause_count, "hand_loss_count": hand_lost_count,
            "false_gripper_trigger_count": explicitly_annotated_false_triggers,
            "collision_count": collisions, "contact_loss_count": contact_loss,
            "mean_relative_ee_error_m": finite_mean(relative), "max_relative_ee_error_m": metric_max(rows, "relative_pose_error_m"),
            "mean_input_disagreement_m": finite_mean(disagreement), "object_path_length_m": object_path}


def evaluate_roots(roots: list[Path]) -> tuple[list[dict], dict]:
    result, seen = [], set()
    for supplied in roots:
        root = supplied.expanduser().resolve()
        if root in seen:
            raise ValueError(f"Dataset was supplied twice: {root}")
        seen.add(root)
        for entry in read_jsonl(root / "research" / "episodes.jsonl"):
            result.append(summarize_episode(root, entry))
    if not result:
        raise ValueError("No saved Task 3 episodes; start recording and save a result first")
    count = len(result)
    histogram = Counter(row["failure_reason"] or "unclassified_failure" for row in result if not row["success"])
    config_hashes = sorted({row["config_hash"] for row in result if row["config_hash"]})
    summary = {"episodes": count, "successes": sum(row["success"] for row in result),
               "failures": sum(not row["success"] for row in result),
               "success_rate": sum(row["success"] for row in result) / count,
               "mean_success_completion_time_s": finite_mean([row["wall_duration_s"] for row in result if row["success"]]),
               "mean_all_wall_duration_s": finite_mean([row["wall_duration_s"] for row in result]),
               "collision_rate": sum(row["collision_count"] > 0 for row in result) / count,
               "contact_loss_rate": sum(row["contact_loss_count"] > 0 for row in result) / count,
               "mean_relative_ee_error_m": finite_mean([row["mean_relative_ee_error_m"] for row in result]),
               "mean_object_path_length_m": finite_mean([row["object_path_length_m"] for row in result]),
               "failure_reason_histogram": dict(histogram), "config_hashes": config_hashes,
               "notes": ["False gripper trigger counts require operator annotations; missing values mean unmeasured.",
                         "Rejected episode rate cannot include unsaved/discarded attempts; retain all attempts for that analysis.",
                         "A success rate is meaningful only when both success and failure roots for the same session are included."]}
    if len(config_hashes) > 1:
        summary["notes"].append("Multiple configurations included: compare per configuration before reporting experiment results.")
    return result, summary


def write_reports(output: Path, rows: list[dict], summary: dict):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_roots", nargs="+", type=Path, help="Supply success AND failure datasets for unbiased efficiency statistics")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        rows, summary = evaluate_roots(args.dataset_roots)
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from error
    write_reports(args.output_dir.expanduser().resolve(), rows, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
