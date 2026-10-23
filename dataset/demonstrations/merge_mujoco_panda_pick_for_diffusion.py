#!/usr/bin/env python3
"""Merge compatible Panda pick-and-place LeRobot datasets for Diffusion Policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = [
    PROJECT_ROOT / "datasets/mujoco_panda_pick_20261022_174343",
    PROJECT_ROOT / "datasets/mujoco_panda_pick_20261022_172916",
    PROJECT_ROOT / "datasets/mujoco_panda_pick_20261022_165728",
    PROJECT_ROOT / "datasets/mujoco_panda_pick_20261022_155728",
]
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets/mujoco_panda_pick_diffusion_20261022"
REPO_ID = "local/mujoco_panda_pick_diffusion_20261022"
HAND_FIELDS = {
    "sample_index",
    "frame_index_at_receive",
    "raw_wrist_uv",
    "raw_timestamp_ms",
    "filtered_wrist_uv",
    "filtered_timestamp_ms",
    "hand_motion_segment",
    "sent_at_ms",
}
INSPECTION_CAMERAS = ("overview", "camera_2", "camera_3")


def read_info(root: Path) -> dict:
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise ValueError(f"Missing LeRobot metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("fps") != 30 or info.get("robot_type") != "mujoco_panda":
        raise ValueError(f"Unexpected robot type or fps in {root}")
    features = info.get("features", {})
    required = {
        "observation.state",
        "observation.images.overview",
        "action",
        "observation.gesture",
        "observation.hand_delta_uv",
        "observation.hand_delta_valid",
    }
    if not required.issubset(features):
        raise ValueError(f"Missing required Diffusion Policy features in {root}: {sorted(required - features.keys())}")
    if not info.get("total_episodes") or not info.get("total_frames"):
        raise ValueError(f"Empty dataset: {root}")
    return info


def validate_source(root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not root.is_dir():
        raise ValueError(f"Source dataset does not exist: {root}")
    info = read_info(root)
    dataset = LeRobotDataset(
        repo_id=f"local/{root.name}", root=root, video_backend="pyav"
    )
    episodes = pq.read_table(
        root / "meta/episodes/chunk-000/file-000.parquet",
        columns=["episode_index", "length"],
    ).to_pylist()
    lengths = {int(row["episode_index"]): int(row["length"]) for row in episodes}
    expected_indices = set(range(int(info["total_episodes"])))
    if set(lengths) != expected_indices:
        raise ValueError(f"Episode metadata is incomplete in {root}")

    trajectory_dir = root / "hand_trajectories"
    trajectory_files = {
        int(path.stem.removeprefix("episode-")): path
        for path in trajectory_dir.glob("episode-*.jsonl")
    }
    if set(trajectory_files) != expected_indices:
        raise ValueError(f"hand_trajectories is incomplete in {root}")
    for episode_index, path in trajectory_files.items():
        for sample_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            sample = json.loads(line)
            if set(sample) != HAND_FIELDS or sample["sample_index"] != sample_index:
                raise ValueError(f"Invalid hand trajectory row in {path}, line {sample_index + 1}")

    video_root = root / "episode_videos"
    manifest_path = video_root / "initial_positions.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"Missing layout manifest: {manifest_path}")
    manifest = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    if [int(row["episode_index"]) for row in manifest] != list(range(len(episodes))):
        raise ValueError(f"Layout manifest episode indices are incomplete in {root}")
    for episode_index in expected_indices:
        episode_video_dir = video_root / f"episode_{episode_index:06d}"
        for camera in INSPECTION_CAMERAS:
            video = episode_video_dir / f"{camera}.mp4"
            if not video.is_file() or video.stat().st_size == 0:
                raise ValueError(f"Missing or empty inspection video: {video}")

    return dataset, {
        "root": str(root),
        "episodes": len(episodes),
        "frames": int(info["total_frames"]),
        "episode_lengths": lengths,
        "trajectories": trajectory_files,
        "layout_manifest": manifest,
    }


def copy_inspection_assets(sources: list[dict], output: Path) -> list[dict]:
    video_root = output / "episode_videos"
    trajectory_root = output / "hand_trajectories"
    video_root.mkdir(parents=True)
    trajectory_root.mkdir(parents=True)

    manifest_rows = []
    mapping = []
    next_episode = 0
    for source in sources:
        root = Path(source["root"])
        start = next_episode
        for old_episode in range(source["episodes"]):
            trajectory = source["trajectories"][old_episode]
            shutil.copy2(
                trajectory,
                trajectory_root / f"episode-{next_episode:06d}.jsonl",
            )
            shutil.copytree(
                root / "episode_videos" / f"episode_{old_episode:06d}",
                video_root / f"episode_{next_episode:06d}",
            )
            row = dict(source["layout_manifest"][old_episode])
            row["episode_index"] = next_episode
            row["source_dataset"] = root.name
            manifest_rows.append(row)
            mapping.append({
                "source_dataset": root.name,
                "source_episode_index": old_episode,
                "merged_episode_index": next_episode,
                "frames": source["episode_lengths"][old_episode],
            })
            next_episode += 1

    with (video_root / "initial_positions.jsonl").open("w", encoding="utf-8") as stream:
        for row in manifest_rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roots", nargs="+", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without creating the merged dataset")
    args = parser.parse_args()

    roots = [path.expanduser().resolve() for path in args.source_roots]
    output = args.output_root.expanduser().resolve()
    if len(set(roots)) != len(roots):
        parser.error("Source roots must be unique")
    if output in roots:
        parser.error("Output root cannot also be a source root")
    if output.exists():
        parser.error(f"Output already exists; choose a new --output-root: {output}")

    sources = []
    datasets = []
    baseline_features = None
    for root in roots:
        dataset, source = validate_source(root)
        features = dataset.meta.features
        if baseline_features is None:
            baseline_features = features
        elif features != baseline_features:
            raise ValueError(f"Incompatible feature schema: {root}")
        datasets.append(dataset)
        sources.append(source)

    total_episodes = sum(source["episodes"] for source in sources)
    total_frames = sum(source["frames"] for source in sources)
    print(json.dumps({
        "output_root": str(output),
        "repo_id": REPO_ID,
        "sources": [{k: source[k] for k in ("root", "episodes", "frames")} for source in sources],
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "dry_run": args.dry_run,
    }, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    from lerobot.datasets.dataset_tools import merge_datasets

    merged = merge_datasets(
        datasets,
        output_repo_id=REPO_ID,
        output_dir=output,
        concatenate_videos=True,
        concatenate_data=True,
    )
    if merged.meta.total_episodes != total_episodes or merged.meta.total_frames != total_frames:
        raise RuntimeError("Merged LeRobot dataset counts do not match the sources")

    mapping = copy_inspection_assets(sources, output)
    (output / "merge_manifest.json").write_text(
        json.dumps({
            "repo_id": REPO_ID,
            "sources": [{k: source[k] for k in ("root", "episodes", "frames")} for source in sources],
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "episode_mapping": mapping,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Merged {total_episodes} episodes / {total_frames} frames into {output}")


if __name__ == "__main__":
    main()
