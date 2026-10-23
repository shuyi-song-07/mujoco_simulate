#!/usr/bin/env python3
"""Create a hard-linked dataset copy with causally smoothed wrist deltas.

Only ``observation.hand_delta_uv`` is changed. Validity, segment IDs, images,
robot state, actions, and all other labels remain byte-for-byte identical.
The EMA is updated on valid hand reports, carried across frames without a new
report, and reset at episode or hand-motion-segment boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "datasets/mujoco_panda_pick_diffusion_20261022"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets/mujoco_panda_pick_diffusion_20261022_hand_delta_ema"
DELTA_KEY = "observation.hand_delta_uv"
VALID_KEY = "observation.hand_delta_valid"
SEGMENT_KEY = "observation.hand_motion_segment"


def causal_segment_ema(
    delta: np.ndarray, valid: np.ndarray, segment: np.ndarray, episode: np.ndarray, alpha: float
) -> np.ndarray:
    """Filter valid samples without leaking across episodes or re-entry segments."""
    if delta.ndim != 2 or delta.shape[1] != 2:
        raise ValueError(f"Expected Nx2 hand delta, got {delta.shape}")
    if not (len(delta) == len(valid) == len(segment) == len(episode)):
        raise ValueError("Hand-delta columns have different lengths")
    if not 0.0 < alpha <= 1.0 or not math.isfinite(alpha):
        raise ValueError("EMA alpha must be finite and in (0, 1]")

    output = np.zeros_like(delta, dtype=np.float32)
    state: dict[int, tuple[int, np.ndarray]] = {}
    for index in range(len(delta)):
        if not valid[index]:
            continue
        sample = np.asarray(delta[index], dtype=np.float32)
        if not np.isfinite(sample).all() or int(segment[index]) < 0:
            continue
        episode_id = int(episode[index])
        segment_id = int(segment[index])
        previous = state.get(episode_id)
        if previous is None or previous[0] != segment_id:
            smoothed = sample.copy()
        else:
            smoothed = alpha * sample + (1.0 - alpha) * previous[1]
        output[index] = smoothed
        state[episode_id] = (segment_id, smoothed)
    return output


def stats_for(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
        "q01": quantiles[0].tolist(),
        "q10": quantiles[1].tolist(),
        "q50": quantiles[2].tolist(),
        "q90": quantiles[3].tolist(),
        "q99": quantiles[4].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    source = args.source_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"Source dataset does not exist: {source}")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    if not 0.0 < args.alpha <= 1.0 or not math.isfinite(args.alpha):
        parser.error("--alpha must be finite and in (0, 1]")

    # Videos and unchanged metadata are immutable, so hard links avoid another
    # ~850 MB physical copy. Replaced Parquet/stat files are atomically unlinked.
    shutil.copytree(source, output, copy_function=os.link)
    all_filtered = []
    valid_samples = 0
    try:
        for path in sorted((output / "data").rglob("*.parquet")):
            table = pq.read_table(path)
            required = {DELTA_KEY, VALID_KEY, SEGMENT_KEY, "episode_index"}
            missing = required - set(table.column_names)
            if missing:
                raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
            delta = np.asarray(table[DELTA_KEY].to_pylist(), dtype=np.float32)
            valid = np.asarray(table[VALID_KEY].to_pylist(), dtype=np.int64).astype(bool)
            segment = np.asarray(table[SEGMENT_KEY].to_pylist(), dtype=np.int64)
            episode = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
            filtered = causal_segment_ema(delta, valid, segment, episode, args.alpha)
            column_index = table.column_names.index(DELTA_KEY)
            replacement = pa.array(filtered.tolist(), type=table.schema.field(DELTA_KEY).type)
            table = table.set_column(column_index, DELTA_KEY, replacement)
            temporary = path.with_suffix(".filtered.parquet")
            pq.write_table(table, temporary)
            os.replace(temporary, path)
            all_filtered.append(filtered)
            valid_samples += int(valid.sum())

        combined = np.concatenate(all_filtered)
        stats_path = output / "meta/stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats[DELTA_KEY] = stats_for(combined.astype(np.float64))
        temporary_stats = stats_path.with_suffix(".filtered.json")
        temporary_stats.write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")
        os.replace(temporary_stats, stats_path)

        provenance = {
            "source_root": str(source),
            "output_root": str(output),
            "field": DELTA_KEY,
            "filter": "causal_segment_ema",
            "alpha": args.alpha,
            "reset_on": ["episode_change", "hand_motion_segment_change"],
            "carry_state_across_invalid_frames": True,
            "rows": int(len(combined)),
            "valid_samples": valid_samples,
        }
        (output / "hand_delta_filter.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(provenance, indent=2, ensure_ascii=False))
    except Exception:
        shutil.rmtree(output)
        raise


if __name__ == "__main__":
    main()
