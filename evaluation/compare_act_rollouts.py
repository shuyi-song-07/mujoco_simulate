#!/usr/bin/env python3
"""Compare two ACT rollout CSV files episode-by-episode on matching seeds."""

from __future__ import annotations

import argparse
import csv
import json
from math import comb
from pathlib import Path


def read_rows(path: Path) -> dict[int, dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = {int(row["seed"]): row for row in csv.DictReader(stream)}
    if not rows:
        raise ValueError(f"No rollout rows in {path}")
    return rows


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def exact_mcnemar_p(baseline_only: int, gesture_only: int) -> float:
    """Two-sided exact binomial McNemar p-value for paired binary outcomes."""
    discordant = baseline_only + gesture_only
    if discordant == 0:
        return 1.0
    smaller = min(baseline_only, gesture_only)
    tail = sum(comb(discordant, index) for index in range(smaller + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--gesture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline = read_rows(args.baseline.expanduser().resolve())
    gesture = read_rows(args.gesture.expanduser().resolve())
    if set(baseline) != set(gesture):
        missing_gesture = sorted(set(baseline) - set(gesture))
        missing_baseline = sorted(set(gesture) - set(baseline))
        raise ValueError(
            f"Seed sets differ; missing gesture={missing_gesture}, missing baseline={missing_baseline}"
        )

    paired = []
    for seed in sorted(baseline):
        base_success = as_bool(baseline[seed]["success"])
        gesture_success = as_bool(gesture[seed]["success"])
        paired.append({
            "seed": seed,
            "baseline_success": base_success,
            "gesture_success": gesture_success,
            "outcome": (
                "both" if base_success and gesture_success else
                "baseline_only" if base_success else
                "gesture_only" if gesture_success else "neither"
            ),
            "baseline_xy_m": float(baseline[seed]["cube_target_xy_m"]),
            "gesture_xy_m": float(gesture[seed]["cube_target_xy_m"]),
            "baseline_sim_time_s": float(baseline[seed]["sim_time_s"]),
            "gesture_sim_time_s": float(gesture[seed]["sim_time_s"]),
        })

    count = len(paired)
    baseline_successes = sum(row["baseline_success"] for row in paired)
    gesture_successes = sum(row["gesture_success"] for row in paired)
    baseline_only = sum(row["outcome"] == "baseline_only" for row in paired)
    gesture_only = sum(row["outcome"] == "gesture_only" for row in paired)
    both = sum(row["outcome"] == "both" for row in paired)
    neither = sum(row["outcome"] == "neither" for row in paired)
    summary = {
        "episodes": count,
        "seeds": sorted(baseline),
        "success_rule": "red rigid body contacts and remains stably supported by target plate for 0.5 s",
        "baseline": {
            "csv": str(args.baseline.expanduser().resolve()),
            "successes": baseline_successes,
            "success_rate": baseline_successes / count,
        },
        "gesture_act": {
            "csv": str(args.gesture.expanduser().resolve()),
            "successes": gesture_successes,
            "success_rate": gesture_successes / count,
        },
        "absolute_success_rate_delta": (gesture_successes - baseline_successes) / count,
        "relative_success_rate_change": (
            (gesture_successes - baseline_successes) / baseline_successes
            if baseline_successes else None
        ),
        "paired_outcomes": {
            "both": both,
            "baseline_only": baseline_only,
            "gesture_only": gesture_only,
            "neither": neither,
        },
        "exact_mcnemar_p_value": exact_mcnemar_p(baseline_only, gesture_only),
    }

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "paired_rollouts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    (output / "comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
