#!/usr/bin/env python3
"""Train Diffusion Policy variants on the merged 2026-10-22 Panda dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from train_diffusion_variation_baseline import make_parser, run


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "datasets/mujoco_panda_pick_diffusion_20261022"
DEFAULT_REPO_ID = "local/mujoco_panda_pick_diffusion_20261022"


def main() -> None:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument(
        "--mode",
        choices=("baseline", "gesture", "gesture_hand_delta"),
        default="baseline",
    )
    selected, remaining = selector.parse_known_args()

    auxiliary = selected.mode != "baseline"
    motion = selected.mode == "gesture_hand_delta"
    parser = make_parser(auxiliary=auxiliary, motion=motion)
    parser.description = __doc__
    parser.set_defaults(dataset_root=DEFAULT_DATASET, repo_id=DEFAULT_REPO_ID)
    parser.add_argument(
        "--mode",
        choices=("baseline", "gesture", "gesture_hand_delta"),
        default=selected.mode,
        help="Training objective; baseline is the default",
    )
    args = parser.parse_args(remaining)

    if auxiliary:
        from train_diffusion_variation_gesture import install_auxiliary

        if motion and args.hand_delta_weight <= 0:
            parser.error("--hand-delta-weight must be positive for gesture_hand_delta mode")
        run(args, auxiliary=True, install_auxiliary=install_auxiliary)
    else:
        run(args)


if __name__ == "__main__":
    main()
