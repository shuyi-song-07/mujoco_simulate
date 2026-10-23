#!/usr/bin/env python3
"""Train Diffusion Policy with gesture classification and wrist-delta regression.

The wrist target is the filtered, re-anchored webcam-relative displacement.
Raw wrist samples and mapped robot control are recorded for inspection only.
"""

from train_diffusion_variation_baseline import make_parser, run
from train_diffusion_variation_gesture import install_auxiliary


if __name__ == "__main__":
    args = make_parser(auxiliary=True, motion=True).parse_args()
    if args.hand_delta_weight <= 0:
        raise ValueError("--hand-delta-weight must be positive for this experiment")
    run(args, auxiliary=True, install_auxiliary=install_auxiliary)
