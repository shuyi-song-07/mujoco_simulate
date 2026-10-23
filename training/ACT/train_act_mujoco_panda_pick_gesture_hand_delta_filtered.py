#!/usr/bin/env python3
"""Train the controlled ACT variant using EMA-filtered wrist displacement."""

from pathlib import Path

try:
    from .train_act_variation_baseline import make_parser, run
    from .train_act_variation_gesture import install_auxiliary
except ImportError:
    from train_act_variation_baseline import make_parser, run
    from train_act_variation_gesture import install_auxiliary


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "datasets/mujoco_panda_pick_diffusion_20261022_hand_delta_ema"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/act_mujoco_panda_pick_gesture_hand_delta_filtered"


def main() -> None:
    parser = make_parser(auxiliary=True, motion=True)
    parser.set_defaults(
        dataset_root=DEFAULT_DATASET,
        repo_id="local/mujoco_panda_pick_diffusion_20261022_hand_delta_ema",
        output_dir=DEFAULT_OUTPUT,
        steps=100_000,
        batch_size=8,
        num_workers=4,
        eval_split=0.1,
        eval_steps=5000,
        save_freq=10_000,
        chunk_size=100,
        seed=1000,
        learning_rate=1e-5,
        gesture_weight=0.1,
        hand_delta_weight=0.1,
        hand_delta_scale=0.03,
    )
    run(parser.parse_args(), auxiliary=True, install_auxiliary=install_auxiliary)


if __name__ == "__main__":
    main()
