#!/usr/bin/env python
"""Launch the isolated Panda Push-T demonstration task."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from record_mujoco_panda import main


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _has_option(name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:])


def _add_default(name: str, value: str) -> None:
    if not _has_option(name):
        sys.argv.extend([name, value])


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _add_default("--task-mode", "push_t")
    _add_default("--model-path", str(HERE / "assets/franka_emika_panda/push_t_scene.xml"))
    _add_default("--dataset-root", str(REPO_ROOT / "datasets" / f"mujoco_panda_push_t_{timestamp}"))
    _add_default(
        "--failure-dataset-root",
        str(REPO_ROOT / "datasets" / f"mujoco_panda_push_t_failures_{timestamp}"),
    )
    _add_default("--repo-id", "local/mujoco_panda_push_t")
    _add_default("--task", "Push the red T-shaped rigid body into the white T-shaped goal")

    # Conservative Panda workspace for the Push-T task.
    _add_default("--cube-x-min", "0.46")
    _add_default("--cube-x-max", "0.54")
    _add_default("--cube-y-min", "-0.10")
    _add_default("--cube-y-max", "0.10")
    _add_default("--plate-x-min", "0.57")
    _add_default("--plate-x-max", "0.66")
    _add_default("--plate-y-min", "-0.14")
    _add_default("--plate-y-max", "0.14")
    _add_default("--object-min-separation", "0.14")

    # Curriculum stage one: randomized object yaw, fixed goal orientation.
    _add_default("--cube-yaw-min-deg", "-30")
    _add_default("--cube-yaw-max-deg", "30")
    _add_default("--push-t-success-coverage", "0.90")
    _add_default("--push-t-exit-coverage", "0.85")
    _add_default("--target-dwell-s", "0.5")
    _add_default("--cube-half-size", "0.015")

    main()
