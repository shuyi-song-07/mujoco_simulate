"""Validated Task 3 configuration; all geometry and control limits are explicit."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "task3_dual_arm.json"
DEFAULT_SCENE_PATH = PROJECT_ROOT / "simulation" / "mujoco" / "assets" / "dual_arm" / "dual_arm_scene.xml"


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(source=None) -> dict:
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    supplied = None
    if isinstance(source, dict):
        supplied = source
    elif source is not None and Path(source).resolve() != DEFAULT_CONFIG_PATH.resolve():
        with Path(source).open(encoding="utf-8") as handle:
            supplied = json.load(handle)
    if supplied is not None:
        # Full dataset/checkpoint configurations are frozen research records.
        # Adding an optional UI default later must not alter their config hash.
        complete = {"version", "task", "control", "cooperation", "scene", "recording", "evaluation"}
        config = copy.deepcopy(supplied) if complete.issubset(supplied) else _merge(config, supplied)
    if config["cooperation"]["mode"] != "common_translation":
        raise ValueError("Task 3 v1 supports only common_translation")
    if config["recording"].get("sampling_policy", "fixed_phase_v1") != "fixed_phase_v1":
        raise ValueError("Unsupported recording.sampling_policy")
    positive = [("scene", "timestep"), ("scene", "control_hz"), ("control", "max_ee_step_m"), ("control", "dls_damping"), ("recording", "fps")]
    for section, key in positive:
        if not isinstance(config[section][key], (float, int)) or config[section][key] <= 0:
            raise ValueError(f"{section}.{key} must be positive")
    for key in ("gripper_confidence_min", "gripper_open_confidence_min", "gripper_close_confidence_min",
                "motion_confidence_min", "calibration_gesture_confidence_min", "handedness_confidence_min"):
        if key in config["control"]:
            value = config["control"][key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"control.{key} must be a confidence fraction greater than 0 and at most 1")
    for side in ("left", "right"):
        workspace = config["scene"][f"{side}_workspace"]
        if len(workspace) != 3 or any(len(axis) != 2 or axis[0] >= axis[1] for axis in workspace):
            raise ValueError(f"Invalid {side} workspace")
    sizes = config["scene"].get("table_half_size_xy", [0.5, 0.55])
    if len(sizes) != 2 or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in sizes):
        raise ValueError("scene.table_half_size_xy needs two positive half lengths in meters")
    for name, camera in config["scene"]["cameras"].items():
        projection = camera.get("projection", "perspective")
        span = camera["fovy"]
        if projection not in {"perspective", "orthographic"}:
            raise ValueError(f"Unknown {name} camera projection")
        if not isinstance(span, (int, float)) or not math.isfinite(span) or span <= 0 or (projection == "perspective" and span >= 180):
            raise ValueError(f"Invalid {name} fovy: degrees for perspective, meters for orthographic")
    alpha = config.get("operator_view", {}).get("top_robot_alpha", 1.0)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("operator_view.top_robot_alpha must be between 0 (invisible) and 1 (opaque)")
    return config


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
