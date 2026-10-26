"""Validated Task 3 configuration; all geometry and control limits are explicit."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from .training_camera import validate_recording_cameras

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "simulation_base.json"
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
    if config["recording"].get("sampling_policy", "fixed_phase_v1") not in {"fixed_phase_v1", "raw_fixed_phase_v2"}:
        raise ValueError("Unsupported recording.sampling_policy")
    if config["recording"].get("sampling_policy") == "raw_fixed_phase_v2" and config["recording"].get("schema_version") != "task3-papera-v2":
        raise ValueError("Raw timeline requires the explicit Paper A v2 schema")
    validate_recording_cameras(config)
    gripper = config.get("gripper_control", {})
    if not isinstance(gripper, dict) or gripper.get("mode", "legacy_binary") not in {"legacy_binary", "continuous_independent"}:
        raise ValueError("Unsupported gripper_control.mode")
    if gripper.get("mode") == "continuous_independent":
        speed = gripper.get("max_width_speed_mps")
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0:
            raise ValueError("gripper_control.max_width_speed_mps must be positive and finite")
    contact = config.get("contact_control", {})
    level_gestures = config.get('gesture_control', {}).get('gripper_trigger_policy') == 'palm_fist_level_v2'
    if contact.get("enabled"):
        if gripper.get("mode") != "continuous_independent":
            raise ValueError("contact_control requires continuous_independent grippers")
        for key in ("support_weight_fraction", "support_height_tolerance_m", "support_stable_seconds", "release_width_epsilon_m", "max_tilt_rad"):
            value = contact.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"contact_control.{key} must be positive and finite")
        if "support_penetration_tolerance_m" in contact:
            value = contact["support_penetration_tolerance_m"]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("contact_control.support_penetration_tolerance_m must be positive and finite")
    if contact.get("control_protocol") == "gesture-rate-gripper-v3":
        gesture = config.get("gesture_control", {})
        if gesture.get("gripper_trigger_policy", "level_v1") not in {"level_v1", "palm_fist_edge_v1", "palm_fist_edge_v2", "palm_fist_level_v2"}:
            raise ValueError("Unsupported gesture_control.gripper_trigger_policy")
        for key in ("open_score_min", "close_score_min", "motion_score_min"):
            value = gesture.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"gesture_control.{key} must be a confidence fraction in (0, 1]")
        duration = gesture.get("mode_confirm_ms")
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0 or (not level_gestures and duration == 0):
            raise ValueError("gesture_control.mode_confirm_ms must be finite; zero is allowed only for palm_fist_level_v2")
        if level_gestures and contact.get('release_confirm_seconds') != 0:
            raise ValueError('palm_fist_level_v2 uses mode_confirm_ms only; release_confirm_seconds must be zero')
    if "rigid_control" in config:
        if not contact.get("enabled"):
            raise ValueError("rigid_control requires contact_control")
        for key in ("max_yaw_rad", "max_yaw_step_rad", "axial_conflict_m", "ik_position_tolerance_m", "ik_angle_tolerance_rad"):
            value = config["rigid_control"].get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"rigid_control.{key} must be positive and finite")
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
    randomization = config['scene']['randomization']
    for key in ('object_xy_range_m', 'target_xy_range_m'):
        values = randomization.get(key)
        if (not isinstance(values, (list, tuple)) or len(values) != 2 or
                any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)):
            raise ValueError(f'scene.randomization.{key} requires two finite nonnegative half-ranges')
    yaw = randomization.get('object_yaw_range_rad')
    if type(yaw) not in (int, float) or not math.isfinite(yaw) or yaw < 0:
        raise ValueError('scene.randomization.object_yaw_range_rad must be finite and nonnegative')
    if type(config['scene'].get('blockers_follow_initial_pose', False)) is not bool:
        raise ValueError('scene.blockers_follow_initial_pose must be boolean')
    sizes = config["scene"].get("table_half_size_xy", [0.5, 0.55])
    paper = config["scene"].get("paper_object")
    if paper is not None:
        if paper.get("kind") not in {"tray", "blocked_tray"}:
            raise ValueError("Unsupported scene.paper_object.kind")
        for key in ("pan_width_m", "pan_length_m", "handle_width_m", "handle_length_m", "bottom_thickness_m", "wall_thickness_m"):
            value = paper.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"scene.paper_object.{key} must be positive and finite")
        if paper["bottom_thickness_m"] >= 2*config["scene"]["object_half_size"][2] or 2*paper["wall_thickness_m"] >= min(paper["pan_width_m"], paper["pan_length_m"]):
            raise ValueError("Tray wall/bottom leaves no interior")
    if len(sizes) != 2 or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in sizes):
        raise ValueError("scene.table_half_size_xy needs two positive half lengths in meters")
    platform = config['scene'].get('placement_platform')
    if platform is not None:
        if not paper or paper['kind'] != 'tray':
            raise ValueError('placement_platform requires the tray scene')
        if set(platform) != {'height_m', 'extra_size_xy_m', 'containment_tolerance_m'}:
            raise ValueError('Unknown or missing placement_platform parameters')
        for key in ('height_m', 'containment_tolerance_m'):
            value = platform[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'placement_platform.{key} must be positive and finite')
        extra = platform['extra_size_xy_m']
        if not isinstance(extra, list) or len(extra) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in extra):
            raise ValueError('placement_platform.extra_size_xy_m needs two nonnegative lengths')
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
