"""Validated settings for the single evidence-based curation policy."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

VERSION = "papera-intent-aware"
CATEGORY_POLICY = "recorded_category_no_score_v1"
FPS = 30.
CONTROL_HZ = 60.
MAXIMUM_FRAME_GAP_S = .05
MAXIMUM_NEXT_CONTROL_GAP_S = .03
GROUPS = ("state_joint_rad", "state_aperture_m", "action_position_m",
          "action_rotation_rad", "action_aperture_m")


def effective_config(config):
    """Explanation text never changes the semantic configuration fingerprint."""
    if isinstance(config, dict):
        return {k: effective_config(v) for k, v in config.items() if k != "_help"}
    if isinstance(config, list):
        return [effective_config(v) for v in config]
    return copy.deepcopy(config)


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("裁剪参数必须是JSON对象")
    result = copy.deepcopy(config)
    value = effective_config(result)
    schema = {
        "version": VERSION,
        "sa": {"window_frames": (int, 1, 3000), "thresholds": {
            "state_joint_rad": (float, 1e-9, math.pi),
            "state_aperture_m": (float, 1e-9, .08),
            "action_position_m": (float, 1e-9, 1.),
            "action_rotation_rad": (float, 1e-9, math.pi),
            "action_aperture_m": (float, 1e-9, .08)}},
        "intent": {"stable_frames": (int, 1, 60), "min_distinct_inputs": (int, 1, 60),
                   "recenter_search_frames": (int, 1, 300)},
        "context": {"before_frames": (int, 0, 300), "after_frames": (int, 0, 300)},
        "waits": {**{k: (float, 0., 600.) for k in
                  ("minimum_prefix_s", "minimum_middle_s", "minimum_recenter_s")},
                  "same_intent_frames": (int, 2, 600), "changed_intent_frames": (int, 2, 600),
                  "post_release_frames": (int, 0, 300)},
        "protection": {"height_threshold_m": (float, .001, 3.),
                       "settle_seconds": (float, 1e-6, 10.),
                       "aperture_stability_m": (float, 1e-9, .08),
                       "relative_position_stability_m": (float, 1e-9, 1.),
                       "relative_rotation_stability_rad": (float, 1e-9, math.pi),
                       "open_width_tolerance_m": (float, 1e-9, .02)},
        "physical_motion": {"position_m": (float, 1e-9, 1.),
                            "rotation_rad": (float, 1e-9, math.pi),
                            "joint_rad": (float, 1e-9, math.pi),
                            "aperture_m": (float, 1e-9, .08),
                            "linear_speed_mps": (float, 1e-9, 10.),
                            "angular_speed_radps": (float, 1e-9, 100.),
                            "joint_speed_radps": (float, 1e-9, 100.)},
        "quality": {"excluded_episodes": [{"episode_key": str, "reason": str}]},
        "alignment": {"time_fallback": {"enabled": bool, "max_control_periods": (int, 1, 1)}},
        "auxiliary": {"maximum_age_ms": (float, 1., 1000.),
                      "minimum_identity_score": (float, 0., 1.),
                      "image_margin": (float, 0., .49),
                      "category_minimum_scores": {k: (float, 0., 1.) for k in
                                                   ("ILoveYou", "Open_Palm", "Closed_Fist")},
                      "unknown_task_geometry": bool},
    }

    def check(actual, expected, path="config"):
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(actual) != set(expected):
                raise ValueError(f"{path}字段必须为 {', '.join(expected)}")
            for key, sub in expected.items():
                check(actual[key], sub, f"{path}.{key}")
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) > 10000:
                raise ValueError(f"{path}必须是列表，最多10000项")
            for i, item in enumerate(actual):
                check(item, expected[0], f"{path}[{i}]")
        elif expected is str:
            if not isinstance(actual, str) or not actual.strip() or len(actual) > 8000:
                raise ValueError(f"{path}必须是非空文本，最多8000字符")
        elif isinstance(expected, tuple):
            kind, lower, upper = expected
            correct = type(actual) is int if kind is int else type(actual) in (int, float)
            if not correct or not math.isfinite(actual) or not lower <= actual <= upper:
                raise ValueError(f"{path}必须是{kind.__name__}，范围[{lower}, {upper}]")
        elif expected is bool:
            if type(actual) is not bool:
                raise ValueError(f"{path}必须是布尔值")
        elif actual != expected:
            raise ValueError(f"{path}必须为{expected}")

    # Historical frozen manifests lack this key; do not inject defaults into
    # them, since their configuration fingerprints must remain reproducible.
    if isinstance(value.get('sa'), dict) and 'minimum_static_frames' in value['sa']:
        schema['sa']['minimum_static_frames'] = (int, 1, 3000)
    if isinstance(value.get('waits'), dict) and 'minimum_transition_s' in value['waits']:
        schema['waits']['minimum_transition_s'] = (float, 0., 600.)
    check(value, schema)
    if value["intent"]["min_distinct_inputs"] > value["intent"]["stable_frames"]:
        raise ValueError("min_distinct_inputs不能大于stable_frames")
    if value["intent"]["recenter_search_frames"] < value["intent"]["stable_frames"]:
        raise ValueError("recenter_search_frames不能小于stable_frames")
    excluded = [e["episode_key"] for e in value["quality"]["excluded_episodes"]]
    if len(set(excluded)) != len(excluded):
        raise ValueError("排除示范的episode_key不能重复")
    return result


def excluded_episode_reason(config, episode_key):
    return next((e["reason"] for e in config["quality"]["excluded_episodes"]
                 if e["episode_key"] == episode_key), None)


def load_config(path=None):
    path = Path(path) if path is not None else Path(__file__).resolve().parents[2] / "config/curation.json"
    return validate_config(json.loads(path.read_text(encoding="utf-8")))
