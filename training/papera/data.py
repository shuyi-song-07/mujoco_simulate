"""Read an immutable exploratory pack without crossing temporal discontinuities.

Human labels are returned as auxiliary targets only. They are never part of the
robot observation. Images stay in read-only uint8 memory maps until sampled.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

PACK_VERSION = "papera-exploratory-pack-v1"
ALIGNMENT = "next_control_tick_at_30hz_anchors_v1"
CAMERAS = ("top", "front", "side")
GESTURES = ("ILoveYou", "Open_Palm", "Closed_Fist")
STAGES = ("free_stationary", "free_motion", "contact_hold", "contact_motion",
          "partial_grasp", "dual_grasp_hold", "dual_grasp_motion", "releasing",
          "terminal_success", "terminal_failure")
ANGLE_COLUMNS = (3, 4, 5, 10, 11, 12)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inside(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if root not in path.parents:
        raise ValueError(f"Path must be inside pack: {path}")
    return path


def wrap_actions(action, reference, angle_columns=ANGLE_COLUMNS):
    values = np.asarray(action, dtype=np.float64).copy()
    ref = np.asarray(reference, dtype=np.float64)
    for i in angle_columns:
        values[..., i] = ref[i] + (values[..., i] - ref[i] + np.pi) % (2 * np.pi) - np.pi
    return values


def fit_statistics(episodes, training_keys, *, schema=None):
    """Fit on training episodes only; held-out data cannot affect scaling."""
    states, actions = [], []
    for key in training_keys:
        ep = episodes[key]
        states.append(np.asarray(ep["state"], np.float64))
        actions.append(np.asarray(ep["action"], np.float64)[ep["action_mask"].astype(bool)])
    state = np.concatenate(states)
    action = np.concatenate(actions)
    if not len(action) or not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Cannot normalize missing/nonfinite training targets")
    from tasks.data_schema import schema_for
    layout = schema_for(schema)
    if state.ndim != 2 or state.shape[1] != layout.state_dim or action.ndim != 2 or action.shape[1] != layout.action_dim:
        raise ValueError('Normalization input differs from the declared robot schema')
    angles = tuple(offset + i for offset in range(0, layout.action_dim, 7) for i in (3,4,5))
    reference = np.zeros(layout.action_dim, np.float64)
    for i in angles:
        reference[i] = np.arctan2(np.sin(action[:, i]).mean(), np.cos(action[:, i]).mean())
    unwrapped = wrap_actions(action, reference, angles)
    return {"version": "train-only-zscore-circular-rpy-v1", "training_keys": list(training_keys),
            "state_count": len(state), "action_count": len(action), "angle_columns": list(angles),
            "action_angle_reference": reference.tolist(),
            "state_mean": state.mean(0).tolist(), "state_std": np.maximum(state.std(0), .01).tolist(),
            "action_mean": unwrapped.mean(0).tolist(), "action_std": np.maximum(unwrapped.std(0), .01).tolist(),
            "std_floor": .01, "fit_split": "train"}


class Normalizer:
    def __init__(self, stats):
        self.stats = stats
        self.reference = np.asarray(stats["action_angle_reference"])
        self.state_mean = np.asarray(stats["state_mean"])
        self.state_std = np.asarray(stats["state_std"])
        self.action_mean = np.asarray(stats["action_mean"])
        self.action_std = np.asarray(stats["action_std"])
        self.angle_columns = tuple(stats.get('angle_columns', ANGLE_COLUMNS))
        expected_angles = (3,4,5) if self.reference.shape == (7,) else ANGLE_COLUMNS
        if self.angle_columns != expected_angles or self.reference.shape not in ((7,), (14,)):
            raise ValueError('Unsupported action dimensions or angle columns in normalization')

    def encode_state(self, value):
        return ((value - self.state_mean) / self.state_std).astype(np.float32)

    def encode_action(self, value):
        return ((wrap_actions(value, self.reference, self.angle_columns) - self.action_mean) / self.action_std).astype(np.float32)

    def decode_action(self, value):
        action = np.asarray(value) * self.action_std + self.action_mean
        action[..., self.angle_columns] = (action[..., self.angle_columns] + np.pi) % (2 * np.pi) - np.pi
        return action.astype(np.float32)

