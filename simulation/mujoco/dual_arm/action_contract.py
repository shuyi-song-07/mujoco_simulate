"""Versioned Task3 units and explicit legacy adapters (no physics side effects)."""
from __future__ import annotations

import numpy as np

V1 = "task3-v1"
V2 = "task3-papera-v2"
GRASP_HOLD_RELEASE_V1 = "grasp-hold-release-v1"
GESTURE_GRASP_RELEASE_V2 = "gesture-grasp-release-v2"
GESTURE_RATE_GRIPPER_V3 = "gesture-rate-gripper-v3"
INTERACTION_MODES = ("XY_HOLD", "Z_HOLD", "Z_APPROACH", "PINCH", "RELEASE", "OPENING", "HOLD")
RATE_INTERACTION_MODES = ("XY_HOLD", "Z_OPEN", "Z_CLOSE", "HOLD")
SIDES = ("left", "right")
STATE_NAMES = [f"{side}_{name}" for side in SIDES for name in
               [*(f"joint{i}" for i in range(1, 8)), "finger_width"]]
ACTION_NAMES_V1 = [f"{side}_{name}" for side in SIDES for name in
                   ("x", "y", "z", "roll", "pitch", "yaw", "gripper")]
ACTION_NAMES_V2 = [name.replace("_gripper", "_gripper_width_m") for name in ACTION_NAMES_V1]


def vector(value, size=14):
    try:
        raw = np.asarray(value)
        if raw.shape != (size,) or raw.dtype.kind not in "iuf":
            raise ValueError
        result = raw.astype(np.float64)
        if not np.isfinite(result).all():
            raise ValueError
        return result
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"Expected finite numeric {size}D vector") from error


def validate_action(action, version, maximum_widths):
    if version != V2:
        raise ValueError("action_schema_mismatch")
    result = vector(action)
    limits = vector(maximum_widths, 2)
    if np.any(limits <= 0) or np.any(result[[6, 13]] < 0) or np.any(result[[6, 13]] > limits + 1e-8):
        raise ValueError("gripper_width_out_of_range")
    return result


def validate_release_intent(value):
    """Separate supervision/action head; numeric width is not release intent."""
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(v) is not bool for v in value):
        raise ValueError("release_intent_must_be_two_booleans")
    return list(value)


def validate_interaction_modes(value, *, protocol=GESTURE_GRASP_RELEASE_V2):
    allowed = RATE_INTERACTION_MODES if protocol == GESTURE_RATE_GRIPPER_V3 else INTERACTION_MODES
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(v) is not str or v not in allowed for v in value)):
        raise ValueError("interaction_modes_must_be_two_known_modes")
    return list(value)


def legacy_to_metric(action, maximum_widths, *, version):
    """Unit conversion only. Never silently threshold model predictions here."""
    if version != V1:
        raise ValueError("legacy_schema_mismatch")
    result = vector(action)
    if np.any(result[[6, 13]] < 0) or np.any(result[[6, 13]] > 255):
        raise ValueError("legacy_gripper_out_of_range")
    result[[6, 13]] *= vector(maximum_widths, 2) / 255.
    return result


def metric_to_legacy(action, maximum_widths, *, version):
    result = validate_action(action, version, maximum_widths).copy()
    result[[6, 13]] *= 255. / vector(maximum_widths, 2)
    return result


def describe(maximum_widths, *, continuous_gripper=False):
    return {"version": V2, "legacy_dataset_version": V1, "arm_order": list(SIDES),
            "coordinate_frame": "task_frame", "rotation_convention": "Rz(yaw) @ Ry(pitch) @ Rx(roll)",
            "action_names": ACTION_NAMES_V2, "action_units": ["m", "m", "m", "rad", "rad", "rad", "m"] * 2,
            "state_names": STATE_NAMES, "state_units": ["rad"] * 7 + ["m"] + ["rad"] * 7 + ["m"],
            "gripper_max_width_m": list(map(float, maximum_widths)),
            "normalized_width": "requested_width_m / model_max_total_width_m",
            "capabilities": {"continuous_gripper": continuous_gripper, "independent_gripper": continuous_gripper, "free_orientation": False},
            "execution_scope": "fixed_arm_gripper_validation" if continuous_gripper else "legacy_task",
            "reserved_hand_fields": {"measurement_id": None, "capture_time_ms": None,
                "left": {"image_landmarks_21": None, "world_landmarks_21": None, "coordinate_type": None,
                         "identity": None, "valid": None, "aperture_normalized": None},
                "right": {"image_landmarks_21": None, "world_landmarks_21": None, "coordinate_type": None,
                          "identity": None, "valid": None, "aperture_normalized": None}}}


def require_legacy_checkpoint(provenance):
    """Reject same-dimensional but differently scaled checkpoints before load."""
    task = provenance.get("task3", {}) if isinstance(provenance, dict) else {}
    protocols = (task.get("control_interface_version"), task.get("control_protocol"),
                 task.get("policy_contract_version"),
                 (task.get("config", {}).get("contact_control", {}) or {}).get("control_protocol"))
    if any(isinstance(p, str) and p.startswith("gesture-edge-mode-policy-") for p in protocols):
        raise ValueError("gesture-edge-mode-policy checkpoints are not supported by the legacy ACT checkpoint loader")
    if any(p in {GESTURE_RATE_GRIPPER_V3, "gesture-rate-mode-policy-v3"} for p in protocols):
        raise ValueError("gesture-rate-mode-policy-v3 is not supported by the legacy ACT checkpoint loader")
    if task.get("schema_version") != V1:
        raise ValueError("Checkpoint schema mismatch: the current ACT entry requires explicit task3-v1 metadata")
    expected_units = ["m", "m", "m", "rad", "rad", "rad", "0..255"] * 2
    if "action_units" in task and task["action_units"] != expected_units:
        raise ValueError("Checkpoint gripper/action units do not match task3-v1")


def legacy_policy_filter(action, previous, status, max_step_m=0.008):
    """Explicit v1 compatibility adapter; v2 execution never calls this filter."""
    value, previous = vector(action), vector(previous)
    if not np.isfinite(max_step_m) or max_step_m <= 0:
        raise ValueError("Invalid policy step limit")
    filtered = value.copy()
    deltas = [value[o:o+3]-previous[o:o+3] for o in (0, 7)]
    coupled = status.get("task_state") in {"DUAL_GRASPED", "PLACEMENT"}
    if coupled:
        common = .5 * (deltas[0]+deltas[1])
        deltas = [common.copy(), common.copy()]
    clamped = 0
    for offset, delta in zip((0, 7), deltas):
        norm = np.linalg.norm(delta)
        if norm > max_step_m:
            delta = delta * max_step_m/norm
            clamped += 1
        filtered[offset:offset+3] = previous[offset:offset+3] + delta
        filtered[offset+3:offset+6] = previous[offset+3:offset+6]
    filtered[[6, 13]] = 255. if float(np.clip(value[[6, 13]], 0, 255).mean()) >= 127.5 else 0.
    return filtered.astype(np.float32), {"adapter": "explicit_task3_v1_policy", "motion_clamp_count": clamped,
        "cooperative_filter": coupled, "orientation_prediction_error_rad": float(np.linalg.norm(
            value[[3, 4, 5, 10, 11, 12]]-previous[[3, 4, 5, 10, 11, 12]]))}
