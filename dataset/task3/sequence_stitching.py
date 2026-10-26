"""Certify retimed Dataset sequences without changing robot targets or ACT.

Automatic and manual cuts are merely proposals.  Every interior cut is checked
against all original controls, physical observations and causal targets again.
The two clocks are deliberately separate: original_time_s is immutable, while
training_time_s is the retained observation index divided by the recorded 30 Hz.
"""
from __future__ import annotations

import copy
import math
from collections import Counter

import numpy as np

from dataset.task3.curation_config import effective_config, load_config, validate_config
from dataset.task3.frame_evidence import _rotation_distance

VERSION = "papera-safe-sequence-1"
SIDES = ("left", "right")
ROW_FIELDS = ("training_frame_index", "training_time_s", "train_sequence_id",
              "sequence_frame_index", "sequence_time_s", "causal_target_valid",
              "causal_target_reason", "stitch_gap_before", "virtual_clock_valid")
DESCRIPTOR = {
    "version": VERSION,
    "training_frame_index": "zero-based retained observation index within episode",
    "training_time_s": "training_frame_index / frozen observation fps; virtual, not simulation time",
    "train_sequence_id": "zero-based sequence; may cross certified gaps only",
    "sequence_frame_index": "zero-based retained observation index within train_sequence_id",
    "sequence_time_s": "sequence_frame_index / frozen observation fps",
    "causal_action": "unchanged target from next legal original 60Hz control tick",
    "raw_time": "original_time_s and all original source IDs are preserved",
    "sampler": "data contract only; ACT sampler adaptation remains required",
    "unavailable_clock": "unproven frozen 30Hz rows have null virtual time and isolated hard sequences",
}
TRACKING_PAUSES = {"hand_lost", "hand_near_frame_edge", "tracking_recovery",
                   "camera_stalled", "stale_recognition", "command_timeout",
                   "transport_recovery"}
ALLOWED_EVENTS = {"request_accepted", "accepted_intent_changed", "pause_changed",
                  "raw_request_received", "raw_request_queued", "raw_state",
                  "raw_request_dropped"}
CONTACT_FIELDS = ("fingers", "any_finger", "double_finger", "qualified", "stable")
PHYSICAL_EVENTS = {"contact_changed", "grasp_stable_changed", "task_state_changed",
                   "gripper_open_executed", "gripper_close_executed", "gripper_aborted"}


def _number(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def _array(value, shape):
    try:
        value = np.asarray(value, dtype=float)
        return value if value.shape == shape and np.isfinite(value).all() else None
    except (TypeError, ValueError):
        return None


def _facts(row):
    return row.get("evidence") or {}


def _action(facts):
    values = []
    features = facts.get("features") or {}
    for side in SIDES:
        for key, shape in (("target_position", (3,)), ("target_rpy", (3,)),
                           ("target_aperture", (1,))):
            field = features.get(f"{side}.{key}")
            vector = _array(field[1], shape) if isinstance(field, (list, tuple)) and len(field) == 2 else None
            if vector is None:
                return None
            values.extend(vector.tolist())
    return np.asarray(values)


def _state(facts):
    values = []
    for side in SIDES:
        for key, shape in (("joints", (7,)), ("actual_aperture", (1,))):
            field = facts.get("features", {}).get(f"{side}.{key}")
            vector = _array(field[1], shape) if isinstance(field, (list, tuple)) and len(field) == 2 else None
            if vector is None:
                return None
            values.extend(vector.tolist())
    return np.asarray(values)


def _causal_target(row):
    """Validate the recorded next-tick identity, never a compressed array index."""
    target = row.get("next_control_target")
    if target is None:
        return False, "no_next_control_target"
    current, following = _facts(row), _facts(row).get("next_control")
    if not isinstance(following, dict):
        return False, "missing_next_control_evidence"
    source = following.get("source") or {}
    action = _array(target.get("action"), (14,))
    actual = _action(following)
    if action is None or actual is None or not np.array_equal(action, actual):
        return False, "next_target_action_mismatch"
    if type(row.get("source_control_tick")) is not int:
        return False, "missing_observation_control_tick"
    if (target.get("source_tick") != row["source_control_tick"] + 1
            or following.get("tick") != target.get("source_tick")):
        return False, "next_target_not_next_original_tick"
    if (source.get("valid") is not True or source.get("available") is not True
            or source.get("hard_boundary") or source.get("reasons")):
        return False, "next_target_source_invalid"
    for key in ("control_epoch",):
        if row.get(key) is None or target.get(key) != row[key] or following.get(key) != row[key]:
            return False, "next_target_epoch_mismatch"
    if following.get("session") != row.get("session"):
        return False, "next_target_session_mismatch"
    for key in ("action_event_id", "action_measurement_id"):
        if target.get(key) != following.get(key):
            return False, "next_target_causal_id_mismatch"
    line, following_line = row.get("source_line"), following.get("source_line")
    if (type(line) is not int or type(following_line) is not int
            or following_line <= line or target.get("source_line") != following_line):
        return False, "next_target_source_order_invalid"
    now, future = _number(row.get("original_time_s")), _number(following.get("sim_time"))
    if (now is None or future is None or not 0 < future - now <= .03
            or target.get("sim_time") != following.get("sim_time")
            or target.get("alignment") != "next_observed_control_tick_after_post_step_snapshot"):
        return False, "next_target_clock_invalid"
    if current.get("source", {}).get("valid") is not True:
        return False, "observation_source_invalid"
    return True, "verified_next_original_control_tick"


def _feature_delta(first, second, key):
    a, b = first.get("features", {}).get(key), second.get("features", {}).get(key)
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) != 2 or len(b) != 2:
        return None
    kind = a[0]
    try:
        av, bv = np.asarray(a[1], dtype=float), np.asarray(b[1], dtype=float)
        if not np.isfinite(av).all() or not np.isfinite(bv).all() or a[0] != b[0]:
            return None
        if kind == "rotation":
            from dataset.task3.frame_evidence import _rotation_matrix
            av, bv = _rotation_matrix(av), _rotation_matrix(bv)
            if av is None or bv is None:
                return None
            return float(_rotation_distance(av, bv))
        if av.shape != bv.shape:
            return None
        expected = (7,) if kind == "joint" else (1,) if kind == "aperture" else (3,)
        if av.shape != expected:
            return None
        delta = av - bv
        if kind == "rpy":
            return float(np.linalg.norm((delta + np.pi) % (2 * np.pi) - np.pi))
        return float(np.max(np.abs(delta)) if kind in ("joint", "aperture") else np.linalg.norm(delta))
    except (ValueError, TypeError):
        return None


def _features_config(cfg):
    p, a = cfg["physical_motion"], cfg["sa"]["thresholds"]
    fields = {"object.position": p["position_m"], "object.rotation": p["rotation_rad"]}
    for side in SIDES:
        fields.update({f"{side}.{key}": limit for key, limit in (
            ("joints", p["joint_rad"]), ("actual_aperture", p["aperture_m"]),
            ("actual_position", p["position_m"]), ("actual_rotation", p["rotation_rad"]),
            ("target_position", a["action_position_m"]), ("target_rpy", a["action_rotation_rad"]),
            ("target_aperture", a["action_aperture_m"]))})
    return fields


def _fact_failures(facts, cfg, *, require_rotation=False):
    """Check facts themselves, independently of curation tags/keep reasons."""
    reasons = []
    source, motion = facts.get("source") or {}, facts.get("motion") or {}
    if (source.get("valid") is not True or source.get("available") is not True
            or source.get("hard_boundary") or source.get("reasons") or facts.get("invalid")):
        reasons.append("source_missing_or_boundary")
    if source.get("observation_period_s") != 1 / 30 or source.get("control_period_s") != 1 / 60:
        reasons.append("frozen_sampling_contract_missing")
    if motion.get("available") is not True:
        reasons.append("physical_motion_evidence_missing")
    if motion.get("moving") is not False:
        reasons.append("actual_motion")
    obj = facts.get("object") or {}
    for label, value, shape, limit in (
            ("object_linear_velocity", obj.get("linear_velocity"), (3,), cfg["physical_motion"]["linear_speed_mps"]),
            ("object_angular_velocity", obj.get("angular_velocity"), (3,), cfg["physical_motion"]["angular_speed_radps"]),
            ("ee_velocity", motion.get("ee_speeds_mps"), (2,), cfg["physical_motion"]["linear_speed_mps"])):
        velocity = _array(value, shape)
        if velocity is None:
            reasons.append("velocity_evidence_missing:" + label)
        elif (np.max(np.abs(velocity)) if label == "ee_velocity" else np.linalg.norm(velocity)) > limit:
            reasons.append("actual_velocity:" + label)
    for key in _features_config(cfg):
        if key == "object.rotation" and not require_rotation:
            continue
        if _feature_delta(facts, facts, key) is None:
            reasons.append("physical_or_target_feature_missing:" + key)
    execution = facts.get("execution") or {}
    if execution.get("available") is not True:
        reasons.append("execution_evidence_missing")
    for side in SIDES:
        item = execution.get("sides", {}).get(side, {})
        if item.get("pending_authorization") is not False:
            reasons.append("pending_gripper_authorization:" + side)
        if item.get("release_confirmation_pending") is not False:
            reasons.append("pending_release_confirmation:" + side)
        if item.get("pending") is not False:
            reasons.append("pending_execution:" + side)
        gripper = facts.get("grippers", {}).get(side, {})
        for name in ("actual_width_m", "command_width_m", "goal_width_m"):
            if _number(gripper.get(name)) is None:
                reasons.append("gripper_width_evidence_missing:" + side)
        if any(gripper.get(key) is True for key in ("executed_open", "executed_close", "aborted")):
            reasons.append("gripper_execution_event:" + side)
        contact = facts.get("contact", {}).get(side, {})
        if (not isinstance(contact.get("fingers"), list) or len(contact["fingers"]) != 2
                or any(type(x) is not bool for x in contact["fingers"])
                or any(type(contact.get(key)) is not bool for key in CONTACT_FIELDS[1:])):
            reasons.append("contact_evidence_missing:" + side)
    if facts.get("paused") is None:
        reasons.append("pause_evidence_missing")
    elif facts.get("paused") is True and facts.get("pause_reason") not in TRACKING_PAUSES:
        reasons.append("safety_pause")
    return sorted(set(reasons))


def _event_failures(events, anchor, cfg, previous_line, current_line):
    failures = []
    for event in events:
        kind = event.get("kind")
        if kind not in ALLOWED_EVENTS:
            failures.append("causal_event:" + str(kind))
        elif kind == "pause_changed" and event.get("after") is True and event.get("reason") not in TRACKING_PAUSES:
            failures.append("safety_pause_event")
        elif kind == "request_accepted" and event.get("command") != "continuous_input":
            failures.append("non_motion_control_request:" + str(event.get("command")))
        elif kind == "request_accepted":
            execution = event.get("execution") or {}
            if execution.get("available") is not True:
                failures.append("accepted_request_execution_unknown")
            if any(execution.get("sides", {}).get(side, {}).get("pending") is not False for side in SIDES):
                failures.append("accepted_request_pending_execution")
            target, accepted = _action(anchor), _array(event.get("accepted_target"), (14,))
            if target is None or accepted is None:
                failures.append("accepted_request_target_missing")
            else:
                for j, side in enumerate(SIDES):
                    delta = accepted[j*7:j*7+7] - target[j*7:j*7+7]
                    for key, distance in (("action_position_m", np.linalg.norm(delta[:3])),
                                          ("action_rotation_rad", np.linalg.norm((delta[3:6]+np.pi) % (2*np.pi)-np.pi)),
                                          ("action_aperture_m", abs(delta[6]))):
                        if distance >= cfg["sa"]["thresholds"][key]:
                            failures.append("accepted_request_target_excursion:" + side + "." + key)
            if execution.get("strategy") in ("palm_fist_edge_v1", "palm_fist_edge_v2"):
                for side in SIDES:
                    edge, original = event.get("gripper_edges", {}).get(side), anchor.get("grippers", {}).get(side, {})
                    if not isinstance(edge, dict):
                        failures.append("accepted_request_edge_missing:" + side)
                        continue
                    for key, factual in (("baseline_mode", "baseline_mode"), ("baseline_source", "baseline_source"),
                                         ("authorized_mode", "authorized_mode"), ("edge_count", "edge_count"),
                                         ("event_id", "edge_event_id")):
                        if key not in edge or edge[key] != original.get(factual):
                            failures.append("accepted_request_execution_transition:" + side + "." + key)
        elif kind == "raw_request_received" and event.get("command") not in ("continuous_input", "telemetry"):
            failures.append("non_motion_request_received:" + str(event.get("command")))
        elif kind == "raw_request_dropped" and not (event.get("reason") == "superseded_telemetry"
                                                    and event.get("command") in ("continuous_input", "telemetry")):
            failures.append("dropped_request_requires_review")
        if type(event.get("source_line")) is not int:
            failures.append("event_source_missing")
        elif (type(previous_line) is not int or type(current_line) is not int
              or not previous_line < event["source_line"] <= current_line):
            failures.append("event_source_outside_interval")
        if any(event.get(key) != anchor.get(key) for key in ("session", "control_epoch")):
            failures.append("event_source_context_mismatch")
    return failures


def _gap_check(rows, left, right, cfg):
    """Check one whole proposed gap, including both retained boundary states."""
    failures, details, maxima = [], [], {}
    start, end = rows[left], rows[right]
    anchor = _facts(start)
    all_facts = [anchor]
    owner_indices = [left]
    for i in range(left, right + 1):
        row = rows[i]
        if any(row.get(key) != start.get(key) for key in
               ("source_episode", "session", "episode_index", "control_epoch", "task_segment_id")):
            failures.append("source_or_epoch_boundary")
        if row.get("raw_frame_index") != start.get("raw_frame_index", -1) + i - left:
            failures.append("missing_original_frame")
        if row.get("frame_role") == "INVALID" or _facts(row).get("source", {}).get("valid") is not True:
            failures.append("invalid_original_frame")
        reasons = _fact_failures(_facts(row), cfg, require_rotation=True)
        for key, factual, shape in (("state_recorded", _state(_facts(row)), (16,)),
                                    ("action_recorded", _action(_facts(row)), (14,))):
            recorded = _array(row.get(key), shape)
            if recorded is None or factual is None or not np.array_equal(recorded, factual):
                reasons.append("recorded_array_evidence_mismatch:" + key)
        if reasons:
            details.append({"raw_frame_index": row["raw_frame_index"], "reasons": reasons})
            failures.extend(reasons)
        if i == left:
            continue
        interval = _facts(row).get("interval") or {}
        controls = interval.get("controls")
        if not isinstance(controls, list) or not controls:
            failures.append("missing_intermediate_controls")
            continue
        if (controls[-1].get("tick") != row.get("source_control_tick")
                or controls[-1].get("source_line") != row.get("source_line")):
            failures.append("image_control_alignment_mismatch")
        event_failures = _event_failures(interval.get("events") or [], anchor, cfg,
                                         rows[i-1].get("source_line"), row.get("source_line"))
        if event_failures:
            details.append({"raw_frame_index": row["raw_frame_index"], "reasons": sorted(set(event_failures))})
            failures.extend(event_failures)
        for control in controls:
            reasons = _fact_failures(control, cfg)
            if reasons:
                details.append({"raw_frame_index": row["raw_frame_index"],
                                "control_tick": control.get("tick"), "reasons": reasons})
                failures.extend(reasons)
            all_facts.append(control)
            owner_indices.append(i)
    limits = _features_config(cfg)
    for i, (facts, owner) in enumerate(zip(all_facts, owner_indices)):
        if i:
            prev = all_facts[i - 1]
            t, p = _number(facts.get("sim_time")), _number(prev.get("sim_time"))
            if (type(prev.get("tick")) is not int or facts.get("tick") != prev["tick"] + 1
                    or t is None or p is None or not 0 < t - p <= .03
                    or type(facts.get("source_line")) is not int
                    or type(prev.get("source_line")) is not int
                    or facts["source_line"] <= prev["source_line"]):
                failures.append("control_tick_or_clock_discontinuity")
            if any(facts.get(k) != anchor.get(k) for k in ("session", "control_epoch")):
                failures.append("control_epoch_or_session_change")
            for side in SIDES:
                g, old = facts.get("grippers", {}).get(side, {}), prev.get("grippers", {}).get(side, {})
                for key in ("baseline_mode", "baseline_source", "authorized_mode", "edge_event_id", "edge_count"):
                    if g.get(key) != old.get(key):
                        failures.append("execution_state_transition:" + side + "." + key)
                delta = _feature_delta(facts, prev, side + ".joints")
                if delta is not None and t is not None and p is not None and t > p:
                    if delta / (t-p) > cfg["physical_motion"]["joint_speed_radps"]:
                        failures.append("actual_joint_velocity:" + side)
        for key, limit in limits.items():
            delta = _feature_delta(facts, anchor, key)
            if delta is None and key == "object.rotation" and owner != left:
                # Real 60 Hz rows have no full object rotation.  30 Hz snapshots
                # are checked below; all 60 Hz angular velocities are required.
                continue
            if delta is None:
                failures.append("feature_comparison_missing:" + key)
            else:
                maxima[key] = max(maxima.get(key, 0.), delta)
                if delta >= limit if ".target_" in key else delta > limit:
                    failures.append("cumulative_change:" + key)
        for side in SIDES:
            contact, old = facts.get("contact", {}).get(side, {}), anchor.get("contact", {}).get(side, {})
            if any(contact.get(key) != old.get(key) for key in CONTACT_FIELDS):
                failures.append("contact_change:" + side)
            goal = _number(facts.get("grippers", {}).get(side, {}).get("goal_width_m"))
            original = _number(anchor.get("grippers", {}).get(side, {}).get("goal_width_m"))
            if goal is None or original is None:
                failures.append("goal_evidence_missing:" + side)
            elif abs(goal - original) >= cfg["sa"]["thresholds"]["action_aperture_m"]:
                failures.append("goal_change:" + side)
    for row in rows[left:right + 1]:
        delta = _feature_delta(_facts(row), anchor, "object.rotation")
        if delta is None:
            failures.append("snapshot_object_rotation_missing")
        elif delta > cfg["physical_motion"]["rotation_rad"]:
            failures.append("cumulative_change:object.rotation")
        if delta is not None:
            maxima["object.rotation"] = max(maxima.get("object.rotation", 0.), delta)
    before, after = cfg["context"]["before_frames"], cfg["context"]["after_frames"]
    for i in range(max(0, left+1-after), min(len(rows), right+before)):
        events = _facts(rows[i]).get("interval", {}).get("events") or []
        if any(event.get("kind") in PHYSICAL_EVENTS for event in events):
            for protected in range(max(left+1, i-before), min(right, i+after+1)):
                failures.append("physical_event_context")
                details.append({"raw_frame_index": rows[protected]["raw_frame_index"],
                                "reasons": ["physical_event_context"]})
    causal = {}
    for label, row in (("left", start), ("right", end)):
        ok, reason = _causal_target(row)
        causal[label] = {"valid": ok, "reason": reason,
                         "observation_control_tick": row.get("source_control_tick"),
                         "next_target_control_tick": (row.get("next_control_target") or {}).get("source_tick"),
                         "next_target_source_line": (row.get("next_control_target") or {}).get("source_line")}
        if not ok:
            failures.append("causal_boundary:" + reason)
    target_delta = {}
    if all(item["valid"] for item in causal.values()):
        a, b = np.asarray(start["next_control_target"]["action"]), np.asarray(end["next_control_target"]["action"])
        for side_index, side in enumerate(SIDES):
            d = a[side_index*7:side_index*7+7] - b[side_index*7:side_index*7+7]
            for key, delta in (("action_position_m", np.linalg.norm(d[:3])),
                               ("action_rotation_rad", np.linalg.norm((d[3:6]+np.pi) % (2*np.pi)-np.pi)),
                               ("action_aperture_m", abs(d[6]))):
                target_delta[f"{side}.{key}"] = float(delta)
                if delta >= cfg["sa"]["thresholds"][key]:
                    failures.append("causal_target_jump:" + side + "." + key)
    endpoint_speed = {}
    period = anchor.get("source", {}).get("observation_period_s")
    if period == 1/30:
        for key in ("object.position", "object.rotation", "left.actual_position", "right.actual_position",
                    "left.joints", "right.joints", "left.actual_aperture", "right.actual_aperture"):
            delta = _feature_delta(_facts(end), anchor, key)
            if delta is not None:
                speed = delta / period
                endpoint_speed[key] = speed
                limit = cfg["physical_motion"]["angular_speed_radps" if key == "object.rotation" else
                                               "joint_speed_radps" if key.endswith(".joints") else "linear_speed_mps"]
                if speed > limit:
                    failures.append("compressed_boundary_speed:" + key)
    return {"passed": not failures, "reasons": sorted(set(failures)),
            "checked_control_count": len(all_facts), "checked_raw_frame_count": right-left+1,
            "maximum_change_from_left": maxima, "causal_target_deltas": target_delta,
            "compressed_boundary_speed": endpoint_speed, "causal_targets": causal,
            "failing_rows": details}


def _kept(row):
    return bool(row.get("effective_keep", row.get("keep", True)))


def _gaps(rows):
    previous = None
    for i, row in enumerate(rows):
        if _kept(row):
            if previous is not None and i > previous+1:
                yield previous, i
            previous = i


def _restore_edges(rows, cfg):
    """Restore actual motion before certifying a proposed automatic deletion.

    A hard boundary alone cannot repair the loss of a real movement inside a
    removed gap. Preserve that whole automatic gap conservatively. Otherwise a
    bounded edge extension can move a seam off an event/convergence boundary.
    Manual exclusions and invalid rows are never overridden.
    """
    def restore(indices, left, right, reasons, reason):
        for i in indices:
            if rows[i].get("human_decision") == "exclude" or rows[i].get("frame_role") == "INVALID":
                continue
            rows[i].update(keep=True, effective_keep=True, soft_curation_gap=False,
                stitch_restoration={"reason": reason,
                    "original_gap_start_raw_frame_index": rows[left+1]["raw_frame_index"],
                    "original_gap_end_raw_frame_index": rows[right-1]["raw_frame_index"],
                    "initial_failure_reasons": reasons})

    for left, right in list(_gaps(rows)):
        result = _gap_check(rows, left, right, cfg)
        if result["passed"]:
            continue
        physical = ("actual_velocity:", "actual_joint_velocity:", "compressed_boundary_speed:",
                    "cumulative_change:object.", "cumulative_change:left.actual_",
                    "cumulative_change:right.actual_", "cumulative_change:left.joints",
                    "cumulative_change:right.joints")
        actual_motion = any(reason.startswith(physical) for reason in result["reasons"])
        if "actual_motion" in result["reasons"]:
            actual_motion |= any(
                facts.get("motion", {}).get("moving") is True
                for row in rows[left:right+1]
                for facts in [_facts(row), *(_facts(row).get("interval", {}).get("controls") or [])])
        # A small accumulated endpoint displacement may be caused solely by
        # settling at the edge. Try the existing bounded edge restoration
        # before restoring the whole wait; never relax the certification.
        endpoint_only = actual_motion and all(
            reason.startswith('compressed_boundary_speed:') for reason in result['reasons'])
        if actual_motion and not endpoint_only:
            restore(range(left+1, right), left, right, result["reasons"], "restore_original_physical_motion")
            continue
        if any(r.get("human_decision") == "exclude" for r in rows[left+1:right]):
            continue
        # Do not use restoration to conceal missing/illegal/epoch provenance.
        fatal = ("source_", "control_epoch", "missing_", "invalid_", "control_tick_", "event_source_",
                 "causal_event:request_rejected", "execution_evidence_missing")
        if any(reason.startswith(fatal) for reason in result["reasons"]):
            continue
        chosen = None
        before, after = cfg["context"]["before_frames"]+1, cfg["context"]["after_frames"]+1
        if any(left + before < i < right - after
               for i, row in enumerate(rows[left:right+1], left)
               if any(detail.get("raw_frame_index") == row["raw_frame_index"]
                      for detail in result["failing_rows"])):
            continue
        for total in range(1, before + after + 1):
            for advance in range(max(0, total-after), min(before, total)+1):
                retreat = total-advance
                a, b = left+advance, right-retreat
                if b <= a+1:
                    continue
                if any(row.get("frame_role") == "INVALID" for row in rows[left+1:a+1] + rows[b:right]):
                    continue
                if _gap_check(rows, a, b, cfg)["passed"]:
                    chosen = a, b
                    break
            if chosen:
                break
        if chosen:
            a, b = chosen
            restore(list(range(left+1, a+1)) + list(range(b, right)), left, right,
                    result["reasons"], "restore_original_boundary_context")
        elif endpoint_only:
            restore(range(left+1, right), left, right, result['reasons'], 'restore_original_physical_motion')


def _raw_boundary(previous, row):
    if previous is None:
        return True
    if any(row.get(key) != previous.get(key) for key in
           ("source_episode", "session", "episode_index", "control_epoch", "task_segment_id")):
        return True
    a, b = _number(previous.get("original_time_s")), _number(row.get("original_time_s"))
    return bool(row.get("hard_boundary") or _facts(row).get("source", {}).get("hard_boundary")
                or _facts(row).get("source", {}).get("valid") is not True
                or _facts(previous).get("source", {}).get("valid") is not True
                or _facts(row).get("source", {}).get("available") is not True
                or _facts(previous).get("source", {}).get("available") is not True
                or row.get("frame_role") == "INVALID" or previous.get("frame_role") == "INVALID"
                or a is None or b is None or not 0 < b-a <= .05
                or row.get("raw_frame_index") != previous.get("raw_frame_index", -2)+1)


def build_sequence_contract(rows, config=None, *, restore_boundaries=True):
    """Return copied raw rows, gaps, sequences and independently certified clocks."""
    cfg = effective_config(validate_config(config) if config is not None else load_config())
    result = copy.deepcopy(rows)
    if not result:
        raise ValueError("sequence contract needs the complete original frame index")
    for row in result:
        if not _kept(row) or row.get("human_decision") in ("keep", "exclude"):
            row.pop("stitch_restoration", None)
    if restore_boundaries:
        _restore_edges(result, cfg)
    gaps, by_right = [], {}
    for left, right in _gaps(result):
        check = _gap_check(result, left, right, cfg)
        record = {"gap_id": len(gaps), "left_raw_frame_index": result[left]["raw_frame_index"],
            "right_raw_frame_index": result[right]["raw_frame_index"],
            "deleted_start_raw_frame_index": result[left+1]["raw_frame_index"],
            "deleted_end_raw_frame_index": result[right-1]["raw_frame_index"],
            "deleted_frame_count": right-left-1,
            "original_time_gap_s": (result[right]["original_time_s"]-result[left]["original_time_s"])
                if _number(result[right].get("original_time_s")) is not None
                   and _number(result[left].get("original_time_s")) is not None else None,
            "virtual_time_gap_s": 1/30 if check["passed"] else None,
            "decision_origin": "manual" if any(r.get("human_decision") == "exclude" for r in result[left+1:right]) else "automatic",
            "kind": "safe_stitch" if check["passed"] else "hard_boundary",
            "checks": check}
        gaps.append(record)
        by_right[right] = record
    sequences, previous, sequence, local_index, global_index = [], None, -1, -1, -1
    for i, row in enumerate(result):
        for key in ROW_FIELDS:
            row[key] = None
        if not _kept(row):
            continue
        causal_ok, causal_reason = _causal_target(row)
        period = _facts(row).get("source", {}).get("observation_period_s")
        clock_valid = period == 1/30
        if not clock_valid:
            causal_ok, causal_reason = False, "frozen_sampling_contract_missing"
        gap = by_right.get(i)
        boundary = (previous is None or (gap is not None and gap["kind"] != "safe_stitch")
                    or (gap is None and _raw_boundary(result[previous] if previous is not None else None, row))
                    or not clock_valid
                    or (previous is not None and not result[previous]["causal_target_valid"]))
        if boundary:
            sequence += 1
            local_index = -1
            sequences.append({"train_sequence_id": sequence, "frame_count": 0,
                "first_raw_frame_index": row["raw_frame_index"], "last_raw_frame_index": row["raw_frame_index"],
                "first_training_frame_index": global_index+1, "last_training_frame_index": global_index+1,
                "causal_action_valid_frames": 0, "safe_stitches": 0})
        global_index += 1
        local_index += 1
        row.update(training_frame_index=global_index, training_time_s=global_index*period if clock_valid else None,
                   train_sequence_id=sequence, sequence_frame_index=local_index,
                   sequence_time_s=local_index*period if clock_valid else None, causal_target_valid=causal_ok,
                   causal_target_reason=causal_reason, stitch_gap_before=gap["gap_id"] if gap else None,
                   virtual_clock_valid=clock_valid)
        item = sequences[-1]
        item.update(frame_count=item["frame_count"]+1, last_raw_frame_index=row["raw_frame_index"],
                    last_training_frame_index=global_index)
        item["causal_action_valid_frames"] += int(causal_ok)
        item["safe_stitches"] += int(bool(gap and gap["kind"] == "safe_stitch"))
        previous = i
    valid_run, windows = 0, 0
    for row in result:
        if not _kept(row):
            continue
        if row["sequence_frame_index"] == 0:
            valid_run = 0
        valid_run = valid_run+1 if row["causal_target_valid"] else 0
        windows += int(valid_run >= 30)
    reason_counts = Counter(reason for gap in gaps if gap["kind"] == "hard_boundary"
                            for reason in gap["checks"]["reasons"])
    metrics = {"retained_frames": global_index+1, "sequence_count": len(sequences),
        "safe_stitch_count": sum(g["kind"] == "safe_stitch" for g in gaps),
        "hard_gap_count": sum(g["kind"] == "hard_boundary" for g in gaps),
        "deleted_middle_frames": sum(g["deleted_frame_count"] for g in gaps),
        "boundary_restored_frames": sum(bool(row.get("stitch_restoration")) for row in result),
        "physical_motion_restored_frames": sum((row.get("stitch_restoration") or {}).get("reason")
                                               == "restore_original_physical_motion" for row in result),
        "invalid_virtual_clock_frames": sum(_kept(row) and not row["virtual_clock_valid"] for row in result),
        "full_valid_action_windows_30": windows, "hard_gap_reasons": dict(sorted(reason_counts.items()))}
    return {"version": VERSION, "descriptor": copy.deepcopy(DESCRIPTOR), "rows": result,
            "gaps": gaps, "sequences": sequences, "metrics": metrics}


def verify_sequence_contract(rows, sequence, config=None):
    """Recompute all seam certificates and per-row clocks; reject stale metadata."""
    expected = build_sequence_contract(rows, config, restore_boundaries=False)
    for old, new in zip(rows, expected["rows"]):
        if any(old.get(key) != new.get(key) for key in ROW_FIELDS):
            raise ValueError(f"sequence row contract mismatch at raw frame {old.get('raw_frame_index')}")
    for key in ("version", "descriptor", "gaps", "sequences", "metrics"):
        if sequence.get(key) != expected[key]:
            raise ValueError("sequence contract mismatch: " + key)
    expected.pop("rows")
    return expected
