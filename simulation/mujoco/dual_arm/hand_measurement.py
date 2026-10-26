"""Validate observed hand data without promoting observations to authorization."""
import math
import numpy as np

GESTURE_PROTOCOL = "gesture-grasp-release-v2"
RATE_PROTOCOL = "gesture-rate-gripper-v3"
RATE_PARAMETERS = {"mode_confirm_ms": 200, "open_score_min": .55, "close_score_min": .80, "motion_score_min": .75}
GESTURE_PARAMETERS = {
    "mode_confirm_ms": 200, "wrist_still_ms": 200, "wrist_motion_threshold": .008,
    "aperture_takeover_tolerance_m": .012, "palm_score_min": .6, "motion_score_min": .75,
    "extended_angle_deg": 150, "flexed_angle_deg": 130,
    "extended_straightness": .80, "flexed_straightness": .65,
    "tip_mcp_ratio_min": 1.1, "min_bone_px": 3,
    "thumb_angle_deg": 145, "thumb_straightness": .78,
}


def finger_geometry(hand, width, height, parameters=None):
    """Recompute camera-mode evidence from pixels, never cached frontend flags.

    Projected angles are an engineering gate, not proof of true 3D joint angles.
    Thumb/index aperture does not participate in the three idle finger gate.
    """
    p = {**GESTURE_PARAMETERS, **(parameters or {})}
    result = {"method": "image-pixel-angles-v1", "fingers": {},
              "idle_extended": False, "full_palm": False}
    try:
        points = np.asarray(hand["image_landmarks_21"], dtype=float)
        size = np.asarray([width, height], dtype=float)
        if points.shape != (21, 3) or not np.isfinite(points).all() or not np.isfinite(size).all() or np.any(size <= 0):
            return result
        xy = points[:, :2] * size
    except (KeyError, TypeError, ValueError):
        return result
    for name, start in (("index", 5), ("middle", 9), ("ring", 13), ("little", 17), ("thumb", 1)):
        chain = xy[start:start+4]
        lengths = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        f = {"valid": False, "state": "UNKNOWN", "pip_angle_deg": None,
             "dip_angle_deg": None, "straightness": None, "tip_mcp_ratio": None}
        if np.all(lengths+1e-9 >= p["min_bone_px"]):
            def angle(a, b):
                return float(np.degrees(np.arccos(np.clip(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))))
            a = angle(chain[0]-chain[1], chain[2]-chain[1])
            b = angle(chain[1]-chain[2], chain[3]-chain[2])
            direct = float(np.linalg.norm(chain[3]-chain[0]))
            straight, ratio = direct/float(lengths.sum()), direct/float(lengths[0])
            extended_angle = p["thumb_angle_deg"] if name == "thumb" else p["extended_angle_deg"]
            extended_straight = p["thumb_straightness"] if name == "thumb" else p["extended_straightness"]
            state = ("EXTENDED" if a+1e-9 >= extended_angle and b+1e-9 >= extended_angle and straight+1e-9 >= extended_straight
                     and ratio+1e-9 >= p["tip_mcp_ratio_min"] else "FLEXED" if a <= p["flexed_angle_deg"]+1e-9
                     or b <= p["flexed_angle_deg"]+1e-9 or straight <= p["flexed_straightness"]+1e-9 else "UNKNOWN")
            f.update(valid=True, state=state, pip_angle_deg=a, dip_angle_deg=b, straightness=straight, tip_mcp_ratio=ratio)
        result["fingers"][name] = f
    result["idle_extended"] = all(result["fingers"][n]["state"] == "EXTENDED" for n in ("middle", "ring", "little"))
    score = hand.get("gesture_score")
    score_valid = type(score) in (int, float) and math.isfinite(score)
    result["full_palm"] = bool(hand.get("gesture") == "Open_Palm" and score_valid and score >= p["palm_score_min"]
                               and all(f["state"] == "EXTENDED" for f in result["fingers"].values()))
    return result


def inspect_gesture_input(payload, *, parameters=None):
    """Stateless v2 camera evidence check; temporal/physical gates are separate."""
    p = {**GESTURE_PARAMETERS, **(parameters or {})}
    if payload.get("control_protocol") != GESTURE_PROTOCOL:
        return "gesture_control_protocol_required"
    modes = payload.get("interaction_modes")
    allowed = {"XY_HOLD", "Z_HOLD", "Z_APPROACH", "PINCH", "RELEASE", "OPENING", "HOLD"}
    if not isinstance(modes, (list, tuple)) or len(modes) != 2 or any(not isinstance(v, str) or v not in allowed for v in modes):
        return "invalid_interaction_modes"
    intent = payload.get("release_intent")
    if not isinstance(intent, (list, tuple)) or len(intent) != 2 or any(type(v) is not bool for v in intent):
        return "invalid_release_intent"
    if any(flag != (mode == "RELEASE") for flag, mode in zip(intent, modes)):
        return "release_intent_mode_mismatch"
    plane = payload.get("mode")
    if plane not in {"XY", "Z"} or any((m == "XY_HOLD" and plane != "XY") or
                                        (m in {"Z_HOLD", "Z_APPROACH", "PINCH", "RELEASE"} and plane != "Z") for m in modes):
        return "interaction_plane_mismatch"
    t = payload.get("telemetry")
    rec = t.get("recognition_result") if isinstance(t, dict) else None
    m = rec.get("measurement") if isinstance(rec, dict) else None
    if not isinstance(m, dict) or not isinstance(m.get("hands"), list):
        return "hand_schema_required"
    for side, mode in zip(("left", "right"), modes):
        pair = [h for h in m["hands"] if isinstance(h, dict) and h.get("bound_side") == side]
        if len(pair) != 1 or pair[0].get("valid") is not True:
            return "untrusted_hand_measurement"
        h = pair[0]
        g = finger_geometry(h, m.get("width"), m.get("height"), p)
        fs = g["fingers"]
        score = h.get("gesture_score", 0)
        confident = type(score) in (int, float) and math.isfinite(score) and score >= p["motion_score_min"]
        fist = bool(confident and h.get("gesture") == "Closed_Fist" and all(fs.get(n, {}).get("state") == "FLEXED" for n in ("index", "middle", "ring", "little")))
        ily = bool(confident and h.get("gesture") == "ILoveYou"
                   and all(fs.get(n, {}).get("state") == "EXTENDED" for n in ("index", "little"))
                   and all(fs.get(n, {}).get("state") == "FLEXED" for n in ("middle", "ring")))
        if ((mode == "XY_HOLD" and not ily) or (mode == "Z_HOLD" and not fist)
                or (mode in {"PINCH", "Z_APPROACH"} and not g["idle_extended"])
                or (mode == "RELEASE" and not g["full_palm"])
                or (mode == "OPENING" and (fist or ily or len(fs) != 5
                                          or not all(f.get('valid') for f in fs.values())))):
            return "interaction_geometry_mismatch"
        try:
            delta = np.asarray(payload["delta_task_m"][side], dtype=float)
            if delta.shape != (3,) or not np.isfinite(delta).all():
                return "invalid_continuous_input"
        except (KeyError, TypeError, ValueError):
            return "invalid_continuous_input"
        if ((mode in {"PINCH", "RELEASE", "OPENING", "HOLD"} and np.any(np.abs(delta) > 1e-12))
                or (plane == "Z" and np.any(np.abs(delta[:2]) > 1e-12))
                or (plane == "XY" and abs(delta[2]) > 1e-12)):
            return "interaction_motion_forbidden"
    return None


def inspect_rate_input(payload, *, parameters=None):
    """V3 checks raw category evidence; it does not infer finger angles/aperture."""
    p = {**RATE_PARAMETERS, **(parameters or {})}
    if payload.get("control_protocol") != RATE_PROTOCOL:
        return "gesture_control_protocol_required"
    modes = payload.get("interaction_modes")
    if (not isinstance(modes, (list, tuple)) or len(modes) != 2 or
            any(type(m) is not str or m not in {"XY_HOLD", "Z_OPEN", "Z_CLOSE", "HOLD"} for m in modes)):
        return "invalid_interaction_modes"
    intent = payload.get("release_intent")
    if not isinstance(intent, (list, tuple)) or len(intent) != 2 or any(type(v) is not bool for v in intent):
        return "invalid_release_intent"
    if any(flag != (mode == "Z_OPEN") for flag, mode in zip(intent, modes)):
        return "release_intent_mode_mismatch"
    plane = payload.get("mode")
    if plane not in {"XY", "Z"} or any((m == "XY_HOLD" and plane != "XY") or
                                        (m in {"Z_OPEN", "Z_CLOSE"} and plane != "Z") for m in modes):
        return "interaction_plane_mismatch"
    telemetry = payload.get("telemetry")
    rec = telemetry.get("recognition_result") if isinstance(telemetry, dict) else None
    measurement = rec.get("measurement") if isinstance(rec, dict) else None
    if not isinstance(measurement, dict) or not isinstance(measurement.get("hands"), list):
        return "hand_schema_required"
    calibration = telemetry.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("palm_binding_guard") is not False:
        return "palm_binding_guard"
    expected = {"XY_HOLD": ("ILoveYou", p["motion_score_min"]),
                "Z_OPEN": ("Open_Palm", p["open_score_min"]), "Z_CLOSE": ("Closed_Fist", p["close_score_min"])}
    for side, mode in zip(("left", "right"), modes):
        pair = [h for h in measurement["hands"] if isinstance(h, dict) and h.get("bound_side") == side]
        if len(pair) != 1 or pair[0].get("valid") is not True:
            return "untrusted_hand_measurement"
        if mode != "HOLD":
            name, threshold = expected[mode]
            score = pair[0].get("gesture_score")
            if (pair[0].get("gesture") != name or type(score) not in (int, float) or
                    not math.isfinite(score) or not threshold <= score <= 1):
                return "interaction_gesture_mismatch"
        try:
            delta = np.asarray(payload["delta_task_m"][side], dtype=float)
            if delta.shape != (3,) or not np.isfinite(delta).all():
                return "invalid_continuous_input"
        except (KeyError, TypeError, ValueError):
            return "invalid_continuous_input"
        if ((mode == "HOLD" and np.any(np.abs(delta) > 1e-12)) or
                (plane == "Z" and np.any(np.abs(delta[:2]) > 1e-12)) or
                (plane == "XY" and abs(delta[2]) > 1e-12)):
            return "interaction_motion_forbidden"
    return None


def inspect_release_intent(payload, *, threshold=.85):
    """Camera adapter only: verify explicit intent against raw 2D geometry.

    A policy/script uses the same public intent interface without pretending
    to observe human landmarks. This is not proof of psychological intention.
    """
    intent = payload.get('release_intent', [False, False])
    if not isinstance(intent, (list, tuple)) or len(intent) != 2 or any(type(v) is not bool for v in intent):
        return 'invalid_release_intent'
    if not any(intent):
        return None
    telemetry = payload.get('telemetry') or {}
    calibration = telemetry.get('aperture_calibration') or {}
    endpoints = calibration.get('endpoints') or {}
    m = (telemetry.get('recognition_result') or {}).get('measurement') or {}
    for side, requested in zip(('left', 'right'), intent):
        if not requested:
            continue
        e = endpoints.get(side) if isinstance(endpoints, dict) else None
        if not isinstance(e, dict) or any(type(e.get(k)) not in (int, float) or not math.isfinite(e[k]) for k in ('closed', 'open')):
            return 'release_calibration_required'
        if calibration.get('phase') != 'READY' or not 0 <= e['closed'] < .45 or e['open'] <= .8 or e['open']-e['closed'] <= .4:
            return 'release_calibration_required'
        hand = next((h for h in m.get('hands', []) if h.get('bound_side') == side), None)
        if not hand or hand.get('valid') is not True:
            return 'release_intent_geometry_mismatch'
        try:
            points = np.asarray(hand['image_landmarks_21'], dtype=float)
            size = np.asarray([m['width'], m['height']], dtype=float)
            palm = np.linalg.norm((points[5, :2]-points[17, :2])*size)
            ratio = np.linalg.norm((points[4, :2]-points[8, :2])*size)/palm if palm >= 12 else float('nan')
            normalized = (ratio-e['closed'])/(e['open']-e['closed'])
        except (KeyError, ValueError, TypeError, IndexError):
            return 'release_intent_geometry_mismatch'
        if not math.isfinite(normalized) or normalized+1e-9 < threshold:
            return 'release_intent_geometry_mismatch'
    return None


def inspect_measurement(measurement, *, session, epoch):
    if not isinstance(measurement, dict) or measurement.get("schema_version") != "task3-hand-v1":
        return "hand_schema_required"
    if not isinstance(measurement.get("measurement_id"), str) or not 1 <= len(measurement["measurement_id"]) <= 180:
        return "measurement_id_required"
    if measurement.get("source_session") != session or type(measurement.get("source_control_epoch")) is not int or measurement["source_control_epoch"] != epoch:
        return "stale_measurement_context"
    hands = measurement.get("hands")
    if not isinstance(hands, list) or len(hands) > 2:
        return "invalid_hand_list"
    for field in ("capture_browser_monotonic_ms", "capture_epoch_ms", "width", "height"):
        value = measurement.get(field)
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            return "invalid_measurement_time_or_dimensions"
    for hand in hands:
        if not isinstance(hand, dict):
            return "invalid_hand"
        for field in ("image_landmarks_21", "hand_center_landmarks_m_21"):
            value = hand.get(field)
            if value is not None:
                if not isinstance(value, list) or len(value) != 21 or any(not isinstance(p, list) or len(p) != 3 for p in value):
                    return "invalid_landmark_shape"
                if any(v is not None and (type(v) not in {int, float} or not math.isfinite(v)) for p in value for v in p):
                    return "invalid_landmark_number"
        if hand.get("valid"):
            image = hand.get("image_landmarks_21")
            if image is None or any(v is None for p in image for v in p):
                return "valid_hand_has_missing_points"
            points = np.asarray(image)
            if np.any(points[:, :2] < 0) or np.any(points[:, :2] > 1):
                return "valid_hand_out_of_frame"
            dimensions = np.array([measurement["width"], measurement["height"]])
            if np.linalg.norm((points[5, :2]-points[17, :2])*dimensions) < 12:
                return "palm_too_small"
            score = hand.get("identity_score")
            if type(score) not in {int, float} or not math.isfinite(score) or not .8 <= score <= 1:
                return "invalid_identity_score"
    return None


class HandInputGuard:
    """Independent backend enablement for continuous HTTP camera input."""
    def __init__(self):
        self.context = None
        self.seen = set()
        self.invalidate()

    def invalidate(self):
        self.last_capture = None
        self.started = None
        self.samples = 0
        self.last_points = None
        self.still_points = None
        self.still_since = None
        self.reason = "tracking_recovery"

    def check(self, payload, *, session, epoch, now_ms, release_threshold=.85,
              control_protocol=None, gesture_parameters=None):
        if self.context != (session, epoch):
            self.context = (session, epoch); self.seen.clear(); self.invalidate()
        telemetry = payload.get("telemetry", {})
        recognition = telemetry.get('recognition_result') if isinstance(telemetry,dict) else None
        m = recognition.get('measurement') if isinstance(recognition,dict) else None
        reason = inspect_measurement(m, session=session, epoch=epoch)
        if reason:
            self.invalidate(); self.reason = reason; return reason
        mid = m["measurement_id"]
        if mid in self.seen:
            self.invalidate(); self.reason = "duplicate_measurement"; return self.reason
        if len(self.seen) >= 65536:
            self.invalidate(); self.reason = "measurement_history_full_reset_scene"; return self.reason
        self.seen.add(mid)
        stamp = m["capture_epoch_ms"]
        pair = [next((h for h in m["hands"] if h.get("bound_side") == s), None) for s in ("left", "right")]
        rate = control_protocol == RATE_PROTOCOL
        aperture = telemetry.get('aperture_calibration')
        calibration_good = (isinstance(aperture, dict) and
            (aperture == {"phase": "NOT_REQUIRED", "reason": "categorical_rate_control"} if rate
             else aperture.get("phase") == "READY"))
        good = (payload.get("measurement_id") == mid and payload.get("sentAt") == stamp and 0 <= now_ms-stamp <= 150
                and m.get("accepted_recognition") is True and len(m["hands"]) == 2
                and all(h and h.get("valid") is True for h in pair) and telemetry.get("calibrated") is True
                and calibration_good)
        if not good:
            self.invalidate(); self.reason = "untrusted_hand_measurement"; return self.reason
        # Recompute the browser's 6% control margin from raw coordinates.
        # Do not reject historical observations at the schema-inspection layer.
        for hand in pair:
            image = np.asarray(hand["image_landmarks_21"])
            if np.any(image[:, :2] < .06) or np.any(image[:, :2] > .94):
                self.invalidate(); self.reason = "hand_near_frame_edge"; return self.reason
        gesture = control_protocol == GESTURE_PROTOCOL
        reason = (inspect_rate_input(payload, parameters=gesture_parameters) if rate else
                  inspect_gesture_input(payload, parameters=gesture_parameters) if gesture
                  else inspect_release_intent(payload, threshold=release_threshold))
        if reason:
            self.invalidate(); self.reason = reason; return reason
        points = np.asarray([h["image_landmarks_21"][0][:2] for h in pair])
        if self.last_capture is not None:
            if stamp <= self.last_capture:
                self.invalidate(); self.reason = "out_of_order_measurement"; return self.reason
            if stamp-self.last_capture > 250:
                self.invalidate()
            elif self.last_points is not None and np.max(np.linalg.norm(points-self.last_points, axis=1)) > .2:
                self.invalidate(); self.reason = "identity_jump"; return self.reason
        if self.started is None:
            self.started = stamp
        if gesture:
            p = {**GESTURE_PARAMETERS, **(gesture_parameters or {})}
            if self.still_points is None:
                self.still_points = points.copy()
                self.still_since = np.full(2, stamp, dtype=float)
            moved = np.linalg.norm(points-self.still_points, axis=1) > p["wrist_motion_threshold"]
            self.still_points[moved] = points[moved]
            self.still_since[moved] = stamp
            for i, mode in enumerate(payload["interaction_modes"]):
                if mode in {"PINCH", "RELEASE"} and stamp-self.still_since[i]+1e-6 < p["wrist_still_ms"]:
                    # Keep valid recovery history: awaiting stillness is a hold,
                    # not an excuse to perpetually restart temporal recovery.
                    self.samples += 1; self.last_capture = stamp; self.last_points = points
                    self.reason = "gesture_wrist_not_still"
                    return self.reason
                if mode == "OPENING" and moved[i]:
                    self.reason = "opening_wrist_motion"
                    self.samples += 1; self.last_capture = stamp; self.last_points = points
                    return self.reason
        self.samples += 1; self.last_capture = stamp; self.last_points = points
        self.reason = None if self.samples >= 5 and stamp-self.started >= 200 else "tracking_recovery"
        return self.reason

    def status(self):
        return {"reason": self.reason, "samples": self.samples, "stable_ms": 0 if self.last_capture is None else self.last_capture-self.started}
