"""Causal source facts for Paper A frames; this module never selects or deletes frames.

Times belong to their recorded clock domains. JSONL source order, event IDs and
session/epoch matching take precedence over rounded control ticks. Missing
physical facts remain unavailable, never inferred from gesture or mode names.
"""
from __future__ import annotations

import copy
import bisect
import numpy as np

SIDES = ('left', 'right')
VERSION = 'papera-frame-evidence-v2-execution'


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def _array(value, shape=None):
    try:
        a = np.asarray(value, dtype=float)
        return a if a.size and np.isfinite(a).all() and (shape is None or a.shape == shape) else None
    except (ValueError, TypeError):
        return None

def _number(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (ValueError, TypeError):
        return None

def _snapshot(snapshots, index):
    if isinstance(snapshots, dict):
        return {k: v[index] for k, v in snapshots.items() if np.ndim(v) > 0 and len(v) > index}
    return snapshots[index] if index < len(snapshots) else {}

def _rotation_distance(a, b):
    if a.shape == (4,):
        aa, bb = np.linalg.norm(a), np.linalg.norm(b)
        if aa < 1e-9 or bb < 1e-9:
            return float('inf')
        return float(2 * np.arccos(np.clip(abs(np.dot(a, b) / aa / bb), 0., 1.)))
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1., 1.)))

def _features(frame, snap, metadata):
    status = _dict(frame.get('status'))
    features = {}
    missing = []; invalid = False
    state = _array(snap.get('state', snap.get('state16', frame.get('state'))), (16,))
    action = _array(snap.get('action', snap.get('action14', frame.get('action'))), (14,))
    if state is None or action is None:
        return features, ['invalid_state_or_action'], True
    for j, side in enumerate(SIDES):
        features[f'{side}.joints'] = ('joint', state[j * 8:j * 8 + 7])
        features[f'{side}.actual_aperture'] = ('aperture', state[j * 8 + 7:j * 8 + 8])
        features[f'{side}.target_position'] = ('position', action[j * 7:j * 7 + 3])
        features[f'{side}.target_rpy'] = ('rpy', action[j * 7 + 3:j * 7 + 6])
        features[f'{side}.target_aperture'] = ('aperture', action[j * 7 + 6:j * 7 + 7])
        actual = _dict(_dict(status.get('actual_ee')).get(side))
        for k, kind, shape in [('position', 'position', (3,)), ('rotation', 'rotation', (3, 3))]:
            value = _array(actual.get(k), shape)
            if value is None:
                missing.append(f'{side}.actual_{k}')
                invalid |= k in actual and actual[k] is not None
            else:
                features[f'{side}.actual_{k}'] = (kind, value)
    qpos = _array(snap.get('qpos'))
    adr = metadata.get('object_qpos_adr')
    if isinstance(adr, int) and qpos is not None and len(qpos) >= adr + 7:
        features['object.position'] = ('position', qpos[adr:adr + 3])
        quat = qpos[adr + 3:adr + 7]
        if np.linalg.norm(quat) > 1e-9:
            features['object.rotation'] = ('rotation', quat)
        else:
            missing.append('object.rotation')
    else:
        pos = _array(status.get('object_position'), (3,))
        rotation = _array(status.get('object_rotation'), (3, 3))
        if pos is not None:
            features['object.position'] = ('position', pos)
        else:
            missing.append('object.position')
        if rotation is not None:
            features['object.rotation'] = ('rotation', rotation)
        else:
            missing.append('object.full_rotation')
    invalid |= any(k in snap and _array(snap[k]) is None for k in ('qpos', 'qvel', 'ctrl'))
    invalid |= any(k in status and status[k] is not None and _array(status[k]) is None for k in
                   ('object_position', 'object_linear_velocity_mps', 'object_angular_velocity_radps', 'ee_speeds_mps'))
    return features, missing, invalid

def _changed_keys(a, b, cfg):
    changed = set()
    for key in a.keys() & b.keys():
        kind, x = a[key]; _, y = b[key]
        if kind == 'rotation':
            delta, limit = _rotation_distance(x, y), cfg['rotation_rad']
        elif kind == 'rpy':
            delta, limit = np.linalg.norm((x - y + np.pi) % (2 * np.pi) - np.pi), cfg['rotation_rad']
        else:
            delta = np.max(np.abs(x - y)) if kind in ('joint', 'aperture') else np.linalg.norm(x - y)
            limit = cfg[{'joint': 'joint_rad', 'aperture': 'aperture_m', 'position': 'position_m'}[kind]]
        if delta > limit:
            changed.add(key)
    return changed

def _robot_phase(frame, moving):
    s = _dict(frame.get('status')); pc = _dict(s.get('physical_contact'))
    stable = _dict(pc.get('stable_grasps')); contacts = _dict(s.get('contacts'))
    if s.get('success') is True or s.get('task_state') == 'DONE':
        return 'terminal_success'
    if s.get('task_state') == 'FAIL':
        return 'terminal_failure'
    if any(_dict(pc.get('release_opening')).values()):
        return 'releasing'
    if all(stable.get(side) is True for side in SIDES):
        return 'dual_grasp_motion' if moving else 'dual_grasp_hold'
    if any(stable.get(side) is True for side in SIDES):
        return 'partial_grasp'
    if any(contacts.get(side) is True or (isinstance(contacts.get(f'{side}_fingers'), list) and
            any(v is True for v in contacts[f'{side}_fingers'])) for side in SIDES):
        return 'contact_motion' if moving else 'contact_hold'
    return 'free_motion' if moving else 'free_stationary'

def _original_auxiliary(frame, measurements, cfg):
    mid = frame.get('observation_measurement_id'); m = _dict(measurements.get(mid))
    if m.get('kind') == 'hand_measurement':
        m = _dict(m.get('measurement'))
    wall = _number(frame.get('wall_time_ms')); capture = _number(m.get('capture_epoch_ms'))
    recognized = _number(m.get('recognition_epoch_ms'))
    age = wall - capture if wall is not None and capture is not None else None
    causal = (age is not None and 0 <= age <= cfg['maximum_age_ms'] and
              recognized is not None and capture <= recognized <= wall and m.get('accepted_recognition') is True and
              m.get('source_session') == frame.get('session') and m.get('source_control_epoch') == frame.get('control_epoch'))
    output = {}
    for side in SIDES:
        hands = [h for h in (m.get('hands') if isinstance(m.get('hands'), list) else [])
                 if isinstance(h, dict) and h.get('bound_side') == side]
        h = hands[0] if len(hands) == 1 else {}
        identity = _number(h.get('identity_score')); score = _number(h.get('gesture_score'))
        image = _array(h.get('image_landmarks_21'), (21, 3))
        world = _array(h.get('hand_center_landmarks_m_21'), (21, 3))
        margin = cfg['image_margin']
        inside = image is not None and bool(np.all((image[:, :2] >= margin) & (image[:, :2] <= 1 - margin)))
        trusted = bool(causal and h.get('valid') is True and identity is not None and
                       identity >= cfg['minimum_identity_score'] and inside and not h.get('near_edge') and
                       _dict(frame.get('hand_measurement_mask')).get(side) is True)
        category = h.get('gesture'); threshold = cfg['category_minimum_scores'].get(category)
        category_mask = bool(trusted and threshold is not None and score is not None and score >= threshold)
        geometry_mask = bool(trusted and h.get('world_valid') is True and world is not None and
                             np.linalg.norm(world[5] - world[17]) > 1e-6)
        # Unknown is not inferred to be intentional recentering. Its task
        # applicability stays pending, independently of raw geometry validity.
        task_mask = bool(geometry_mask and not _dict(frame.get('status')).get('paused') and
                         (category_mask or cfg['unknown_task_geometry']))
        reason = ('measurement_unavailable_or_not_causal' if not causal else
                  'hand_untrusted' if not trusted else 'unknown_or_low_score' if not category_mask else 'trusted_category')
        output[side] = {'measurement_id': mid, 'bound_side': side, 'raw_label': h.get('raw_label'),
                        'category': category, 'category_score': score, 'identity_score': identity,
                        'category_mask': category_mask, 'geometry_mask': geometry_mask,
                        'task_geometry_mask': task_mask, 'task_geometry_review': 'candidate' if task_mask else 'pending_or_inapplicable',
                        'reason': reason, 'age_ms': age,
                        'geometry_reference': {'measurement_id': mid, 'bound_side': side,
                            'image_field': 'image_landmarks_21', 'model_world_field': 'hand_center_landmarks_m_21',
                            'normalization': 'wrist_relative_divide_distance_landmarks_5_17'},
                        'old_masks': {key: _dict(frame.get(key)).get(side) for key in
                                      ('hand_measurement_mask', 'hand_supervision_mask', 'direct_action_alignment_mask', 'morphology_applicable_mask')}}
    return output


def _dict(value):
    return value if isinstance(value, dict) else {}


def _bool(value):
    return value if type(value) is bool else None


def _line(record):
    value = record.get('source_line')
    return value if type(value) is int and value > 0 else None


def _before(source, target):
    """Source order proves causality even when ticks and timestamps coincide."""
    a, b = _line(source), _line(target)
    return a is not None and b is not None and a < b


def _clock_age(frame, measurement, pair=None):
    """Map browser capture to server epoch with the health round-trip bounds.

    Browser and server monotonic clocks are never subtracted. A present but
    malformed pair is not silently replaced with a same-clock assumption.
    Historical records without a pair retain their recorded epoch convention.
    """
    wall = _number(frame.get('wall_time_ms'))
    capture = _number(measurement.get('capture_epoch_ms'))
    if wall is None or capture is None:
        return {'age_ms': None, 'age_bounds_ms': None, 'clock_basis': 'unavailable'}
    if pair is None:
        return {'age_ms': wall - capture, 'age_bounds_ms': [wall - capture] * 2,
                'clock_basis': 'recorded_epoch_without_clock_pair'}
    pair = _dict(pair)
    request = _number(pair.get('browser_request_epoch_ms'))
    receive = _number(pair.get('browser_receive_epoch_ms'))
    server = _number(pair.get('server_epoch_ms'))
    mono = _number(pair.get('server_monotonic_ns'))
    frame_mono = _number(frame.get('server_monotonic_ns'))
    if (None in (request, receive, server) or request > receive or
            (mono is not None and frame_mono is not None and mono > frame_mono)):
        return {'age_ms': None, 'age_bounds_ms': None, 'clock_basis': 'invalid_clock_pair'}
    low = wall - capture - (server - request)
    high = wall - capture - (server - receive)
    return {'age_ms': (low + high) / 2, 'age_bounds_ms': [low, high],
            'clock_basis': 'health_request_response_epoch_bounds'}


def _auxiliary(frame, measurements, cfg, measurement_sources=None):
    """Category-only mask ignores scores; the existing geometry masks do not."""
    output = _original_auxiliary(frame, measurements, cfg)
    mid = frame.get('observation_measurement_id')
    source = (measurement_sources or {}).get(mid)
    ordered = source is None or _before(source, frame)
    for item in output.values():
        item['previous_category_mask'] = item['category_mask']
        item['previous_reason'] = item['reason']
        current = (mid is not None and item['age_ms'] is not None and
                   0 <= item['age_ms'] <= cfg['maximum_age_ms'] and
                   item['reason'] != 'measurement_unavailable_or_not_causal' and ordered)
        item['category_mask'] = bool(current and item['category'] in
                                     ('ILoveYou', 'Open_Palm', 'Closed_Fist'))
        item['category_policy'] = 'recorded_category_no_score_v1'
        item['reason'] = ('recorded_category_no_score' if item['category_mask'] else
                          'recorded_unknown_category' if current else
                          'measurement_unavailable_or_not_causal')
        item['source_line'] = _line(source) if source is not None else None
        if not ordered:
            item['geometry_mask'] = item['task_geometry_mask'] = False
            item['task_geometry_review'] = 'pending_or_inapplicable'
    return output


def _rotation_matrix(value):
    v = _array(value)
    if v is None:
        return None
    if v.shape == (3, 3):
        if np.allclose(v.T @ v, np.eye(3), atol=1e-5, rtol=0) and abs(np.linalg.det(v)-1) < 1e-5:
            return v
        return None
    if v.shape != (4,) or np.linalg.norm(v) < 1e-9:
        return None
    w, x, y, z = v / np.linalg.norm(v)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _accepted_intent(control, accepted, episode, alignment=None):
    status = _dict(control.get('status')); pc = _dict(status.get('physical_contact'))
    modes = _dict(pc.get('interaction_modes'))
    ids = _dict(pc.get('interaction_mode_event_id'))
    labels = _dict(control.get('control_labels'))
    maximum_age = _number(_dict(_dict(episode.get('metadata', {}).get('config')).get('control')).get('max_command_age_ms'))
    result = {}
    for side_index, side in enumerate(SIDES):
        mode = modes.get(side)
        event_id = ids.get(side, _dict(labels.get('interaction_mode_event_id')).get(side))
        source = accepted.get(event_id)
        method = 'explicit_event_id'
        fallback = _dict(_dict(alignment).get('time_fallback'))
        if not event_id and fallback.get('enabled') is True:
            max_periods = _number(fallback.get('max_control_periods'))
            control_hz = _number(_dict(episode.get('metadata', {}).get('sampling_policy')).get('control_hz'))
            now = _number(control.get('sim_time')); candidates = []
            if max_periods is not None and 0 < max_periods <= 1 and now is not None and control_hz is not None and control_hz > 0:
                for candidate in accepted.values():
                    then = _number(candidate.get('sim_time'))
                    if (then is not None and 0 < now-then <= max_periods/control_hz+1e-9 and
                        candidate.get('session') == control.get('session') and
                        candidate.get('control_epoch') == control.get('control_epoch') and
                        (_before(candidate, control) or (_line(candidate) is None and _line(control) is None))):
                        candidates.append(candidate)
            if len(candidates) == 1:
                source = candidates[0]
                event_id = _dict(source.get('request')).get('eventId')
                method = 'bounded_prior_control_time_fallback'
            else:
                method = 'time_fallback_unavailable_or_ambiguous'
        request = _dict((source or {}).get('request'))
        response = _dict((source or {}).get('result'))
        mid = request.get('measurement_id')
        measurement = _dict(episode.get('measurements', {}).get(mid))
        if measurement.get('kind') == 'hand_measurement':
            measurement = _dict(measurement.get('measurement'))
        pair = _dict(request.get('telemetry')).get('clock_pair')
        age = _clock_age(control, measurement, pair)
        reasons = []
        if source is None or response.get('accepted') is not True:
            reasons.append('accepted_source_unavailable')
        elif not _before(source, control) and method != 'bounded_prior_control_time_fallback':
            reasons.append('accepted_source_not_before_observation')
        if source is not None and (source.get('session') != control.get('session') or
                                   source.get('control_epoch') != control.get('control_epoch')):
            reasons.append('accepted_source_context_mismatch')
        if source is not None and (request.get('session') != source.get('session') or
                                   request.get('control_epoch') != source.get('control_epoch')):
            reasons.append('request_context_mismatch')
        accepted_modes = response.get('interaction_modes')
        accepted_mode = (accepted_modes.get(side) if isinstance(accepted_modes, dict) else
                         accepted_modes[side_index] if isinstance(accepted_modes, list) and len(accepted_modes) == 2 else None)
        if mode not in ('XY', 'XY_HOLD', 'Z_OPEN', 'Z_CLOSE', 'HOLD') or accepted_mode != mode:
            reasons.append('accepted_mode_unavailable_or_mismatch')
        if not measurement or measurement.get('accepted_recognition') is not True:
            reasons.append('measurement_unavailable_or_not_accepted')
        if (measurement.get('source_session') != control.get('session') or
                measurement.get('source_control_epoch') != control.get('control_epoch')):
            reasons.append('measurement_context_mismatch')
        ms = episode.get('measurement_sources', {}).get(mid)
        if ms is not None and source is not None and not _before(ms, source):
            reasons.append('measurement_not_before_accepted_command')
        bounds = age['age_bounds_ms']
        if maximum_age is None:
            reasons.append('frozen_freshness_limit_unavailable')
        elif bounds is None or not 0 <= bounds[0] <= bounds[1] <= maximum_age:
            reasons.append('measurement_age_not_proven_current')
        recognized = _number(measurement.get('recognition_epoch_ms'))
        capture = _number(measurement.get('capture_epoch_ms'))
        if recognized is None or capture is None or recognized < capture:
            reasons.append('recognition_time_invalid')
        else:
            recognition_age = _clock_age(control, {'capture_epoch_ms': recognized}, pair)['age_bounds_ms']
            if recognition_age is None or recognition_age[0] < 0:
                reasons.append('recognition_not_proven_before_observation')
        hands = [h for h in measurement.get('hands', [])
                 if isinstance(h, dict) and h.get('bound_side') == side] if isinstance(measurement.get('hands'), list) else []
        hand = hands[0] if len(hands) == 1 else {}
        recorded_hand = _dict(_dict(status.get('hands')).get(side))
        # Preserve the controller's recorded validity decision per side. Do
        # not recompute identity/category confidence using today's settings.
        if len(hands) != 1:
            reasons.append('side_hand_unavailable_or_ambiguous')
        elif hand.get('valid') is not True:
            reasons.append('side_hand_recorded_invalid')
        invalid_reason = hand.get('invalid_reason')
        condition = ('measurement_unavailable' if not measurement else
                     'hand_missing' if not hands else
                     'identity_invalid' if len(hands) != 1 or
                         (isinstance(invalid_reason, str) and 'identity' in invalid_reason) else
                     'out_of_frame' if hand.get('near_edge') is True else
                     'recorded_invalid' if hand.get('valid') is False else
                     'unknown_gesture' if hand.get('gesture') not in ('ILoveYou', 'Open_Palm', 'Closed_Fist') else
                     'recorded_recognized')
        result[side] = {'mode': mode, 'reliable': not reasons, 'event_id': event_id,
                        'measurement_id': mid, 'source_line': _line(source or {}),
                        'hand_condition': condition, 'recorded_category': hand.get('gesture'),
                        'recorded_hand_valid': _bool(hand.get('valid')),
                        'recorded_invalid_reason': invalid_reason,
                        'recorded_hand_visible': _bool(recorded_hand.get('visible')),
                        'recorded_near_edge': _bool(hand.get('near_edge')),
                        'trusted_hold': not reasons and mode == 'HOLD',
                        'global_paused': _bool(status.get('paused')),
                        'pause_reason': status.get('pause_reason'),
                        'alignment_method': method,
                        'reason': ';'.join(reasons) or 'accepted_current_input',
                        'reasons': reasons, 'maximum_age_ms': maximum_age,
                        'clock_pair': copy.deepcopy(pair), **age}
    return result


def _physics(control, snapshot, metadata, previous, physical_cfg):
    status = _dict(control.get('status')); pc = _dict(status.get('physical_contact'))
    features, missing, invalid = _features(control, snapshot, metadata)
    state = _array(snapshot.get('state', control.get('state')), (16,))
    action = _array(snapshot.get('action', control.get('action')), (14,))
    time = _number(control.get('sim_time')); dt = None
    if previous is not None:
        previous_time = _number(previous.get('sim_time'))
        if time is not None and previous_time is not None and time > previous_time:
            dt = time - previous_time
    contacts = _dict(status.get('contacts'))
    grippers, contact = {}, {}
    for j, side in enumerate(SIDES):
        g = _dict(_dict(status.get('grippers')).get(side))
        edge = _dict(_dict(pc.get('gripper_edge')).get(side))
        actual = _number(g.get('actual_width_m'))
        if actual is None and state is not None:
            actual = float(state[j*8+7])
        command = _number(g.get('command_width_m'))
        if command is None and action is not None:
            command = float(action[j*7+6])
        old_width = (previous or {}).get('grippers', {}).get(side, {}).get('actual_width_m')
        velocity = (actual - old_width)/dt if dt is not None and actual is not None and old_width is not None else None
        max_width = _number(g.get('max_width_m'))
        fingers = contacts.get(f'{side}_fingers')
        fingers_valid = isinstance(fingers, list) and len(fingers) == 2 and all(type(v) is bool for v in fingers)
        forces = _array(_dict(contacts.get('finger_normal_force_n')).get(side), (2,))
        if forces is None:
            forces = _array(contacts.get(f'{side}_grip_forces_n'), (2,))
        contact[side] = {
            'fingers': copy.deepcopy(fingers) if fingers_valid else None,
            'any_finger': any(fingers) if fingers_valid else None,
            'double_finger': all(fingers) if fingers_valid else None,
            'qualified': _bool(contacts.get(side)),
            'stable': _bool(_dict(pc.get('stable_grasps')).get(side)),
            'finger_forces_n': forces,
        }
        grippers[side] = {'actual_width_m': actual, 'command_width_m': command,
            'goal_width_m': _number(g.get('goal_width_m')), 'max_width_m': max_width,
            'requested_width_m': _number(g.get('requested_width_m')),
            'width_velocity_mps': velocity,
            'opening': velocity > 1e-6 if velocity is not None else None,
            'closing': velocity < -1e-6 if velocity is not None else None,
            'edge_state': edge.get('state'), 'edge_event_id': edge.get('event_id'),
            'baseline_mode': edge.get('baseline_mode'), 'baseline_source': edge.get('baseline_source'),
            'edge_reason': edge.get('reason'),
            'edge_count': edge.get('edge_count'), 'authorized_mode': edge.get('authorized_mode'),
            'rate_enabled': _bool(_dict(pc.get('gripper_rate_enabled')).get(side)),
            'rate_direction': _number(_dict(pc.get('gripper_rate_direction')).get(side)),
            'interrupted_release_hold': _bool(_dict(pc.get('rate_interrupted_release_hold')).get(side)),
            'hold_width_m': _number(_dict(pc.get('hold_width_m')).get(side)),
            'executed_open': False, 'executed_close': False, 'aborted': False}
    position = features.get('object.position', (None, None))[1]
    rotation = _rotation_matrix(features.get('object.rotation', (None, None))[1])
    linear = _array(status.get('object_linear_velocity_mps'), (3,))
    angular = _array(status.get('object_angular_velocity_radps'), (3,))
    relative = {}
    for side in SIDES:
        p = features.get(f'{side}.actual_position', (None, None))[1]
        r = _rotation_matrix(features.get(f'{side}.actual_rotation', (None, None))[1])
        available = p is not None and r is not None and position is not None and rotation is not None
        relative[side] = {'available': available,
            'position': rotation.T @ (p-position) if available else None,
            'rotation': rotation.T @ r if available else None,
            'basis': 'actual_ee_in_actual_object_frame' if available else 'unavailable'}
    ee_speeds = _array(status.get('ee_speeds_mps'), (2,))
    reasons = []
    for label, v, threshold in (
        ('object_linear_velocity', linear, physical_cfg['linear_speed_mps']),
        ('object_angular_velocity', angular, physical_cfg['angular_speed_radps']),
        ('ee_velocity', ee_speeds, physical_cfg['linear_speed_mps'])):
        if v is not None and (np.linalg.norm(v) if label.startswith('object_') else max(abs(v))) > threshold:
            reasons.append(label)
    if previous is not None and dt is not None:
        old_features = {k: (v[0], np.asarray(v[1])) for k, v in previous.get('features', {}).items()}
        reasons.extend(sorted(key for key in _changed_keys(features, old_features, physical_cfg)
                              if '.target_' not in key))
        for side in SIDES:
            key = f'{side}.joints'
            if key in features and key in old_features and np.max(
                    np.abs(features[key][1]-old_features[key][1]))/dt > physical_cfg['joint_speed_radps']:
                reasons.append(f'{side}.joint_velocity')
    for side in SIDES:
        v = grippers[side]['width_velocity_mps']
        if v is not None and abs(v) > physical_cfg['linear_speed_mps']:
            reasons.append(f'{side}.finger_velocity')
    release = {new: copy.deepcopy(pc.get(old)) for new, old in (
        ('intent','release_intent'),('armed','release_armed'),('progress_s','release_progress_s'),
        ('executed','release_executed'),('opening','release_opening'),('event_id','release_event_id'),
        ('cycle_state','release_cycle_state'),('support_ready','support_ready'),
        ('reason','release_reason'),('reentry_pending','release_reentry_pending'),
        ('intent_event_id','release_intent_event_id'),
        ('released','released'),('ever_dual','ever_dual'))}
    return {'features': features, 'missing': missing, 'invalid': invalid,
        'grippers': grippers, 'contact': contact, 'relative_pose': relative,
        'object': {'position': position, 'rotation': rotation, 'linear_velocity': linear,
                   'angular_velocity': angular,
                   'linear_speed_mps': float(np.linalg.norm(linear)) if linear is not None else None,
                   'angular_speed_radps': float(np.linalg.norm(angular)) if angular is not None else None},
        'motion': {'moving': bool(reasons), 'available': linear is not None and angular is not None and
                   ee_speeds is not None and not invalid, 'reasons': reasons, 'ee_speeds_mps': ee_speeds},
        'release': release, 'execution': _execution_state(pc, metadata, grippers),
        'task': {'state': status.get('task_state'), 'success': _bool(status.get('success')),
                 'failure': status.get('failure_reason'), 'done': status.get('task_state') == 'DONE'},
        'paused': _bool(status.get('paused')),
        'pause_reason': status.get('pause_reason') if isinstance(status.get('pause_reason'), str) else None}


def _execution_state(pc, metadata, grippers):
    """Recorded authorization is distinct from categorical Z_OPEN/Z_CLOSE.

    V2 trusted HOLD retains an unfinished edge. A missing frozen policy or
    missing execution flags is unknown, never a proof that no work is pending.
    The level policy has no edge authorization; its recorded ramp/confirmation
    fields provide the evidence. No gesture score is thresholded here.
    """
    frozen = metadata.get('gripper_trigger_policy')
    configured = _dict(_dict(metadata.get('config')).get('gesture_control')).get('gripper_trigger_policy')
    strategy = frozen if frozen is not None else configured
    recorded = pc.get('gripper_trigger_policy')
    shared_reasons = []
    if strategy not in ('level_v1', 'palm_fist_edge_v1', 'palm_fist_edge_v2', 'palm_fist_level_v2'):
        shared_reasons.append('frozen_gripper_strategy_unavailable')
    if frozen is not None and configured is not None and frozen != configured:
        shared_reasons.append('frozen_gripper_strategy_mismatch')
    if recorded is not None and strategy != recorded:
        shared_reasons.append('recorded_gripper_strategy_mismatch')
    sides = {}
    for side in SIDES:
        reasons = list(shared_reasons); g = grippers[side]
        edge = _dict(_dict(pc.get('gripper_edge')).get(side))
        cycle = _dict(pc.get('release_cycle_state')).get(side)
        progress = _number(_dict(pc.get('release_progress_s')).get(side))
        opening = _bool(_dict(pc.get('release_opening')).get(side))
        armed = _bool(_dict(pc.get('release_armed')).get(side))
        if cycle is None or progress is None or opening is None or armed is None or g['rate_enabled'] is None:
            reasons.append('execution_status_unavailable')
        authorization = None
        if strategy in ('palm_fist_edge_v1', 'palm_fist_edge_v2'):
            if not all(key in edge for key in ('authorized_mode', 'state', 'baseline_mode', 'edge_count', 'event_id')):
                reasons.append('edge_authorization_unavailable')
            else:
                authorization = edge['authorized_mode'] in ('Z_OPEN', 'Z_CLOSE')
                if edge['authorized_mode'] not in (None, 'Z_OPEN', 'Z_CLOSE'):
                    reasons.append('edge_authorization_invalid')
        elif strategy in ('level_v1', 'palm_fist_level_v2'):
            authorization = False  # Level execution is described by actual rate flags.
        confirming = cycle == 'CONFIRMING' or (progress is not None and progress > 0 and opening is not True)
        # Some historical edge controllers retain a CLOSE edge after the
        # physical grasp latched. It cannot continue closing while the verified
        # hold is active. An OPEN edge is never treated as completed this way.
        hold = _number(_dict(pc.get('hold_width_m')).get(side))
        completed_close_hold = bool(not reasons and edge.get('authorized_mode') == 'Z_CLOSE' and
            edge.get('state') == 'HOLD' and edge.get('reason') == 'grasp_hold' and
            _dict(pc.get('stable_grasps')).get(side) is True and hold is not None and
            g.get('goal_width_m') == hold and g.get('command_width_m') == hold and
            g['rate_enabled'] is False and opening is False and armed is False and not confirming)
        authorization_present = authorization
        if completed_close_hold:
            authorization = False
        pending = bool(authorization or g['rate_enabled'] is True or opening is True or
                       armed is True or confirming)
        sides[side] = {'available': not reasons, 'reasons': reasons,
            'pending_authorization': authorization,
            'authorization_present': authorization_present,
            'completed_close_hold': completed_close_hold,
            'release_confirmation_pending': confirming,
            'release_opening': opening, 'release_armed': armed,
            'release_cycle_state': cycle, 'release_progress_s': progress,
            'pending': True if pending else False if not reasons else None,
            'trusted_hold_resumable': strategy == 'palm_fist_edge_v2' and authorization is True and
                edge.get('state') == 'HOLD' and edge.get('reason') == 'gripper_edge_transition_hold' and
                _dict(pc.get('interaction_modes')).get(side) == 'HOLD'}
    return {'strategy': strategy, 'recorded_strategy': recorded,
            'available': all(v['available'] for v in sides.values()), 'sides': sides}


def _event(record, kind, side=None, **extra):
    request = _dict(record.get('request'))
    return {'kind': kind, 'side': side, 'source_line': _line(record),
            'session': record.get('session'), 'control_epoch': record.get('control_epoch'),
            'control_tick': record.get('tick'), 'sim_time': _number(record.get('sim_time')),
            'wall_time_ms': _number(record.get('wall_time_ms')),
            'server_monotonic_ns': record.get('server_monotonic_ns'),
            'event_id': request.get('eventId'), 'measurement_id': request.get('measurement_id'),
            **extra}


def _request_events(episode):
    events, seen, warnings = [], set(), []
    for record in episode.get('executed_events', []):
        request = _dict(record.get('request')); response = _dict(record.get('result'))
        accepted = response.get('accepted') is True
        pc = _dict(response.get('physical_contact'))
        execution = (_execution_state(pc, episode.get('metadata', {}),
            {side: {'rate_enabled': _bool(_dict(pc.get('gripper_rate_enabled')).get(side)),
                    'goal_width_m': _number(_dict(_dict(response.get('grippers')).get(side)).get('goal_width_m')),
                    'command_width_m': _number(_dict(_dict(response.get('grippers')).get(side)).get('command_width_m'))}
             for side in SIDES})
            if pc else None)
        events.append(_event(record, 'request_accepted' if accepted else 'request_rejected',
            accepted=accepted, command=request.get('command'), reason=response.get('reason'),
            requested_interaction_modes=copy.deepcopy(request.get('interaction_modes')),
            accepted_interaction_modes=copy.deepcopy(response.get('interaction_modes')),
            delta_task_m=copy.deepcopy(request.get('delta_task_m')),
            requested_widths_m=copy.deepcopy(request.get('widths_m')),
            requested_release_intent=copy.deepcopy(request.get('release_intent')),
            client_id=request.get('client_id'), sequence=request.get('sequence'),
            execution=execution,
            gripper_edges=copy.deepcopy(pc.get('gripper_edge')),
            actual_state_phase=record.get('actual_state_phase'),
            requested_target=response.get('requested_target'),
            projected_target=response.get('projected_target'),
            accepted_target=response.get('accepted_target')))
        eid = request.get('eventId')
        if not accepted or not eid:
            continue
        if eid in seen:
            warnings.append(f'duplicate_accepted_event_id:{eid}')
            continue
        seen.add(eid)
        for side in SIDES:
            g = _dict(_dict(response.get('grippers')).get(side))
            edge = _dict(_dict(pc.get('gripper_edge')).get(side))
            goal, command = _number(g.get('goal_width_m')), _number(g.get('command_width_m'))
            common = {'goal_width_m': goal, 'command_width_m': command,
                      'actual_state_phase': record.get('actual_state_phase'),
                      'basis': 'accepted_gripper_goal_and_execution_authorization'}
            close = (edge.get('event_id') == eid and edge.get('authorized_mode') == 'Z_CLOSE' and
                     goal is not None and command is not None and goal < command-1e-7)
            release = (_dict(pc.get('release_event_id')).get(side) == eid and
                       _dict(pc.get('release_executed')).get(side) is True)
            open_edge = (edge.get('event_id') == eid and edge.get('authorized_mode') == 'Z_OPEN' and
                         goal is not None and command is not None and goal > command+1e-7)
            if close:
                events.append(_event(record, 'gripper_close_executed', side, **common))
            if release or open_edge:
                events.append(_event(record, 'gripper_open_executed', side, **common))
    for raw in episode.get('timeline_events', []):
        request = _dict(raw.get('request'))
        events.append(_event(raw, 'raw_' + str(raw.get('kind')), raw.get('side'),
            reason=raw.get('reason', raw.get('drop_reason')), stage=raw.get('stage'),
            command=request.get('command'),
            event_id=raw.get('event_id', request.get('eventId')),
            measurement_id=raw.get('measurement_id', request.get('measurement_id'))))
    return events, warnings


def _next_target(current, following):
    """The very next observed 60 Hz row, never a later image or reused tick."""
    if following is None:
        return None
    a = _array(following.get('action'), (14,))
    t0, t1 = _number(current.get('sim_time')), _number(following.get('sim_time'))
    line0, line1 = _line(current), _line(following)
    if (type(current.get('tick')) is not int or following.get('tick') != current['tick'] + 1 or
        t0 is None or t1 is None or not 0 < t1-t0 <= .03 or a is None or
        current.get('session') != following.get('session') or
        current.get('control_epoch') != following.get('control_epoch') or
        (line0 is not None and line1 is not None and line1 <= line0)):
        return None
    return {'action': a.tolist(), 'source_tick': following['tick'], 'source_line': line1,
            'sim_time': t1, 'control_epoch': following.get('control_epoch'),
            'wall_time_ms': following.get('wall_time_ms'), 'action_event_id': following.get('action_event_id'),
            'action_measurement_id': following.get('action_measurement_id'),
            'alignment': 'next_observed_control_tick_after_post_step_snapshot', 'executor_label_frozen': False}


def _command_values(control):
    """Recorded accepted targets, not hand intent or requested GUI targets."""
    action = _array(control.get('action'), (14,))
    grippers = _dict(_dict(control.get('status')).get('grippers'))
    goals = [_number(_dict(grippers.get(side)).get('goal_width_m')) for side in SIDES]
    return action, goals


def _command_changes(current, reference, thresholds):
    changed = []
    action, goals = _command_values(current)
    old_action, old_goals = _command_values(reference)
    if action is not None and old_action is not None:
        for arm, side in enumerate(SIDES):
            delta = action[arm*7:arm*7+7] - old_action[arm*7:arm*7+7]
            for label, value, threshold in (
                ('target_position', np.linalg.norm(delta[:3]), thresholds['action_position_m']),
                ('target_rpy', np.linalg.norm((delta[3:6]+np.pi) % (2*np.pi)-np.pi),
                 thresholds['action_rotation_rad']),
                ('target_aperture', abs(delta[6]), thresholds['action_aperture_m']),
            ):
                if value >= threshold:
                    changed.append(f'{side}.{label}')
    for side, value, old_value in zip(SIDES, goals, old_goals):
        if value is not None and old_value is not None and abs(value-old_value) >= thresholds['action_aperture_m']:
            changed.append(f'{side}.goal_aperture')
    return changed


def _command_activity(control, previous, initial, source, thresholds):
    """Separate causal command evidence; never relabel it as actual motion."""
    action, goals = _command_values(control)
    reasons = []
    for label, record in (('current', control), ('previous', previous), ('initial', initial)):
        if record is None:
            reasons.append(f'{label}_command_unavailable')
            continue
        values, widths = _command_values(record)
        if values is None or any(width is None for width in widths):
            reasons.append(f'{label}_target_or_goal_unavailable')
        if _line(record) is None:
            reasons.append(f'{label}_source_line_unavailable')
    if not source.get('valid') or not source.get('available') or source.get('reasons'):
        reasons.append('source_continuity_or_sampling_unavailable')
    from_previous = _command_changes(control, previous, thresholds) if previous is not None else []
    from_initial = _command_changes(control, initial, thresholds) if initial is not None else []
    known = not reasons and action is not None and all(width is not None for width in goals)
    return {'available': known,
            'changed': True if from_previous or from_initial else False if known else None,
            'changed_from_previous': from_previous, 'changed_from_initial': from_initial,
            'baseline_source_line': _line(initial) if initial is not None else None,
            'previous_source_line': _line(previous) if previous is not None else None,
            'reasons': reasons}


def build_frame_evidence(episode, config):
    """Return complete raw-frame facts and compact, ordered control/event tables.

    ``interval`` owns records after the preceding image through this image,
    inclusive of the current control and exclusive of any later same-tick event.
    Controls after the final image remain in the tables; terminal duplicates are
    never substituted for image rows. No keep/delete/quality approval is made.
    """
    cfg = config or {}
    from dataset.task3.curation_config import load_config
    command_thresholds = cfg.get('sa', {}).get('thresholds') or load_config()['sa']['thresholds']
    physical_cfg = cfg.get('physical_motion', cfg.get('stability', {
        'position_m': .0001, 'rotation_rad': .001, 'joint_rad': .001, 'aperture_m': .0001,
        'linear_speed_mps': .005, 'angular_speed_radps': .02, 'joint_speed_radps': .03}))
    auxiliary_cfg = cfg.get('auxiliary', {'maximum_age_ms': 150., 'minimum_identity_score': .8,
        'image_margin': .06, 'category_minimum_scores': {'ILoveYou': .75, 'Open_Palm': .55, 'Closed_Fist': .8},
        'unknown_task_geometry': False})
    metadata = episode.get('metadata', {}); ref = episode.get('ref', {})
    sampling = _dict(metadata.get('sampling_policy'))
    fps, control_hz = _number(sampling.get('fps')), _number(sampling.get('control_hz'))
    observation_period = 1/fps if fps is not None and fps > 0 else None
    control_period = 1/control_hz if control_hz is not None and control_hz > 0 else None
    sampling_valid = fps == 30 and control_hz == 60
    missing_sampling = any(sampling.get(key) is None for key in ('fps', 'control_hz'))
    invalid_sampling = any(sampling.get(key) is not None and
                           (number is None or number <= 0) for key, number in
                           (('fps', fps), ('control_hz', control_hz)))
    sampling_reason = ('invalid_frozen_sampling_contract' if invalid_sampling else
                       'missing_frozen_sampling_contract' if missing_sampling else
                       'unsupported_frozen_sampling_contract' if not sampling_valid else None)
    raw_controls, frames = episode.get('controls', []), episode.get('frames', [])
    accepted = episode.get('accepted_events', {})
    events, warnings = _request_events(episode)
    if not raw_controls:
        warnings.append('missing_60hz_controls')
    controls = []; previous = None; previous_raw = None; initial_command = None
    frame_by_line = {_line(f): i for i, f in enumerate(frames) if _line(f) is not None}
    # Source-line-free synthetic/legacy data can match an exact image identity,
    # but cannot gain accepted intent from tick equality alone.
    frame_by_key = {(f.get('frame_index'), f.get('tick'), f.get('sim_time')): i for i, f in enumerate(frames)}
    image_controls = {}
    for ci, raw in enumerate(raw_controls):
        fi = frame_by_line.get(_line(raw)) if _line(raw) is not None else frame_by_key.get(
            (raw.get('frame_index'), raw.get('tick'), raw.get('sim_time')))
        snap = _snapshot(episode.get('snapshots', {}), fi) if fi is not None else {}
        facts = _physics(raw, snap, metadata, previous, physical_cfg)
        reasons = []
        if sampling_reason is not None:
            reasons.append(sampling_reason)
        if facts['invalid']:
            reasons.append('invalid_numeric')
        if _array(raw.get('state'), (16,)) is None or _array(raw.get('action'), (14,)) is None:
            reasons.append('invalid_state_or_action')
        if _number(raw.get('sim_time')) is None:
            reasons.append('invalid_sim_time')
        if fi is not None:
            for name, shape in (('state', (16,)), ('action', (14,))):
                observed = _array(raw.get(name), shape)
                image_value = _array(frames[fi].get(name), shape)
                snapshot_value = _array(snap.get(name), shape)
                if observed is not None and (image_value is None or not np.array_equal(observed, image_value)):
                    reasons.append(f'invalid_image_control_{name}_mismatch')
                if observed is not None and snapshot_value is not None and not np.allclose(
                        observed, snapshot_value, atol=1e-6, rtol=0):
                    reasons.append(f'invalid_snapshot_control_{name}_mismatch')
        if _line(raw) is None:
            warnings.append('source_line_unavailable')
        if _number(raw.get('log_dropped')) not in (None, 0):
            reasons.append('recorded_log_dropped')
        if previous_raw is not None:
            if raw.get('session') != previous_raw.get('session'):
                reasons.append('session_change')
            if raw.get('control_epoch') != previous_raw.get('control_epoch'):
                reasons.append('control_epoch_change')
            t0, t1 = _number(previous_raw.get('sim_time')), _number(raw.get('sim_time'))
            if t0 is not None and t1 is not None:
                if t1 < t0 or t1-t0 > .03:
                    reasons.append('control_clock_gap_or_regression')
                elif t1 == t0:
                    warnings.append('duplicate_nonadvancing_control_rows')
            if _line(raw) is not None and _line(previous_raw) is not None and _line(raw) <= _line(previous_raw):
                reasons.append('source_line_not_increasing')
        intent = _accepted_intent(raw, accepted, episode, cfg.get('alignment'))
        facts.update(index=ci, source_line=_line(raw), tick=raw.get('tick'),
            physics_tick=raw.get('physics_tick'), log_dropped=raw.get('log_dropped'),
            action_event_id=raw.get('action_event_id'), action_measurement_id=raw.get('action_measurement_id'),
            observation_measurement_id=raw.get('observation_measurement_id'),
            sim_time=_number(raw.get('sim_time')), wall_time_ms=_number(raw.get('wall_time_ms')),
            server_monotonic_ns=raw.get('server_monotonic_ns'), session=raw.get('session'),
            control_epoch=raw.get('control_epoch'), frame_index=raw.get('frame_index'),
            accepted_intent=intent, source={'valid': not facts['invalid'] and not any(
                r.startswith('invalid_') for r in reasons), 'hard_boundary': bool(reasons),
                'available': sampling_valid,
                'reasons': reasons, 'source_line': _line(raw),
                'observation_period_s': observation_period, 'control_period_s': control_period,
                'clock_pair': {s: intent[s]['clock_pair'] for s in SIDES}})
        if (initial_command is None or any(raw.get(key) != initial_command.get(key)
                                           for key in ('session', 'control_epoch'))):
            initial_command = raw
            command_previous = raw
        else:
            command_previous = previous_raw
        facts['command_activity'] = _command_activity(
            raw, command_previous, initial_command, facts['source'], command_thresholds)
        if previous is not None:
            for side in SIDES:
                comparisons = (
                    ('contact_changed', facts['contact'][side], previous['contact'][side],
                     ('fingers', 'double_finger', 'any_finger', 'qualified')),
                    ('grasp_stable_changed', facts['contact'][side], previous['contact'][side], ('stable',)),
                    ('accepted_intent_changed', facts['accepted_intent'][side], previous['accepted_intent'][side], ('mode',)),
                    ('gripper_authorization_changed', facts['grippers'][side], previous['grippers'][side],
                     ('authorized_mode', 'edge_event_id', 'edge_count')),
                    ('gripper_baseline_changed', facts['grippers'][side], previous['grippers'][side],
                     ('baseline_mode', 'baseline_source')),
                    ('gripper_execution_changed', facts['grippers'][side], previous['grippers'][side],
                     ('edge_state', 'rate_enabled', 'rate_direction', 'interrupted_release_hold')),
                    ('release_confirmation_changed', facts['execution']['sides'][side], previous['execution']['sides'][side],
                     ('release_confirmation_pending', 'release_armed')),
                )
                for kind, now, old, fields in comparisons:
                    if any(now.get(k) != old.get(k) for k in fields):
                        events.append(_event(raw, kind, side,
                            before={k: old.get(k) for k in fields}, after={k: now.get(k) for k in fields},
                            event_id=intent[side]['event_id'], measurement_id=intent[side]['measurement_id']))
                before_cycle = _dict(previous['release'].get('cycle_state')).get(side)
                cycle = _dict(facts['release'].get('cycle_state')).get(side)
                if cycle != before_cycle:
                    events.append(_event(raw, 'release_cycle_changed', side, before=before_cycle, after=cycle))
                oldedge, edge = previous['grippers'][side]['edge_state'], facts['grippers'][side]['edge_state']
                if edge in ('ABORTED', 'CANCELLED') and edge != oldedge:
                    events.append(_event(raw, 'gripper_aborted', side, before=oldedge, after=edge))
            if facts['task']['state'] != previous['task']['state']:
                events.append(_event(raw, 'task_state_changed', before=previous['task']['state'], after=facts['task']['state']))
            if facts['paused'] != previous['paused']:
                events.append(_event(raw, 'pause_changed', before=previous['paused'], after=facts['paused'],
                                     reason=facts['pause_reason']))
        controls.append(_json(facts))
        if fi is not None:
            if fi in image_controls:
                warnings.append(f'duplicate_image_control:{fi}')
            else:
                image_controls[fi] = ci
        previous, previous_raw = facts, raw
    # Missing source lines remain visible, but cannot be placed before an image.
    events.sort(key=lambda e: (e['source_line'] is None, e['source_line'] or 0))
    for i, event in enumerate(events):
        event['index'] = i
    located_events = [e for e in events if e['source_line'] is not None]
    event_lines = [e['source_line'] for e in located_events]
    previous_control_line = 0
    for control in controls:
        line = control['source_line']
        control_events = (located_events[bisect.bisect_right(event_lines, previous_control_line):
                          bisect.bisect_right(event_lines, line)] if line is not None else [])
        control['event_indices'] = [e['index'] for e in control_events]
        for side in SIDES:
            for flag, kind in (('executed_close', 'gripper_close_executed'),
                               ('executed_open', 'gripper_open_executed'), ('aborted', 'gripper_aborted')):
                control['grippers'][side][flag] = any(
                    e['kind'] == kind and e['side'] == side for e in control_events)
        if line is not None:
            previous_control_line = line
    rows = []; previous_ci = -1; previous_line = 0
    for fi, frame in enumerate(frames):
        ci = image_controls.get(fi); line = _line(frame)
        if ci is None:
            facts = _physics(frame, _snapshot(episode.get('snapshots', {}), fi), metadata, None, physical_cfg)
            facts.update(accepted_intent=_accepted_intent(frame, accepted, episode, cfg.get('alignment')),
                source={'valid': False, 'available': False, 'hard_boundary': True,
                        'reasons': ['image_control_unavailable'],
                        'observation_period_s': observation_period, 'control_period_s': control_period,
                        'source_line': line, 'clock_pair': None})
            indices = []
        else:
            facts = copy.deepcopy(controls[ci]); indices = list(range(previous_ci+1, ci+1))
        interval_events = (located_events[bisect.bisect_right(event_lines, previous_line):
                            bisect.bisect_right(event_lines, line)] if line is not None else [])
        interval = {'control_indices': indices, 'event_indices': [e['index'] for e in interval_events],
            'events': interval_events,
            'mode_sequence': [{'tick': controls[j]['tick'], 'source_line': controls[j]['source_line'],
                'left': controls[j]['accepted_intent']['left']['mode'],
                'right': controls[j]['accepted_intent']['right']['mode'],
                'left_reliable': controls[j]['accepted_intent']['left']['reliable'],
                'right_reliable': controls[j]['accepted_intent']['right']['reliable'],
                'left_measurement_id': controls[j]['accepted_intent']['left']['measurement_id'],
                'right_measurement_id': controls[j]['accepted_intent']['right']['measurement_id'],
                'left_event_id': controls[j]['accepted_intent']['left']['event_id'],
                'right_event_id': controls[j]['accepted_intent']['right']['event_id'],
                'left_hand_condition': controls[j]['accepted_intent']['left']['hand_condition'],
                'right_hand_condition': controls[j]['accepted_intent']['right']['hand_condition']}
                for j in indices],
            'safety_pause': any(controls[j]['paused'] is True for j in indices),
            'pause_reasons': list(dict.fromkeys(controls[j]['pause_reason'] for j in indices
                                                if controls[j]['paused'] is True))}
        command_known = bool(indices) and all(controls[j]['command_activity']['available'] for j in indices)
        command_changed = any(controls[j]['command_activity']['changed'] is True for j in indices)
        command_step_changed = any(controls[j]['command_activity']['changed_from_previous'] for j in indices)
        interval.update(
            command_activity=True if command_changed else False if command_known else None,
            command_changed_from_previous=True if command_step_changed else False if command_known else None,
            command_activity_available=command_known,
            command_activity_details=[
                {'tick': controls[j]['tick'], 'source_line': controls[j]['source_line'],
                 **copy.deepcopy(controls[j]['command_activity'])}
                for j in indices if (controls[j]['command_activity']['changed'] is True
                                     or not controls[j]['command_activity']['available'])])
        for side in SIDES:
            for flag, kind in (('executed_close','gripper_close_executed'),
                               ('executed_open','gripper_open_executed'),('aborted','gripper_aborted')):
                facts['grippers'][side][flag] = any(e['kind'] == kind and e['side'] == side for e in interval_events)
        boundaries = {r for j in indices for r in controls[j]['source']['reasons']}
        boundaries.update(facts['source']['reasons'])
        facts['source']['hard_boundary'] = bool(boundaries)
        facts['source']['reasons'] = sorted(boundaries)
        if any(not controls[j]['source']['valid'] for j in indices):
            facts['source']['valid'] = False
        facts['interval'] = interval; facts['events'] = interval_events
        # Self-contained source facts let Prepared re-check human edits and
        # Pack independently verify every deleted 60 Hz interval. Never use a
        # compressed array position as a raw control tick.
        interval['controls'] = [copy.deepcopy(controls[j]) for j in indices]
        interval['execution_available'] = bool(indices) and all(controls[j]['execution']['available'] for j in indices)
        interval['execution_pending'] = any(
            controls[j]['execution']['sides'][side]['pending'] is True for j in indices for side in SIDES)
        facts['next_control'] = (copy.deepcopy(controls[ci+1])
                                if ci is not None and ci+1 < len(controls) else None)
        if indices and any(controls[j]['motion']['moving'] for j in indices):
            facts['motion']['moving'] = True
            facts['motion']['reasons'] = sorted(set(facts['motion']['reasons']) | {'intermediate_control_motion'})
        status = _dict(frame.get('status')); pc = _dict(status.get('physical_contact'))
        snap = _snapshot(episode.get('snapshots', {}), fi)
        row = {'source_episode': ref.get('key'), 'session': ref.get('session', frame.get('session')),
            'episode_index': ref.get('episode_index'), 'raw_frame_index': frame.get('frame_index', fi),
            'source_line': line, 'source_control_tick': frame.get('tick'), 'control_epoch': frame.get('control_epoch'),
            'original_time_s': _number(frame.get('sim_time')), 'original_wall_time_ms': frame.get('wall_time_ms'),
            'original_monotonic_ns': frame.get('server_monotonic_ns'),
            'recorded_video_time_s': fi*observation_period if observation_period is not None else None,
            'state_recorded': snap.get('state', frame.get('state')),
            'action_recorded': snap.get('action', frame.get('action')),
            'action_recorded_semantics': 'post_step_issued_target_metres_not_frozen_executor_label',
            'next_control_target': _next_target(raw_controls[ci], raw_controls[ci+1] if ci+1<len(raw_controls) else None)
                if ci is not None and sampling_valid and controls[ci]['source']['valid'] and
                   (ci+1 >= len(controls) or controls[ci+1]['source']['valid']) else None,
            'robot_phase': _robot_phase(frame, facts['motion']['moving']),
            'status_summary': {'paused': status.get('paused'), 'pause_reason': status.get('pause_reason'),
                'task_state': status.get('task_state'), 'hand_measurement_mask': frame.get('hand_measurement_mask')},
            'auxiliary': _auxiliary(frame, episode.get('measurements', {}), auxiliary_cfg,
                                    episode.get('measurement_sources')),
            'source_refs': {'observation_measurement_id': frame.get('observation_measurement_id'),
                'action_measurement_id': frame.get('action_measurement_id'), 'action_event_id': frame.get('action_event_id'),
                'camera_frame_index': frame.get('frame_index', fi),
                'previous_image_control_tick': frames[fi-1].get('tick') if fi else None,
                'current_image_control_tick': frame.get('tick'),
                'intermediate_events_source': 'JSONL source lines after previous image through current image',
                'previous_image_source_line': previous_line or None, 'current_image_source_line': line},
            'source_release': {k: copy.deepcopy(pc.get(k)) for k in ('release_intent','release_armed',
                'release_progress_s','release_executed','release_opening','release_event_id','release_cycle_state','gripper_edge')},
            'original_control_labels': copy.deepcopy(frame.get('control_labels')),
            'terminal': {'actual_done': status.get('task_state') in ('DONE','FAIL'),
                         'success': status.get('success') is True or status.get('task_state') == 'DONE', 'truncated': False},
            'review_status': 'pending', 'training_approved': False, 'evidence': facts}
        rows.append(_json(row))
        if ci is not None:
            previous_ci = ci
        if line is not None:
            previous_line = line
    return _json({'version': VERSION, 'rows': rows, 'controls': controls, 'events': events,
                  'warnings': sorted(set(warnings)), 'training_approved': False})
