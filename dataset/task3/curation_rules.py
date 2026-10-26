"""Single curation policy: physical evidence first, uncertainty is retained.

These are offline indexing decisions, never robot commands or ACT algorithms.
Every original observation survives in the index, including excluded rows.
"""
from __future__ import annotations

import copy
import math
import numpy as np

from dataset.task3.curation_config import (
    CATEGORY_POLICY, GROUPS, VERSION,
    effective_config, load_config, validate_config,
)

SIDES = ('left', 'right')
ACTIVE_MODES = {'XY_HOLD', 'Z_OPEN', 'Z_CLOSE'}
PHYSICAL_EVENTS = {'contact_changed', 'grasp_stable_changed', 'task_state_changed'}
CAUSAL_EVENTS = PHYSICAL_EVENTS | {
    'gripper_open_executed', 'gripper_close_executed', 'gripper_aborted',
    'gripper_authorization_changed', 'gripper_baseline_changed',
    'gripper_execution_changed', 'release_confirmation_changed', 'release_cycle_changed',
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _array(value, shape=None):
    try:
        a = np.asarray(value, dtype=float)
        return a if (shape is None or a.shape == shape) and np.isfinite(a).all() else None
    except (ValueError, TypeError):
        return None


def group_deltas(state, action, past_states, past_actions):
    """Unchanged s/a units and five-group mathematics; no gestures here."""
    ds = np.abs(np.asarray(past_states).reshape(-1, 2, 8) - np.asarray(state).reshape(2, 8))
    da = np.asarray(past_actions).reshape(-1, 2, 7) - np.asarray(action).reshape(2, 7)
    angles = (da[:, :, 3:6] + np.pi) % (2 * np.pi) - np.pi
    return dict(zip(GROUPS, map(float, (
        ds[:, :, :7].max(), ds[:, :, 7].max(),
        np.linalg.norm(da[:, :, :3], axis=2).max(),
        np.linalg.norm(angles, axis=2).max(), np.abs(da[:, :, 6]).max()))))


def _physical_state_changed(row, anchor, cfg):
    """Cumulative actual joint/width motion at the existing physical tolerances."""
    delta = np.abs(np.asarray(row['state_recorded']).reshape(2, 8)
                   - np.asarray(anchor['state_recorded']).reshape(2, 8))
    return bool(delta[:, :7].max() >= cfg['physical_motion']['joint_rad']
                or delta[:, 7].max() >= cfg['physical_motion']['aperture_m'])


def _facts(row):
    return row.get('evidence') or {}


def _events(row):
    f = _facts(row)
    return f.get('events') or f.get('interval', {}).get('events') or []


def execution_hazards(row):
    """Factual causal hazards, shared by curation and seam verification.

    Modes are deliberately absent: a recorded Z_CLOSE is not an authorization,
    and a V2 trusted HOLD can still carry an unfinished authorization.
    """
    f = _facts(row)
    interval, execution = f.get('interval', {}), f.get('execution', {})
    reasons = [e['kind'] for e in _events(row) if e.get('kind') in CAUSAL_EVENTS]
    if any((e.get('execution') or {}).get('sides', {}).get(side, {}).get('pending') is True
           for e in _events(row) for side in SIDES):
        reasons.append('execution_authorized_between_control_ticks')
    if interval.get('execution_pending') is True:
        reasons.append('pending_execution_between_images')
    for side in SIDES:
        if execution.get('sides', {}).get(side, {}).get('pending') is True:
            reasons.append(side + '_unfinished_execution_or_authorization')
    return sorted(set(reasons))


def physical_evidence_missing(row):
    """Missing physical/control facts are unknown, never zero motion."""
    f, missing = _facts(row), []
    if f.get('source', {}).get('available') is not True:
        missing.append('source')
    if f.get('motion', {}).get('available') is not True:
        missing.append('motion')
    interval = f.get('interval', {})
    if interval.get('command_activity_available') is not True:
        missing.append('control_commands')
    if f.get('execution', {}).get('available') is not True or interval.get('execution_available') is not True:
        missing.append('execution_authorization')
    obj = f.get('object', {})
    if (_array(obj.get('position'), (3,)) is None
            or _rotation_distance(obj.get('rotation'), obj.get('rotation')) is None
            or any(_number(obj.get(k)) is None for k in ('linear_speed_mps', 'angular_speed_radps'))):
        missing.append('object_pose_or_velocity')
    for side in SIDES:
        pose, grip = f.get('relative_pose', {}).get(side, {}), f.get('grippers', {}).get(side, {})
        if (pose.get('available') is not True or _array(pose.get('position'), (3,)) is None
                or _rotation_distance(pose.get('rotation'), pose.get('rotation')) is None):
            missing.append(side + '_relative_pose')
        if any(_number(grip.get(k)) is None for k in ('actual_width_m', 'command_width_m', 'goal_width_m')):
            missing.append(side + '_gripper_width')
        if any(type(f.get('contact', {}).get(side, {}).get(k)) is not bool
               for k in ('any_finger', 'double_finger', 'stable')):
            missing.append(side + '_contact')
    return missing


def _duration(rows):
    """Observation coverage from frozen sampling facts, not current defaults."""
    if not rows:
        return 0.
    periods = [_number(_facts(r).get('source', {}).get('observation_period_s')) for r in rows]
    if any(p is None or p <= 0 for p in periods):
        return None
    return rows[-1]['original_time_s'] - rows[0]['original_time_s'] + periods[-1]


def _runs(flags):
    start = None
    for i, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            yield start, i
            start = None


def _rotation_distance(a, b):
    a, b = _array(a), _array(b)
    if a is None or b is None or a.shape != b.shape:
        return None
    if a.shape == (3, 3):
        return float(np.arccos(np.clip((np.trace(a.T @ b) - 1.) / 2., -1., 1.)))
    if a.shape == (4,) and min(np.linalg.norm(a), np.linalg.norm(b)) > 1e-9:
        return float(2. * np.arccos(np.clip(abs(np.dot(a, b)) / np.linalg.norm(a) / np.linalg.norm(b), 0., 1.)))
    return None


def _window(rows, index, seconds):
    begin = index
    while begin > 0 and rows[begin - 1]['task_segment_id'] == rows[index]['task_segment_id']:
        begin -= 1
        if rows[index]['original_time_s'] - rows[begin]['original_time_s'] >= seconds - 1e-9:
            return rows[begin:index + 1]
    return []


def _stable_width(window, side, tolerance):
    widths = [_number(_facts(r).get('grippers', {}).get(side, {}).get('actual_width_m')) for r in window]
    return bool(widths and all(w is not None for w in widths) and max(widths) - min(widths) <= tolerance)


def _stable_relative(window, side, cfg):
    poses = [_facts(r).get('relative_pose', {}).get(side, {}) for r in window]
    if not poses or not all(p.get('available') is True for p in poses):
        return False
    positions = [_array(p.get('position'), (3,)) for p in poses]
    if any(p is None for p in positions):
        return False
    # Pairwise extrema avoid accepting opposite deviations around an anchor.
    if any(np.linalg.norm(a - b) > cfg['relative_position_stability_m'] for a in positions for b in positions):
        return False
    for a in poses:
        for b in poses:
            angle = _rotation_distance(a.get('rotation'), b.get('rotation'))
            if angle is None or angle > cfg['relative_rotation_stability_rad']:
                return False
    return True


def _object_settled(window, cfg):
    if not window:
        return False
    objects = [_facts(r).get('object', {}) for r in window]
    for obj in objects:
        ls, rs = _number(obj.get('linear_speed_mps')), _number(obj.get('angular_speed_radps'))
        if ls is None or rs is None or ls > cfg['physical_motion']['linear_speed_mps'] or rs > cfg['physical_motion']['angular_speed_radps']:
            return False
    positions = [_array(o.get('position'), (3,)) for o in objects]
    if any(p is None for p in positions):
        return False
    if any(np.linalg.norm(a - b) > cfg['protection']['relative_position_stability_m'] for a in positions for b in positions):
        return False
    for first in objects:
        for second in objects:
            angle = _rotation_distance(first.get('rotation'), second.get('rotation'))
            if angle is None or angle > cfg['protection']['relative_rotation_stability_rad']:
                return False
    return True


def _stable_intent(rows, cfg):
    if len(rows) != cfg['intent']['stable_frames'] or len({r['task_segment_id'] for r in rows}) != 1:
        return None
    pair = []
    for side in SIDES:
        values = [_facts(r).get('accepted_intent', {}).get(side, {}) for r in rows]
        modes = {v.get('mode') for v in values}
        if (len(modes) != 1 or not modes.issubset(ACTIVE_MODES)
                or not all(v.get('reliable') is True and v.get('event_id') and v.get('measurement_id') for v in values)):
            pair.append(None)
            continue
        needed = cfg['intent']['min_distinct_inputs']
        if min(len({v['event_id'] for v in values}), len({v['measurement_id'] for v in values})) < needed:
            pair.append(None)
            continue
        pair.append(next(iter(modes)))
    return tuple(pair) if any(mode is not None for mode in pair) else None


def _internal_intents(rows):
    sequence, unreliable, safety = [], False, False
    for row in rows:
        evidence = _facts(row)
        interval = evidence.get('interval', {})
        safety |= bool(interval.get('safety_pause') or row.get('status_summary', {}).get('paused'))
        inside = interval.get('mode_sequence') or []
        if not inside:
            values = evidence.get('accepted_intent', {})
            inside = [{s: values.get(s, {}).get('mode') for s in SIDES} |
                      {s + '_reliable': values.get(s, {}).get('reliable', False) for s in SIDES}]
        for item in inside:
            modes = tuple(item.get(s) for s in SIDES)
            if not sequence or modes != sequence[-1]:
                sequence.append(modes)
            for side, mode in zip(SIDES, modes):
                if mode not in ACTIVE_MODES | {'HOLD'} or (mode != 'HOLD' and item.get(side + '_reliable') is not True):
                    unreliable = True
    return sequence, unreliable, safety


def _pause_context(rows, phase):
    """Explain a normal initial/final stop only in its verified task phase."""
    reasons = []
    for row in rows:
        interval = _facts(row).get('interval', {})
        current = list(interval.get('pause_reasons') or [])
        if row.get('status_summary', {}).get('paused') is True:
            current.append(row['status_summary'].get('pause_reason'))
        if interval.get('safety_pause') and not current:
            current.append(None)
        for reason in current:
            if reason not in reasons:
                reasons.append(reason)
    allowed = ({'record_start_baseline', 'operation_pending', 'hand_lost',
                'hand_near_frame_edge', 'tracking_recovery', 'camera_stalled',
                'stale_recognition', 'command_timeout', 'palm_binding_guard',
                'identity_conflict', 'transport_recovery'} if phase == 'prefix'
               else {'task_done', 'hand_lost', 'operation_pending',
                     'hand_near_frame_edge', 'tracking_recovery'})
    return {'phase': phase, 'raw_pause_reasons': reasons,
            'interpretation': 'before_first_robot_activity' if phase == 'prefix'
                              else 'after_verified_done_and_settled',
            'explained_normal_wait': all(r in allowed for r in reasons)}


def _initial_activity_boundary(rows, cfg):
    """Bound preparation against a fixed initial pose, not a rolling window.

    A slow cumulative departure or a between-image accepted target excursion
    ends preparation even when each image delta is below the motion threshold.
    Unknown physical/control evidence cannot prove an expendable prefix.
    """
    initial = rows[0]
    for i, row in enumerate(rows):
        f = _facts(row)
        interval = f.get('interval', {})
        if (row['frame_role'] == 'INVALID' or (i and row['hard_boundary'])
                or f.get('source', {}).get('available') is False
                or f.get('motion', {}).get('available') is not True
                or interval.get('command_activity_available') is False):
            return None
        obj = f.get('object', {})
        if (any(f.get('relative_pose', {}).get(s, {}).get('available') is not True for s in SIDES)
                or _array(obj.get('position'), (3,)) is None
                or _rotation_distance(obj.get('rotation'), obj.get('rotation')) is None):
            return None
        contact = [f.get('contact', {}).get(s, {}).get('any_finger') for s in SIDES]
        if any(value is None for value in contact):
            return None
        if (row['sa_changed'] or f['motion'].get('moving') is True
                or row.get('execution_grasp_phase') or row.get('execution_release_phase') or any(contact)
                or interval.get('command_activity') is True
                or any(e.get('kind') in PHYSICAL_EVENTS |
                       {'gripper_open_executed', 'gripper_close_executed'}
                       for e in _events(row))):
            return i
        delta = group_deltas(row['state_recorded'], row['action_recorded'],
                             [initial['state_recorded']], [initial['action_recorded']])
        if (any(delta[key] >= cfg['sa']['thresholds'][key] for key in GROUPS)
                or _physical_state_changed(row, initial, cfg)):
            return i
        # Objects/actual end effectors can drift without changed target labels.
        for key in ('object', 'left.actual', 'right.actual'):
            if key == 'object':
                now, start = f.get('object', {}), _facts(initial).get('object', {})
                p, p0 = _array(now.get('position'), (3,)), _array(start.get('position'), (3,))
                angle = _rotation_distance(now.get('rotation'), start.get('rotation'))
            else:
                now, start = f.get('features', {}), _facts(initial).get('features', {})
                p = _array(now.get(key + '_position', (None, None))[1], (3,))
                p0 = _array(start.get(key + '_position', (None, None))[1], (3,))
                angle = _rotation_distance(now.get(key + '_rotation', (None, None))[1],
                                           start.get(key + '_rotation', (None, None))[1])
            if p is not None and p0 is not None and np.linalg.norm(p - p0) >= cfg['physical_motion']['position_m']:
                return i
            if angle is not None and angle >= cfg['physical_motion']['rotation_rad']:
                return i
    return None


def _nearby_intent(rows, edge, direction, cfg, segment):
    count, search = cfg['intent']['stable_frames'], cfg['intent']['recenter_search_frames']
    starts = (range(edge-count, max(-1, edge-search-1), -1) if direction < 0
              else range(edge, min(len(rows)-count+1, edge+search-count+1)))
    sides = {}
    for begin in starts:
        if begin < 0:
            continue
        window = rows[begin:begin+count]
        if not window or any(r['task_segment_id'] != segment for r in window):
            break
        pair = _stable_intent(window, cfg)
        if pair:
            for side, mode in zip(SIDES, pair):
                if mode is not None and side not in sides:
                    sides[side] = {'mode': mode, 'start_index': begin, 'stop_index': begin+count,
                                   'raw_frames': [r['raw_frame_index'] for r in window]}
            if len(sides) == len(SIDES):
                break
    if not sides:
        return None
    return {'modes': [sides.get(side, {}).get('mode') for side in SIDES],
            'start_index': min(v['start_index'] for v in sides.values()),
            'stop_index': max(v['stop_index'] for v in sides.values()),
            'raw_frames': sorted({i for value in sides.values() for i in value['raw_frames']}),
            'by_side': sides}


def _recenter_context(rows, start, stop, cfg, diagnostics=None):
    """Verified static core with explicit before/after intent, including transitions."""
    def reject(reason):
        if diagnostics is not None:
            diagnostics['reason'] = reason
        return None

    part = rows[start:stop]
    segment = part[0]['task_segment_id']
    before = _nearby_intent(rows, start, -1, cfg, segment)
    after = _nearby_intent(rows, stop, 1, cfg, segment)
    if diagnostics is not None:
        diagnostics.update(pre_intent_evidence=before, post_intent_evidence=after)
    if not before:
        return reject('missing_pre_intent_context')
    if not after:
        return reject('missing_post_intent_context')
    comparable = [side for side in SIDES if side in before['by_side'] and side in after['by_side']]
    if not comparable:
        return reject('no_same_side_comparable_intent')
    # Include all the space between independently found side contexts. A valid
    # left context cannot conceal a right-side event or reset between images.
    bridge_start = min(v['stop_index'] for v in before['by_side'].values())
    bridge = rows[bridge_start:max(v['start_index'] for v in after['by_side'].values())]
    sequence, _, _ = _internal_intents(bridge)
    changed = any(before['by_side'][s]['mode'] != after['by_side'][s]['mode'] for s in comparable)
    has_hold = any(value in (None, 'HOLD') for pair in sequence for value in pair)
    # Each side may switch from its old mode to its new one once. An unrelated
    # third mode or a reversal remains a real semantic boundary for review.
    for side in range(2):
        old, new = before['modes'][side], after['modes'][side]
        if SIDES[side] not in comparable:
            # No cross-hand substitution. Unobserved context is allowed only
            # with actual commands, authorization and physical facts checked.
            observed = {pair[side] for pair in sequence if pair[side] in ACTIVE_MODES}
            if len(observed) > 1:
                return reject('uncomparable_side_mode_excursion')
            continue
        switched = False
        for pair in sequence:
            mode = pair[side]
            if mode in (None, 'HOLD'):
                continue
            if mode not in (old, new) or (old != new and switched and mode == old):
                return reject('unexplained_mode_excursion')
            switched |= old != new and mode == new
    tracking_reasons = {'hand_lost', 'hand_near_frame_edge', 'tracking_recovery',
                        'camera_stalled', 'stale_recognition', 'command_timeout',
                        'transport_recovery'}
    reasons = []
    retained_boundary_events = []
    for bridge_index, row in enumerate(bridge, bridge_start):
        f = _facts(row)
        interval = f.get('interval', {})
        pauses = list(interval.get('pause_reasons') or [])
        if row.get('status_summary', {}).get('paused'):
            pauses.append(row['status_summary'].get('pause_reason'))
        if interval.get('safety_pause') and not pauses:
            pauses.append(None)
        reasons.extend(pauses)
        if (row['task_segment_id'] != segment or row['frame_role'] == 'INVALID'
                or f.get('source', {}).get('available') is False):
            return reject('incomplete_source_evidence')
        if any(p not in tracking_reasons for p in pauses):
            return reject('safety_pause_requires_review')
        if pauses and not has_hold:
            return reject('tracking_pause_without_hold_evidence')
        hazards = execution_hazards(row)
        if hazards or row.get('height_protected'):
            if start <= bridge_index < stop:
                return reject('physical_event_in_wait_context')
            # The event itself and its six-frame context were removed from
            # candidates above. It cannot invalidate an earlier/later fully
            # stable core merely because intent windows reach across it.
            retained_boundary_events.append({'raw_frame_index': row['raw_frame_index'],
                                             'reasons': hazards})
    # Forward, fixed-anchor intervals: a later restart closes the current
    # interval without reclassifying an already stationary earlier interval.
    def fits(index, anchor):
        row = rows[index]
        f, origin = _facts(row), _facts(anchor)
        interval = f.get('interval', {})
        if (physical_evidence_missing(row) or f['motion'].get('moving')
                or interval.get('command_changed_from_previous') is True
                or any(f.get('relative_pose', {}).get(s, {}).get('available') is not True for s in SIDES)):
            return False
        # Current A/S stationarity is a consecutive adjacent comparison, not
        # distance to the last sample of the entire pause.
        sa_anchor = rows[index - 1] if index else row
        delta = group_deltas(row['state_recorded'], row['action_recorded'],
                             [sa_anchor['state_recorded']], [sa_anchor['action_recorded']])
        if any(delta[key] >= cfg['sa']['thresholds'][key] for key in GROUPS):
            return False
        # Per-image motion flags and the short A/S window can miss a slow
        # cumulative null-space joint change or servo opening. Compare the
        # candidate interval against its fixed first frame as well.
        if _physical_state_changed(row, anchor, cfg):
            return False
        for side in SIDES:
            goal = _number(f.get('grippers', {}).get(side, {}).get('goal_width_m'))
            initial_goal = _number(origin.get('grippers', {}).get(side, {}).get('goal_width_m'))
            previous_goal = _number(_facts(rows[index-1] if index else row).get('grippers', {}).get(side, {}).get('goal_width_m'))
            if (goal is None or initial_goal is None
                    or previous_goal is None or abs(goal-previous_goal) > 1e-12
                    or abs(goal - initial_goal) >= cfg['sa']['thresholds']['action_aperture_m']):
                return False
        for key in ('object', *SIDES):
            now = f.get('object', {}) if key == 'object' else f.get('relative_pose', {}).get(key, {})
            old = origin.get('object', {}) if key == 'object' else origin.get('relative_pose', {}).get(key, {})
            p, p0 = _array(now.get('position'), (3,)), _array(old.get('position'), (3,))
            angle = _rotation_distance(now.get('rotation'), old.get('rotation'))
            if (p is None or p0 is None or angle is None
                    or np.linalg.norm(p - p0) >= cfg['physical_motion']['position_m']
                    or angle >= cfg['physical_motion']['rotation_rad']):
                return False
        return True

    cores = []
    core_start = None
    def finish(end):
        if core_start is not None:
            cores.append({'start_index': core_start, 'stop_index': end,
                'start_raw_frame': rows[core_start]['raw_frame_index'],
                'end_raw_frame': rows[end-1]['raw_frame_index'],
                'duration_s': _duration(rows[core_start:end])})
    for index in range(start, stop):
        if core_start is not None and not fits(index, rows[core_start]):
            finish(index)
            core_start = None
        if core_start is None and fits(index, rows[index]):
            core_start = index
    finish(stop)
    if not cores:
        return reject('no_verified_stationary_core')
    stable_start = cores[0]['start_index']
    phase = 'transition' if changed else 'recenter' if has_hold else 'same_intent'
    return {'phase': phase, 'interpretation': 'verified_static_core_between_accepted_intents',
            'comparable_sides': comparable, 'intent_changed': changed,
            'retained_events_outside_core': retained_boundary_events,
            'human_reposition_inferred': has_hold, 'pre_intent_evidence': before,
            'post_intent_evidence': after, 'raw_pause_reasons': list(dict.fromkeys(reasons)),
            'stationary_start_index': stable_start,
            'stationary_start_raw_frame': rows[stable_start]['raw_frame_index'],
            'stationary_duration_s': sum(c['duration_s'] or 0. for c in cores),
            'stationary_cores': cores,
            'explained_normal_wait': True}


def _wait_exclusions(start, stop, stable_start, budget):
    """Only the stable core consumes the budget; all settling stays outside it."""
    remaining = budget
    first, last = remaining // 2, remaining - remaining // 2
    return list(range(min(stop, stable_start + first), max(stable_start, stop - last)))


def _protection(rows, cfg):
    """Protect actual execution until its physical end; no fixed frame cap."""
    intervals, warnings = [], []
    active = {side: None for side in SIDES}
    ever_contact = {side: False for side in SIDES}
    seen = set()

    def finish(side, end, reason, event=None, uncertain=False):
        cycle = active[side]
        if cycle is None:
            return
        record = cycle | {'side': side, 'end_raw_frame_index': rows[end]['raw_frame_index'],
                          'end_time_s': rows[end]['original_time_s'], 'end_reason': reason,
                          'end_event': event, 'review_required': uncertain}
        intervals.append(record)
        for j in range(cycle['start_index'], end + 1):
            r = rows[j]
            field = 'grasp_protected' if cycle['phase'] == 'grasp' else 'release_protected'
            r[field] = True
            r['protection_by_side'][side].append(cycle['phase'])
            r['protection_start_event'].append(cycle['start_event'])
            if event is not None:
                r['protection_end_event'].append(event)
            r['review_required'] |= uncertain
        active[side] = None

    for i, row in enumerate(rows):
        f = _facts(row)
        if i and row['task_segment_id'] != rows[i - 1]['task_segment_id']:
            for side in SIDES:
                if active[side]:
                    finish(side, i - 1, 'hard_boundary_before_physical_end', uncertain=True)
            ever_contact = {side: False for side in SIDES}
        if row['frame_role'] == 'INVALID':
            continue
        events = _events(row)
        for side in SIDES:
            contact = f.get('contact', {}).get(side, {})
            ever_contact[side] |= contact.get('any_finger') is True or contact.get('stable') is True
            gripper = f.get('grippers', {}).get(side, {})
            execution = [e for e in events if e.get('side') in (side, 'both') and e.get('kind') in
                         {'gripper_close_executed', 'gripper_open_executed', 'gripper_aborted'}]
            for event in execution:
                identity = (row.get('control_epoch'), side, event['kind'], event.get('event_id') or event.get('control_tick'))
                if identity in seen:
                    continue
                seen.add(identity)
                if active[side]:
                    finish(side, i, 'aborted_or_reversed_execution', event, uncertain=event['kind'] == 'gripper_aborted')
                if event['kind'] == 'gripper_aborted':
                    continue
                phase = ('grasp' if event['kind'] == 'gripper_close_executed' else
                         'release' if ever_contact[side] else 'open_adjustment')
                active[side] = {'phase': phase, 'start_index': i, 'start_raw_frame_index': row['raw_frame_index'],
                                'start_time_s': row['original_time_s'], 'start_event': copy.deepcopy(event)}
            cycle = active[side]
            if cycle is None:
                continue
            window = _window(rows, i, cfg['protection']['settle_seconds'])
            if not window or window[0]['raw_frame_index'] < cycle['start_raw_frame_index']:
                continue
            stable_width = _stable_width(window, side, cfg['protection']['aperture_stability_m'])
            all_contact = [_facts(r).get('contact', {}).get(side, {}) for r in window]
            if cycle['phase'] == 'grasp':
                if (stable_width and all(c.get('stable') is True and c.get('double_finger') is True for c in all_contact)
                        and _stable_relative(window, side, cfg['protection'])):
                    finish(side, i, 'actual_grasp_and_relative_pose_stable', {'kind': 'physical_grasp_verified', 'control_tick': row.get('source_control_tick')})
                else:
                    width = _number(gripper.get('actual_width_m'))
                    if stable_width and width is not None and width <= cfg['protection']['aperture_stability_m'] and all(c.get('any_finger') is False for c in all_contact):
                        finish(side, i, 'empty_grasp_closed_and_stable', {'kind': 'empty_grasp_physical_end', 'control_tick': row.get('source_control_tick')}, uncertain=True)
            else:
                opened = []
                for r in window:
                    g = _facts(r).get('grippers', {}).get(side, {})
                    actual, maximum = _number(g.get('actual_width_m')), _number(g.get('max_width_m'))
                    opened.append(actual is not None and maximum is not None and maximum > 0
                                  and actual >= maximum - cfg['protection']['open_width_tolerance_m'])
                clear = all(c.get('any_finger') is False for c in all_contact)
                completed = (cycle['phase'] == 'open_adjustment' or
                             (f.get('task', {}).get('done') is True and f.get('task', {}).get('success') is True and _object_settled(window, cfg)))
                if stable_width and all(opened) and clear and completed:
                    finish(side, i, 'actual_open_contact_clear_and_stable' if cycle['phase'] == 'release' else 'empty_gripper_open_adjustment_complete',
                           {'kind': 'physical_release_verified' if cycle['phase'] == 'release' else 'physical_open_adjustment_complete', 'control_tick': row.get('source_control_tick')})
    for side in SIDES:
        if active[side]:
            finish(side, len(rows) - 1, 'missing_reliable_physical_end', uncertain=True)
            warnings.append(f'{side}:physical_protection_end_unavailable')
    return intervals, warnings


def _terminal_release(rows, cfg):
    """Find actual bilateral completion, never an open request or DONE alone.

    A recorded open execution must follow real contact. Verify measured opening,
    contact clearance and object settling on the original clock. An ABORTED
    controller label after completion neither proves nor disproves these facts.
    Only the final source segment can supply a terminal cut; a later retry or
    task regression disqualifies an earlier apparent completion.
    """
    final_segment = rows[-1]['task_segment_id']
    final_indices = [i for i, r in enumerate(rows) if r['task_segment_id'] == final_segment]
    last_execution = max((i for i in final_indices if any(
        e.get('kind') in {'gripper_open_executed', 'gripper_close_executed'}
        for e in _events(rows[i]))), default=-1)
    good_suffix = [False] * (len(rows) + 1)
    good_suffix[-1] = True
    for i in reversed(final_indices):
        f = _facts(rows[i])
        object_state = f.get('object', {})
        linear = _number(object_state.get('linear_speed_mps'))
        angular = _number(object_state.get('angular_speed_radps'))
        clear_open = True
        for side in SIDES:
            g = f.get('grippers', {}).get(side, {})
            width, maximum = _number(g.get('actual_width_m')), _number(g.get('max_width_m'))
            clear_open &= (
                f.get('contact', {}).get(side, {}).get('any_finger') is False
                and width is not None and maximum is not None and maximum > 0
                and width >= maximum - cfg['protection']['open_width_tolerance_m'])
        good_suffix[i] = bool(
            good_suffix[i + 1] and rows[i]['frame_role'] != 'INVALID'
            and f.get('source', {}).get('available') is True
            and f.get('task', {}).get('done') is True
            and f.get('task', {}).get('success') is True and clear_open
            and linear is not None and angular is not None
            and linear <= cfg['physical_motion']['linear_speed_mps']
            and angular <= cfg['physical_motion']['angular_speed_radps'])
    ever_contact = {s: False for s in SIDES}
    closing = {s: False for s in SIDES}
    release_event = None
    for i in final_indices:
        f = _facts(rows[i])
        if f.get('source', {}).get('available') is not True:
            ever_contact = {s: False for s in SIDES}
            release_event = None
            continue
        for side in SIDES:
            ever_contact[side] |= f.get('contact', {}).get(side, {}).get('any_finger') is True
        for event in _events(rows[i]):
            for side in SIDES:
                if event.get('side') in (side, 'both'):
                    if event.get('kind') == 'gripper_close_executed':
                        closing[side] = True
                    elif event.get('kind') == 'gripper_open_executed':
                        closing[side] = False
            if (event.get('kind') == 'gripper_open_executed'
                    and any(ever_contact[s] for s in SIDES if event.get('side') in (s, 'both'))):
                release_event = copy.deepcopy(event)
        if release_event is None or any(closing.values()) or i < last_execution or not good_suffix[i]:
            continue
        window = _window(rows, i, cfg['protection']['settle_seconds'])
        if (not window or window[0]['raw_frame_index'] < rows[last_execution]['raw_frame_index']
                or not _object_settled(window, cfg)):
            continue
        complete = all(
            _stable_width(window, side, cfg['protection']['aperture_stability_m'])
            for side in SIDES)
        for r in window:
            facts = _facts(r)
            complete &= facts.get('source', {}).get('available') is True
            for side in SIDES:
                g = facts.get('grippers', {}).get(side, {})
                actual, maximum = _number(g.get('actual_width_m')), _number(g.get('max_width_m'))
                complete &= (
                    facts.get('contact', {}).get(side, {}).get('any_finger') is False
                    and actual is not None and maximum is not None and maximum > 0
                    and actual >= maximum - cfg['protection']['open_width_tolerance_m'])
        if complete and _object_settled(rows[i:], cfg):
            last_kept = min(len(rows) - 1, i + cfg['waits']['post_release_frames'])
            return {
                'completion_index': i, 'completion_raw_frame': rows[i]['raw_frame_index'],
                'completion_time_s': rows[i]['original_time_s'],
                'task_segment_id': final_segment, 'release_event': release_event,
                'post_release_frames': cfg['waits']['post_release_frames'],
                'retained_post_frames': last_kept - i, 'last_kept_index': last_kept,
                'last_kept_raw_frame': rows[last_kept]['raw_frame_index'],
                'excluded_frames': len(rows) - last_kept - 1,
                'evidence': 'actual_bilateral_open_contact_clear_object_settled_and_done',
            }
    return None


def _height_regions(rows, cfg, terminal=None):
    """Both sites below threshold is a fallback for absent physical evidence."""
    threshold = cfg['protection']['height_threshold_m']
    for i, row in enumerate(rows):
        features = _facts(row).get('features', {})
        positions = {s: _array(features.get(s + '.actual_position', (None, None))[1], (3,))
                     for s in SIDES}
        available = all(value is not None for value in positions.values())
        row['height_evidence'] = {
            'available': available, 'threshold_m': threshold,
            'left_z_m': None if positions['left'] is None else float(positions['left'][2]),
            'right_z_m': None if positions['right'] is None else float(positions['right'][2]),
            'comparison': 'both_strictly_below',
            'coordinate': 'actual_end_effector_site_world_m',
        }
        low = available and all(value[2] < threshold for value in positions.values())
        missing = physical_evidence_missing(row)
        tail_excluded = bool(terminal and i > terminal['last_kept_index'])
        row['height_evidence']['both_below_threshold'] = bool(low)
        row['height_evidence']['physical_evidence_missing'] = missing
        row['height_evidence']['fallback_required'] = bool(missing)
        row['height_protected'] = bool(low and missing and not tail_excluded)
        row['height_protection_overridden_by_terminal'] = bool(low and missing and tail_excluded)
        row['terminal_tail'] = bool(terminal and i > terminal['completion_index'])
        row['terminal_tail_excluded'] = tail_excluded
        row['terminal_completion_raw_frame'] = terminal['completion_raw_frame'] if terminal else None
        row['terminal_post_release_frames'] = cfg['waits']['post_release_frames']
        # Retain diagnostic aliases, with actual execution once again the
        # primary retention rule regardless of absolute end-effector height.
        row['execution_grasp_phase'] = row['grasp_protected']
        row['execution_release_phase'] = row['release_protected']
        row['execution_phase_by_side'] = copy.deepcopy(row['protection_by_side'])
        if row['height_protected']:
            for side in SIDES:
                row['protection_by_side'][side].append('height_fallback')
    intervals = []
    start = None
    for i in range(len(rows) + 1):
        protected = i < len(rows) and rows[i]['height_protected']
        boundary = i < len(rows) and rows[i]['hard_boundary']
        if start is not None and (not protected or boundary):
            intervals.append({
                'phase': 'height', 'side': 'both', 'start_index': start,
                'start_raw_frame_index': rows[start]['raw_frame_index'],
                'end_raw_frame_index': rows[i-1]['raw_frame_index'],
                'start_time_s': rows[start]['original_time_s'],
                'end_time_s': rows[i-1]['original_time_s'],
                'threshold_m': threshold, 'comparison': 'both_strictly_below',
                'end_reason': 'height_region_or_source_boundary', 'review_required': True,
            })
            start = None
        if protected and start is None:
            start = i
    return intervals


def _confirm_stationary_runs(rows, minimum):
    """Offline confirmation backfills the first low-change row of each run.

    Each comparison uses the history configured above (one previous raw frame
    in the current policy). Source boundaries have no valid comparison and
    cannot contribute confirmation evidence. This never overrides physics.
    """
    for row in rows:
        low = (row['frame_role'] != 'INVALID'
               and row['sa_window']['maximum_changes'] is not None
               and not row['sa_changed'])
        row['sa_stagnant'] = False
        row['sa_stationarity'] = dict(below_threshold=low, confirmed=False,
                                      run_length=0, minimum_frames=minimum,
                                      start_raw_frame=None, end_raw_frame=None)
    start = 0
    while start < len(rows):
        if not rows[start]['sa_stationarity']['below_threshold']:
            start += 1
            continue
        stop = start + 1
        while (stop < len(rows) and rows[stop]['sa_stationarity']['below_threshold']
               and rows[stop]['task_segment_id'] == rows[start]['task_segment_id']):
            stop += 1
        confirmed = stop - start >= minimum
        for row in rows[start:stop]:
            row['sa_stagnant'] = confirmed
            row['sa_stationarity'].update(confirmed=confirmed, run_length=stop-start,
                start_raw_frame=rows[start]['raw_frame_index'],
                end_raw_frame=rows[stop-1]['raw_frame_index'])
            if not confirmed:
                row['policy_reason'] = ['insufficient_consecutive_low_change_frames']
        start = stop


def select_rows(evidence_rows, config=None):
    cfg = effective_config(validate_config(config) if config is not None else load_config())
    rows = copy.deepcopy(evidence_rows)
    if not rows:
        raise ValueError('完整证据帧索引不能为空')
    segment, history_start = -1, 0
    for i, r in enumerate(rows):
        r.pop('sa_stationarity', None)
        f = _facts(r)
        source = f.get('source', {})
        valid = (source.get('valid') is True and _array(r.get('state_recorded'), (16,)) is not None
                 and _array(r.get('action_recorded'), (14,)) is not None and _number(r.get('original_time_s')) is not None)
        hard = i == 0 or source.get('hard_boundary') is True or not valid
        if i:
            p = rows[i - 1]
            hard |= any(r.get(k) != p.get(k) for k in ('source_episode', 'session', 'episode_index', 'control_epoch'))
            t, previous_t = _number(r.get('original_time_s')), _number(p.get('original_time_s'))
            period = _number(source.get('observation_period_s'))
            hard |= (t is None or previous_t is None or t <= previous_t or
                     (period is not None and t - previous_t > period * 1.5 + 1e-9)
                     or r.get('raw_frame_index') != p.get('raw_frame_index', -2) + 1 or p['frame_role'] == 'INVALID')
        if hard:
            segment += 1; history_start = i
        begin = max(history_start, i - cfg['sa']['window_frames'])
        delta = group_deltas(r['state_recorded'], r['action_recorded'],
                            [p['state_recorded'] for p in rows[begin:i]], [p['action_recorded'] for p in rows[begin:i]]) if valid and begin < i else None
        changed = [k for k in GROUPS if delta is not None and delta[k] >= cfg['sa']['thresholds'][k]]
        r.update(task_segment_id=segment, hard_boundary=bool(hard), soft_curation_gap=False,
                 keep=True, effective_keep=True, policy_decision='keep', policy_reason=[], reason=[],
                 frame_role='INVALID' if not valid else 'ACTIVE_MOTION' if changed else 'AMBIGUOUS_IDLE',
                 stagnant_run_id=None, grasp_protected=False, release_protected=False,
                 protection_by_side={s: [] for s in SIDES}, protection_start_event=[], protection_end_event=[],
                 review_required=False, training_approved=False, curation_approved=False, review_status='pending',
                 accepted_left_intent=f.get('accepted_intent', {}).get('left', {}).get('mode'),
                 accepted_right_intent=f.get('accepted_intent', {}).get('right', {}).get('mode'),
                 pre_intent=None, post_intent=None, phase_pause_interpretation=None,
                 sa_window={'history_start_raw_frame': rows[begin]['raw_frame_index'], 'history_count': i-begin,
                            'maximum_changes': delta, 'changed_groups': changed},
                 sa_changed=bool(changed), sa_stagnant=not bool(changed))
        if not valid:
            r.update(keep=False, effective_keep=False, policy_decision='exclude', policy_reason=['invalid_numeric_or_source'], review_required=True)
        elif changed:
            r['policy_reason'] = ['state_or_action_changed']
    if 'minimum_static_frames' in cfg['sa']:
        _confirm_stationary_runs(rows, cfg['sa']['minimum_static_frames'])
    execution_intervals, warnings = _protection(rows, cfg)
    terminal = _terminal_release(rows, cfg)
    intervals = execution_intervals + _height_regions(rows, cfg, terminal)
    required = [r['frame_role'] in ('INVALID', 'ACTIVE_MOTION') or
                (r.get('sa_stationarity', {}).get('below_threshold') is True
                 and not r['sa_stationarity']['confirmed']) for r in rows]
    for i, r in enumerate(rows):
        if r['frame_role'] == 'INVALID':
            continue
        f = _facts(r)
        if r['grasp_protected'] or r['release_protected']:
            r['frame_role'] = 'GRASP_EXECUTION' if r['grasp_protected'] else 'RELEASE_EXECUTION'
            r['policy_reason'] = ['complete_physical_execution_phase']
            required[i] = True
        elif r['height_protected']:
            r['frame_role'] = 'LOW_HEIGHT_PROTECTED'
            r['policy_reason'] = ['both_actual_ee_low_missing_physical_evidence_fallback']
            required[i] = True
        elif f.get('motion', {}).get('moving') is True:
            r['frame_role'] = 'ACTIVE_MOTION' if r['sa_changed'] else 'SEMANTIC_PHYSICAL_HOLD'
            r['policy_reason'] = ['actual_physical_motion']
            required[i] = True
    contexts = []
    for cycle in execution_intervals:
        a, b = cycle['start_index'], next(j for j, r in enumerate(rows) if r['raw_frame_index'] == cycle['end_raw_frame_index'])
        role = 'PRE_GRASP_HOLD' if cycle['phase'] == 'grasp' else 'PRE_RELEASE_HOLD'
        contexts.extend((j, role, 'physical_event_pre_context') for j in range(max(0, a-cfg['context']['before_frames']), a) if rows[j]['task_segment_id'] == rows[a]['task_segment_id'])
        contexts.extend((j, 'SEMANTIC_PHYSICAL_HOLD', 'physical_event_post_context') for j in range(b+1, min(len(rows), b+cfg['context']['after_frames']+1)) if rows[j]['task_segment_id'] == rows[b]['task_segment_id'])
    for i, r in enumerate(rows):
        hazards = execution_hazards(r)
        if hazards:
            contexts.extend((j, 'SEMANTIC_PHYSICAL_HOLD', 'physical_or_task_event_context') for j in range(max(0,i-cfg['context']['before_frames']),min(len(rows),i+cfg['context']['after_frames']+1)) if rows[j]['task_segment_id']==r['task_segment_id'])
        if hazards:
            required[i] = True
            r['policy_reason'] = sorted(set(r['policy_reason'] + hazards))
            if r['frame_role'] == 'AMBIGUOUS_IDLE':
                r['frame_role'] = 'SEMANTIC_PHYSICAL_HOLD'
    for i, role, reason in contexts:
        if rows[i]['frame_role'] == 'INVALID':
            continue
        if not required[i]:
            rows[i]['frame_role'] = role
            rows[i]['policy_reason'] = [reason]
        required[i] = True
    for i, row in enumerate(rows):
        if row['frame_role'] != 'INVALID' and _facts(row).get('source', {}).get('available') is False:
            required[i] = True
            row['review_required'] = True
            row['policy_reason'].append('incomplete_source_evidence')
        if row['frame_role'] != 'INVALID' and not row['height_evidence']['available']:
            required[i] = True
            row['review_required'] = True
            row['policy_reason'].append('missing_actual_ee_height')
        missing = physical_evidence_missing(row)
        if row['frame_role'] != 'INVALID' and missing:
            required[i] = True
            row['review_required'] = True
            row['policy_reason'].append('missing_physical_or_execution_evidence')
    prefix_boundary = _initial_activity_boundary(rows, cfg)
    if prefix_boundary is not None and not required[prefix_boundary]:
        # Split a subthreshold rolling run at its cumulative/hidden departure.
        required[prefix_boundary] = True
        rows[prefix_boundary].update(frame_role='ACTIVE_MOTION',
            policy_reason=['initial_pose_or_accepted_target_departure'])
    # Ordinary A/S-stagnant runs are descriptive; physical protection can occupy
    # part of one run without turning its entire descent or tail into protection.
    stagnant = []
    start = 0
    while start < len(rows):
        if not rows[start]['sa_stagnant'] or rows[start]['frame_role']=='INVALID':
            start += 1; continue
        stop = start+1
        while stop<len(rows) and rows[stop]['sa_stagnant'] and rows[stop]['frame_role']!='INVALID' and rows[stop]['task_segment_id']==rows[start]['task_segment_id']:
            stop += 1
        rid=len(stagnant)
        for r in rows[start:stop]:r['stagnant_run_id']=rid
        sequence, uncertain, safety=_internal_intents(rows[start:stop])
        k=cfg['intent']['stable_frames']
        pre=_stable_intent(rows[max(0,start-k):start],cfg)
        post=_stable_intent(rows[stop:stop+k],cfg)
        if pre and rows[max(0,start-k)]['task_segment_id']!=rows[start]['task_segment_id']:pre=None
        if post and rows[stop]['task_segment_id']!=rows[start]['task_segment_id']:post=None
        events=[e for r in rows[start:stop] for e in _events(r)]
        stagnant.append({'id':rid,'start_raw_frame_index':rows[start]['raw_frame_index'],'end_raw_frame_index':rows[stop-1]['raw_frame_index'],
                         'duration_s':_duration(rows[start:stop]),
                         'pre_intent':list(pre) if pre else None,'post_intent':list(post) if post else None,
                         'task_events':[e for e in events if e.get('kind')=='task_state_changed'],
                         'physical_events':[e for e in events if e.get('kind') in PHYSICAL_EVENTS |
                                            {'gripper_open_executed','gripper_close_executed','gripper_aborted'}],
                         'intent_sequence_inside_run':[list(v) for v in sequence], 'safety_pause':safety,'unreliable_intent':uncertain})
        start=stop
    activity=[i == prefix_boundary or r['sa_changed'] or _facts(r).get('motion',{}).get('moving') is True or r['execution_grasp_phase'] or r['execution_release_phase'] for i,r in enumerate(rows)]
    for a,b in _runs([not x for x in required]):
        # Never classify across a genuine source boundary.
        cuts=[a]+[i for i in range(a+1,b) if rows[i]['task_segment_id']!=rows[i-1]['task_segment_id']]+[b]
        for start,stop in zip(cuts,cuts[1:]):
            part=rows[start:stop];k=cfg['intent']['stable_frames']
            before=_stable_intent(rows[max(0,start-k):start],cfg);after=_stable_intent(rows[stop:stop+k],cfg)
            if before and rows[max(0,start-k)]['task_segment_id']!=part[0]['task_segment_id']:before=None
            if after and rows[stop]['task_segment_id']!=part[0]['task_segment_id']:after=None
            seq,unreliable,safety=_internal_intents(part)
            changed_inside=bool(before and any(any(v is not None and v!='HOLD' and v!=before[j] for j,v in enumerate(pair)) for pair in seq))
            same_segment=[i for i,r in enumerate(rows) if r['task_segment_id']==part[0]['task_segment_id']]
            earlier=any(activity[i] for i in same_segment if i<start);later=any(activity[i] for i in same_segment if i>=stop)
            available=all(_facts(r).get('motion',{}).get('available') is True for r in part)
            duration=_duration(part)
            role='AMBIGUOUS_IDLE';reason='insufficient_reliable_evidence';remove=[]
            task_done=all(_facts(r).get('task',{}).get('done') is True and _facts(r).get('task',{}).get('success') is True for r in part)
            settled=_object_settled(_window(rows,start,cfg['protection']['settle_seconds']),cfg)
            changed_anywhere = any(len({pair[side] for pair in seq if pair[side] in ACTIVE_MODES}) > 1
                                   for side in range(2))
            no_prefix_contact = all(
                _facts(rows[i]).get('contact', {}).get(side, {}).get('any_finger') is False
                for i in same_segment if i < stop for side in SIDES)
            no_new_physical_event = not any(e.get('kind') in PHYSICAL_EVENTS |
                {'gripper_open_executed', 'gripper_close_executed', 'gripper_aborted'}
                for r in part for e in _events(r))
            prefix = (available and duration is not None and not earlier and later
                      and no_prefix_contact and prefix_boundary is not None and stop <= prefix_boundary
                      and part[0]['task_segment_id'] == rows[0]['task_segment_id'])
            tail = available and duration is not None and task_done and settled and not later and no_new_physical_event
            pause_context = _pause_context(part, 'prefix' if prefix else 'tail') if prefix or tail else None
            wait_diagnostics = {}
            recenter = (_recenter_context(rows, start, stop, cfg, wait_diagnostics)
                        if not prefix and not tail and available and duration is not None and earlier and later else None)
            normal_phase_pause = bool(pause_context and pause_context['explained_normal_wait'])
            if recenter:
                phase = recenter['phase']
                role, reason = {
                    'recenter': ('RECENTER_IDLE', 'stationary_reposition_same_intent'),
                    'transition': ('INTENT_TRANSITION_IDLE', 'stationary_intent_transition'),
                    'same_intent': ('SAME_INTENT_IDLE', 'same_fresh_intent_without_physical_event'),
                }[phase]
                pause_context = recenter
                before, after = tuple(recenter['pre_intent_evidence']['modes']), tuple(recenter['post_intent_evidence']['modes'])
                stable_start = recenter['stationary_start_index']
                minimum = (cfg['waits']['minimum_middle_s'] if phase == 'same_intent' else
                           cfg['waits'].get('minimum_transition_s', cfg['waits']['minimum_recenter_s'])
                           if phase == 'transition' else cfg['waits']['minimum_recenter_s'])
                budget = cfg['waits']['changed_intent_frames' if recenter['intent_changed'] else 'same_intent_frames']
                for core in recenter['stationary_cores']:
                    if core['duration_s'] is not None and core['duration_s'] + 1e-9 >= minimum:
                        remove.extend(_wait_exclusions(core['start_index'], core['stop_index'], core['start_index'], budget))
                core_frames = sum(c['stop_index']-c['start_index'] for c in recenter['stationary_cores'])
                recenter.update(retained_frame_budget=budget,
                                stable_core_retained_frames=core_frames-len(remove),
                                settling_retained_frames=stop-start-core_frames,
                                retained_frames=stop-start-len(remove), minimum_duration_s=minimum)
            elif safety and not normal_phase_pause:
                reason='safety_pause_requires_review'
            elif prefix:
                # Before any robot activity, absent/stale hand recognition and
                # mode changes describe preparation, not an in-task transition.
                # Physical motion, execution/contact events and missing source
                # evidence have already vetoed this path. Mid-task waits still
                # require same-side comparable intent plus bilateral physics.
                role='PREFIX_IDLE';reason='pre_activity_redundant_wait'
                if duration>=cfg['waits']['minimum_prefix_s']:remove=list(range(start,max(start,stop-cfg['context']['before_frames'])))
            elif before and after and (before!=after or changed_inside):
                role='INTENT_TRANSITION_HOLD';reason='accepted_intent_transition_context'
            elif changed_anywhere:
                role='INTENT_TRANSITION_HOLD';reason='accepted_intent_transition_context'
            elif unreliable and not (tail and normal_phase_pause):
                reason='insufficient_reliable_evidence'
            elif tail:
                role='AMBIGUOUS_IDLE';reason='missing_actual_release_completion'
            if not recenter and wait_diagnostics:
                pause_context = wait_diagnostics
                if role == 'AMBIGUOUS_IDLE':
                    reason = wait_diagnostics.get('reason', reason)
                pre = wait_diagnostics.get('pre_intent_evidence')
                post = wait_diagnostics.get('post_intent_evidence')
                before = tuple(pre['modes']) if pre else before
                after = tuple(post['modes']) if post else after
            for i in range(start,stop):
                r=rows[i];r.update(frame_role=role,policy_reason=[reason],pre_intent=list(before) if before else None,
                                  post_intent=list(after) if after else None,
                                  review_required=role=='AMBIGUOUS_IDLE' or (role=='INTENT_TRANSITION_HOLD' and unreliable),
                                  phase_pause_interpretation=copy.deepcopy(pause_context))
                if i in remove:r.update(keep=False,effective_keep=False,policy_decision='exclude',soft_curation_gap=True)
                elif role in ('PREFIX_IDLE','TAIL_IDLE','SAME_INTENT_IDLE','RECENTER_IDLE','INTENT_TRANSITION_IDLE'):r['policy_reason'].append('wait_context_or_below_minimum_duration')
                if recenter and not any(c['start_index'] <= i < c['stop_index'] for c in recenter['stationary_cores']):
                    r.update(frame_role='SEMANTIC_PHYSICAL_HOLD',
                             policy_reason=['settling_or_target_change_before_stationary_recenter'])
    if terminal:
        for i in range(terminal['completion_index'], len(rows)):
            r = rows[i]
            excluded = i > terminal['last_kept_index']
            r.update(
                keep=not excluded, effective_keep=not excluded,
                policy_decision='exclude' if excluded else 'keep',
                soft_curation_gap=excluded, review_required=False,
                frame_role=('TAIL_IDLE' if excluded else 'RELEASE_TAIL_CONTEXT'
                            if i > terminal['completion_index'] else 'SEMANTIC_PHYSICAL_HOLD'),
                policy_reason=['after_verified_release_tail_budget' if excluded
                               else 'verified_release_post_context' if i > terminal['completion_index']
                               else 'verified_bilateral_release_complete'],
                phase_pause_interpretation=copy.deepcopy(terminal))
    for r in rows:
        r['reason'] = list(r['policy_reason'])
    segments, cuts = resegment_rows(rows)
    from dataset.task3.sequence_stitching import build_sequence_contract
    sequence_contract = build_sequence_contract(rows, cfg)
    rows = sequence_contract.pop('rows')
    # Seam repair can restore original boundary rows. Refresh fragment metadata
    # without conflating raw fragments with independently certified sequences.
    segments, cuts = resegment_rows(rows)
    retained_durations=[_duration([r]) for r in rows if r['keep']]
    filtered_durations=[_duration([r]) for r in rows if not r['keep']]
    metrics={'version':VERSION,'raw_frames':len(rows),'kept_frames':sum(r['keep'] for r in rows),'filtered_frames':sum(not r['keep'] for r in rows),
             'segment_count':len(segments),'task_segment_count':segment+1,'stagnant_runs':len(stagnant),'training_approved':False,
             'retained_physics_s':None if any(v is None for v in retained_durations) else sum(retained_durations),
             'filtered_physics_s':None if any(v is None for v in filtered_durations) else sum(filtered_durations),
             'unknown_sampling_period_frames':sum(_duration([r]) is None for r in rows),
             'frame_role_counts':{role:sum(r['frame_role']==role for r in rows) for role in sorted({r['frame_role'] for r in rows})},
             'grasp_protected_frames':sum(r['grasp_protected'] for r in rows),'release_protected_frames':sum(r['release_protected'] for r in rows),
             'height_protected_frames':sum(r['height_protected'] for r in rows),
             'execution_grasp_frames':sum(r['execution_grasp_phase'] for r in rows),
             'execution_release_frames':sum(r['execution_release_phase'] for r in rows),
             'ambiguous_frames':sum(r['frame_role']=='AMBIGUOUS_IDLE' for r in rows),
             'same_intent_idle_frames':sum(r['frame_role']=='SAME_INTENT_IDLE' for r in rows),
             'recenter_idle_frames':sum(r['frame_role']=='RECENTER_IDLE' for r in rows),
             'intent_transition_idle_frames':sum(r['frame_role']=='INTENT_TRANSITION_IDLE' for r in rows),
             'terminal_tail_excluded_frames':sum(r['terminal_tail_excluded'] for r in rows),
             'intent_transition_hold_frames':sum(r['frame_role']=='INTENT_TRANSITION_HOLD' for r in rows),
             'soft_gap_count':sum(c['kind']=='soft_curation_gap' for c in cuts),
             'review_required_frames':sum(r['review_required'] for r in rows)}
    return {'rows':rows,'metrics':metrics,'segments':segments,'cuts':cuts,'stagnant_runs':stagnant,
            'protected_intervals':intervals,'execution_intervals':execution_intervals,'terminal_release':terminal,
            'events':[e for r in rows for e in _events(r)],'warnings':warnings,
            'sequence_contract': sequence_contract}


def resegment_rows(rows):
    """Rebuild effective contiguous fragments in place, leaving policy immutable.

    Human exclusion is an explicit non-bridgeable boundary. Automated valid
    exclusions are soft gaps; neither changes the factual task segment.
    """
    segments=[];previous=None;fragment=-1
    for i,r in enumerate(rows):
        r.pop('gap_before', None)
        if r.get('human_decision') == 'exclude':
            r.update(keep=False, effective_keep=False, soft_curation_gap=False)
        elif r.get('human_decision') == 'keep' and r.get('frame_role') != 'INVALID':
            r.update(keep=True, effective_keep=True, soft_curation_gap=False)
        elif r.get('frame_role') == 'INVALID':
            r.update(keep=False, effective_keep=False, soft_curation_gap=False)
        else:
            r['effective_keep'] = bool(r['keep'])
        if not r['keep']:
            r.update(segment_id=None,contiguous_fragment_id=None,cut_reasons=[]);continue
        boundary=previous is None or i!=previous+1 or r['task_segment_id']!=rows[previous]['task_segment_id']
        if boundary:
            fragment+=1
            segments.append({'segment_id':fragment,'contiguous_fragment_id':fragment,'task_segment_id':r['task_segment_id'],
                             'source_episode':r.get('source_episode'),'start_raw_frame_index':r['raw_frame_index'],'end_raw_frame_index':r['raw_frame_index'],
                             'start_time_s':r['original_time_s'],'end_time_s':r['original_time_s'],'control_epoch':r.get('control_epoch'),'training_approved':bool(r.get('training_approved'))})
        if previous is not None and i>previous+1:
            excluded = rows[previous+1:i]
            kind = ('hard_boundary' if r['task_segment_id']!=rows[previous]['task_segment_id']
                    else 'human_exclusion' if any(x.get('human_decision')=='exclude' for x in excluded)
                    else 'soft_curation_gap')
            r['gap_before']={'kind':kind,
                             'start_raw_frame_index':rows[previous+1]['raw_frame_index'],'end_raw_frame_index':rows[i-1]['raw_frame_index']}
        r.update(segment_id=fragment,contiguous_fragment_id=fragment,
                 cut_reasons=['hard_boundary'] if r.get('hard_boundary') else [r.get('gap_before', {}).get('kind','filtered_gap')] if boundary and previous is not None else ['first_retained_frame'] if previous is None else [])
        segments[-1].update(end_raw_frame_index=r['raw_frame_index'],end_time_s=r['original_time_s']);previous=i
    cuts=[]
    for a,b in _runs([not r['keep'] for r in rows]):
        ends=[a]+[i for i in range(a+1,b) if rows[i]['task_segment_id']!=rows[i-1]['task_segment_id']]+[b]
        for start,stop in zip(ends,ends[1:]):
            part=rows[start:stop]
            kind=('hard_boundary' if any(r.get('frame_role')=='INVALID' for r in part)
                  else 'human_exclusion' if any(r.get('human_decision')=='exclude' for r in part)
                  else 'soft_curation_gap')
            cuts.append({'start_raw_frame_index':part[0]['raw_frame_index'],'end_raw_frame_index':part[-1]['raw_frame_index'],
                         'start_time_s':part[0]['original_time_s'],'end_time_s':part[-1]['original_time_s'],
                         'kind':kind,'reasons':sorted({reason for r in part for reason in r.get('policy_reason', [])}),
                         'review_status':'pending'})
    return segments, cuts


def derive_episode(episode, config=None):
    from dataset.task3.frame_evidence import build_frame_evidence
    cfg=validate_config(config) if config is not None else load_config()
    evidence=build_frame_evidence(episode,effective_config(cfg))
    out=select_rows(evidence['rows'],cfg)
    out['events']=evidence.get('events',out['events'])
    out['warnings']=sorted(set(out['warnings']+evidence.get('warnings',[])))
    return out


def build_windows(rows, length=30, observation_horizon=1, stride=1, pad=True):
    """Return bounded, masked source-index windows; never synthesize terminals.

    Passing all index rows or only reviewed retained rows is supported. Each
    maximal contiguous run is rechecked so later manual cuts cannot leak into
    an old segment. Repeated boundary indices are storage padding only.
    """
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in (length, observation_horizon, stride)):
        raise ValueError('positive integer horizons and stride required')
    groups = []; current = []
    for row in rows:
        if not row.get('keep', True) or row.get('segment_id') is None:
            if current:
                groups.append(current); current = []
            continue
        if current:
            prev = current[-1]
            keys = ('source_episode', 'session', 'episode_index', 'segment_id', 'control_epoch')
            a = _number(prev.get('original_time_s')); b = _number(row.get('original_time_s'))
            time_gap = a is not None and b is not None and not (0 < b - a <= .05 + 1e-9)
            if any(row.get(k) != prev.get(k) for k in keys) or row['raw_frame_index'] != prev['raw_frame_index'] + 1 or time_gap:
                groups.append(current); current = []
        current.append(row)
    if current:
        groups.append(current)
    windows = []
    for group in groups:
        for i in range(0, len(group), stride):
            future = group[i:i + length]
            if not pad and len(future) < length:
                continue
            valid = len(future)
            future = future + [future[-1]] * (length - valid)
            a = max(0, i - observation_horizon + 1); history = group[a:i + 1]
            history_pad = observation_horizon - len(history)
            history = [history[0]] * history_pad + history
            r = group[i]
            windows.append({'source_episode': r.get('source_episode'), 'session': r.get('session'),
                            'episode_index': r.get('episode_index'), 'segment_id': r.get('segment_id'),
                            'control_epoch': r.get('control_epoch'), 'anchor_raw_frame_index': r['raw_frame_index'],
                            'source_indices': [x['raw_frame_index'] for x in future],
                            'source_control_ticks': [x.get('source_control_tick') for x in future],
                            'is_pad': [j >= valid for j in range(length)], 'loss_mask': [j < valid for j in range(length)],
                            'observation_indices': [x['raw_frame_index'] for x in history],
                            'observation_is_pad': [j < history_pad for j in range(observation_horizon)],
                            'actual_terminal_mask': [j < valid and bool(x.get('terminal', {}).get('actual_done')) for j, x in enumerate(future)],
                            'truncated': valid < length and not future[valid - 1].get('terminal', {}).get('actual_done', False),
                            'executor_labels_frozen': False})
    return windows
