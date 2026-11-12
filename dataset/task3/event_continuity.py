"""Classify execution effects from recorded events and bracketing control ticks.

This is shared by curation and Safe Stitch. An unchanged event is only one
piece of evidence: the seam still checks every physical state and causal label.
Missing evidence is never synthesized from an event name or a rejected flag.
"""
from __future__ import annotations

import math
import numpy as np

POLICY = 'verified_execution_v1'
SIDES = ('left', 'right')
TRACKING_PAUSES = frozenset(('hand_lost', 'hand_near_frame_edge', 'tracking_recovery',
    'camera_stalled', 'stale_recognition', 'command_timeout', 'transport_recovery'))
RECOVERY_REJECTIONS = TRACKING_PAUSES | {'stale_command', 'stale_measurement'}
RAW_EVENTS = {'raw_state', 'raw_request_received', 'raw_request_queued',
              'raw_request_dropped', 'raw_request_rejected'}
METADATA_EVENTS = {'accepted_intent_changed', 'pause_changed', 'gripper_baseline_changed',
                   'raw_release_confirmation_cancelled'}
CONTROL_COMMANDS = {'continuous_input', 'telemetry'}


def enabled(config):
    return (config or {}).get('stitching', {}).get('event_policy') == POLICY


def action_from_facts(facts):
    values = []
    for side in SIDES:
        for key, size in (('target_position', 3), ('target_rpy', 3), ('target_aperture', 1)):
            field = facts.get('features', {}).get(f'{side}.{key}')
            if not isinstance(field, (list, tuple)) or len(field) != 2:
                return None
            a = vector(field[1], (size,))
            if a is None:
                return None
            values.extend(a.tolist())
    return np.asarray(values)


def vector(value, shape):
    try:
        a = np.asarray(value, dtype=float)
        return a if a.shape == shape and np.isfinite(a).all() else None
    except (TypeError, ValueError):
        return None


def target_changed(a, b, cfg):
    limits = cfg['sa']['thresholds']
    for i in (0, 7):
        d = a[i:i+7] - b[i:i+7]
        if (np.linalg.norm(d[:3]) >= limits['action_position_m']
                or np.linalg.norm((d[3:6]+np.pi) % (2*np.pi)-np.pi) >= limits['action_rotation_rad']
                or abs(d[6]) >= limits['action_aperture_m']):
            return True
    return False


def settled_execution(facts):
    execution = facts.get('execution') or {}
    return (execution.get('available') is True and all(
        execution.get('sides', {}).get(s, {}).get(k) is False
        for s in SIDES for k in ('pending_authorization', 'release_confirmation_pending', 'pending')))


def interval_controls(facts):
    interval = facts.get('interval') or {}
    before = interval.get('previous_control')
    return ([before] if isinstance(before, dict) else []) + list(interval.get('controls') or [])


def classify_event(event, controls, cfg):
    kind = event.get('kind')
    result = {'kind': kind, 'source_line': event.get('source_line'),
              'event_id': event.get('event_id'), 'effect': 'unknown', 'reasons': []}
    def reject(reason, changed=False):
        result['effect'] = 'changed' if changed else 'unknown'
        result['reasons'].append(reason)
        return result
    if kind not in RAW_EVENTS | METADATA_EVENTS | {'request_accepted', 'request_rejected'}:
        return reject('causal_event:' + str(kind), changed=True)
    command, reason = event.get('command'), event.get('reason')
    if kind == 'pause_changed' and event.get('after') is True and reason not in TRACKING_PAUSES:
        return reject('safety_pause_event')
    tracking_pause = command == 'pause' and event.get('requested_pause_reason') in TRACKING_PAUSES
    if kind in {'request_accepted', 'request_rejected', 'raw_request_received',
                'raw_request_queued', 'raw_request_dropped', 'raw_request_rejected'} and command not in CONTROL_COMMANDS and not tracking_pause:
        return reject('non_motion_control_request:' + str(command))
    if kind == 'raw_release_confirmation_cancelled' and reason not in TRACKING_PAUSES:
        return reject('release_cancellation_requires_review')
    if kind in ('request_rejected', 'raw_request_rejected') and reason not in RECOVERY_REJECTIONS:
        return reject('unexplained_rejection:' + str(reason))
    if kind == 'raw_request_dropped' and reason not in {'superseded_telemetry', 'superseded_motion', 'newer_telemetry'}:
        return reject('dropped_request_requires_review')
    line = event.get('source_line')
    if type(line) is not int:
        return reject('event_source_missing')
    before = [c for c in controls if type(c.get('source_line')) is int and c['source_line'] < line]
    after = [c for c in controls if type(c.get('source_line')) is int and c['source_line'] >= line]
    if not before or not after:
        return reject('event_bracketing_controls_missing')
    a, b = max(before, key=lambda c:c['source_line']), min(after, key=lambda c:c['source_line'])
    result['bracketing_control_ticks'] = [a.get('tick'), b.get('tick')]
    result['bracketing_source_lines'] = [a['source_line'], b['source_line']]
    if any(event.get(k) != a.get(k) or b.get(k) != a.get(k) for k in ('session', 'control_epoch')):
        return reject('event_source_context_mismatch')
    if type(a.get('tick')) is not int or b.get('tick') != a['tick']+1:
        return reject('event_control_tick_gap')
    times = [c.get('sim_time') for c in (a,b)]
    if (any(type(t) not in (int,float) or not math.isfinite(t) for t in times)
            or not 0 < times[1]-times[0] <= .03):
        return reject('event_control_clock_gap')
    for c in (a,b):
        s = c.get('source') or {}
        if (s.get('valid') is not True or s.get('available') is not True
                or s.get('hard_boundary') or s.get('reasons') or c.get('invalid')):
            return reject('event_control_source_invalid')
        if not settled_execution(c):
            return reject('event_execution_pending_or_unknown')
    targets = [action_from_facts(c) for c in (a,b)]
    if any(t is None for t in targets):
        return reject('event_control_target_missing')
    if target_changed(*targets, cfg):
        return reject('event_control_target_changed', changed=True)
    for side in SIDES:
        ga, gb = [c.get('grippers',{}).get(side,{}) for c in (a,b)]
        for key in ('goal_width_m', 'command_width_m', 'actual_width_m'):
            x,y = ga.get(key),gb.get(key)
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in (x,y)):
                return reject('event_gripper_evidence_missing:' + side)
            if abs(x-y) >= cfg['sa']['thresholds']['action_aperture_m']:
                return reject('event_gripper_changed:' + side, changed=True)
        for key in ('authorized_mode', 'edge_event_id', 'edge_count'):
            if key not in ga or key not in gb or ga[key] != gb[key]:
                return reject('event_authorization_changed_or_missing:' + side)
        if any(g.get(k) is True for g in (ga,gb) for k in ('executed_open','executed_close','aborted')):
            return reject('event_gripper_execution:' + side, changed=True)
        for key in ('fingers','any_finger','double_finger','qualified','stable'):
            ca,cb = [c.get('contact',{}).get(side,{}) for c in (a,b)]
            if key not in ca or key not in cb or ca[key] != cb[key] or ca[key] is None:
                return reject('event_contact_changed_or_missing:' + side)
    if kind in ('request_accepted','request_rejected'):
        execution = event.get('execution')
        if execution is not None and not settled_execution({'execution': execution}):
            return reject('accepted_request_pending_execution')
        observed = vector(event.get('accepted_target'), (14,))
        if observed is None:
            return reject('event_post_execution_target_missing')
        if any(target_changed(t, observed, cfg) for t in targets):
            return reject('event_post_execution_target_changed', changed=True)
        if event.get('recorded_post_target') is not None:
            recorded = vector(event['recorded_post_target'], (14,))
            if recorded is None or target_changed(observed, recorded, cfg):
                return reject('event_post_execution_target_mismatch')
        if kind == 'request_accepted' and command == 'continuous_input' and execution is None:
            return reject('accepted_request_execution_unknown')
        # A between-tick authorization must not hide behind unchanged tick endpoints.
        edges = event.get('gripper_edges')
        edge_policy = (execution or {}).get('strategy') in ('palm_fist_edge_v1', 'palm_fist_edge_v2')
        if edge_policy and not isinstance(edges, dict):
            return reject('accepted_request_edge_missing')
        if isinstance(edges,dict):
            for side in SIDES:
                edge, previous = edges.get(side,{}) or {}, a['grippers'][side]
                for name, key in (('authorized_mode','authorized_mode'), ('event_id','edge_event_id'), ('edge_count','edge_count')):
                    if edge_policy and name not in edge:
                        return reject('accepted_request_edge_missing:' + side + '.' + name)
                    if name in edge and edge[name] != previous.get(key):
                        return reject('event_authorization_changed:' + side, changed=True)
    result.update(effect='unchanged', reasons=['verified_unchanged_execution_between_controls'])
    return result
