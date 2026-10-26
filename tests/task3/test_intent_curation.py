"""Intent curation is offline and conservative; fixtures are never raw data."""
import copy
import json
import unittest

import numpy as np

from dataset.task3.curation_rules import (
    build_windows, effective_config, group_deltas, load_config, resegment_rows,
    select_rows, validate_config,
)


def evidence_rows(count=180):
    result = []
    for i in range(count):
        result.append({
            'source_episode': 'fixture/0', 'session': 'fixture', 'episode_index': 0,
            'control_epoch': 1, 'raw_frame_index': i, 'source_control_tick': i * 2,
            'original_time_s': i / 30., 'state_recorded': [0.] * 16,
            'action_recorded': [0.] * 14, 'status_summary': {'paused': False},
            'terminal': {'actual_done': False}, 'auxiliary': {},
            'evidence': {
                'source': {'valid': True, 'available': True, 'hard_boundary': False,
                           'observation_period_s': 1/30, 'control_period_s': 1/60},
                'accepted_intent': {s: {'mode': 'XY_HOLD', 'reliable': True,
                    'event_id': f'{s}e{i//2}', 'measurement_id': f'm{i//2}'}
                    for s in ('left', 'right')},
                'interval': {'mode_sequence': [], 'safety_pause': False,
                             'command_activity_available': True, 'command_activity': False,
                             'command_changed_from_previous': False,
                             'execution_available': True, 'execution_pending': False},
                'execution': {'available': True, 'strategy': 'palm_fist_edge_v2',
                              'sides': {s: {'available': True, 'pending': False}
                                        for s in ('left', 'right')}},
                'events': [], 'motion': {'moving': False, 'available': True},
                'features': {
                    **{f'{s}.actual_position': ('position', [0., y, .5])
                       for s, y in (('left', -.2), ('right', .2))},
                    **{f'{s}.actual_rotation': ('rotation', np.eye(3).tolist())
                       for s in ('left', 'right')},
                },
                'grippers': {s: {'actual_width_m': .03, 'command_width_m': .03,
                    'goal_width_m': .03, 'max_width_m': .08, 'edge_state': 'IDLE'}
                    for s in ('left', 'right')},
                'contact': {s: {'stable': False, 'double_finger': False, 'any_finger': False}
                    for s in ('left', 'right')},
                'relative_pose': {s: {'available': True, 'position': [0., 0., 0.],
                    'rotation': np.eye(3).tolist()} for s in ('left', 'right')},
                'object': {'position': [0., 0., .4], 'rotation': [1., 0., 0., 0.],
                    'linear_speed_mps': 0., 'angular_speed_radps': 0.},
                'task': {'done': False, 'success': False},
            },
        })
    return result


def active(rows, a, b):
    for row in rows[a:b]:
        row['evidence']['motion']['moving'] = True


def execution(rows, i, side='left', opening=False, event='edge'):
    rows[i]['evidence']['events'].append({
        'kind': 'gripper_open_executed' if opening else 'gripper_close_executed',
        'side': side, 'event_id': event, 'control_tick': i*2, 'source_line': i*5+2})


class ConfigTests(unittest.TestCase):
    def test_config_help_is_nonsemantic_and_input_not_mutated(self):
        config = load_config()
        before = copy.deepcopy(config)
        config.setdefault('_help', {})['new_help'] = '文档变化'
        self.assertEqual(effective_config(config), effective_config(before))
        validate_config(config)
        self.assertEqual(config['_help']['new_help'], '文档变化')

    def test_reject_unsafe_and_misspelled_settings(self):
        for section, key, value in (
            ('intent', 'stable_frames', True), ('intent', 'min_distinct_inputs', 4),
            ('sa', 'window_frames', 0), ('context', 'before_frames', -1),
            ('protection', 'settle_seconds', float('nan')),
            ('physical_motion', 'position_m', float('inf')),
            ('intent', 'maximum_age_ms', 150), ('waits', 'post_release_frames', -1),
        ):
            with self.subTest(section=section, key=key):
                config = load_config(); config[section][key] = value
                with self.assertRaises(ValueError):
                    validate_config(config)

    def test_sa_units_and_rotation_wrap_unchanged(self):
        state = np.zeros(16); state[0] = .003; state[15] = .0006
        action = np.zeros(14); action[0:3] = [.0003, .0004, 0.]
        action[3] = 2*np.pi-.003; action[13] = .0007
        delta = group_deltas(state, action, np.zeros((3, 16)), np.zeros((3, 14)))
        self.assertAlmostEqual(delta['state_joint_rad'], .003)
        self.assertAlmostEqual(delta['state_aperture_m'], .0006)
        self.assertAlmostEqual(delta['action_position_m'], .0005)
        self.assertAlmostEqual(delta['action_rotation_rad'], .003)
        self.assertAlmostEqual(delta['action_aperture_m'], .0007)


class IntentCurationTests(unittest.TestCase):
    def test_same_fresh_intent_wait_only_removes_middle_and_keeps_context(self):
        rows = evidence_rows(160); active(rows, 0, 10); active(rows, 140, 160)
        out = select_rows(rows)
        self.assertEqual(out['rows'][60]['frame_role'], 'SAME_INTENT_IDLE')
        self.assertFalse(out['rows'][60]['keep'])
        self.assertEqual(sum(r['keep'] for r in out['rows'][10:140]), 3)
        self.assertEqual(out['metrics']['task_segment_count'], 1)
        self.assertEqual(len(out['segments']), 2)
        self.assertEqual(out['rows'][138]['gap_before']['kind'], 'soft_curation_gap')
        self.assertTrue(out['rows'][60]['soft_curation_gap'])
        self.assertEqual(out['metrics']['intent_transition_hold_frames'], 0)

    def test_unknown_context_does_not_delete_middle_wait(self):
        rows = evidence_rows(160); active(rows, 0, 10); active(rows, 140, 160)
        for row in rows[:10]:
            for side in ('left', 'right'):
                row['evidence']['accepted_intent'][side]['reliable'] = False
        out = select_rows(rows)
        self.assertTrue(out['rows'][60]['keep'])
        self.assertEqual(out['rows'][60]['frame_role'], 'AMBIGUOUS_IDLE')

    def test_repeated_single_input_is_not_three_reliable_context_frames(self):
        rows = evidence_rows(160); active(rows, 0, 10); active(rows, 140, 160)
        for row in rows:
            for side in ('left', 'right'):
                row['evidence']['accepted_intent'][side].update(event_id='old', measurement_id='old')
        out = select_rows(rows)
        self.assertTrue(out['rows'][60]['keep'])
        self.assertTrue(out['rows'][60]['review_required'])

    def test_static_transition_compresses_but_hidden_mode_excursion_stays(self):
        for hidden in (False, True):
            rows = evidence_rows(160); active(rows, 0, 10); active(rows, 140, 160)
            if hidden:
                rows[70]['evidence']['interval']['mode_sequence'] = [{
                    'left': 'Z_CLOSE', 'right': 'XY_HOLD', 'left_reliable': True,
                    'right_reliable': True}, {
                    'left': 'XY_HOLD', 'right': 'XY_HOLD', 'left_reliable': True,
                    'right_reliable': True}]
            else:
                for row in rows[140:]:
                    row['evidence']['accepted_intent']['left']['mode'] = 'Z_CLOSE'
            out = select_rows(rows)
            if hidden:
                self.assertTrue(all(r['keep'] for r in out['rows'][10:140]))
                self.assertEqual(out['rows'][60]['frame_role'], 'INTENT_TRANSITION_HOLD')
            else:
                self.assertEqual(sum(r['keep'] for r in out['rows'][10:140]), 10)
                self.assertEqual(out['rows'][60]['frame_role'], 'INTENT_TRANSITION_IDLE')

    def test_unreliable_hidden_mode_change_is_retained_for_review(self):
        rows=evidence_rows(160);active(rows,0,10);active(rows,140,160)
        rows[70]['evidence']['interval']['mode_sequence']=[{
            'left':'Z_CLOSE','right':'XY_HOLD','left_reliable':False,'right_reliable':True}]
        out=select_rows(rows)
        self.assertTrue(out['rows'][60]['keep'])
        self.assertEqual(out['rows'][60]['frame_role'],'INTENT_TRANSITION_HOLD')
        self.assertTrue(out['rows'][60]['review_required'])

    def test_unknown_pause_or_missing_sampling_prefix_never_auto_deleted(self):
        for key in ('pause', 'period_missing'):
            rows = evidence_rows(100); active(rows, 80, 100)
            if key == 'pause':
                rows[20]['evidence']['interval']['safety_pause'] = True
            else:
                for row in rows:
                    row['evidence']['source'].pop('observation_period_s')
            out = select_rows(rows)
            self.assertTrue(out['rows'][20]['keep'], key)
            self.assertEqual(out['rows'][20]['frame_role'], 'AMBIGUOUS_IDLE', key)
            if key=='period_missing':
                self.assertIsNone(out['metrics']['retained_physics_s'])

    def test_initial_tracking_wait_is_trimmed_without_reliable_hand_intent(self):
        rows = evidence_rows(120); active(rows, 100, 120)
        reasons = ('record_start_baseline', 'operation_pending', 'hand_lost',
                   'hand_near_frame_edge', 'tracking_recovery')
        for i, row in enumerate(rows[:100]):
            reason = reasons[(i // 20) % len(reasons)]
            row['status_summary'].update(paused=True, pause_reason=reason)
            row['evidence']['interval'].update(safety_pause=True, pause_reasons=[reason])
            for side in ('left', 'right'):
                row['evidence']['accepted_intent'][side].update(
                    mode=None if i < 40 else 'HOLD', reliable=False,
                    event_id=None, measurement_id=None)
            # Tiny noncumulative numerical jitter is not useful preparation.
            row['state_recorded'][0] = (i % 2) * 1e-7
        out = select_rows(rows)
        self.assertTrue(all(not r['keep'] for r in out['rows'][:94]))
        self.assertTrue(all(r['frame_role'] == 'PREFIX_IDLE' for r in out['rows'][:94]))
        self.assertTrue(all(r['keep'] for r in out['rows'][94:]))

    def test_stationary_initial_mode_changes_do_not_become_task_transition(self):
        rows = evidence_rows(120); active(rows, 100, 120)
        for i, row in enumerate(rows[:100]):
            for side in ('left', 'right'):
                row['evidence']['accepted_intent'][side].update(
                    mode=('XY_HOLD', 'HOLD', 'Z_CLOSE', None)[i // 25],
                    reliable=False)
        out = select_rows(rows)
        self.assertTrue(all(not r['keep'] for r in out['rows'][:94]))
        self.assertTrue(all(r['keep'] for r in out['rows'][94:]))

    def test_same_tracking_pause_after_motion_remains_in_task_review(self):
        rows = evidence_rows(150); active(rows, 0, 10); active(rows, 130, 150)
        for row in rows[10:130]:
            row['status_summary'].update(paused=True, pause_reason='hand_lost')
            row['evidence']['interval'].update(safety_pause=True, pause_reasons=['hand_lost'])
        out = select_rows(rows)
        self.assertTrue(all(r['keep'] for r in out['rows'][10:130]))

    def test_initial_abort_without_execution_does_not_preserve_preparation(self):
        rows = evidence_rows(120); active(rows, 100, 120)
        rows[99]['evidence']['events'] = [
            {'kind': 'gripper_aborted', 'side': 'left',
             'before': 'WAIT_BASELINE', 'after': 'ABORTED'}]
        out = select_rows(rows)
        self.assertFalse(out['rows'][0]['keep'])
        self.assertTrue(all(r['keep'] for r in out['rows'][94:]))

    def test_slow_cumulative_action_cannot_extend_initial_wait(self):
        rows = evidence_rows(200); active(rows, 180, 200)
        for i in range(60, 200):
            rows[i]['action_recorded'][0] = min((i - 59) * .00005, .003)
        out = select_rows(rows)
        # Rolling three-frame differences stay <0.5mm; total crosses at 69.
        self.assertFalse(out['rows'][68]['sa_changed'])
        self.assertFalse(out['rows'][20]['keep'])
        self.assertTrue(all(r['keep'] for r in out['rows'][63:70]))
        self.assertTrue(all(r['frame_role'] != 'PREFIX_IDLE' for r in out['rows'][69:]))
        self.assertEqual(out['rows'][69]['policy_reason'],
                         ['initial_pose_or_accepted_target_departure'])

    def test_cumulative_object_drift_is_not_stationary_preparation(self):
        rows = evidence_rows(200); active(rows, 180, 200)
        for i in range(60, 200):
            rows[i]['evidence']['object']['position'][0] = min((i - 59) * .00001, .001)
        cfg = load_config(); cfg["physical_motion"]["position_m"] = .0001
        out = select_rows(rows, cfg)
        self.assertFalse(out['rows'][20]['keep'])
        self.assertTrue(all(r['keep'] for r in out['rows'][63:70]))
        self.assertTrue(all(r['frame_role'] != 'PREFIX_IDLE' for r in out['rows'][69:]))

    def test_hidden_target_excursion_ends_prefix_and_missing_control_blocks_it(self):
        for known in (True, False):
            rows = evidence_rows(150); active(rows, 130, 150)
            rows[60]['evidence']['interval'].update(
                command_activity=True if known else None,
                command_activity_available=known)
            out = select_rows(rows)
            self.assertEqual(out['rows'][20]['keep'], not known)
            self.assertTrue(all(r['keep'] for r in out['rows'][54:61]))
            self.assertTrue(all(r['frame_role'] != 'PREFIX_IDLE' for r in out['rows'][60:]))

    def test_first_gripper_execution_context_is_retained(self):
        rows = evidence_rows(120); execution(rows, 80, side='right')
        out = select_rows(rows)
        self.assertFalse(out['rows'][10]['keep'])
        self.assertTrue(all(r['keep'] for r in out['rows'][74:87]))
        self.assertTrue(out['rows'][80]['execution_grasp_phase'])
        self.assertFalse(out['rows'][80]['height_protected'])
        self.assertEqual(out['execution_intervals'][0]['start_raw_frame_index'], 80)

    def test_missing_actual_pose_blocks_prefix_even_with_velocity_available(self):
        rows = evidence_rows(120); active(rows, 100, 120)
        rows[10]['evidence']['relative_pose']['left']['available'] = False
        out = select_rows(rows)
        self.assertTrue(all(r['keep'] for r in out['rows'][:100]))

    def test_unknown_hand_reposition_wait_trims_only_stationary_interior(self):
        for unknown_sides in (('left',), ('right',), ('left', 'right')):
            rows = evidence_rows(240); active(rows, 0, 20); active(rows, 220, 240)
            for row in rows[20:220]:
                row['status_summary'].update(paused=True, pause_reason='hand_lost')
                row['evidence']['interval'].update(safety_pause=True, pause_reasons=['hand_lost'])
                for side in unknown_sides:
                    row['evidence']['accepted_intent'][side].update(mode='HOLD', reliable=False)
            out = select_rows(rows)
            self.assertFalse(out['rows'][100]['keep'], unknown_sides)
            self.assertEqual(out['rows'][100]['frame_role'], 'RECENTER_IDLE')
            self.assertEqual(sum(r['keep'] for r in out['rows'][20:220]), 3)

    def test_reposition_does_not_hide_new_mode_or_contact_fault(self):
        for fault in ('contact_loss', 'hidden_close', 'missing_source',
                      'hidden_command', 'slow_drift', 'gripper_abort', 'slow_goal'):
            rows = evidence_rows(240); active(rows, 0, 20); active(rows, 220, 240)
            for row in rows[20:220]:
                row['status_summary'].update(paused=True, pause_reason='hand_lost')
                row['evidence']['interval'].update(safety_pause=True, pause_reasons=['hand_lost'])
                row['evidence']['accepted_intent']['left'].update(mode='HOLD', reliable=False)
            if fault == 'contact_loss':
                rows[100]['evidence']['interval']['pause_reasons'].append('contact_loss')
            elif fault == 'hidden_close':
                rows[100]['evidence']['interval']['mode_sequence'] = [
                    {'left':'Z_CLOSE', 'right':'XY_HOLD', 'left_reliable':True, 'right_reliable':True}]
            elif fault == 'missing_source':
                rows[100]['evidence']['source']['available'] = False
            elif fault == 'hidden_command':
                rows[100]['evidence']['interval']['command_changed_from_previous'] = True
            elif fault == 'gripper_abort':
                rows[100]['evidence']['events'] = [
                    {'kind': 'gripper_aborted', 'side': 'left',
                     'before': 'CLOSING', 'after': 'ABORTED'}]
            elif fault == 'slow_goal':
                for i in range(20, 220):
                    rows[i]['evidence']['grippers']['left']['goal_width_m'] += (i - 20) * .00001
            else:
                for i in range(20, 220):
                    rows[i]['evidence']['object']['position'][0] = (i - 20) * .0002
            out = select_rows(rows)
            self.assertTrue(out['rows'][100]['keep'], fault)

    def test_reposition_search_reaches_intent_before_residual_motion(self):
        rows = evidence_rows(240); active(rows, 0, 20); active(rows, 220, 240)
        for row in rows[17:220]:
            row['evidence']['accepted_intent']['left'].update(mode='HOLD', reliable=False)
        out = select_rows(rows)
        self.assertFalse(out['rows'][100]['keep'])
        proof = out['rows'][100]['phase_pause_interpretation']
        self.assertEqual(proof['pre_intent_evidence']['by_side']['left']['raw_frames'], [14, 15, 16])
        cfg = load_config(); cfg['intent']['recenter_search_frames'] = 3
        short = select_rows(rows, cfg)
        self.assertFalse(short['rows'][100]['keep'])
        self.assertEqual(short['rows'][100]['phase_pause_interpretation']['comparable_sides'], ['right'])

    def test_stable_held_contact_can_reposition_but_grasp_execution_keeps_context(self):
        rows = evidence_rows(240); active(rows, 0, 20); active(rows, 220, 240)
        for row in rows:
            for side in ('left', 'right'):
                row['evidence']['contact'][side].update(stable=True, double_finger=True, any_finger=True)
        for row in rows[20:220]:
            row['evidence']['accepted_intent']['left'].update(mode='HOLD', reliable=False)
        out = select_rows(rows)
        self.assertFalse(out['rows'][100]['keep'])
        execution(rows, 80)
        protected = select_rows(rows)
        self.assertTrue(all(r['keep'] for r in protected['rows'][74:87]))
        self.assertTrue(protected['rows'][80]['execution_grasp_phase'])
        self.assertFalse(protected['rows'][80]['height_protected'])

    def test_residual_settling_is_preserved_without_blocking_later_reposition_wait(self):
        rows = evidence_rows(240); active(rows, 0, 20); active(rows, 220, 240)
        for i in range(20, 220):
            rows[i]['evidence']['accepted_intent']['left'].update(mode='HOLD', reliable=False)
            rows[i]['evidence']['relative_pose']['left']['position'][0] = min((i - 20) * .00008, .0008)
        cfg = load_config(); cfg["physical_motion"]["position_m"] = .0001
        out = select_rows(rows, cfg)
        self.assertTrue(out['rows'][20]['keep'])
        self.assertFalse(out['rows'][100]['keep'])
        # Short early intervals remain; the later stable interval is independent.
        self.assertIn('wait_context_or_below_minimum_duration', out['rows'][20]['policy_reason'])
        self.assertTrue(out['rows'][219]['keep'])
        self.assertTrue(all(r['keep'] for r in out['rows'][20:29]))

    def test_source_availability_false_keeps_valid_numeric_rows_for_review(self):
        rows=evidence_rows(120);active(rows,100,120)
        for row in rows:
            row['evidence']['source'].update(available=False,observation_period_s=.05)
        out=select_rows(rows)
        self.assertTrue(all(r['keep'] and r['review_required'] for r in out['rows']))
        self.assertTrue(all('incomplete_source_evidence' in r['policy_reason'] for r in out['rows']))

    def test_prefix_trims_but_done_and_settling_without_release_cannot_trim_tail(self):
        rows = evidence_rows(200); active(rows, 70, 90)
        for row in rows[100:]:
            row['evidence']['task'] = {'done': True, 'success': True}
        # A task-state transition supplies context exactly as real evidence does.
        rows[100]['evidence']['events'] = [{'kind': 'task_state_changed', 'side': None}]
        out = select_rows(rows)
        self.assertFalse(out['rows'][20]['keep'])
        self.assertTrue(out['rows'][180]['keep'])
        self.assertTrue(all(r['keep'] for r in out['rows'][64:70]))
        self.assertNotEqual(out['rows'][180]['frame_role'], 'TAIL_IDLE')
        rows[150]['evidence']['object']['linear_speed_mps'] = .2
        rows[150]['evidence']['motion']['moving'] = True
        out = select_rows(rows)
        self.assertTrue(out['rows'][150]['keep'])

    def test_initial_preparation_pause_does_not_prove_done_retraction_is_released(self):
        rows=evidence_rows(230);active(rows,70,90);active(rows,160,180)
        for i,row in enumerate(rows):
            if i<70 or 100<=i<150 or i>=180:
                reason='record_start_baseline' if i<70 else 'hand_lost'
                row['status_summary'].update(paused=True,pause_reason=reason)
                row['evidence']['interval'].update(safety_pause=True,pause_reasons=[reason])
            if i<70:
                for side in ('left','right'):
                    row['evidence']['accepted_intent'][side].update(mode='HOLD',reliable=False)
            if i>=180:
                row['evidence']['task']={'done':True,'success':True}
                for side in ('left','right'):
                    row['evidence']['accepted_intent'][side].update(reliable=False)
        out=select_rows(rows)
        self.assertFalse(out['rows'][20]['keep'])
        self.assertTrue(out['rows'][120]['keep'])
        self.assertTrue(out['rows'][220]['keep'])
        self.assertNotEqual(out['rows'][220]['frame_role'],'TAIL_IDLE')
        self.assertEqual(out['rows'][120]['phase_pause_interpretation']['reason'],
                         'tracking_pause_without_hold_evidence')

    def test_hidden_dangerous_pause_prevents_normal_prefix_exception(self):
        rows=evidence_rows(110);active(rows,90,110)
        for row in rows[:90]:
            row['status_summary'].update(paused=True,pause_reason='operation_pending')
            row['evidence']['interval'].update(safety_pause=True,pause_reasons=['operation_pending'])
        rows[40]['evidence']['interval']['pause_reasons'].append('contact_loss')
        out=select_rows(rows)
        self.assertTrue(out['rows'][20]['keep'])
        self.assertFalse(out['rows'][20]['phase_pause_interpretation']['explained_normal_wait'])

    def test_done_without_release_never_overrides_tracking_recovery_review(self):
        for done in (False, True):
            rows=evidence_rows(150);active(rows,0,20)
            for row in rows[20:]:
                row['evidence']['task']={'done':done,'success':done}
                row['status_summary'].update(paused=True,pause_reason='hand_near_frame_edge')
                row['evidence']['interval'].update(safety_pause=True,
                    pause_reasons=['hand_near_frame_edge','tracking_recovery'])
                for side in ('left','right'):
                    row['evidence']['accepted_intent'][side]['reliable']=False
            out=select_rows(rows)
            self.assertTrue(out['rows'][100]['keep'])
            self.assertEqual(out['rows'][100]['frame_role'],'AMBIGUOUS_IDLE')

    def test_authorization_and_close_mode_are_not_real_execution(self):
        rows = evidence_rows(40)
        for row in rows:
            row['evidence']['grippers']['left'].update(edge_state='CLOSING')
            row['evidence']['accepted_intent']['left']['mode'] = 'Z_CLOSE'
        out = select_rows(rows)
        self.assertFalse(any(r['execution_grasp_phase'] for r in out['rows']))
        self.assertEqual(out['execution_intervals'], [])

    def test_grasp_phase_finishes_only_after_width_contact_and_relative_stable(self):
        rows = evidence_rows(90); execution(rows, 10)
        for row in rows[15:]:
            row['evidence']['contact']['left'].update(stable=True,double_finger=True,any_finger=True)
        # Stable accepted goal is insufficient: actual contact pose still drifts.
        for i in range(15, 40):
            rows[i]['evidence']['relative_pose']['left']['position'][0] = i * .001
        for row in rows[40:]:
            row['evidence']['relative_pose']['left']['position'][0] = .04
        out = select_rows(rows)
        cycle = out['execution_intervals'][0]
        self.assertEqual(cycle['end_raw_frame_index'], 44)
        self.assertEqual(cycle['end_reason'], 'actual_grasp_and_relative_pose_stable')
        self.assertTrue(all(r['execution_grasp_phase'] for r in out['rows'][10:45]))
        self.assertFalse(out['rows'][45]['execution_grasp_phase'])
        self.assertTrue(all(r['keep'] for r in out['rows'][4:17] + out['rows'][45:51]))
        self.assertEqual(out['protected_intervals'], out['execution_intervals'])

    def test_left_and_right_cycles_are_independent_and_no_duration_cap(self):
        rows = evidence_rows(350); execution(rows, 5, 'left', event='l')
        execution(rows, 12, 'right', event='r')
        for row in rows[20:]:
            row['evidence']['contact']['right'].update(stable=True,double_finger=True,any_finger=True)
        out = select_rows(rows)
        self.assertTrue(out['rows'][300]['execution_grasp_phase'])
        self.assertEqual(out['rows'][300]['execution_phase_by_side']['left'], ['grasp'])
        self.assertEqual(out['rows'][300]['execution_phase_by_side']['right'], [])
        left_cycle = next(c for c in out['execution_intervals'] if c['side'] == 'left')
        self.assertTrue(left_cycle['review_required'])
        self.assertEqual(left_cycle['end_raw_frame_index'], 349)
        self.assertEqual(left_cycle['end_reason'], 'missing_reliable_physical_end')
        self.assertIn('left:physical_protection_end_unavailable', out['warnings'])

    def test_hold_is_not_abort_but_explicit_reverse_ends_cycle(self):
        rows = evidence_rows(60); execution(rows, 5)
        for row in rows[10:]:
            row['evidence']['accepted_intent']['left']['mode'] = 'HOLD'
        execution(rows, 30, opening=True, event='reverse')
        out = select_rows(rows)
        self.assertTrue(out['rows'][20]['execution_grasp_phase'])
        self.assertEqual(out['execution_intervals'][0]['end_raw_frame_index'], 30)
        self.assertEqual(out['execution_intervals'][0]['end_reason'], 'aborted_or_reversed_execution')
        self.assertTrue(all(r['keep'] for r in out['rows'][24:37]))

    def test_release_actual_open_contacts_clear_and_done_are_all_required(self):
        rows = evidence_rows(100)
        for row in rows[:20]:
            row['evidence']['contact']['left'].update(stable=True,double_finger=True,any_finger=True)
        execution(rows, 20, opening=True)
        for row in rows[30:]:
            row['evidence']['grippers']['left']['actual_width_m'] = .08
        for row in rows[50:]:
            row['evidence']['task'] = {'done': True, 'success': True}
        out = select_rows(rows)
        cycle = out['execution_intervals'][0]
        self.assertEqual(cycle['phase'], 'release')
        self.assertEqual(cycle['end_raw_frame_index'], 50)
        self.assertTrue(out['rows'][40]['execution_release_phase'])
        self.assertFalse(out['rows'][51]['execution_release_phase'])
        self.assertTrue(all(r['keep'] for r in out['rows'][14:27]))
        for row in rows[30:]:
            row['evidence']['contact']['left']['any_finger'] = True
        out = select_rows(rows)
        self.assertEqual(out['execution_intervals'][0]['end_reason'], 'missing_reliable_physical_end')

    def test_empty_opening_does_not_wait_for_future_task_done(self):
        rows = evidence_rows(60); execution(rows, 10, opening=True)
        for row in rows[20:]:
            row['evidence']['grippers']['left']['actual_width_m'] = .08
        out = select_rows(rows)
        self.assertEqual(out['execution_intervals'][0]['phase'], 'open_adjustment')
        self.assertEqual(out['execution_intervals'][0]['end_raw_frame_index'], 24)

    def test_invalid_maximum_width_cannot_verify_release(self):
        for maximum in (0., -.01, None):
            rows=evidence_rows(30);execution(rows,5,opening=True)
            for row in rows:
                row['evidence']['grippers']['left'].update(actual_width_m=.08,max_width_m=maximum)
            out=select_rows(rows)
            self.assertEqual(out['execution_intervals'][0]['end_reason'],'missing_reliable_physical_end')
            self.assertTrue(out['rows'][-1]['execution_release_phase'])

    def test_hard_epoch_boundary_does_not_merge_wait_or_execution_phases(self):
        rows = evidence_rows(100); execution(rows, 10)
        for row in rows[50:]:
            row['control_epoch'] = 2
        execution(rows, 60, event='edge')  # Same ID in another epoch is not replay.
        out = select_rows(rows)
        self.assertEqual(out['metrics']['task_segment_count'], 2)
        self.assertEqual(len(out['execution_intervals']), 2)
        self.assertEqual(out['execution_intervals'][0]['end_raw_frame_index'], 49)

    def test_physical_motion_and_aux_confidence_independent(self):
        rows = evidence_rows(50); active(rows, 0, 50)
        for row in rows:
            row['auxiliary'] = {'left': {'category_mask': False, 'gesture_score': 0.}}
        out = select_rows(rows)
        self.assertTrue(all(r['keep'] for r in out['rows']))
        self.assertEqual(out['metrics']['filtered_frames'], 0)

    def test_invalid_always_excluded_and_never_manual_forced(self):
        rows = evidence_rows(30); rows[10]['state_recorded'][0] = None
        out = select_rows(rows)
        self.assertEqual(out['rows'][10]['frame_role'], 'INVALID')
        out['rows'][10]['human_decision'] = 'keep'
        resegment_rows(out['rows'])
        self.assertFalse(out['rows'][10]['keep'])
        self.assertFalse(any(10 in w['source_indices'] for w in build_windows(out['rows'])))

    def test_manual_cut_resegments_without_mutating_automatic_decisions(self):
        rows = evidence_rows(50); active(rows, 0, 50)
        out = select_rows(rows); selected=out['rows']
        original = [(r['policy_decision'],list(r['policy_reason'])) for r in selected]
        for row in selected[20:23]:
            row['human_decision']='exclude'; row['human_reason']='人工移除'
        resegment_rows(selected)
        self.assertEqual(original,[(r['policy_decision'],r['policy_reason']) for r in selected])
        self.assertEqual(selected[23]['gap_before']['kind'],'human_exclusion')
        self.assertEqual(selected[0]['task_segment_id'],selected[49]['task_segment_id'])
        self.assertNotEqual(selected[0]['segment_id'],selected[49]['segment_id'])
        self.assertFalse(any(19 in w['source_indices'] and 23 in w['source_indices'] for w in build_windows(selected)))

    def test_rows_and_json_integrity_are_preserved(self):
        rows=evidence_rows(80);active(rows,60,80);before=copy.deepcopy(rows)
        out=select_rows(rows)
        self.assertEqual(rows,before)
        self.assertEqual(len(out['rows']),len(rows))
        self.assertFalse(out['metrics']['training_approved'])
        json.dumps(out,allow_nan=False)


if __name__ == '__main__':
    unittest.main()
