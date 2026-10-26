"""Causal fact extraction with no frame-selection policy or simulator imports."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dataset.task3.frame_evidence import build_frame_evidence
from dataset.task3 import curation_io


def episode():
    state = [0.] * 16
    state[7] = state[15] = .08
    action = [0., -.2, .45, 0., 0., 0., .08, 0., .2, .45, 0., 0., 0., .08]
    controls = []
    for tick in range(7):
        controls.append({'kind': 'control_frame', 'source_line': 10+tick*10,
            'tick': tick, 'frame_index': tick//2 if tick%2 == 0 else None,
            'session': 's', 'control_epoch': 1, 'sim_time': tick/60,
            'wall_time_ms': 1000+tick*1000/60, 'server_monotonic_ns': 1000000000+int(tick*1e9/60),
            'state': list(state), 'action': list(action),
            'status': {'paused': False, 'task_state': 'PREGRASP', 'success': False,
                'actual_ee': {s: {'position': [0., y, .45], 'rotation': np.eye(3).tolist()}
                              for s, y in [('left', -.2), ('right', .2)]},
                'object_position': [0., 0., .4],
                'object_linear_velocity_mps': [0., 0., 0.],
                'object_angular_velocity_radps': [0., 0., 0.], 'ee_speeds_mps': [0., 0.],
                'contacts': {'left': False, 'right': False,
                             'left_fingers': [False, False], 'right_fingers': [False, False]},
                'grippers': {s: {'actual_width_m': .08, 'command_width_m': .08,
                                'goal_width_m': .08, 'max_width_m': .08} for s in ('left', 'right')},
                'physical_contact': {'interaction_modes': {'left': 'HOLD', 'right': 'HOLD'},
                    'interaction_mode_event_id': {'left': 'e', 'right': 'e'},
                    'stable_grasps': {'left': False, 'right': False},
                    'release_cycle_state': {'left': 'IDLE', 'right': 'IDLE'}}}})
    frames = [c for c in controls if c['frame_index'] is not None]
    qpos = np.zeros((4, 25)); qpos[:, 18:21] = [0., 0., .4]; qpos[:, 21] = 1.
    world = np.zeros((21, 3)); world[5] = [.02, 0., 0.]; world[17] = [-.02, 0., 0.]
    measurement = {'measurement_id': 'm', 'source_session': 's', 'source_control_epoch': 1,
        'capture_epoch_ms': 980., 'recognition_epoch_ms': 990., 'accepted_recognition': True,
        'hands': [{'bound_side': s, 'gesture': 'Open_Palm', 'gesture_score': .99,
            'identity_score': .99, 'valid': True, 'world_valid': True, 'near_edge': False,
            'image_landmarks_21': [[.5, .5, 0.]]*21, 'hand_center_landmarks_m_21': world.tolist()}
            for s in ('left', 'right')]}
    event = {'kind': 'request_executed', 'source_line': 5, 'tick': 0, 'sim_time': 0.,
        'session': 's', 'control_epoch': 1, 'wall_time_ms': 995.,
        'request': {'eventId': 'e', 'command': 'continuous_input', 'measurement_id': 'm',
                    'session': 's', 'control_epoch': 1},
        'result': {'accepted': True, 'interaction_modes': ['HOLD', 'HOLD']}}
    for c in controls:
        c.update(observation_measurement_id='m', action_event_id='e', action_measurement_id='m',
                 hand_measurement_mask={'left': True, 'right': True})
    return {'ref': {'key': 's/0', 'session': 's', 'episode_index': 0},
        'metadata': {'object_qpos_adr': 18, 'sampling_policy': {'fps': 30, 'control_hz': 60},
                     'config': {'control': {'max_command_age_ms': 150.}}},
        'frames': frames, 'controls': controls,
        'snapshots': {'time': np.arange(4)/30, 'state': np.array([state]*4),
                      'action': np.array([action]*4), 'qpos': qpos, 'qvel': np.zeros((4,24))},
        'accepted_events': {'e': event}, 'executed_events': [event],
        'measurements': {'m': measurement}, 'measurement_sources': {'m': {'source_line': 2}}}


def close_event(ep, line=15):
    event = copy.deepcopy(ep['executed_events'][0])
    event.update(source_line=line, wall_time_ms=1002., actual_state_phase='before_next_physics_step')
    event['request']['eventId'] = 'close'
    event['result'].update(interaction_modes=['Z_CLOSE', 'HOLD'],
        grippers={'left': {'goal_width_m': 0., 'command_width_m': .08}},
        physical_contact={'gripper_edge': {'left': {'event_id': 'close', 'authorized_mode': 'Z_CLOSE'}}})
    ep['accepted_events']['close'] = event; ep['executed_events'].append(event)
    return event


def execution_episode(policy='palm_fist_edge_v2'):
    ep = episode()
    ep['metadata']['gripper_trigger_policy'] = policy
    ep['metadata']['config']['gesture_control'] = {'gripper_trigger_policy': policy}
    for c in ep['controls']:
        pc = c['status']['physical_contact']
        pc.update(gripper_trigger_policy=policy,
            gripper_rate_enabled={s: False for s in ('left', 'right')},
            gripper_rate_direction={s: 0 for s in ('left', 'right')},
            release_armed={s: False for s in ('left', 'right')},
            release_opening={s: False for s in ('left', 'right')},
            release_progress_s={s: 0. for s in ('left', 'right')},
            gripper_edge={s: {'state': 'READY', 'baseline_mode': 'Z_OPEN',
                'baseline_source': 'accepted_mode', 'authorized_mode': None,
                'event_id': None, 'edge_count': 0} for s in ('left', 'right')})
    return ep


class FrameEvidenceTests(unittest.TestCase):
    def test_validity_is_independent_per_side_without_score_rethreshold(self):
        ep = episode()
        ep['measurements']['m']['hands'][0]['gesture_score'] = 0.001
        ep['measurements']['m']['hands'][1].update(valid=False, invalid_reason='duplicate_identity')
        intent = build_frame_evidence(ep, {})['rows'][0]['evidence']['accepted_intent']
        self.assertTrue(intent['left']['reliable'])
        self.assertFalse(intent['right']['reliable'])
        self.assertEqual(intent['right']['hand_condition'], 'identity_invalid')
        self.assertEqual(intent['right']['recorded_invalid_reason'], 'duplicate_identity')

    def test_unknown_trusted_hold_differs_from_missing_and_global_pause(self):
        ep = episode()
        ep['measurements']['m']['hands'][0]['gesture'] = 'None'
        ep['measurements']['m']['hands'].pop()
        ep['controls'][0]['status']['paused'] = True
        intent = build_frame_evidence(ep, {})['rows'][0]['evidence']['accepted_intent']
        self.assertTrue(intent['left']['trusted_hold'])
        self.assertEqual(intent['left']['hand_condition'], 'unknown_gesture')
        self.assertTrue(intent['left']['global_paused'])
        self.assertEqual(intent['right']['hand_condition'], 'hand_missing')
        self.assertFalse(intent['right']['reliable'])

    def test_v2_trusted_hold_retains_actual_pending_authorization(self):
        ep = execution_episode()
        ep['controls'][1]['status']['physical_contact']['gripper_edge']['left'].update(
            state='HOLD', authorized_mode='Z_CLOSE', event_id='close', edge_count=1,
            reason='gripper_edge_transition_hold')
        out = build_frame_evidence(ep, {})
        ctrl = out['controls'][1]['execution']['sides']['left']
        self.assertTrue(ctrl['available'])
        self.assertTrue(ctrl['pending'])
        self.assertTrue(ctrl['trusted_hold_resumable'])
        self.assertTrue(out['rows'][1]['evidence']['interval']['execution_pending'])
        self.assertFalse(out['rows'][1]['evidence']['execution']['sides']['left']['pending'])
        self.assertTrue(any(e['kind']=='gripper_authorization_changed' for e in out['events']))

    def test_verified_completed_close_hold_does_not_mask_pending_open_edge(self):
        ep = execution_episode()
        for c in ep['controls']:
            pc = c['status']['physical_contact']
            pc['hold_width_m'] = {'left': .08, 'right': .08}
            pc['stable_grasps'] = {'left': True, 'right': True}
            for side, authorization in [('left', 'Z_CLOSE'), ('right', 'Z_OPEN')]:
                pc['gripper_edge'][side].update(state='HOLD', reason='grasp_hold', authorized_mode=authorization)
        evidence = build_frame_evidence(ep, {})['controls'][0]['execution']['sides']
        self.assertTrue(evidence['left']['authorization_present'])
        self.assertTrue(evidence['left']['completed_close_hold'])
        self.assertFalse(evidence['left']['pending'])
        self.assertFalse(evidence['right']['completed_close_hold'])
        self.assertTrue(evidence['right']['pending'])
        ep['controls'][0]['status']['grippers']['left']['goal_width_m'] = 0.
        self.assertTrue(build_frame_evidence(ep, {})['controls'][0]['execution']['sides']['left']['pending'])

    def test_level_mode_is_not_automatic_gripper_execution(self):
        ep = execution_episode('level_v1')
        for c in ep['controls']:
            pc = c['status']['physical_contact']
            pc.pop('gripper_edge')
            pc['interaction_modes']['left'] = 'Z_CLOSE'
        out = build_frame_evidence(ep, {})
        self.assertTrue(out['controls'][0]['execution']['available'])
        self.assertFalse(out['controls'][0]['execution']['sides']['left']['pending'])
        ep['controls'][1]['status']['physical_contact']['gripper_rate_enabled']['left'] = True
        self.assertTrue(build_frame_evidence(ep, {})['rows'][1]['evidence']['interval']['execution_pending'])

    def test_release_confirmation_pending_but_completed_latch_not_pending(self):
        ep = execution_episode()
        for c in ep['controls']:
            pc = c['status']['physical_contact']
            pc['release_executed'] = {'left': True, 'right': True}
            pc['release_cycle_state'] = {'left': 'COMPLETE', 'right': 'COMPLETE'}
        ep['controls'][1]['status']['physical_contact']['release_cycle_state']['right'] = 'CONFIRMING'
        out = build_frame_evidence(ep, {})
        self.assertFalse(out['controls'][0]['execution']['sides']['right']['pending'])
        self.assertTrue(out['controls'][1]['execution']['sides']['right']['release_confirmation_pending'])
        self.assertTrue(any(e['kind']=='release_confirmation_changed' for e in out['events']))

    def test_missing_or_conflicting_frozen_execution_proof_is_unavailable(self):
        ep = execution_episode()
        ep['controls'][0]['status']['physical_contact']['gripper_edge']['left'].pop('authorized_mode')
        left = build_frame_evidence(ep, {})['controls'][0]['execution']['sides']['left']
        self.assertFalse(left['available'])
        self.assertIsNone(left['pending'])
        ep = execution_episode()
        ep['metadata']['config']['gesture_control']['gripper_trigger_policy'] = 'level_v1'
        fact = build_frame_evidence(ep, {})['controls'][0]['execution']
        self.assertFalse(fact['available'])
        self.assertIn('frozen_gripper_strategy_mismatch',fact['sides']['left']['reasons'])

    def test_interval_and_following_control_retain_exact_causal_sources(self):
        ep = execution_episode()
        out = build_frame_evidence(ep, {})
        row = out['rows'][1]
        self.assertEqual(row['evidence']['interval']['controls'],out['controls'][1:3])
        self.assertEqual(row['evidence']['next_control'],out['controls'][3])
        self.assertEqual(row['next_control_target']['source_tick'],out['controls'][3]['tick'])
        row['evidence']['interval']['controls'][0]['grippers']['left']['goal_width_m'] = .04
        self.assertEqual(out['controls'][1]['grippers']['left']['goal_width_m'], .08)

    def test_hidden_accepted_request_exposes_execution_before_sampled_control(self):
        ep = execution_episode()
        event = close_event(ep)
        pc = copy.deepcopy(ep['controls'][0]['status']['physical_contact'])
        pc['gripper_edge']['left'].update(authorized_mode='Z_CLOSE', event_id='close', edge_count=1, state='HOLD')
        event['result']['physical_contact'] = pc
        out = build_frame_evidence(ep, {})
        executed = next(e for e in out['events'] if e['kind']=='request_accepted' and e['event_id']=='close')
        self.assertTrue(executed['execution']['sides']['left']['pending'])
        self.assertEqual(out['rows'][0]['evidence']['interval']['execution_pending'],False)
        self.assertTrue(any(e['event_id']=='close' for e in out['rows'][1]['evidence']['events']))

    def test_baseline_rebuild_and_recorded_log_loss_stay_visible(self):
        ep = execution_episode()
        ep['controls'][1]['status']['physical_contact']['gripper_edge']['right']['baseline_mode'] = None
        ep['controls'][1]['log_dropped'] = 1
        out = build_frame_evidence(ep, {})
        self.assertTrue(any(e['kind']=='gripper_baseline_changed' and e['side']=='right' for e in out['events']))
        self.assertIn('recorded_log_dropped',out['rows'][1]['evidence']['source']['reasons'])

    def test_initial_command_baseline_and_stationary_controls_are_not_activity(self):
        out = build_frame_evidence(episode(), {})
        for row in out['rows']:
            interval = row['evidence']['interval']
            self.assertIs(interval['command_activity'], False)
            self.assertTrue(interval['command_activity_available'])

    def test_hidden_command_excursion_is_causal_even_without_physical_motion(self):
        ep = episode()
        ep['controls'][1]['action'][0] += .002
        out = build_frame_evidence(ep, {})
        self.assertIs(out['rows'][0]['evidence']['interval']['command_activity'], False)
        interval = out['rows'][1]['evidence']['interval']
        self.assertIs(interval['command_activity'], True)
        self.assertIs(interval['command_changed_from_previous'], True)
        self.assertFalse(out['rows'][1]['evidence']['motion']['moving'])
        self.assertEqual([d['tick'] for d in interval['command_activity_details']], [1, 2])
        self.assertIn('left.target_position', interval['command_activity_details'][0]['changed_from_initial'])
        self.assertEqual(out['rows'][1]['action_recorded'], out['rows'][0]['action_recorded'])

    def test_slow_command_accumulation_uses_fixed_initial_baseline(self):
        ep = episode()
        for control in ep['controls']:
            control['action'][0] = control['tick'] * .0001
            if control['frame_index'] is not None:
                ep['snapshots']['action'][control['frame_index'], 0] = control['action'][0]
        out = build_frame_evidence(ep, {})
        interval = out['rows'][-1]['evidence']['interval']
        self.assertIs(interval['command_activity'], True)
        self.assertIs(interval['command_changed_from_previous'], False)
        detail = interval['command_activity_details'][-1]
        self.assertEqual(detail['changed_from_previous'], [])
        self.assertIn('left.target_position', detail['changed_from_initial'])
        self.assertEqual(detail['baseline_source_line'], 10)

    def test_hidden_gripper_goal_is_activity_before_command_ramp(self):
        ep = episode()
        ep['controls'][1]['status']['grippers']['right']['goal_width_m'] = 0.
        out = build_frame_evidence(ep, {})
        interval = out['rows'][1]['evidence']['interval']
        self.assertIs(interval['command_activity'], True)
        self.assertIn('right.goal_aperture', interval['command_activity_details'][0]['changed_from_previous'])
        self.assertFalse(out['rows'][1]['evidence']['motion']['moving'])

    def test_missing_goal_or_command_source_cannot_prove_no_activity(self):
        for missing in ('goal', 'line'):
            ep = episode()
            if missing == 'goal':
                ep['controls'][1]['status']['grippers']['left'].pop('goal_width_m')
            else:
                ep['controls'][1]['source_line'] = None
            out = build_frame_evidence(ep, {})
            interval = out['rows'][1]['evidence']['interval']
            self.assertIsNone(interval['command_activity'])
            self.assertIsNone(interval['command_changed_from_previous'])
            self.assertFalse(interval['command_activity_available'])

    def test_stationary_displaced_target_is_not_a_new_command_change(self):
        ep = episode()
        for control in ep['controls'][1:]:
            control['action'][0] = .01
            if control['frame_index'] is not None:
                ep['snapshots']['action'][control['frame_index'], 0] = .01
        out = build_frame_evidence(ep, {})
        self.assertIs(out['rows'][1]['evidence']['interval']['command_changed_from_previous'], True)
        for row in out['rows'][2:]:
            interval = row['evidence']['interval']
            self.assertIs(interval['command_activity'], True)
            self.assertIs(interval['command_changed_from_previous'], False)
            self.assertTrue(interval['command_activity_available'])

    def test_late_command_after_final_image_does_not_backfill(self):
        ep = episode()
        late = copy.deepcopy(ep['controls'][-1])
        late.update(source_line=80, frame_index=None)
        late['action'][0] += .02
        ep['controls'].append(late)
        out = build_frame_evidence(ep, {})
        self.assertIs(out['rows'][-1]['evidence']['interval']['command_activity'], False)
        self.assertTrue(out['controls'][-1]['command_activity']['changed'])

    def test_command_rotation_comparison_wraps_angles(self):
        ep = episode()
        ep['controls'][1]['action'][3] = 2*np.pi - .0001
        out = build_frame_evidence(ep, {})
        self.assertIs(out['rows'][1]['evidence']['interval']['command_activity'], False)

    def test_same_tick_event_after_image_never_backfills(self):
        ep = episode(); close_event(ep)
        for c in ep['controls']:
            pc = c['status']['physical_contact']
            pc['interaction_modes']['left'] = 'Z_CLOSE'
            pc['interaction_mode_event_id']['left'] = 'close'
        out = build_frame_evidence(ep, {})
        first, second = out['rows'][:2]
        self.assertFalse(first['evidence']['accepted_intent']['left']['reliable'])
        self.assertFalse(first['evidence']['grippers']['left']['executed_close'])
        self.assertTrue(second['evidence']['accepted_intent']['left']['reliable'])
        self.assertTrue(second['evidence']['grippers']['left']['executed_close'])
        self.assertEqual(out['controls'][1]['event_indices'][-1],
                         next(e['index'] for e in out['events'] if e['kind'] == 'gripper_close_executed'))

    def test_duplicate_terminal_tick_retained_without_replacing_image(self):
        ep = episode(); last = ep['controls'][-1]
        for line, wall in [(80, 1200.), (90, 14000.)]:
            duplicate = copy.deepcopy(last)
            duplicate.update(source_line=line, frame_index=None, wall_time_ms=wall)
            duplicate['status']['task_state'] = 'DONE'
            ep['controls'].append(duplicate)
        out = build_frame_evidence(ep, {})
        self.assertEqual(len(out['rows']), 4); self.assertEqual(len(out['controls']), 9)
        self.assertEqual(out['rows'][-1]['source_line'], 70)
        self.assertIsNone(out['rows'][-1]['next_control_target'])
        self.assertFalse(out['rows'][-1]['terminal']['actual_done'])
        self.assertIn('duplicate_nonadvancing_control_rows', out['warnings'])
        self.assertIsNone(out['controls'][-1]['grippers']['left']['width_velocity_mps'])

    def test_clock_pair_bounds_cross_machine_epochs(self):
        ep = episode()
        for c in ep['controls']:
            c['wall_time_ms'] += 500.
        ep['executed_events'][0]['request']['telemetry'] = {'clock_pair': {
            'browser_request_epoch_ms': 900., 'browser_receive_epoch_ms': 904.,
            'server_epoch_ms': 1402., 'server_monotonic_ns': 500000000}}
        intent = build_frame_evidence(ep,{})['rows'][0]['evidence']['accepted_intent']['left']
        self.assertTrue(intent['reliable'], intent)
        np.testing.assert_allclose(intent['age_bounds_ms'], [18.,22.])
        self.assertEqual(intent['clock_basis'], 'health_request_response_epoch_bounds')
        ep['executed_events'][0]['request']['telemetry']['clock_pair']['browser_receive_epoch_ms'] = 899.
        intent = build_frame_evidence(ep,{})['rows'][0]['evidence']['accepted_intent']['left']
        self.assertFalse(intent['reliable'])
        self.assertEqual(intent['clock_basis'], 'invalid_clock_pair')

    def test_next_control_is_immediate_not_next_image_or_duplicate(self):
        ep = episode()
        ep['controls'][1]['action'][0] = .01
        out = build_frame_evidence(ep,{})
        target = out['rows'][0]['next_control_target']
        self.assertEqual(target['source_tick'], 1)
        self.assertEqual(target['source_line'], 20)
        self.assertEqual(target['action'][0], .01)
        ep['controls'][1]['tick'] = 0
        self.assertIsNone(build_frame_evidence(ep,{})['rows'][0]['next_control_target'])

    def test_relative_pose_uses_full_actual_object_rotation(self):
        ep = episode()
        ep['snapshots']['qpos'][:,21:25] = [np.sqrt(.5),0.,0.,np.sqrt(.5)]
        relative = build_frame_evidence(ep,{})['rows'][0]['evidence']['relative_pose']['left']
        self.assertTrue(relative['available'])
        np.testing.assert_allclose(relative['position'], [-.2, 0., .05], atol=1e-10)
        np.testing.assert_allclose(relative['rotation'],
                                   [[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]], atol=1e-10)
        ep['snapshots']['qpos'][:,21:25] = 0.
        self.assertFalse(build_frame_evidence(ep,{})['rows'][0]['evidence']['relative_pose']['left']['available'])

    def test_invalid_numeric_and_missing_physics_are_not_false_certainty(self):
        ep = episode()
        ep['snapshots']['action'][0,0] = np.nan
        ep['controls'][0]['status']['contacts']['left_fingers'] = [False]
        ep['controls'][0]['status']['grippers']['left'].pop('max_width_m')
        out = build_frame_evidence(ep,{})
        row = out['rows'][0]
        self.assertFalse(row['evidence']['source']['valid'])
        self.assertIsNone(row['action_recorded'][0])
        self.assertIsNone(row['evidence']['contact']['left']['any_finger'])
        self.assertIsNone(row['evidence']['grippers']['left']['max_width_m'])
        self.assertNotIn('keep', row)
        json.dumps(out, allow_nan=False)

    def test_category_no_score_preserves_geometry_mask_semantics(self):
        ep = episode()
        ep['measurements']['m']['hands'][0]['gesture_score'] = .01
        out = build_frame_evidence(ep,{})
        left = out['rows'][0]['auxiliary']['left']
        self.assertTrue(left['category_mask'])
        self.assertFalse(left['previous_category_mask'])
        self.assertTrue(left['geometry_mask'])
        self.assertFalse(left['task_geometry_mask'])
        self.assertEqual(left['category_score'], .01)
        ep['measurement_sources']['m']['source_line'] = 11
        left = build_frame_evidence(ep,{})['rows'][0]['auxiliary']['left']
        self.assertFalse(left['category_mask']); self.assertFalse(left['geometry_mask'])

    def test_current_close_mode_without_execution_does_not_invent_event(self):
        ep = episode()
        ep['controls'][0]['status']['physical_contact']['interaction_modes']['left'] = 'Z_CLOSE'
        out = build_frame_evidence(ep,{})
        self.assertFalse(out['rows'][0]['evidence']['grippers']['left']['executed_close'])
        self.assertFalse(any(e['kind'] == 'gripper_close_executed' for e in out['events']))

    def test_duplicate_accepted_id_does_not_repeat_physical_execution(self):
        ep = episode(); event = close_event(ep)
        duplicate = copy.deepcopy(event); duplicate['source_line'] = 25
        ep['executed_events'].append(duplicate)
        out = build_frame_evidence(ep,{})
        self.assertEqual(sum(e['kind']=='gripper_close_executed' for e in out['events']),1)
        self.assertIn('duplicate_accepted_event_id:close',out['warnings'])

    def test_intermediate_motion_and_pause_visible_at_next_image(self):
        ep = episode()
        ep['controls'][1]['status']['paused'] = True
        ep['controls'][1]['status']['object_linear_velocity_mps'][0] = .02
        out = build_frame_evidence(ep,{})
        row = out['rows'][1]
        self.assertEqual(row['evidence']['interval']['control_indices'],[1,2])
        self.assertTrue(row['evidence']['interval']['safety_pause'])
        self.assertEqual(row['evidence']['interval']['pause_reasons'], [None])
        self.assertTrue(row['evidence']['motion']['moving'])
        self.assertFalse(row['evidence']['source']['hard_boundary'])
        self.assertTrue(any(e['kind']=='pause_changed' for e in row['evidence']['events']))
        self.assertFalse(out['rows'][0]['evidence']['interval']['safety_pause'])
        self.assertEqual(out['rows'][0]['evidence']['interval']['pause_reasons'], [])

    def test_input_not_mutated_and_public_provenance_retained(self):
        ep = episode()
        before = copy.deepcopy(ep['controls'])
        out = build_frame_evidence(ep,{})
        self.assertEqual(ep['controls'],before)
        self.assertFalse(out['training_approved'])
        for name in ('state_recorded','action_recorded','source_release','original_control_labels',
                     'source_refs','status_summary','auxiliary','terminal','original_time_s','source_line'):
            self.assertIn(name,out['rows'][0])

    def test_missing_frozen_period_is_unavailable_not_nominal_clock(self):
        ep = episode(); ep['metadata'].pop('sampling_policy')
        row = build_frame_evidence(ep,{})['rows'][0]
        self.assertTrue(row['evidence']['source']['valid'])
        self.assertFalse(row['evidence']['source']['available'])
        self.assertIn('missing_frozen_sampling_contract',row['evidence']['source']['reasons'])
        self.assertIsNone(row['evidence']['source']['observation_period_s'])
        self.assertIsNone(row['recorded_video_time_s'])
        self.assertIsNone(row['next_control_target'])

    def test_unsupported_positive_sampling_is_uncertain_not_invalid(self):
        ep = episode(); ep['metadata']['sampling_policy'] = {'fps': 25, 'control_hz': 50}
        row = build_frame_evidence(ep,{})['rows'][0]
        source = row['evidence']['source']
        self.assertTrue(source['valid']); self.assertFalse(source['available'])
        self.assertIn('unsupported_frozen_sampling_contract',source['reasons'])
        self.assertEqual(source['observation_period_s'],.04)
        self.assertEqual(source['control_period_s'],.02)
        self.assertIsNone(row['next_control_target'])

    def test_explicitly_illegal_sampling_values_remain_invalid(self):
        for value in (0,-1,float('nan'),float('inf'),True):
            with self.subTest(value=value):
                ep = episode(); ep['metadata']['sampling_policy']['fps'] = value
                row = build_frame_evidence(ep,{})['rows'][0]
                self.assertFalse(row['evidence']['source']['valid'])
                self.assertIn('invalid_frozen_sampling_contract',row['evidence']['source']['reasons'])
                self.assertIsNone(row['next_control_target'])

    def test_time_fallback_is_opt_in_prior_bounded_and_unambiguous(self):
        ep = episode()
        for c in ep['controls']:
            c['status']['physical_contact']['interaction_mode_event_id'] = {}
        self.assertFalse(build_frame_evidence(ep,{})['controls'][1]['accepted_intent']['left']['reliable'])
        cfg = {'alignment': {'time_fallback': {'enabled': True, 'max_control_periods': 1}}}
        result = build_frame_evidence(ep,cfg)
        self.assertFalse(result['controls'][0]['accepted_intent']['left']['reliable'])
        self.assertTrue(result['controls'][1]['accepted_intent']['left']['reliable'])
        self.assertFalse(result['controls'][2]['accepted_intent']['left']['reliable'])
        duplicate = copy.deepcopy(ep['accepted_events']['e']); duplicate['request']['eventId'] = 'e2'
        ep['accepted_events']['e2'] = duplicate
        self.assertFalse(build_frame_evidence(ep,cfg)['controls'][1]['accepted_intent']['left']['reliable'])

    def test_malformed_optional_fields_produce_unavailable_evidence(self):
        ep = episode()
        ep['controls'][0]['status']['actual_ee'] = []
        ep['measurements']['m']['hands'] = [False, None, 'invalid']
        row = build_frame_evidence(ep,{})['rows'][0]
        self.assertFalse(row['evidence']['relative_pose']['left']['available'])
        self.assertFalse(row['auxiliary']['left']['category_mask'])

    def test_loader_keeps_line_provenance_and_first_duplicate_source_without_writes(self):
        from tests.task3.test_curation_io import disk_fixture
        with tempfile.TemporaryDirectory() as temp:
            root = disk_fixture(Path(temp))
            ref = curation_io.discover_episodes(root)[0]
            path = ref['research_path']
            originals = path.read_text(encoding='utf-8')
            measurement = {'kind': 'hand_measurement', 'measurement': {'measurement_id': 'm'}}
            event = {'kind': 'request_executed', 'request': {'eventId': 'e'},
                     'result': {'accepted': True}}
            reused = copy.deepcopy(measurement); reused['measurement']['changed_later'] = True
            path.write_text(''.join(json.dumps(r)+'\n' for r in (measurement,event,event))+originals+
                            json.dumps(reused)+'\n',
                            encoding='utf-8')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            def consume(rows, metadata):
                list(rows)
                return [], {}
            with patch.object(curation_io,'audit_hand_timeline',side_effect=consume):
                loaded = curation_io.load_episode(ref)
            self.assertEqual(loaded['frames'][0]['source_line'],4)
            self.assertEqual(loaded['accepted_events']['e']['source_line'],2)
            self.assertEqual([e['source_line'] for e in loaded['executed_events']],[2,3])
            self.assertEqual(loaded['measurement_sources']['m']['source_line'],1)
            self.assertIn('accepted_event_id_reused:e',loaded['load_errors'])
            self.assertIn('measurement_id_reused:m',loaded['load_errors'])
            self.assertNotIn('changed_later',loaded['measurements']['m'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)


if __name__ == '__main__':
    unittest.main()
