"""Recorded categorical evidence replayed in a small synthetic schema envelope.

The two scores, modes, IDs and hand categories come from the accepted requests
in 20261103_174224_102861/episode_000002. Landmark and timing envelopes are
synthetic; no original camera data or research logs are needed by these tests.
"""
import copy
import unittest

from dataset.task3.validate_recordings import audit_hand_timeline
from simulation.mujoco.dual_arm.hand_measurement import inspect_rate_input
from simulation.mujoco.dual_arm.recording_timeline import (
    INTENT_TRAINING_TIMELINE, LEVEL_POLICY_VERSION, LEVEL_RELEASE_VERSION,
    LEVEL_TRIGGER_POLICY, RATE_CALIBRATION, RATE_INTERFACE_VERSION,
    RATE_MODE_CONTRACT, RATE_SUPERVISION_VERSION,
)


PARAMETERS = {
    'gripper_trigger_policy': 'palm_fist_level_v2', 'mode_confirm_ms': 100,
    'open_score_min': .5, 'close_score_min': .6,
    'left_close_score_min': .5, 'motion_score_min': .75,
}
OVERRIDE = {'left': {'Z_CLOSE': .5}}
CASES = (
    ('browser-1793698945269-rtrec96r21-2572',
     '0ca84c11-3fb9-4d09-99f2-f7158760f70c:3127',
     .5465172529220581, 'Z_CLOSE', 'Closed_Fist', .8805418014526367),
    ('browser-1793698945269-rtrec96r21-2612',
     '0ca84c11-3fb9-4d09-99f2-f7158760f70c:3168',
     .5126628875732422, 'HOLD', 'None', .7822887301445007),
)


def recorded_payload(index=0):
    event_id, mid, left_score, right_mode, right_gesture, right_score = CASES[index]
    return {
        'command': 'continuous_input', 'eventId': event_id,
        'measurement_id': mid, 'control_protocol': RATE_INTERFACE_VERSION,
        'interaction_modes': ['Z_CLOSE', right_mode],
        'release_intent': [False, False], 'mode': 'Z',
        'delta_task_m': {'left': [0, 0, 0], 'right': [0, 0, 0]},
        'telemetry': {
            'calibration': {'palm_binding_guard': False},
            'recognition_result': {'measurement': {'hands': [
                {'bound_side': 'right', 'valid': True, 'gesture': right_gesture,
                 'gesture_score': right_score},
                {'bound_side': 'left', 'valid': True, 'gesture': 'Closed_Fist',
                 'gesture_score': left_score},
            ]}},
        },
    }


def frozen_metadata():
    return {
        'session': 'recorded-fixture', 'control_epoch': 2,
        'config': {'contact_control': {'control_protocol': RATE_INTERFACE_VERSION,
                                      'release_confirm_seconds': 0},
                   'gesture_control': copy.deepcopy(PARAMETERS)},
        'control_interface_version': RATE_INTERFACE_VERSION,
        'hand_supervision_version': RATE_SUPERVISION_VERSION,
        'confirmation_gate': {'categorical_gesture': True,
                              'continuous_aperture': False, 'release_intent': False},
        'interaction_mode_contract': RATE_MODE_CONTRACT,
        'policy_contract_version': LEVEL_POLICY_VERSION,
        'gripper_trigger_policy': LEVEL_TRIGGER_POLICY,
        'gripper_trigger_semantics': {
            'version': LEVEL_TRIGGER_POLICY, 'independent_sides': True,
            'pause_recovery': 'fresh_confirmed_current_mode',
            'categorical_confirm_ms': 100,
        },
        'release_authorization_semantics': {
            'version': LEVEL_RELEASE_VERSION, 'independent_sides': True,
            'extra_confirmation_seconds': 0, 'support_required': False,
            'stationarity_required': False,
        },
        'aperture_calibration': RATE_CALIBRATION,
        'release_intent_contract': {
            'names': ['left.release_intent', 'right.release_intent'],
            'shape': [2], 'type': 'bool', 'required_for_policy': True,
        },
        'intent_training_timeline': INTENT_TRAINING_TIMELINE,
        'hand_action_timing_version': 'guard-check-v1',
    }


def research_envelope(index=0):
    """Only schema/timing scaffolding is synthetic; categorical evidence is real."""
    payload = recorded_payload(index)
    capture = 100000 + index * 1000
    payload.update(session='recorded-fixture', control_epoch=2, sentAt=capture)
    measurement = payload['telemetry']['recognition_result']['measurement']
    measurement.update(
        schema_version='task3-hand-v1', measurement_id=payload['measurement_id'],
        source_session='recorded-fixture', source_control_epoch=2,
        capture_epoch_ms=capture, capture_browser_monotonic_ms=index * 1000,
        accepted_recognition=True, width=640, height=480,
    )
    for hand in measurement['hands']:
        hand.update(identity_score=.99, image_landmarks_21=[[.3, .5, 0.] for _ in range(21)])
        hand['image_landmarks_21'][5][0] = .25
        hand['image_landmarks_21'][17][0] = .35
    payload['telemetry'].update(calibrated=True, aperture_calibration=RATE_CALIBRATION)
    payload['telemetry']['recognition_result']['measurement_id'] = payload['measurement_id']
    common = {'session': 'recorded-fixture', 'control_epoch': 2}
    return [
        {**common, 'kind': 'hand_measurement', 'schema_valid': True,
         'measurement': measurement},
        {**common, 'kind': 'request_executed', 'request': payload,
         'wall_time_ms': capture + 10, 'server_monotonic_ns': capture + 10,
         'hand_guard_check': {'epoch_ms': capture + 5, 'monotonic_ns': capture + 5},
         'result': {'accepted': True, 'control_protocol': RATE_INTERFACE_VERSION,
                    'release_intent': [False, False],
                    'interaction_modes': payload['interaction_modes'],
                    'requested_target': [0.] * 14, 'projected_target': [0.] * 14}},
    ]


class RecordedRateAuditTests(unittest.TestCase):
    def test_real_frozen_left_scores_pass_the_actual_timeline_audit(self):
        for index in range(len(CASES)):
            with self.subTest(event_id=CASES[index][0]):
                errors, metrics = audit_hand_timeline(research_envelope(index), frozen_metadata())
                self.assertEqual(errors, [])
                self.assertEqual(metrics['intent_source_events'], 1)
                self.assertEqual(metrics['hand_freshness_audit']['exact_checks'], 1)

    def test_explicit_offline_override_passes_both_recorded_payloads(self):
        for index in range(len(CASES)):
            with self.subTest(event_id=CASES[index][0]):
                self.assertIsNone(inspect_rate_input(
                    recorded_payload(index), parameters=PARAMETERS,
                    score_thresholds_by_side=OVERRIDE))

    def test_live_default_remains_strict_even_when_parameters_contain_left_field(self):
        for index in range(len(CASES)):
            self.assertEqual(inspect_rate_input(recorded_payload(index), parameters=PARAMETERS),
                             'interaction_gesture_mismatch')

    def test_right_close_is_never_lowered_by_left_override(self):
        payload = recorded_payload()
        next(h for h in payload['telemetry']['recognition_result']['measurement']['hands']
             if h['bound_side'] == 'right')['gesture_score'] = .55
        self.assertEqual(inspect_rate_input(payload, parameters=PARAMETERS,
                         score_thresholds_by_side=OVERRIDE), 'interaction_gesture_mismatch')
        right = next(h for h in payload['telemetry']['recognition_result']['measurement']['hands']
                     if h['bound_side'] == 'right')
        right['gesture_score'] = .6
        self.assertIsNone(inspect_rate_input(payload, parameters=PARAMETERS,
                          score_thresholds_by_side=OVERRIDE))

    def test_left_lower_bound_is_inclusive_and_remains_enforced(self):
        payload = recorded_payload()
        left = next(h for h in payload['telemetry']['recognition_result']['measurement']['hands']
                    if h['bound_side'] == 'left')
        left['gesture_score'] = .5
        self.assertIsNone(inspect_rate_input(payload, parameters=PARAMETERS,
                          score_thresholds_by_side=OVERRIDE))
        left['gesture_score'] = .499
        self.assertEqual(inspect_rate_input(payload, parameters=PARAMETERS,
                         score_thresholds_by_side=OVERRIDE), 'interaction_gesture_mismatch')

    def test_explicit_confidence_fraction_accepts_zero_and_one(self):
        for threshold in (0, 1):
            payload = recorded_payload()
            left = next(h for h in payload['telemetry']['recognition_result']['measurement']['hands']
                        if h['bound_side'] == 'left')
            left['gesture_score'] = threshold
            self.assertIsNone(inspect_rate_input(payload, parameters=PARAMETERS,
                              score_thresholds_by_side={'left': {'Z_CLOSE': threshold}}))

    def test_missing_frozen_override_preserves_timeline_rejection(self):
        frozen = frozen_metadata()
        del frozen['config']['gesture_control']['left_close_score_min']
        errors, _ = audit_hand_timeline(research_envelope(), frozen)
        self.assertEqual(errors, ['accepted rate mode disagrees with recorded hand input: interaction_gesture_mismatch'])

    def test_malformed_override_is_rejected(self):
        invalid = [False, [], {'right': {'Z_CLOSE': .5}}, {'left': {'Z_OPEN': .5}},
                   {'left': .5}, {'left': {'Z_CLOSE': None}}]
        invalid += [{'left': {'Z_CLOSE': value}} for value in (-.1, 1.1, True, float('nan'))]
        for override in invalid:
            with self.subTest(override=override):
                self.assertEqual(inspect_rate_input(recorded_payload(), parameters=PARAMETERS,
                                 score_thresholds_by_side=override), 'invalid_score_thresholds_by_side')

    def test_invalid_frozen_override_fails_the_timeline_audit(self):
        for value in (None, True, float('nan'), -.1, 1.1):
            frozen = frozen_metadata()
            frozen['config']['gesture_control']['left_close_score_min'] = value
            errors, _ = audit_hand_timeline(research_envelope(), frozen)
            self.assertIn('accepted rate mode disagrees with recorded hand input: invalid_score_thresholds_by_side', errors)

    def test_offline_override_preserves_other_payload_constraints(self):
        for fault, expected in (
            ('intent', 'release_intent_mode_mismatch'),
            ('category', 'interaction_gesture_mismatch'),
            ('identity', 'untrusted_hand_measurement'),
            ('motion', 'interaction_motion_forbidden'),
        ):
            payload = recorded_payload()
            if fault == 'intent':
                payload['release_intent'][0] = True
            elif fault == 'category':
                payload['telemetry']['recognition_result']['measurement']['hands'][1]['gesture'] = 'Victory'
            elif fault == 'identity':
                payload['telemetry']['recognition_result']['measurement']['hands'][1]['valid'] = False
            else:
                payload['delta_task_m']['left'][0] = .001
            with self.subTest(fault=fault):
                self.assertEqual(inspect_rate_input(payload, parameters=PARAMETERS,
                                 score_thresholds_by_side=OVERRIDE), expected)


if __name__ == '__main__':
    unittest.main()
