"""Operator recovery must not split a physically unchanged training sequence."""
import copy
import unittest

from dataset.task3.curation_config import load_config
from dataset.task3.sequence_stitching import build_sequence_contract
from dataset.task3.curation_rules import execution_hazards
from dataset.task3.frame_evidence import _request_events
from tests.task3.test_sequence_stitching import observations, cut


class EventContinuityTests(unittest.TestCase):
    def rows(self, kind='request_accepted', command='telemetry', reason=None):
        rows = cut(observations())
        facts = rows[20]['evidence']
        facts['interval']['previous_control'] = copy.deepcopy(rows[19]['evidence'])
        edges = {side: {**copy.deepcopy(facts['grippers'][side]), 'event_id': None}
                 for side in ('left', 'right')}
        facts['interval']['events'] = [{
            'kind': kind, 'command': command, 'reason': reason,
            'accepted': kind == 'request_accepted', 'source_line': 399,
            'session': 's', 'control_epoch': 1,
            'execution': copy.deepcopy(facts['execution']),
            'accepted_target': rows[20]['action_recorded'][:], 'gripper_edges': edges,
        }]
        return rows

    def build(self, rows, config=None):
        return build_sequence_contract(rows, config or load_config(), restore_boundaries=False)

    def result(self, *args):
        return self.build(self.rows(*args))

    def test_noop_accepted_telemetry_does_not_split(self):
        out = self.result('request_accepted', 'telemetry')
        self.assertEqual(out['metrics']['safe_stitch_count'], 1, out['gaps'][0]['checks'])
        self.assertEqual(out['metrics']['full_valid_action_windows_30'], 21)

    def test_rejected_recovery_with_complete_unchanged_execution_does_not_split(self):
        out = self.result('request_rejected', 'continuous_input', 'tracking_recovery')
        self.assertEqual(out['metrics']['safe_stitch_count'], 1, out['gaps'][0]['checks'])
        self.assertEqual(out['metrics']['sequence_count'], 1)

    def test_curator_and_stitcher_share_event_evidence(self):
        rows = self.rows('gripper_baseline_changed', None)
        self.assertEqual(execution_hazards(rows[20], load_config()), [])
        check = self.build(rows)['gaps'][0]['checks']['event_continuity_checks'][0]
        self.assertEqual(check['effect'], 'unchanged')
        self.assertEqual(check['bracketing_control_ticks'], [38,39])
        rows[20]['evidence']['interval']['controls'][0]['features']['left.target_position'][1][0] += .02
        self.assertTrue(execution_hazards(rows[20], load_config()))
        self.assertEqual(self.build(rows)['metrics']['safe_stitch_count'], 0)

    def test_tracking_pause_and_redundant_cancellation_are_noop_only_with_evidence(self):
        for kind in ('request_accepted', 'raw_request_received', 'raw_release_confirmation_cancelled'):
            with self.subTest(kind=kind):
                rows = self.rows(kind, 'pause', 'hand_lost')
                event = rows[20]['evidence']['interval']['events'][0]
                event['requested_pause_reason'] = 'hand_lost'
                self.assertEqual(self.build(rows)['metrics']['safe_stitch_count'], 1)
                rows[20]['evidence']['interval']['controls'][0]['execution']['sides']['left']['pending'] = True
                self.assertEqual(self.build(rows)['metrics']['safe_stitch_count'], 0)

    def test_metadata_rebuild_without_authorization_preserves_continuity(self):
        rows = self.rows('gripper_baseline_changed', None)
        rows[20]['evidence']['interval']['controls'][0]['grippers']['left']['baseline_mode'] = None
        self.assertEqual(self.build(rows)['metrics']['safe_stitch_count'], 1)
        rows[20]['evidence']['interval']['controls'][0]['grippers']['left']['authorized_mode'] = 'Z_CLOSE'
        self.assertEqual(self.build(rows)['metrics']['safe_stitch_count'], 0)

    def test_unknown_sources_execution_and_target_changes_never_pass(self):
        for mutation in ('post_missing','post_mismatch','target_jump','pending','auth','missing_edge',
                         'contact','tick','source','clock','epoch','line','gripper'):
            with self.subTest(mutation=mutation):
                rows = self.rows('request_accepted','continuous_input')
                e = rows[20]['evidence']['interval']['events'][0]
                c = rows[20]['evidence']['interval']['controls'][0]
                if mutation == 'post_missing': e.pop('accepted_target')
                if mutation == 'post_mismatch':
                    e['recorded_post_target'] = e['accepted_target'][:]
                    e['recorded_post_target'][0] += .01
                if mutation == 'target_jump': e['accepted_target'][0] += .01
                if mutation == 'pending': e['execution']['sides']['left']['pending'] = True
                if mutation == 'auth': e['gripper_edges']['left']['authorized_mode'] = 'Z_CLOSE'
                if mutation == 'missing_edge': e['gripper_edges']['left'].pop('edge_count')
                if mutation == 'contact': c['contact']['right']['any_finger'] = True
                if mutation == 'tick': c['tick'] += 1
                if mutation == 'source': c['source']['valid'] = False
                if mutation == 'clock': c['sim_time'] += .5
                if mutation == 'epoch': e['control_epoch'] += 1
                if mutation == 'line': e['source_line'] = 9000
                if mutation == 'gripper': c['grippers']['left']['goal_width_m'] -= .01
                self.assertEqual(self.build(rows)['metrics']['safe_stitch_count'], 0)

    def test_safety_conflicts_reset_and_manual_pause_remain_hard(self):
        for kind, command, reason in (
                ('request_rejected','continuous_input','cooperative_axial_conflict'),
                ('request_rejected','continuous_input','workspace_limit'),
                ('request_rejected','continuous_input','palm_binding_guard'),
                ('request_accepted','pause','operator_pause'),
                ('request_accepted','reset',None),
                ('raw_release_confirmation_cancelled',None,'scene_reset')):
            with self.subTest(kind=kind, command=command, reason=reason):
                self.assertEqual(self.result(kind,command,reason)['metrics']['safe_stitch_count'], 0)

    def test_one_arm_hold_never_hides_other_arm_motion(self):
        rows = self.rows('accepted_intent_changed', None)
        c = rows[20]['evidence']['interval']['controls'][0]
        c['features']['right.actual_position'][1][0] += .02
        c['motion']['moving'] = True
        out = build_sequence_contract(rows, load_config())
        self.assertTrue(all(r['keep'] for r in out['rows'][10:40]))
        self.assertGreater(out['metrics']['physical_motion_restored_frames'], 0)

    def test_old_frozen_policy_reproduces_event_hard_cuts(self):
        cfg = load_config(); cfg.pop('stitching')
        out = self.build(self.rows(), cfg)
        self.assertEqual(out['metrics']['safe_stitch_count'], 0)
        self.assertIn('non_motion_control_request:telemetry', out['gaps'][0]['checks']['reasons'])
        self.assertNotIn('event_continuity_checks', out['gaps'][0]['checks'])

    def test_actual_raw_record_locations_and_legacy_extraction(self):
        ep = {'timeline_events': [{'kind':'request_queued', 'command':'telemetry', 'eventId':'q'}],
              'executed_events': [{'request':{'command':'telemetry'}, 'result':{'accepted':True},
                                   'accepted_target_m':[0.]*14}]}
        new, _ = _request_events(ep, execution_evidence=True)
        old, _ = _request_events(ep)
        self.assertEqual(new[0]['accepted_target'], [0.]*14)
        self.assertEqual(new[0]['accepted_target_basis'], 'request_executed.accepted_target_m')
        self.assertEqual(new[1]['command'], 'telemetry')
        self.assertEqual(new[1]['event_id'], 'q')
        self.assertIsNone(old[0]['accepted_target'])
        self.assertIsNone(old[1]['command'])

    def test_queue_pause_reason_needs_earlier_matching_request_not_other_epoch(self):
        events = [{'kind':'request_received', 'source_line':10, 'session':'s','control_epoch':1,
                   'request':{'command':'pause','eventId':'p','reason':'hand_lost'}},
                  {'kind':'request_queued','source_line':11,'session':'s','control_epoch':1,
                   'command':'pause','eventId':'p'}]
        out, _ = _request_events({'timeline_events':events}, execution_evidence=True)
        self.assertEqual(out[1]['requested_pause_reason'], 'hand_lost')
        self.assertEqual(out[1]['requested_pause_source_line'], 10)
        for field, value in (('control_epoch',2),('source_line',9),('eventId','other')):
            changed=copy.deepcopy(events); changed[1][field]=value
            out, _ = _request_events({'timeline_events':changed}, execution_evidence=True)
            self.assertIsNone(out[1]['requested_pause_reason'])


if __name__ == '__main__':
    unittest.main()
