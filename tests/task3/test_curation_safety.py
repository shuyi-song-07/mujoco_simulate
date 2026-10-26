"""Independent causal and bilateral checks for the sole curation policy."""
import copy
import unittest

from dataset.task3.curation_rules import select_rows, load_config, execution_hazards
from tests.task3.test_intent_curation import evidence_rows, active, execution


def middle():
    rows = evidence_rows(180)
    active(rows, 0, 20)
    active(rows, 160, 180)
    return rows


class CurationSafetyTests(unittest.TestCase):
    def test_same_side_context_suffices_but_different_hands_cannot_substitute(self):
        for same_side in (True, False):
            rows = middle()
            for i, row in enumerate(rows):
                invalid = 'right' if same_side or i < 160 else 'left'
                row['evidence']['accepted_intent'][invalid].update(mode=None, reliable=False)
            out = select_rows(rows)
            self.assertEqual(out['rows'][80]['keep'], not same_side)
            if same_side:
                self.assertEqual(out['rows'][80]['phase_pause_interpretation']['comparable_sides'], ['left'])
                self.assertEqual(sum(r['keep'] for r in out['rows'][20:160]), 3)
            else:
                self.assertEqual(out['rows'][80]['phase_pause_interpretation']['reason'],
                                 'no_same_side_comparable_intent')

    def test_other_unknown_hand_authorization_is_never_assumed_idle(self):
        rows = middle()
        for row in rows:
            row['evidence']['accepted_intent']['right'].update(mode='HOLD', reliable=False)
        for row in rows[30:150]:
            row['evidence']['execution']['sides']['right'].update(pending=True,
                pending_authorization=True, trusted_hold_resumable=True)
        out = select_rows(rows)
        self.assertTrue(all(r['keep'] for r in out['rows'][30:150]))
        self.assertIn('right_unfinished_execution_or_authorization', out['rows'][80]['policy_reason'])

    def test_auth_between_images_is_protected_even_if_observation_has_none(self):
        rows = middle()
        rows[80]['evidence']['interval']['execution_pending'] = True
        out = select_rows(rows)
        self.assertTrue(out['rows'][80]['keep'])
        self.assertTrue(all(r['keep'] for r in out['rows'][74:87]))
        self.assertIn('pending_execution_between_images', execution_hazards(out['rows'][80]))

    def test_accepted_execution_between_ticks_retains_observation(self):
        rows = middle()
        rows[80]['evidence']['events'] = [{'kind': 'request_accepted',
            'execution': {'sides': {'right': {'pending': True}}}}]
        out = select_rows(rows)
        self.assertTrue(out['rows'][80]['keep'])
        self.assertTrue(all(r['keep'] for r in out['rows'][74:87]))

    def test_baseline_and_authorization_events_get_existing_six_frame_context(self):
        for kind in ('gripper_baseline_changed', 'gripper_authorization_changed',
                     'gripper_execution_changed', 'release_confirmation_changed'):
            with self.subTest(kind=kind):
                rows = middle()
                rows[80]['evidence']['events'] = [{'kind': kind, 'side': 'right'}]
                out = select_rows(rows)
                self.assertTrue(all(r['keep'] for r in out['rows'][74:87]))

    def test_mode_alone_is_not_open_or_close_execution_in_any_phase(self):
        for mode in ('XY_HOLD', 'Z_OPEN', 'Z_CLOSE'):
            for held in (False, True):
                with self.subTest(mode=mode, held=held):
                    rows = middle()
                    for row in rows:
                        for side in ('left', 'right'):
                            row['evidence']['accepted_intent'][side]['mode'] = mode
                            row['evidence']['contact'][side].update(
                                stable=held, double_finger=held, any_finger=held)
                    out = select_rows(rows)
                    self.assertEqual(sum(r['keep'] for r in out['rows'][20:160]), 3)
                    self.assertFalse(out['execution_intervals'])

    def test_missing_pending_state_is_unknown_even_when_every_target_matches(self):
        rows = middle()
        for row in rows[20:160]:
            row['evidence']['execution']['available'] = False
        out = select_rows(rows)
        self.assertTrue(all(r['keep'] and r['review_required'] for r in out['rows'][20:160]))

    def test_unfinished_close_survives_high_altitude_and_long_wait(self):
        rows = middle()
        execution(rows, 25)
        out = select_rows(rows)
        self.assertTrue(all(r['keep'] for r in out['rows'][25:]))
        self.assertTrue(out['rows'][150]['grasp_protected'])
        self.assertFalse(out['rows'][150]['height_protected'])

    def test_static_core_budgets_are_independent_and_physical_settings_unchanged(self):
        cfg = load_config()
        physical = copy.deepcopy(cfg['physical_motion'])
        same = select_rows(middle(), cfg)
        rows = middle()
        for row in rows[160:]:
            row['evidence']['accepted_intent']['left']['mode'] = 'Z_OPEN'
        changed = select_rows(rows, cfg)
        self.assertEqual(sum(r['keep'] for r in same['rows'][20:160]), 3)
        self.assertEqual(sum(r['keep'] for r in changed['rows'][20:160]), 10)
        cfg['waits']['same_intent_frames'] = 7
        self.assertEqual(sum(r['keep'] for r in select_rows(middle(), cfg)['rows'][20:160]), 7)
        self.assertEqual(cfg['physical_motion'], physical)

    def test_pending_release_confirmation_is_not_plain_stationary_wait(self):
        rows = middle()
        for row in rows[50:130]:
            row['evidence']['execution']['sides']['left'].update(
                pending=True, release_confirmation_pending=True)
        out = select_rows(rows)
        self.assertTrue(all(r['keep'] for r in out['rows'][50:130]))

    def test_new_baseline_after_static_core_keeps_context_not_entire_wait(self):
        rows = middle()
        for row in rows[20:160]:
            for side in ('left', 'right'):
                row['evidence']['accepted_intent'][side].update(mode='HOLD', reliable=False)
        for row in rows[160:]:
            for side in ('left', 'right'):
                row['evidence']['accepted_intent'][side]['mode'] = 'Z_OPEN'
        rows[160]['evidence']['events'] = [{'kind':'gripper_baseline_changed','side':'left'}]
        out = select_rows(rows)
        self.assertFalse(out['rows'][80]['keep'])
        self.assertEqual(sum(r['keep'] for r in out['rows'][20:154]), 10)
        self.assertTrue(all(r['keep'] for r in out['rows'][154:167]))

    def test_cumulative_joint_and_aperture_motion_never_crosses_each_core_limit(self):
        for channel, increment, limit in ((0, .0000075, .001), (7, .00000075, .0001)):
            with self.subTest(channel=channel):
                rows = evidence_rows(240)
                active(rows, 0, 20)
                active(rows, 220, 240)
                for index in range(20, 220):
                    rows[index]['state_recorded'][channel] = (index - 20) * increment
                out = select_rows(rows)
                # No individual frame/short A/S change reaches a threshold,
                # but the accumulated physical excursion must not be dropped.
                self.assertFalse(any(r['sa_changed'] for r in out['rows'][20:220]))
                kept = [r for r in out['rows'][20:220] if r['keep']]
                for left, right in zip(kept, kept[1:]):
                    if right['raw_frame_index'] > left['raw_frame_index']+1:
                        self.assertLess(abs(right['state_recorded'][channel]-left['state_recorded'][channel]), limit)
                self.assertFalse(out['rows'][150]['keep'])
                kept_left = max(r['raw_frame_index'] for r in out['rows'][:150] if r['keep'])
                kept_right = min(r['raw_frame_index'] for r in out['rows'][150:] if r['keep'])
                self.assertLess(abs(rows[kept_right]['state_recorded'][channel]
                                    - rows[kept_left]['state_recorded'][channel]), limit)

    def test_slow_actual_joint_departure_ends_prefix_at_physical_threshold(self):
        rows = evidence_rows(200)
        active(rows, 180, 200)
        for index in range(60, 200):
            rows[index]['state_recorded'][0] = min((index-59) * .00005, .003)
        out = select_rows(rows)
        self.assertFalse(out['rows'][20]['keep'])
        self.assertTrue(all(r['keep'] for r in out['rows'][73:80]))
        self.assertTrue(all(r['frame_role'] != 'PREFIX_IDLE' for r in out['rows'][79:]))



if __name__ == '__main__':
    unittest.main()
