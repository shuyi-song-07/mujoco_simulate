"""Terminal cropping is downstream of actual release, and overrides height."""
import copy
import unittest

from test_intent_curation import evidence_rows, execution
from dataset.task3.curation_rules import load_config, select_rows


def release_rows(count=40, right_open_at=14):
    rows = evidence_rows(count)
    for i, row in enumerate(rows):
        f = row['evidence']
        f['source']['available'] = True
        for side, open_at in (('left', 14), ('right', right_open_at)):
            f.setdefault('features', {})[side + '.actual_position'] = ('position', [0., 0., .38])
            f['grippers'][side]['actual_width_m'] = .08 if i >= open_at else .03
            contact = i < 10
            f['contact'][side].update(any_finger=contact, double_finger=contact, stable=contact)
        f['task'].update(done=i >= 16, success=i >= 16)
    execution(rows, 10, side='both', opening=True)
    return rows


class TerminalTailTests(unittest.TestCase):
    def test_completion_frame_plus_exactly_three_then_delete_even_when_both_low(self):
        rows = release_rows()
        original = copy.deepcopy(rows)
        out = select_rows(rows)
        end = out['terminal_release']
        self.assertEqual(end['completion_raw_frame'], 18)
        self.assertEqual(end['last_kept_raw_frame'], 21)
        self.assertTrue(all(r['keep'] for r in out['rows'][:22]))
        self.assertFalse(any(r['keep'] for r in out['rows'][22:]))
        self.assertEqual(out['rows'][18]['policy_reason'], ['verified_bilateral_release_complete'])
        self.assertEqual([r['frame_role'] for r in out['rows'][19:22]], ['RELEASE_TAIL_CONTEXT'] * 3)
        self.assertTrue(out['rows'][22]['height_evidence']['both_below_threshold'])
        self.assertFalse(out['rows'][22]['height_protected'])
        self.assertFalse(out['rows'][22]['height_protection_overridden_by_terminal'])
        self.assertTrue(all(c['end_raw_frame_index'] <= 18 for c in out['protected_intervals']))
        self.assertEqual(rows, original)

    def test_waits_for_later_actual_gripper_not_first_open_command(self):
        out = select_rows(release_rows(right_open_at=21))
        self.assertEqual(out['terminal_release']['completion_raw_frame'], 25)
        self.assertTrue(all(r['keep'] for r in out['rows'][10:29]))
        self.assertFalse(out['rows'][29]['keep'])

    def test_short_tail_is_cropped_without_one_second_minimum(self):
        out = select_rows(release_rows(count=25))
        self.assertEqual(out['terminal_release']['excluded_frames'], 3)
        out = select_rows(release_rows(count=20))
        self.assertEqual(out['terminal_release']['retained_post_frames'], 1)
        self.assertTrue(all(r['keep'] for r in out['rows']))

    def test_tail_budget_is_configurable_and_independent_of_event_context(self):
        cfg = load_config()
        cfg['waits']['post_release_frames'] = 1
        cfg['context']['after_frames'] = 20
        out = select_rows(release_rows(), cfg)
        self.assertEqual(out['terminal_release']['last_kept_raw_frame'], 19)
        self.assertFalse(out['rows'][20]['keep'])

    def test_done_or_open_request_without_actual_release_never_crops(self):
        for failure in ('no_event', 'no_contact_history', 'still_contact', 'not_open',
                        'object_moving', 'not_success', 'missing_source', 'missing_width'):
            with self.subTest(failure=failure):
                rows = release_rows()
                if failure == 'no_event':
                    rows[10]['evidence']['events'].clear()
                for row in rows:
                    f = row['evidence']
                    if failure == 'no_contact_history':
                        for side in ('left', 'right'):
                            f['contact'][side].update(any_finger=False, double_finger=False, stable=False)
                    elif failure == 'still_contact':
                        f['contact']['right']['any_finger'] = True
                    elif failure == 'not_open':
                        f['grippers']['right']['actual_width_m'] = .07
                    elif failure == 'object_moving':
                        f['object']['linear_speed_mps'] = .02
                    elif failure == 'not_success':
                        f['task']['success'] = False
                    elif failure == 'missing_source':
                        f['source']['available'] = False
                    elif failure == 'missing_width':
                        f['grippers']['right'].pop('actual_width_m')
                out = select_rows(rows)
                self.assertIsNone(out['terminal_release'])
                self.assertTrue(all(r['keep'] for r in out['rows']))

    def test_later_retry_or_regression_invalidates_apparent_completion(self):
        rows = release_rows()
        execution(rows, 30, side='right')
        self.assertIsNone(select_rows(rows)['terminal_release'])
        rows = release_rows()
        for row in rows[30:]:
            row['evidence']['task'].update(done=False, success=False)
        self.assertIsNone(select_rows(rows)['terminal_release'])

    def test_source_reset_and_invalid_rows_do_not_inherit_old_release(self):
        for boundary in ('epoch', 'invalid'):
            rows = release_rows()
            for row in rows[30:]:
                if boundary == 'epoch':
                    row['control_epoch'] = 2
            if boundary == 'invalid':
                rows[30]['evidence']['source']['valid'] = False
            out = select_rows(rows)
            self.assertIsNone(out['terminal_release'])
            self.assertTrue(all(r['keep'] for r in out['rows'][31:]))

    def test_controller_abort_after_complete_is_not_a_physical_failure_by_itself(self):
        rows = release_rows()
        rows[19]['evidence']['events'].append({
            'kind': 'gripper_aborted', 'side': 'both', 'event_id': 'complete_stop', 'control_tick': 38})
        out = select_rows(rows)
        self.assertEqual(out['terminal_release']['completion_raw_frame'], 18)
        self.assertFalse(out['rows'][22]['keep'])

    def test_post_completion_robot_retreat_is_terminal_not_new_task_motion(self):
        rows = release_rows()
        for row in rows[26:]:
            row['evidence']['motion']['moving'] = True
        out = select_rows(rows)
        self.assertFalse(any(r['keep'] for r in out['rows'][22:]))


if __name__ == '__main__':
    unittest.main()
