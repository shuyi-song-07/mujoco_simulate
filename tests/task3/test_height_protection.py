"""Both actual sites low is an evidence fallback, not a blanket clip veto."""
import unittest

from test_intent_curation import evidence_rows, active, execution
from dataset.task3.curation_rules import select_rows


def stationary_middle(left_z, right_z):
    rows = evidence_rows(180)
    active(rows, 0, 20)
    active(rows, 160, 180)
    for row in rows:
        for side, z in (('left', left_z), ('right', right_z)):
            row['evidence']['features'][side + '.actual_position'] = ('position', [0., 0., z])
    return rows


class HeightProtectionTests(unittest.TestCase):
    def test_complete_low_stationary_evidence_can_compress(self):
        out = select_rows(stationary_middle(.38, .38))
        self.assertFalse(out['rows'][80]['keep'])
        self.assertFalse(out['rows'][80]['height_protected'])
        self.assertTrue(out['rows'][80]['height_evidence']['both_below_threshold'])

    def test_missing_physical_evidence_requires_both_low_for_fallback_label(self):
        for left, right, protected in (
            (.389, .389, True), (.389, .391, False), (.391, .389, False),
            (.390, .389, False), (.389, .390, False), (.390, .390, False),
        ):
            with self.subTest(left=left, right=right):
                rows = stationary_middle(left, right)
                rows[80]['evidence']['motion']['available'] = False
                row = select_rows(rows)['rows'][80]
                self.assertEqual(row['height_protected'], protected)
                self.assertTrue(row['keep'] and row['review_required'])

    def test_actual_pose_controls_fallback_not_requested_target(self):
        rows = stationary_middle(.389, .389)
        rows[80]['evidence']['motion']['available'] = False
        rows[80]['action_recorded'][2] = rows[80]['action_recorded'][9] = .6
        self.assertTrue(select_rows(rows)['rows'][80]['height_protected'])
        rows = stationary_middle(.5, .5)
        rows[80]['evidence']['motion']['available'] = False
        rows[80]['action_recorded'][2] = rows[80]['action_recorded'][9] = .2
        self.assertFalse(select_rows(rows)['rows'][80]['height_protected'])

    def test_missing_height_is_not_replaced_by_target(self):
        rows = stationary_middle(.5, .5)
        rows[80]['evidence']['features'].pop('left.actual_position')
        row = select_rows(rows)['rows'][80]
        self.assertFalse(row['height_evidence']['available'])
        self.assertFalse(row['height_protected'])
        self.assertTrue(row['keep'] and row['review_required'])
        self.assertIn('missing_actual_ee_height', row['policy_reason'])

    def test_high_execution_is_protected_through_physical_end(self):
        rows = stationary_middle(.5, .5)
        execution(rows, 30)
        out = select_rows(rows)
        self.assertTrue(out['rows'][80]['execution_grasp_phase'])
        self.assertTrue(out['rows'][80]['grasp_protected'])
        self.assertFalse(out['rows'][80]['height_protected'])
        self.assertTrue(all(r['keep'] for r in out['rows'][24:]))
        self.assertEqual(out['protected_intervals'], out['execution_intervals'])

    def test_fallback_regions_do_not_cross_epoch_boundary(self):
        rows = stationary_middle(.38, .38)
        for i, row in enumerate(rows):
            row['evidence']['motion']['available'] = False
            if i >= 90:
                row['control_epoch'] = 2
        intervals = select_rows(rows)['protected_intervals']
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0]['end_raw_frame_index'], 89)
        self.assertEqual(intervals[1]['start_raw_frame_index'], 90)


if __name__ == '__main__':
    unittest.main()
