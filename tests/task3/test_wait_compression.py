"""Contract for compressing verified static waits, not physical execution."""
import unittest

from test_intent_curation import evidence_rows, active, execution
from dataset.task3.curation_rules import select_rows, load_config


def wait_rows():
    rows = evidence_rows(160)
    active(rows, 0, 30)
    active(rows, 140, 160)
    for row in rows[30:140]:
        for side in ('left', 'right'):
            row['evidence']['accepted_intent'][side]['mode'] = 'HOLD'
    return rows


class WaitCompressionTests(unittest.TestCase):
    def test_resumed_subthreshold_motion_does_not_erase_earlier_static_core(self):
        rows = wait_rows()
        cfg = load_config()
        cfg['physical_motion']['position_m'] = .0001
        rows[139]['evidence']['object']['position'][0] = .0004
        out = select_rows(rows, cfg)['rows']
        self.assertFalse(out[80]['keep'])
        self.assertTrue(out[139]['keep'])
        self.assertEqual(sum(r['keep'] for r in out[30:139]), 3)

    def test_two_static_islands_are_both_compressed_without_crossing_motion(self):
        rows = wait_rows()
        cfg = load_config()
        cfg['physical_motion']['position_m'] = .0001
        for row in rows[85:140]:
            row['evidence']['object']['position'][0] = .001
        out = select_rows(rows, cfg)['rows']
        self.assertFalse(out[50]['keep'])
        self.assertFalse(out[110]['keep'])
        self.assertTrue(out[84]['keep'])
        self.assertTrue(out[85]['keep'])

    def test_known_mode_change_compresses_static_core_to_total_ten_frames(self):
        rows = wait_rows()
        for row in rows[140:]:
            for side in ('left', 'right'):
                row['evidence']['accepted_intent'][side]['mode'] = 'Z_OPEN'
        out = select_rows(rows)
        part = out['rows'][30:140]
        self.assertEqual(sum(r['keep'] for r in part), 10)
        self.assertEqual(part[50]['frame_role'], 'INTENT_TRANSITION_IDLE')
        self.assertEqual(part[50]['pre_intent'], ['XY_HOLD', 'XY_HOLD'])
        self.assertEqual(part[50]['post_intent'], ['Z_OPEN', 'Z_OPEN'])

    def test_asynchronous_hand_stop_can_reach_matching_pre_context(self):
        rows = wait_rows()
        for row in rows[10:30]:
            row['evidence']['accepted_intent']['right']['mode'] = 'HOLD'
        out = select_rows(rows)
        self.assertEqual(sum(r['keep'] for r in out['rows'][30:140]), 3)
        self.assertEqual(out['rows'][80]['frame_role'], 'RECENTER_IDLE')

    def test_residual_settling_is_additional_to_stable_core_budget(self):
        rows = wait_rows()
        for i in range(30, 140):
            rows[i]['evidence']['relative_pose']['left']['position'][0] = min(i - 30, 3) * .00008
        cfg = load_config()
        cfg['physical_motion']['position_m'] = .0001
        out = select_rows(rows, cfg)
        part = out['rows'][30:140]
        self.assertEqual(sum(r['keep'] for r in part), 5)
        self.assertTrue(all(r['keep'] for r in part[:2]))
        self.assertFalse(part[50]['keep'])
        self.assertTrue(part[-1]['keep'])

    def test_fast_recenter_threshold_and_transition_are_independent(self):
        for count, expected in ((5, 5), (6, 3), (10, 3)):
            rows = wait_rows()
            active(rows, 30+count, 160)
            for row in rows[30+count:]:
                for side in ('left', 'right'):
                    row['evidence']['accepted_intent'][side]['mode'] = 'XY_HOLD'
            out = select_rows(rows)['rows']
            self.assertEqual(sum(r['keep'] for r in out[30:30+count]), expected)
            for row in rows[30+count:]:
                for side in ('left', 'right'):
                    row['evidence']['accepted_intent'][side]['mode'] = 'Z_OPEN'
            out = select_rows(rows)['rows']
            self.assertTrue(all(r['keep'] for r in out[30:30+count]))

    def test_position_tolerance_allows_small_drift_but_partitions_larger_drift(self):
        rows = wait_rows()
        for i in range(30, 140):
            rows[i]['evidence']['relative_pose']['left']['position'][0] = min(i - 30, 3) * .00008
        self.assertEqual(sum(r['keep'] for r in select_rows(rows)['rows'][30:140]), 3)
        # Do not accumulate unbounded subthreshold drift in one static interval.
        for i in range(30, 140):
            rows[i]['evidence']['object']['position'][0] = (i-30) * .0002
        self.assertTrue(all(r['keep'] for r in select_rows(rows)['rows'][30:140]))

    def test_hidden_third_mode_and_reversal_are_not_compressed(self):
        for sequence in (
            [('Z_CLOSE', 'XY_HOLD')],
            [('Z_OPEN', 'Z_OPEN'), ('XY_HOLD', 'XY_HOLD')],
        ):
            rows = wait_rows()
            for row in rows[140:]:
                for side in ('left', 'right'):
                    row['evidence']['accepted_intent'][side]['mode'] = 'Z_OPEN'
            rows[80]['evidence']['interval']['mode_sequence'] = [
                dict(left=l, right=r, left_reliable=True, right_reliable=True)
                for l, r in sequence
            ]
            out = select_rows(rows)
            self.assertTrue(out['rows'][80]['keep'])

    def test_physical_execution_boundary_is_not_capped_by_wait_budget(self):
        rows = wait_rows()
        execution(rows, 40)
        out = select_rows(rows)
        self.assertTrue(all(r['keep'] for r in out['rows'][34:47]))
        self.assertTrue(out['rows'][80]['execution_grasp_phase'])
        self.assertEqual(out['rows'][34]['frame_role'], 'PRE_GRASP_HOLD')

    def test_budget_is_configurable_and_does_not_change_event_context(self):
        rows = wait_rows()
        cfg = load_config()
        cfg['waits']['same_intent_frames'] = 8
        out = select_rows(rows, cfg)
        self.assertEqual(sum(r['keep'] for r in out['rows'][30:140]), 8)


if __name__ == '__main__':
    unittest.main()
