"""Adjacent low-change runs are candidates, not permission to skip physics."""
import unittest

from test_intent_curation import evidence_rows
from test_wait_compression import wait_rows
from dataset.task3.curation_rules import load_config, select_rows, validate_config


class AdjacentStationarityTests(unittest.TestCase):
    def test_adjacent_small_changes_do_not_accumulate_over_three_frames(self):
        rows = evidence_rows(12)
        for i, row in enumerate(rows):
            row['state_recorded'][0] = i * .0015
        out = select_rows(rows)['rows']
        self.assertTrue(out[3]['sa_stagnant'])
        self.assertAlmostEqual(out[3]['sa_window']['maximum_changes']['state_joint_rad'], .0015)
        self.assertEqual(out[3]['sa_window']['history_count'], 1)
        self.assertEqual(out[1]['sa_stationarity']['run_length'], 11)
        self.assertTrue(out[1]['sa_stationarity']['confirmed'])

    def test_short_low_change_run_does_not_qualify(self):
        rows = evidence_rows(7)
        for row, value in zip(rows, [0, .01, .01, .01, .02, .03, .04]):
            row['state_recorded'][0] = value
        out = select_rows(rows)['rows']
        self.assertFalse(out[2]['sa_stagnant'])
        self.assertEqual(out[2]['sa_stationarity']['run_length'], 2)
        self.assertEqual(out[2]['policy_reason'], ['insufficient_consecutive_low_change_frames'])
        self.assertTrue(out[2]['keep'])

    def test_confirmation_is_retroactive_and_boundary_resets_it(self):
        rows = evidence_rows(8)
        for row, value in zip(rows, [0, .01, .01, .01, .01, .02, .02, .02]):
            row['state_recorded'][0] = value
        out = select_rows(rows)['rows']
        self.assertTrue(all(r['sa_stagnant'] for r in out[2:5]))
        self.assertFalse(out[5]['sa_stagnant'])
        rows[3]['evidence']['source']['hard_boundary'] = True
        out = select_rows(rows)['rows']
        self.assertFalse(out[2]['sa_stagnant'])
        self.assertFalse(out[3]['sa_stagnant'])
        self.assertFalse(out[4]['sa_stagnant'])

    def test_stationary_candidate_does_not_override_real_physics(self):
        rows = wait_rows()
        rows[80]['evidence']['motion']['moving'] = True
        out = select_rows(rows)['rows']
        self.assertTrue(out[80]['sa_stagnant'])
        self.assertTrue(out[80]['keep'])
        self.assertIn('actual_physical_motion', out[80]['policy_reason'])

    def test_config_and_historical_config_compatibility(self):
        cfg = load_config()
        self.assertEqual(cfg['sa']['window_frames'], 1)
        self.assertEqual(cfg['sa']['minimum_static_frames'], 3)
        cfg['sa']['minimum_static_frames'] = 0
        with self.assertRaises(ValueError):
            validate_config(cfg)
        del cfg['sa']['minimum_static_frames']
        cfg['sa']['window_frames'] = 3
        self.assertEqual(validate_config(cfg), cfg)
        rows = evidence_rows(8)
        for i, row in enumerate(rows):
            row['state_recorded'][0] = i * .0015
        self.assertTrue(select_rows(rows, cfg)['rows'][3]['sa_changed'])
        for bad in (None, [], 'invalid'):
            cfg = load_config()
            cfg['sa'] = bad
            with self.assertRaises(ValueError):
                validate_config(cfg)


if __name__ == '__main__':
    unittest.main()
