"""Fixed-duration labels use capture time and cannot bridge invalid gaps."""

import unittest

import numpy as np

from simulation.mujoco.hand_motion_window import FixedWindowHandDisplacement


class HandMotionWindowTest(unittest.TestCase):
    def test_interpolates_exact_duration_with_irregular_capture_intervals(self):
        window = FixedWindowHandDisplacement(window_ms=40, max_gap_ms=50)
        for t in (1000, 1010, 1040, 1065):
            window.append(t, np.array([(t - 1000) / 1000, 0.5]))
        label = window.consume(1070)
        self.assertTrue(label["valid"])
        self.assertEqual(label["end_ms"] - label["start_ms"], 40)
        np.testing.assert_allclose(label["uv"], [0.04, 0.0])

    def test_warmup_and_reused_endpoint_are_invalid(self):
        window = FixedWindowHandDisplacement(40, 50)
        window.append(1000, [0.4, 0.5])
        self.assertFalse(window.consume(1001)["valid"])
        window.append(1040, [0.42, 0.5])
        self.assertTrue(window.consume(1045)["valid"])
        self.assertFalse(window.consume(1046)["valid"])

    def test_reliably_stationary_hand_is_valid_zero(self):
        window = FixedWindowHandDisplacement(40, 50)
        for t in (1000, 1020, 1040):
            window.append(t, [0.4, 0.5])
        label = window.consume(1045)
        self.assertTrue(label["valid"])
        np.testing.assert_array_equal(label["uv"], [0.0, 0.0])

    def test_long_capture_gap_requires_a_whole_new_window(self):
        window = FixedWindowHandDisplacement(100, 50)
        for t in (1000, 1040, 1080, 1160):
            window.append(t, [0.4, 0.5])
        self.assertFalse(window.consume(1165)["valid"])
        for t in (1200, 1240):
            window.append(t, [0.4, 0.5])
        self.assertFalse(window.consume(1245)["valid"])
        window.append(1280, [0.4, 0.5])
        self.assertTrue(window.consume(1285)["valid"])

    def test_reset_never_bridges_unknown_or_reentry(self):
        window = FixedWindowHandDisplacement(40, 50)
        window.append(1000, [0.4, 0.5])
        window.append(1040, [0.42, 0.5])
        window.reset()
        window.append(1060, [0.8, 0.5])
        self.assertFalse(window.consume(1065)["valid"])
        window.append(1100, [0.82, 0.5])
        np.testing.assert_allclose(window.consume(1105)["uv"], [0.02, 0])

    def test_stale_and_future_samples_do_not_extrapolate(self):
        window = FixedWindowHandDisplacement(40, 50)
        window.append(1000, [0.4, 0.5])
        window.append(1040, [0.42, 0.5])
        self.assertFalse(window.consume(1039)["valid"])
        self.assertFalse(window.consume(1091)["valid"])

    def test_duplicate_and_out_of_order_samples_cannot_change_a_label(self):
        window = FixedWindowHandDisplacement(40, 50)
        window.append(1000, [0.4, 0.5])
        window.append(1040, [0.42, 0.5])
        window.append(1040, [0.9, 0.5])
        window.append(1030, [0.8, 0.5])
        np.testing.assert_allclose(window.consume(1045)["uv"], [0.02, 0])

    def test_history_is_bounded_and_malformed_points_clear_it(self):
        window = FixedWindowHandDisplacement(40, 50)
        for t in range(1000, 10000, 10):
            window.append(t, [0.4, 0.5])
        self.assertLessEqual(len(window.samples), 6)
        window.append(10000, [float("nan"), 0.5])
        self.assertFalse(window.consume(10005)["valid"])

    def test_bad_configuration_is_rejected(self):
        for duration, gap in ((0, 50), (40, 0), (float("nan"), 50), (40, float("inf"))):
            with self.subTest(duration=duration, gap=gap), self.assertRaises(ValueError):
                FixedWindowHandDisplacement(duration, gap)


if __name__ == "__main__":
    unittest.main()
