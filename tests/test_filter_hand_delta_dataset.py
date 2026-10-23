import unittest

import numpy as np

from dataset.demonstrations.filter_hand_delta_dataset import causal_segment_ema


class CausalSegmentEmaTest(unittest.TestCase):
    def test_carries_across_invalid_frames_and_resets_on_segment(self):
        delta = np.array([[1, 0], [0, 0], [3, 0], [9, 0]], dtype=np.float32)
        valid = np.array([1, 0, 1, 1], dtype=bool)
        segment = np.array([4, -1, 4, 5])
        episode = np.zeros(4, dtype=np.int64)
        actual = causal_segment_ema(delta, valid, segment, episode, alpha=0.5)
        expected = np.array([[1, 0], [0, 0], [2, 0], [9, 0]], dtype=np.float32)
        np.testing.assert_allclose(actual, expected)

    def test_resets_between_episodes(self):
        delta = np.array([[2, 2], [6, 6]], dtype=np.float32)
        valid = np.ones(2, dtype=bool)
        segment = np.zeros(2, dtype=np.int64)
        episode = np.array([0, 1])
        np.testing.assert_allclose(
            causal_segment_ema(delta, valid, segment, episode, alpha=0.5), delta
        )


if __name__ == "__main__":
    unittest.main()
