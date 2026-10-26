"""Causal scheduling, finite validity and rotation-safe ACT aggregation."""
import unittest
import numpy as np

from rollout.action_chunks import ActionChunkScheduler, InvalidActionChunk, LEGACY, _rotation


class ActionChunkTests(unittest.TestCase):
    def test_chunk_offsets_advance_without_extra_model_queries(self):
        calls = []
        def predict():
            origin = len(calls)*5
            calls.append(origin)
            return np.repeat((origin+np.arange(30))[:, None], 14, axis=1)
        scheduler = ActionChunkScheduler("chunked", n_action_steps=5)
        records = [scheduler.next_action(t, predict) for t in range(12)]
        self.assertEqual(calls, [0, 5, 10])
        np.testing.assert_array_equal([a[0] for a, _ in records], np.arange(12))
        self.assertEqual([r["query_origin"] for _, r in records], [0]*5+[5]*5+[10]*2)
        self.assertEqual([r["selected_offset"] for _, r in records], [0,1,2,3,4]*2+[0,1])

    def test_explicit_legacy_first_action_queries_each_observation(self):
        scheduler = ActionChunkScheduler(LEGACY)
        for frame in range(4):
            action, meta = scheduler.next_action(frame, lambda: np.full((30,14), frame))
            np.testing.assert_array_equal(action, np.full(14, frame))
            self.assertEqual(meta["selected_offset"], 0)
            self.assertTrue(meta["queried"])
        self.assertEqual(scheduler.query_count, 4)

    def test_temporal_aggregation_aligns_absolute_time_and_expires_old_chunks(self):
        scheduler = ActionChunkScheduler("temporal_ensemble", chunk_size=3,
                                         n_action_steps=1, temporal_ensemble_coeff=0.)
        records = []
        for frame in range(6):
            chunk = np.zeros((3,14))
            chunk[:, 0] = frame*10 + np.arange(3)
            records.append(scheduler.next_action(frame, lambda: chunk))
        # At t=2: query0/h2=2, query1/h1=11, query2/h0=20.
        self.assertAlmostEqual(records[2][0][0], 11.)
        # Query0 has expired at t=3: 12,21,30; never use its invalid h3.
        self.assertAlmostEqual(records[3][0][0], 21.)
        self.assertEqual(records[3][1]["source_origins"], [1,2,3])
        self.assertEqual(records[5][1]["source_offsets"], [2,1,0])
        self.assertLessEqual(len(scheduler._chunks), 3)

    def test_temporal_coeff_uses_documented_oldest_first_weighting(self):
        scheduler = ActionChunkScheduler("temporal_ensemble", temporal_ensemble_coeff=.1)
        scheduler.next_action(0, lambda: np.zeros((30,14)))
        action, record = scheduler.next_action(1, lambda: np.full((30,14), .2))
        expected = .2*np.exp(-.1)/(1+np.exp(-.1))
        self.assertAlmostEqual(action[0], expected)
        self.assertGreater(record["source_weights"][0], record["source_weights"][1])

    def test_zero_targets_are_valid_predictions(self):
        scheduler = ActionChunkScheduler("temporal_ensemble")
        for frame in range(3):
            action, meta = scheduler.next_action(frame, lambda: np.zeros((30,14)))
            np.testing.assert_array_equal(action, np.zeros(14))
            self.assertEqual(len(meta["source_origins"]), frame+1)

    def test_rotation_average_does_not_turn_pi_aliases_into_zero(self):
        scheduler = ActionChunkScheduler("temporal_ensemble", temporal_ensemble_coeff=0.)
        first, second = np.zeros((30,14)), np.zeros((30,14))
        first[:, [3,10]], second[:, [3,10]] = np.pi-.01, -np.pi+.01
        scheduler.next_action(0, lambda: first)
        action, _ = scheduler.next_action(1, lambda: second)
        for start in (3,10):
            np.testing.assert_allclose(_rotation(action[start:start+3]), _rotation([np.pi,0,0]), atol=1e-12)

    def test_skip_duplicate_negative_and_future_execution_frames_rejected(self):
        scheduler = ActionChunkScheduler()
        for bad in (-1, 1, True, 0.):
            with self.subTest(frame=bad), self.assertRaises(ValueError):
                scheduler.next_action(bad, lambda: np.zeros((30,14)))
        scheduler.next_action(0, lambda: np.zeros((30,14)))
        with self.assertRaises(ValueError):
            scheduler.next_action(0, lambda: np.zeros((30,14)))
        self.assertEqual(scheduler.query_count, 1)

    def test_invalid_prediction_does_not_consume_a_step(self):
        for bad in (np.zeros((29,14)), np.full((30,14), np.nan),
                    np.full((30,14), np.inf), "bad"):
            scheduler = ActionChunkScheduler()
            with self.subTest(kind=str(bad)[:20]), self.assertRaises(InvalidActionChunk):
                scheduler.next_action(0, lambda: bad)
            self.assertEqual(scheduler.query_count, 0)
            action, _ = scheduler.next_action(0, lambda: np.zeros((30,14)))
            np.testing.assert_array_equal(action, np.zeros(14))

    def test_prediction_storage_does_not_alias_callers_array(self):
        chunk = np.zeros((30,14))
        scheduler = ActionChunkScheduler()
        scheduler.next_action(0, lambda: chunk)
        chunk[:] = 99.
        cached = scheduler.latest_chunk
        cached[:] = 22.
        action, _ = scheduler.next_action(1, lambda: self.fail("unexpected model query"))
        np.testing.assert_array_equal(action, np.zeros(14))

    def test_callback_value_error_is_not_reclassified_as_invalid_prediction(self):
        scheduler = ActionChunkScheduler()
        original = ValueError("observation preprocessing failed")
        def fail():
            raise original
        with self.assertRaises(ValueError) as caught:
            scheduler.next_action(0, fail)
        self.assertIs(caught.exception, original)
        self.assertNotIsInstance(caught.exception, InvalidActionChunk)
        self.assertEqual(scheduler.query_count, 0)

    def test_invalid_schedule_parameters_rejected(self):
        for kwargs in ({"mode":"unknown"},{"chunk_size":True},{"n_action_steps":31},
                       {"n_action_steps":0},{"temporal_ensemble_coeff":-1},
                       {"temporal_ensemble_coeff":float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ActionChunkScheduler(**kwargs)


if __name__ == "__main__":
    unittest.main()
