"""Wrist displacement labels must never bridge pauses or re-entry."""

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from simulation.mujoco.record_mujoco_panda import MujocoPandaRecorder
from simulation.mujoco.render_workers import _save_hand_trajectory


class HandMotionRecordingTest(unittest.TestCase):
    def test_raw_and_filtered_paths_save_every_received_sample(self):
        recorder = object.__new__(MujocoPandaRecorder)
        recorder.control_paused = False
        recorder.control_token = "first"
        recorder.hand_visible = True
        recorder.recording = True
        recorder.episode_frames = 3
        recorder.hand_trajectory_samples = []
        recorder.reset_hand_motion()

        def send(x, captured_at, sent_at, ready=True):
            payload = {
                "wristUv": [x, 0.5], "wristCapturedAt": captured_at,
                "sentAt": sent_at, "handVisible": True,
                "handMotionReady": ready, "handMotionToken": "first",
                "handMotionSegment": 1,
            }
            recorder.update_raw_wrist(payload)
            recorder.update_hand_motion(payload)
            recorder.append_hand_trajectory_sample(payload)

        send(0.4, 1000, 1002)
        send(0.43, 1030, 1032)
        send(0.9, 1060, 1062)  # tracking jump: raw moves, filtered holds
        send(0.91, 1090, 1092, ready=False)
        self.assertEqual(len(recorder.hand_trajectory_samples), 4)
        self.assertEqual(recorder.hand_trajectory_samples[2]["raw_wrist_uv"], [0.9, 0.5])
        np.testing.assert_allclose(
            recorder.hand_trajectory_samples[2]["filtered_wrist_uv"], [0.43, 0.5]
        )
        self.assertIsNone(recorder.hand_trajectory_samples[3]["filtered_wrist_uv"])
        with tempfile.TemporaryDirectory() as temporary:
            _save_hand_trajectory(Path(temporary), 0, recorder.hand_trajectory_samples)
            path = Path(temporary) / "hand_trajectories/episode-000000.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertEqual([row["raw_timestamp_ms"] for row in rows],
                         [1000, 1030, 1060, 1090])

    def test_raw_wrist_and_mapped_control_are_separate(self):
        recorder = object.__new__(MujocoPandaRecorder)
        recorder.update_raw_wrist({
            "wristUv": [0.3, 0.7], "wristCapturedAt": 1234.5,
            "handVisible": True,
        })
        self.assertTrue(recorder.raw_wrist_valid)
        np.testing.assert_allclose(recorder.raw_wrist_uv, [0.3, 0.7])
        self.assertEqual(recorder.raw_wrist_timestamp_ms, 1234.5)
        recorder.mapped_control_pending = np.zeros(3)
        recorder.mapped_control_pending_commands = 0
        seen = []
        recorder.move_hand = lambda *delta: seen.append(delta)
        recorder.apply_web_motion(0.01, -0.02, 0.0)
        recorder.apply_web_motion(0.0, 0.0, 0.005)
        self.assertEqual(seen, [(0.01, -0.02, 0.0), (0.0, 0.0, 0.005)])
        np.testing.assert_allclose(recorder.mapped_control_pending, [0.01, -0.02, 0.005])
        self.assertEqual(recorder.mapped_control_pending_commands, 2)
        recorder.update_raw_wrist({"wristUv": None, "handVisible": False})
        self.assertFalse(recorder.raw_wrist_valid)

    def test_reentry_reanchors_and_rejects_tracking_jump(self):
        recorder = object.__new__(MujocoPandaRecorder)
        recorder.control_paused = False
        recorder.control_token = "first"
        recorder.hand_visible = True
        recorder.reset_hand_motion()

        def send(x, y, when, token="first", segment=1, ready=True):
            recorder.update_hand_motion({
                "wristUv": [x, y], "sentAt": when,
                "handMotionToken": token, "handMotionSegment": segment,
                "handMotionReady": ready,
            })

        send(0.40, 0.50, 1000)
        self.assertEqual(recorder.hand_motion_pending_samples, 0)
        send(0.43, 0.48, 1030)
        np.testing.assert_allclose(recorder.hand_motion_pending, [0.03, -0.02])

        send(0.80, 0.80, 1060)
        self.assertEqual(recorder.hand_motion_pending_samples, 0)
        send(0.82, 0.80, 1090)
        np.testing.assert_allclose(recorder.hand_motion_pending, [0.02, 0.0])

        send(0.20, 0.20, 1120, ready=False)
        self.assertEqual(recorder.hand_motion_pending_samples, 0)
        recorder.control_token = "second"
        send(0.10, 0.10, 1150, token="second", segment=2)
        self.assertEqual(recorder.hand_motion_pending_samples, 0)
        send(0.12, 0.11, 1180, token="second", segment=2)
        np.testing.assert_allclose(recorder.hand_motion_pending, [0.02, 0.01])
        self.assertEqual(recorder.hand_motion_segment, 2)


if __name__ == "__main__":
    unittest.main()
