"""V3 raw hand evidence does not require pinch calibration or still wrists."""
import copy
import unittest

from rate_fixture import packet, PROTOCOL
from simulation.mujoco.dual_arm.hand_measurement import inspect_rate_input, HandInputGuard


class RateHandGuardTests(unittest.TestCase):
    def check(self, guard, p, age=10):
        return guard.check(p, session="session", epoch=0, now_ms=p["sentAt"]+age, control_protocol=PROTOCOL)

    def test_moving_fist_and_palm_no_aperture_or_stillness_gate(self):
        g = HandInputGuard()
        for i in range(12):
            p = packet(i, modes=("Z_CLOSE", "Z_OPEN"), dy=.01*i)
            p["delta_task_m"] = {"left":[0,0,.001], "right":[0,0,-.001]}
            reason = self.check(g, p)
            self.assertEqual(reason, "tracking_recovery" if i < 4 else None)

    def test_raw_category_scores_and_intents_checked(self):
        for mode, threshold in (("Z_OPEN", .55), ("Z_CLOSE", .8), ("XY_HOLD", .75)):
            p = packet(modes=(mode, mode))
            h = p["telemetry"]["recognition_result"]["measurement"]["hands"][0]
            h["gesture_score"] = threshold
            self.assertIsNone(inspect_rate_input(p))
            h["gesture_score"] = threshold-.001
            self.assertEqual(inspect_rate_input(p), "interaction_gesture_mismatch")
            h["gesture_score"] = 1.
            h["gesture"] = "Victory"
            self.assertEqual(inspect_rate_input(p), "interaction_gesture_mismatch")
        p = packet(modes=("Z_CLOSE", "Z_CLOSE")); p["release_intent"] = [True, True]
        self.assertEqual(inspect_rate_input(p), "release_intent_mode_mismatch")

    def test_wrong_plane_and_hold_motion_rejected(self):
        p = packet(modes=("XY_HOLD", "Z_OPEN"))
        self.assertEqual(inspect_rate_input(p), "interaction_plane_mismatch")
        p = packet(); p["delta_task_m"]["left"][2] = .001
        self.assertEqual(inspect_rate_input(p), "interaction_motion_forbidden")

    def test_identity_stale_missing_nan_edge_binding_and_old_protocol(self):
        for fault in ("identity", "stale", "missing", "nan", "edge", "binding", "old"):
            g = HandInputGuard()
            for i in range(5): self.check(g, packet(i))
            p = packet(5, modes=("Z_CLOSE", "Z_OPEN"))
            m = p["telemetry"]["recognition_result"]["measurement"]
            if fault == "identity": m["hands"][0]["identity_score"] = .79
            if fault == "missing": m["hands"].pop()
            if fault == "nan": m["hands"][0]["image_landmarks_21"][0][0] = float("nan")
            if fault == "edge": m["hands"][0]["image_landmarks_21"][0][0] = .02
            if fault == "binding": p["telemetry"]["calibration"]["palm_binding_guard"] = True
            if fault == "old": p["control_protocol"] = "gesture-grasp-release-v2"
            self.assertIsNotNone(self.check(g, p, age=200 if fault == "stale" else 10), fault)

    def test_duplicate_and_ordered_recovery(self):
        g = HandInputGuard()
        for i in range(5): self.check(g, packet(i))
        self.assertEqual(self.check(g, packet(4)), "duplicate_measurement")
        for i in range(5, 10):
            self.assertEqual(self.check(g, packet(i)), "tracking_recovery" if i < 9 else None)


if __name__ == "__main__":
    unittest.main()
