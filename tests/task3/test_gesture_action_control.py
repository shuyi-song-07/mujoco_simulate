"""V3 categorical gripper direction; real MuJoCo actuator/contact regression."""
import unittest
import numpy as np

from simulation.mujoco.dual_arm.action_contract import V2
from test_recoverable_contact import controller, settle

PROTOCOL = "gesture-rate-gripper-v3"


def send(c, modes=("HOLD", "HOLD"), *, delta=None, widths=None, protocol=PROTOCOL):
    action = c.action_vector_m()
    if widths is not None:
        action[[6, 13]] = widths
    if delta is not None:
        for offset, change in zip((0, 7), delta):
            action[offset:offset+3] += change
    c._rate_serial = getattr(c, "_rate_serial", 0)+1
    return c.apply_action(action, schema_version=V2, source="scripted_test",
        event_id=f"rate-{c._rate_serial}", mode="XY" if "XY_HOLD" in modes else "Z",
        interaction_modes=list(modes), control_protocol=protocol,
        release_intent=[m == "Z_OPEN" for m in modes])


def tick(c, modes=("HOLD", "HOLD"), **kw):
    result = send(c, modes, **kw)
    c.step(.02)
    return result


def pump(c, modes, steps=50, **kw):
    for _ in range(steps):
        result = tick(c, modes, **kw)
        if not result["accepted"]:
            raise AssertionError(result)
    return result


def grasp():
    c = controller(True, protocol=PROTOCOL)
    settle(c, .3)
    pump(c, ("Z_CLOSE", "Z_CLOSE"), 75)
    assert all(c.contact_control.stable.values()), c.get_status()
    return c


class RateGestureControlTests(unittest.TestCase):
    def test_rate_close_and_open_without_aperture_takeover(self):
        c = controller(protocol=PROTOCOL)
        start = np.array([a.gripper_command_width_m for a in c.arms])
        pump(c, ("Z_CLOSE", "HOLD"), 5, widths=(.08, 0.))
        now = np.array([a.gripper_command_width_m for a in c.arms])
        self.assertAlmostEqual(start[0]-now[0], .008, places=6)
        self.assertAlmostEqual(now[1], start[1])
        pump(c, ("Z_OPEN", "HOLD"), 5, widths=(0., 0.))
        self.assertAlmostEqual(c.left.gripper_command_width_m, start[0], places=6)
        self.assertFalse(any(c.contact_control.aperture_direct_enabled.values()))

    def test_xy_holds_issued_width_and_not_residual_goal(self):
        c = controller(protocol=PROTOCOL)
        pump(c, ("Z_CLOSE", "Z_CLOSE"), 5)
        widths = [a.gripper_command_width_m for a in c.arms]
        pump(c, ("XY_HOLD", "XY_HOLD"), 20, widths=(0., 0.))
        np.testing.assert_allclose([a.gripper_command_width_m for a in c.arms], widths)
        self.assertTrue(all(v is None for v in c.contact_control.operation_hold_width_m.values()))

    def test_z_motion_and_close_are_simultaneous(self):
        c = controller(protocol=PROTOCOL)
        before = c.action_vector_m().copy()
        pump(c, ("Z_CLOSE", "Z_OPEN"), 5, delta=([0, 0, .0005], [0, 0, .0005]))
        self.assertGreater(c.action_vector_m()[2], before[2]+.001)
        self.assertLess(c.left.gripper_command_width_m, before[6]-.005)

    def test_input_gap_stops_free_gripper_and_rejected_old_protocol_stops_it(self):
        c = controller(protocol=PROTOCOL)
        tick(c, ("Z_CLOSE", "Z_CLOSE"))
        settle(c, .4)
        frozen = [a.gripper_command_width_m for a in c.arms]
        settle(c, .4)
        np.testing.assert_allclose([a.gripper_command_width_m for a in c.arms], frozen)
        tick(c, ("Z_CLOSE", "Z_CLOSE"))
        result = send(c, ("Z_CLOSE", "Z_CLOSE"), protocol="gesture-grasp-release-v2")
        self.assertFalse(result["accepted"])
        frozen = [a.gripper_command_width_m for a in c.arms]
        settle(c, .1)
        np.testing.assert_allclose([a.gripper_command_width_m for a in c.arms], frozen)

    def test_dual_release_requires_both_palms_and_confirmed_support(self):
        c = grasp()
        widths = [a.gripper_command_width_m for a in c.arms]
        pump(c, ("Z_OPEN", "Z_CLOSE"), 45)
        np.testing.assert_allclose([a.gripper_command_width_m for a in c.arms], widths)
        self.assertFalse(any(c.contact_control.release_executed.values()))
        pump(c, ("Z_OPEN", "Z_OPEN"), 10)
        self.assertFalse(any(c.contact_control.release_executed.values()))
        pump(c, ("Z_OPEN", "Z_OPEN"), 90)
        self.assertTrue(all(c.contact_control.release_executed.values()))
        self.assertTrue(all(a.gripper_command_width_m >= .079 for a in c.arms))
        self.assertFalse(any(c.contact_control.release_reentry_pending.values()))

    def test_airborne_palm_holds_grip_but_allows_lowering(self):
        c = grasp()
        pump(c, ("Z_CLOSE", "Z_CLOSE"), 80, delta=([0, 0, .001], [0, 0, .001]))
        pump(c, ("Z_CLOSE", "Z_CLOSE"), 15)
        self.assertFalse(c.contact_control.support_ready)
        before = c.action_vector_m().copy()
        widths = [a.gripper_command_width_m for a in c.arms]
        pump(c, ("Z_OPEN", "Z_OPEN"), 10, delta=([0, 0, -.001], [0, 0, -.001]))
        self.assertLess(c.action_vector_m()[2], before[2]-.005)
        np.testing.assert_allclose([a.gripper_command_width_m for a in c.arms], widths)
        self.assertFalse(any(c.contact_control.release_executed.values()))
        self.assertEqual(c.contact_control.release_reason_by_side["left"], "release_requires_stable_table_support")

    def test_release_interruption_holds_and_requires_new_full_confirmation(self):
        c = grasp()
        for _ in range(60):
            tick(c, ("Z_OPEN", "Z_OPEN"))
            if all(c.contact_control.release_opening.values()):
                break
        self.assertTrue(all(c.contact_control.release_opening.values()))
        held = [a.gripper_command_width_m for a in c.arms]
        tick(c, ("XY_HOLD", "XY_HOLD"))
        pump(c, ("XY_HOLD", "XY_HOLD"), 15)
        np.testing.assert_allclose([a.gripper_command_width_m for a in c.arms], held)
        self.assertFalse(any(c.contact_control.release_opening.values()))
        pump(c, ("Z_OPEN", "Z_OPEN"), 15)
        self.assertFalse(any(c.contact_control.release_opening.values()))
        np.testing.assert_allclose([a.gripper_command_width_m for a in c.arms], held)
        pump(c, ("Z_OPEN", "Z_OPEN"), 75)
        self.assertTrue(all(a.gripper_command_width_m >= .079 for a in c.arms))

    def test_rejected_pose_nan_and_mixed_modes_cancel_pending_rate(self):
        for fault in ("pose", "nan", "mixed", "missing_protocol"):
            c = controller(protocol=PROTOCOL)
            pump(c, ("Z_CLOSE", "Z_CLOSE"), 5)
            before = c.action_vector_m().copy()
            widths = [a.gripper_command_width_m for a in c.arms]
            ids = dict(c.contact_control.interaction_mode_event_id)
            if fault == "pose":
                action = c.action_vector_m(); action[3] += .5
                result = c.apply_action(action, schema_version=V2, source="scripted_test", event_id="bad-roll",
                    mode="XY", interaction_modes=["XY_HOLD", "XY_HOLD"], control_protocol=PROTOCOL)
            elif fault == "nan":
                result = send(c, ("Z_CLOSE", "Z_CLOSE"), widths=(float("nan"), .08))
            elif fault == "mixed":
                result = send(c, ("XY_HOLD", "Z_CLOSE"))
            else:
                result = send(c, ("Z_CLOSE", "Z_CLOSE"), protocol=None)
            self.assertFalse(result["accepted"], (fault, result))
            self.assertEqual(c.contact_control.interaction_mode_event_id, ids)
            settle(c, .1)
            np.testing.assert_allclose([a.gripper_command_width_m for a in c.arms], widths)
            np.testing.assert_allclose(c.action_vector_m()[[0,1,2,7,8,9]], before[[0,1,2,7,8,9]])

    def test_unsupported_width_fields_cannot_release_or_start_rate(self):
        c = controller(protocol=PROTOCOL)
        pump(c, ("Z_CLOSE", "Z_CLOSE"), 15)
        widths = [a.gripper_command_width_m for a in c.arms]
        pump(c, ("HOLD", "HOLD"), 15, widths=(.08, .08))
        np.testing.assert_allclose([a.gripper_command_width_m for a in c.arms], widths)
        self.assertFalse(any(c.contact_control.gripper_rate_enabled.values()))

    def test_single_grasp_releases_only_engaged_group_with_support(self):
        c = controller(True, protocol=PROTOCOL)
        settle(c, .3)
        pump(c, ("Z_CLOSE", "HOLD"), 75)
        self.assertTrue(c.contact_control.stable["left"])
        self.assertFalse(c.contact_control.stable["right"])
        pump(c, ("Z_OPEN", "HOLD"), 100)
        self.assertTrue(c.contact_control.release_executed["left"])
        self.assertFalse(c.contact_control.release_executed["right"])
        self.assertGreater(c.left.gripper_command_width_m, .079)

    def test_manual_pause_stops_free_closing_and_recovery_does_not_resume_old_goal(self):
        c = controller(protocol=PROTOCOL)
        pump(c, ("Z_CLOSE", "Z_CLOSE"), 10)
        c.pause("operator_pause")
        widths = [a.gripper_command_width_m for a in c.arms]
        settle(c, .3)
        np.testing.assert_allclose([a.gripper_command_width_m for a in c.arms], widths)
        pump(c, ("HOLD", "HOLD"), 15)
        np.testing.assert_allclose([a.gripper_command_width_m for a in c.arms], widths)

    def test_release_support_loss_before_next_servo_step_holds_current_command(self):
        c = grasp()
        for _ in range(60):
            tick(c, ("Z_OPEN", "Z_OPEN"))
            if all(c.contact_control.release_opening.values()):
                break
        self.assertTrue(all(c.contact_control.release_opening.values()))
        widths = [a.gripper_command_width_m for a in c.arms]
        # Explicit fault injection at physical support seam, not a camera label.
        c.contact_control.support = lambda: setattr(c.contact_control, "support_ready", False)
        c.step(.02)
        self.assertFalse(any(c.contact_control.release_opening.values()))
        np.testing.assert_allclose([a.gripper_command_width_m for a in c.arms], widths)
        self.assertEqual(c.contact_control.release_reason_by_side["left"], "release_requires_stable_table_support")

    def test_failed_ik_cannot_install_a_new_mode_or_release_authorization(self):
        from unittest.mock import patch
        c = controller(protocol=PROTOCOL)
        pump(c, ("Z_CLOSE", "Z_CLOSE"), 10)
        before = c.action_vector_m().copy()
        ids = dict(c.contact_control.interaction_mode_event_id)
        with patch.object(c.rigid_control, "feasible", return_value="ik_unreachable"):
            result = send(c, ("Z_OPEN", "Z_OPEN"), delta=([0, 0, .001], [0, 0, .001]))
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "ik_unreachable")
        self.assertEqual(c.contact_control.interaction_mode_event_id, ids)
        self.assertFalse(any(c.contact_control.release_executed.values()))
        self.assertFalse(any(c.contact_control.gripper_rate_enabled.values()))
        settle(c, .1)
        np.testing.assert_allclose(c.action_vector_m(), before, atol=1e-7)

    def test_supported_palm_opening_can_reverse_to_fist_after_contact_clears(self):
        from unittest.mock import patch
        c = grasp()
        for _ in range(100):
            tick(c, ("Z_OPEN", "Z_OPEN"))
            if (all(c.contact_control.release_opening.values()) and
                    not any(any(c.contacts[f"{s}_fingers"]) for s in ("left", "right"))):
                break
        self.assertTrue(all(c.contact_control.release_opening.values()))
        self.assertTrue(c.contact_control.support_ready)
        width = c.left.gripper_command_width_m
        self.assertLess(width, .075)
        with patch.object(c.rigid_control, "feasible", return_value="ik_unreachable"):
            result = send(c, ("Z_CLOSE", "Z_CLOSE"))
        self.assertFalse(result["accepted"])
        self.assertTrue(all(c.contact_control.rate_interrupted_release_hold.values()))
        np.testing.assert_allclose(list(c.contact_control.hold_width_m.values()), [width, width])
        pump(c, ("Z_CLOSE", "Z_CLOSE"), 50)
        self.assertLess(c.left.gripper_command_width_m, width-.002)
        self.assertTrue(all(c.contact_control.stable.values()))
        self.assertFalse(any(c.contact_control.release_executed.values()))


if __name__ == "__main__":
    unittest.main()
