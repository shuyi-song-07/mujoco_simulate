"""M03 tests run real MuJoCo contacts; fault injection is explicitly labelled."""
import unittest
import numpy as np
from simulation.mujoco.dual_arm.config import load_config
from simulation.mujoco.dual_arm.action_contract import V2
from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
from simulation.mujoco.dual_arm.panda_arm_controller import matrix_to_rpy
from simulation.mujoco.dual_arm.rigid_control import rz


def controller(at_object=False, *, protocol="grasp-hold-release-v1"):
    config = load_config("config/collection.json")
    # These M03/M04/v1 fixtures retain their frozen interaction semantics;
    # gesture-v2 has a separate suite with explicit operation modes.
    config["contact_control"]["control_protocol"] = protocol
    # Historical suites exercise level-triggered v3; edge tests opt in explicitly.
    config.setdefault("gesture_control", {})["gripper_trigger_policy"] = "level_v1"
    config['gesture_control']['mode_confirm_ms'] = 200
    config['contact_control']['release_confirm_seconds'] = .7
    if protocol != "gesture-rate-gripper-v3":
        config["contact_control"]["release_stationary_target_tolerance_m"] = .0002
        config["contact_control"].pop("support_penetration_tolerance_m", None)
    if protocol == "gesture-grasp-release-v2":
        from simulation.mujoco.dual_arm.hand_measurement import GESTURE_PARAMETERS
        config["gesture_control"] = dict(GESTURE_PARAMETERS)
    if protocol == "grasp-hold-release-v1":
        config["contact_control"]["release_intent_aperture_normalized"] = .85
    config["scene"]["randomization"] = {"object_xy_range_m": [0, 0], "target_xy_range_m": [0, 0], "object_yaw_range_rad": 0}
    if at_object:
        config["scene"].update(left_home_ee=[0, -.22, .372], right_home_ee=[0, .22, .372])
    c = DualArmTaskController(config)
    c.set_episode_timeout_enabled(False)
    return c


def request(c, widths=None, positions=None):
    action = c.action_vector_m()
    components = []
    if widths is not None:
        action[[6, 13]] = widths
        components.append("gripper")
    if positions is not None:
        for o, position in zip((0, 7), positions):
            action[o:o+3] = position
        components.append("motion")
    return c.apply_action(action, schema_version=V2, components=components, source="scripted_test")


def settle(c, seconds):
    for _ in range(round(seconds/.02)):
        c._last_command_sim_time = c.data.time
        c.step(.02)


def move(c, delta, seconds=1.):
    start = [a.world_to_task(a.command_position) for a in c.arms]
    for f in np.linspace(0, 1, round(seconds/.02)):
        r = request(c, positions=[p+f*np.asarray(delta) for p in start])
        if not r["accepted"]:
            raise AssertionError(r)
        c.step(.02)
    settle(c, .3)


def release(c, widths=(.08, .08), intent=(True, True)):
    """Public explicit intent, with the same confirmation used by camera input."""
    action = c.action_vector_m(); action[[6, 13]] = widths
    result = c.apply_action(action, schema_version=V2, components=("gripper",), source="scripted_test")
    if not result["accepted"]:
        raise AssertionError(result)
    c.step(.02)
    for _ in range(110):
        result = c.apply_action(action, schema_version=V2, components=("gripper",), source="scripted_test",
                                release_intent=intent)
        if not result["accepted"]:
            raise AssertionError(result)
        c.step(.02)
        if all(not active or (c.contact_control.release_executed[side] and not c.contact_control.release_opening[side]
                              and c.contact_control.hold_width_m[side] is None)
               for side, active in zip(("left", "right"), intent)):
            break
    return result


class RecoverableContactTests(unittest.TestCase):
    def test_closed_grippers_push_with_real_contacts(self):
        cfg = load_config("config/collection.json")
        cfg["contact_control"]["control_protocol"] = "grasp-hold-release-v1"
        cfg["contact_control"]["release_intent_aperture_normalized"] = .85
        cfg["scene"].update(left_home_ee=[-.07, -.22, .375], right_home_ee=[-.07, .22, .375])
        cfg["scene"]["randomization"] = {"object_xy_range_m": [0, 0], "target_xy_range_m": [0, 0], "object_yaw_range_rad": 0}
        c = DualArmTaskController(cfg); c.set_episode_timeout_enabled(False)
        settle(c, .3); request(c, widths=[0, 0]); settle(c, 1.4)
        before = c.data.xpos[c.object_id].copy()
        move(c, [.07, 0, 0], 2.)
        self.assertGreater(c.data.xpos[c.object_id, 0]-before[0], .02)
        self.assertNotEqual(c.task_state, "FAIL")
        self.assertLess(abs(c.data.xpos[c.object_id, 2]-.372), .003)

    def test_empty_closure_and_single_arm_motion_are_recoverable(self):
        c = controller(); settle(c, .3)
        self.assertTrue(request(c, widths=[0, .06])["accepted"])
        settle(c, 1.8)
        self.assertEqual(c.task_state, "PREGRASP")
        right = c.right.command_position.copy()
        positions = [c.left.world_to_task(c.left.command_position)+[.015, 0, 0], c.right.world_to_task(right)]
        for _ in range(15):
            self.assertTrue(request(c, positions=positions)["accepted"]); c.step(.02)
        np.testing.assert_allclose(c.right.command_position, right, atol=1e-7)
        self.assertTrue(request(c, widths=[.08, .08])["accepted"])
        settle(c, 1.3)
        self.assertEqual(c.task_state, "PREGRASP")

    def test_partial_grasp_holds_only_grasping_arm_and_allows_release_retry(self):
        c = controller(True); settle(c, .3)
        request(c, widths=[0, .08]); settle(c, 1.4)
        self.assertEqual(c.task_state, "PARTIAL_CONTACT")
        self.assertEqual(c.contact_control.stable, {"left": True, "right": False})
        positions = [a.world_to_task(a.command_position) for a in c.arms]
        positions[0][2] += .01
        self.assertEqual(request(c, positions=positions)["reason"], "partial_grasp_holds_engaged_arm")
        positions[0][2] -= .01; positions[1][2] += .005
        self.assertTrue(request(c, positions=positions)["accepted"])
        settle(c, .4)
        self.assertTrue(release(c, intent=(True, False))["accepted"])
        settle(c, 1.4)
        self.assertEqual(c.task_state, "PREGRASP")
        self.assertFalse(c.contacts["left"])

    def test_dual_grasp_airborne_release_blocked_and_supported_release(self):
        c = controller(True); settle(c, .3)
        request(c, widths=[0, 0]); settle(c, 1.4)
        self.assertEqual(c.task_state, "DUAL_GRASPED")
        move(c, [0, 0, .08], 2.)
        self.assertGreater(c.data.xpos[c.object_id, 2], .43)
        before = [a.gripper_goal_width_m for a in c.arms]
        result = request(c, widths=[.04, 0])
        self.assertTrue(result["accepted"])
        self.assertTrue(6 in result["intervened_fields"] or 13 in result["intervened_fields"])
        self.assertEqual(before, [a.gripper_goal_width_m for a in c.arms])
        move(c, [0, 0, -.08], 2.)
        settle(c, .5)
        self.assertTrue(c.release_ready, c.get_status())
        self.assertTrue(release(c)["accepted"])
        settle(c, 1.4)
        self.assertEqual(c.task_state, "PREGRASP")  # Away from goal, retryable.
        self.assertIsNone(c.contact_control.reference)

    def test_airborne_sub_epsilon_requests_cannot_accumulate_release(self):
        c = controller(True); settle(c, .3)
        request(c, widths=[0, 0]); settle(c, 1.4)
        move(c, [0, 0, .08], 2.)
        epsilon = c.config["contact_control"]["release_width_epsilon_m"]
        accepted = 0
        for _ in range(250):
            widths = [a.gripper_command_width_m + .95*epsilon for a in c.arms]
            result = request(c, widths=widths)
            accepted += result["accepted"]
            if not result["accepted"]:
                self.assertEqual(result["reason"], "release_requires_stable_table_support")
            c.step(.02)
        self.assertEqual(accepted, 250)
        self.assertTrue(all(abs(a.gripper_command_width_m-c.contact_control.hold_width_m[a.side]) <= 1e-9 for a in c.arms))
        self.assertGreater(c.data.xpos[c.object_id, 2], .43)
        self.assertEqual(c.task_state, "DUAL_GRASPED")
        self.assertTrue(c.contacts["left"] and c.contacts["right"])
        # Support alone no longer unlocks a grasp; explicit intent permits retry.
        move(c, [0, 0, -.08], 2.); settle(c, .5)
        self.assertTrue(release(c)["accepted"])
        settle(c, 1.4)
        self.assertIsNone(c.contact_control.reference)

    def test_partial_grasp_rejects_rotation_without_changing_free_arm(self):
        c = controller(True); settle(c, .3)
        request(c, widths=[0, .08]); settle(c, 1.4)
        self.assertEqual(c.contact_control.stable, {"left": True, "right": False})
        before = c.action_vector_m().copy(); ctrl = c.data.ctrl.copy()
        action = before.copy()
        action[3:6] = matrix_to_rpy(c.left.task_rotation.T @ rz(.002) @ c.left.command_rotation)
        action[7] += .002
        result = c.apply_action(action, schema_version=V2, components=("motion",))
        self.assertEqual(result["reason"], "partial_grasp_holds_engaged_arm")
        np.testing.assert_array_equal(c.action_vector_m(), before)
        np.testing.assert_array_equal(c.data.ctrl, ctrl)
        # The unengaged arm may still change orientation independently.
        action = before.copy()
        action[10:13] = matrix_to_rpy(c.right.task_rotation.T @ rz(.002) @ c.right.command_rotation)
        self.assertTrue(c.apply_action(action, schema_version=V2, components=("motion",))["accepted"])
        np.testing.assert_array_equal(c.action_vector_m()[:7], before[:7])

    def test_opening_ramp_does_not_allow_simultaneous_lift(self):
        c = controller(True); settle(c, .3)
        request(c, widths=[0, 0]); settle(c, 1.4)
        self.assertTrue(c.contact_control.support_ready)
        target = c.action_vector_m(); target[[6, 13]] = .08
        c.apply_action(target, schema_version=V2); c.step(.02)
        for _ in range(80):
            c.apply_action(target, schema_version=V2, release_intent=(True, True)); c.step(.02)
            if all(c.contact_control.release_executed.values()):
                break
        self.assertTrue(all(c.contact_control.release_executed.values()))
        positions = [a.world_to_task(a.command_position)+[0, 0, .003] for a in c.arms]
        action = c.action_vector_m()
        for offset, position in zip((0, 7), positions):
            action[offset:offset+3] = position
        result = c.apply_action(action, schema_version=V2, release_intent=(True, True), source="scripted_test")
        self.assertEqual(result["reason"], "release_opening_in_progress")
        self.assertTrue(c.contact_control.support_ready)
        held = c.action_vector_m()[[6, 13]].copy()
        settle(c, .3)
        np.testing.assert_allclose(c.action_vector_m()[[6, 13]], held)
        release(c); settle(c, .3)
        self.assertFalse(c.contacts["left"] or c.contacts["right"])

    def test_pause_holds_aperture_and_failure_cannot_be_cleared_by_action(self):
        c = controller(); request(c, widths=[0, .04]); settle(c, .2)
        c.pause("hand_lost")
        widths = c.action_vector_m()[[6, 13]].copy(); settle(c, .4)
        np.testing.assert_array_equal(c.action_vector_m()[[6, 13]], widths)
        c.pause("injected_collision", fail=True)
        self.assertEqual(request(c, widths=[.08, .08])["reason"], "episode_ended_reset_required")
        self.assertEqual(c.task_state, "FAIL")

    def test_fault_injection_airborne_contact_loss_is_latched(self):
        c = controller(True); settle(c, .3); request(c, widths=[0, 0]); settle(c, 1.4)
        move(c, [0, 0, .06], 2.)
        original = c._contact_state
        def lost():
            s = original(); s["left"] = False; s["left_fingers"] = [False, False]; return s
        c._contact_state = lost  # Classified synthetic sensor fault, not physics success.
        c.step(.02)
        self.assertTrue(c.paused)
        self.assertEqual(c.pause_reason, "contact_loss")
        settle(c, .4)
        self.assertEqual(c.task_state, "FAIL")
        c._contact_state = original; settle(c, .4)
        self.assertEqual(c.task_state, "FAIL")


if __name__ == "__main__":
    unittest.main()
