"""Direct policy execution: real physics plus explicitly marked guard faults."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np

from rollout.executor import PolicyExecutor
from simulation.mujoco.dual_arm.config import load_config
from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
from simulation.mujoco.dual_arm.panda_arm_controller import matrix_to_rpy
from simulation.mujoco.dual_arm.rigid_control import rz
from simulation.mujoco.dual_arm.task_scenes import scenario_config


def make_controller(at_handles=False):
    cfg = scenario_config(load_config("config/collection.json"), "S1")
    cfg["scene"]["randomization"] = {"object_xy_range_m": [0., 0.], "target_xy_range_m": [0., 0.], "object_yaw_range_rad": 0.}
    if at_handles:
        cfg["scene"].update(left_home_ee=[0., -.22, .372], right_home_ee=[0., .22, .372])
    c = DualArmTaskController(cfg)
    c.set_episode_timeout_enabled(False)
    return c, PolicyExecutor(c)


def run_target(c, executor, action, seconds, *, allow_done=False):
    results = []
    for _ in range(round(seconds*60)):
        if allow_done and str(c.task_state) == "DONE":
            break
        result = executor.apply(action)
        if not result["accepted"]:
            raise AssertionError(result)
        results.append(result)
        c.step(1/60)
        if hasattr(c, "_policy_test_monitor"):
            c._policy_test_monitor_result = c._policy_test_monitor.update(c)
        if str(c.task_state) == "FAIL":
            raise AssertionError(c.get_status())
    return results


def real_grasp(*, staggered=True):
    c, executor = make_controller(True)
    action = c.action_vector_m().astype(float)
    run_target(c, executor, action, .3)
    action[6] = 0.
    action[13] = .08 if staggered else 0.
    run_target(c, executor, action, 1.2)
    if staggered:
        assert c.contact_control.stable["left"] and not c.contact_control.stable["right"]
    action[13] = 0.
    run_target(c, executor, action, 1.2)
    assert all(c.contact_control.stable.values()), c.get_status()
    return c, executor


class PolicyExecutorTests(unittest.TestCase):
    def test_continuous_left_right_width_and_physical_speed(self):
        c, executor = make_controller()
        action = c.action_vector_m().astype(float)
        action[6], action[13] = .04, .065
        run_target(c, executor, action, .1)
        self.assertAlmostEqual(c.left.gripper_command_width_m, .072, places=6)
        self.assertAlmostEqual(c.right.gripper_command_width_m, .072, places=6)
        run_target(c, executor, action, .6)
        self.assertAlmostEqual(c.left.gripper_command_width_m, .04, places=6)
        self.assertAlmostEqual(c.right.gripper_command_width_m, .065, places=6)
        self.assertFalse(c.contact_control.ever_dual)

    def test_nan_and_shape_never_partially_write_targets(self):
        c, executor = make_controller()
        before, ctrl = c.action_vector_m().copy(), c.data.ctrl.copy()
        bad = before.copy(); bad[8] = np.nan
        for action in (bad, [0.]*13, "bad"):
            result = executor.apply(action)
            self.assertFalse(result["accepted"])
            np.testing.assert_array_equal(c.action_vector_m(), before)
            np.testing.assert_array_equal(c.data.ctrl, ctrl)
            json.dumps(result, allow_nan=False)

    def test_gross_orientation_rejected_small_noise_projected(self):
        c, executor = make_controller()
        action = c.action_vector_m().astype(float)
        action[3] += .6
        self.assertEqual(executor.apply(action)["reason"], "policy_orientation_residual_exceeded")
        action[3] -= .55
        self.assertTrue(executor.apply(action)["accepted"])

    def test_equivalent_rpy_turns_are_not_orientation_interventions(self):
        for turns in (-1, 1):
            c, executor = make_controller()
            action = c.action_vector_m().astype(float)
            for arm, offset in zip(c.arms, (0, 7)):
                action[offset+3:offset+6] = matrix_to_rpy(arm.task_rotation.T @ arm.command_rotation)
                action[offset+3:offset+6] += turns*2*np.pi
            result = executor.apply(action)
            self.assertTrue(result["accepted"])
            self.assertFalse(set(result["intervened_fields"]) & {3, 4, 5, 10, 11, 12}, result)
            np.testing.assert_array_equal(result["requested_target"], action)
            self.assertLess(max(abs(result["target_residual_wrapped"][i]) for i in (3, 4, 5, 10, 11, 12)), 1e-7)

    def test_workspace_projection_and_no_object_pose_write(self):
        c, executor = make_controller()
        action = c.action_vector_m().astype(float)
        action[0] = 9.
        before = c.data.qpos.copy()
        result = executor.apply(action)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["intervened"])
        self.assertLessEqual(c.left.world_to_task(c.left.command_position)[0], c.left.workspace[0, 1])
        np.testing.assert_array_equal(c.data.qpos, before)

    def test_invalid_second_arm_ik_is_atomic_including_gripper(self):
        c, executor = make_controller()
        before = c.action_vector_m().copy(); ctrl = c.data.ctrl.copy()
        goals = [a.gripper_goal_width_m for a in c.arms]
        action = before.astype(float); action[0] += .002; action[6] = 0.
        original = c.rigid_control.feasible
        def rejected_right(arm, position, rotation):
            return "ik_unreachable" if arm.side == "right" else original(arm, position, rotation)
        with patch.object(c.rigid_control, "feasible", side_effect=rejected_right):
            result = executor.apply(action)
        self.assertFalse(result["accepted"])
        np.testing.assert_array_equal(c.action_vector_m(), before)
        np.testing.assert_array_equal(c.data.ctrl, ctrl)
        self.assertEqual([a.gripper_goal_width_m for a in c.arms], goals)

    def test_joint_limit_and_collision_rejections_are_atomic(self):
        for name, reason in (("feasible", "joint_limit"), ("collision_reason", "predicted_robot_robot_collision")):
            c, executor = make_controller()
            before = c.action_vector_m().copy(); ctrl = c.data.ctrl.copy()
            action = before.astype(float); action[7] += .001; action[13] = 0.
            with patch.object(c.rigid_control, name, return_value=reason):
                self.assertEqual(executor.apply(action)["reason"], reason)
            np.testing.assert_array_equal(c.action_vector_m(), before)
            np.testing.assert_array_equal(c.data.ctrl, ctrl)

    def test_partial_contact_only_holds_engaged_arm(self):
        c, executor = make_controller(True)
        action = c.action_vector_m().astype(float)
        action[6] = 0.
        run_target(c, executor, action, 1.2)
        self.assertTrue(c.contact_control.stable["left"])
        self.assertFalse(c.contact_control.stable["right"])
        before = c.action_vector_m().copy()
        action[0] += .002; action[7] += .002
        result = executor.apply(action)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["projection"]["partial_hold_sides"], ["left"])
        np.testing.assert_allclose(c.action_vector_m()[:6], before[:6], atol=1e-7)
        self.assertGreater(c.action_vector_m()[7], before[7])

    def test_real_staggered_grasp_common_lift_no_repeat_drift(self):
        c, executor = real_grasp()
        initial_z = float(c.data.xpos[c.object_id, 2])
        # Permit the shared, rate-limited contact target handoff to complete.
        action = c.action_vector_m().astype(float)
        run_target(c, executor, action, .3)
        action = c.action_vector_m().astype(float)
        action[[2, 9]] += .06
        results = run_target(c, executor, action, 2.)
        self.assertTrue(c.contact_control.ever_dual)
        self.assertGreater(c.data.xpos[c.object_id, 2], initial_z+.045)
        self.assertTrue(all(r["projection"]["coupled"] for r in results))
        before = [a.command_position.copy() for a in c.arms]
        run_target(c, executor, action, .5)
        for arm, position in zip(c.arms, before):
            np.testing.assert_allclose(arm.command_position, position, atol=.001)

    def test_independent_airborne_release_has_no_support_gate(self):
        c, executor = real_grasp()
        action = c.action_vector_m().astype(float)
        action[[2, 9]] += .04
        run_target(c, executor, action, 1.5)
        self.assertFalse(c.contact_control.support_ready)
        held_right = c.right.gripper_command_width_m
        action = c.action_vector_m().astype(float)
        action[6] = .08
        result = executor.apply(action)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["projection"]["released_sides"], ["left"])
        self.assertAlmostEqual(c.right.gripper_goal_width_m, held_right)
        self.assertIsNone(c.contact_control.reference)
        c.step(1/60)
        self.assertGreater(c.left.gripper_command_width_m, result["accepted_target"][6])

    def test_small_width_noise_holds_preload(self):
        c, executor = real_grasp()
        widths = [a.gripper_command_width_m for a in c.arms]
        action = c.action_vector_m().astype(float)
        action[[6, 13]] += .001
        result = executor.apply(action)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["projection"]["released_sides"], [])
        np.testing.assert_allclose([a.gripper_goal_width_m for a in c.arms], widths)

    def test_real_contact_transfer_translate_turn_place_and_success(self):
        # Start at the ordinary 0.55 m home, physically descend to the handles;
        # neither the object nor arms are teleported after reset.
        c, executor = make_controller()
        from rollout.protocol import SuccessMonitor
        c._policy_test_monitor = SuccessMonitor(c.config, c.initial_state)
        approach = c.action_vector_m().astype(float)
        approach[[1, 8]] = [-.22, .22]
        run_target(c, executor, approach, .5)
        approach[[2, 9]] = .372
        run_target(c, executor, approach, 3.)
        approach[6] = 0.
        run_target(c, executor, approach, 1.2)
        self.assertTrue(c.contact_control.stable["left"])
        self.assertFalse(c.contact_control.stable["right"])
        approach[13] = 0.
        run_target(c, executor, approach, 1.2)
        self.assertTrue(all(c.contact_control.stable.values()))
        refs = c.contact_control.reference
        def target(center, yaw):
            action = c.action_vector_m().astype(float)
            rotation = rz(yaw)
            for arm, offset, reference in zip(c.arms, (0, 7), refs):
                action[offset:offset+3] = arm.world_to_task(np.asarray(center)+rotation @ reference[0])
                action[offset+3:offset+6] = matrix_to_rpy(arm.task_rotation.T @ rotation @ reference[1])
            return action
        initial = c.data.xpos[c.object_id].copy()
        lifted = initial+np.array([0., 0., .08])
        run_target(c, executor, target(lifted, 0.), 2.)
        self.assertGreater(c.data.xpos[c.object_id, 2], initial[2]+.06)
        lifted[:2] = c.target_position[:2]
        run_target(c, executor, target(lifted, 0.), 2.)
        run_target(c, executor, target(lifted, c.target_yaw), 1.5)
        run_target(c, executor, target(c.target_position, c.target_yaw), 2.)
        action = c.action_vector_m().astype(float)
        action[[6, 13]] = .08
        run_target(c, executor, action, 1.5, allow_done=True)
        if str(c.task_state) != "DONE":
            action[[2, 9]] += .08
            run_target(c, executor, action, 2., allow_done=True)
        self.assertEqual(str(c.task_state), "DONE", c.get_status())
        self.assertTrue(c.contact_control.ever_dual)
        self.assertTrue(c.contact_control.released)
        self.assertTrue(c.contact_control.support_ready)
        self.assertEqual(c.metrics.collision_count, 0)
        # The independent scorer also requires measured (not merely commanded)
        # open jaws. No new policy action may be sent after legacy DONE.
        for _ in range(60):
            if c._policy_test_monitor_result["task_success"]:
                break
            c.step(1/60)
            c._policy_test_monitor_result = c._policy_test_monitor.update(c)
        self.assertTrue(c._policy_test_monitor_result["task_success"], c._policy_test_monitor_result)

    def test_reset_invalidates_adapter_and_terminal_state_rejects(self):
        c, executor = make_controller()
        c.pause("synthetic_failure", fail=True)
        self.assertFalse(executor.apply(c.action_vector_m())["accepted"])
        c.reset(2)
        with self.assertRaisesRegex(RuntimeError, "new_policy_executor"):
            executor.apply(c.action_vector_m())

    def test_non_60hz_requires_explicit_contract_change(self):
        c, executor = make_controller()
        with self.assertRaisesRegex(ValueError, "60_hz"):
            executor.apply(c.action_vector_m(), dt=1/30)


if __name__ == "__main__":
    unittest.main()
