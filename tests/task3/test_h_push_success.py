"""Verify the H-frame scorer through the real controller and MuJoCo state."""
import math
import unittest

import mujoco
import numpy as np

from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
from simulation.mujoco.dual_arm.task_states import TaskState
from tasks.registry import get, hook


class HFrameSuccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.controller = DualArmTaskController(get("h_frame_push_align").config())

    def setUp(self):
        self.c = self.controller
        self.c.reset(42)
        self.monitor = hook(self.c.config, "rollout_monitor")(self.c.config, self.c.initial_state)

    def pose_fixture(self, offset=(0.0, 0.0, 0.0002), yaw=None):
        # These are scorer fixtures, separate from the contact-push validation.
        c = self.c
        yaw = c.target_yaw if yaw is None else yaw
        c.data.qpos[c.object_qpos_adr:c.object_qpos_adr + 7] = np.r_[
            c.target_position + np.asarray(offset), math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)
        ]
        c.data.qvel[c.object_dof_adr:c.object_dof_adr + 6] = 0.0
        mujoco.mj_forward(c.model, c.data)

    def ticks(self, count):
        for _ in range(count):
            self.c.step(1 / 60)
            result = self.monitor.update(self.c)
        return result

    def test_initial_pose_is_not_successful(self):
        result = self.ticks(40)
        self.assertFalse(result["task_success"])
        self.assertFalse(result["final_conditions"]["position_xy"])
        self.assertTrue(result["final_conditions"]["table_support"])

    def test_real_supported_pose_succeeds_after_half_second_without_grasp(self):
        self.pose_fixture()
        self.assertFalse(self.ticks(29)["task_success"])
        self.assertNotEqual(self.c.task_state, TaskState.DONE)
        result = self.ticks(11)
        self.assertTrue(result["task_success"])
        self.assertGreaterEqual(result["alignment_stable_seconds"], 0.5 - 1e-9)
        self.assertEqual(self.c.task_state, TaskState.DONE)
        self.assertFalse(any(self.c.contacts["left_fingers"] + self.c.contacts["right_fingers"]))

    def test_position_without_correct_yaw_is_not_successful(self):
        self.pose_fixture(yaw=self.c.target_yaw + math.radians(20))
        result = self.ticks(40)
        self.assertFalse(result["task_success"])
        self.assertTrue(result["final_conditions"]["position_xy"])
        self.assertFalse(result["final_conditions"]["yaw"])

    def test_yaw_without_correct_position_is_not_successful(self):
        self.pose_fixture(offset=(0.04, 0.0, 0.0002))
        result = self.ticks(40)
        self.assertFalse(result["task_success"])
        self.assertTrue(result["final_conditions"]["yaw"])
        self.assertFalse(result["final_conditions"]["position_xy"])

    def test_wrapped_yaw_matches_across_pi_boundary(self):
        self.c.target_yaw = math.pi - math.radians(2)
        self.pose_fixture(yaw=-math.pi + math.radians(2))
        result = self.ticks(40)
        self.assertTrue(result["task_success"])
        self.assertAlmostEqual(result["target_error"]["orientation_rad"], math.radians(4), places=4)

    def test_alignment_hold_restarts_after_a_bad_pose(self):
        self.pose_fixture()
        self.assertGreater(self.ticks(20)["alignment_stable_seconds"], 0.2)
        self.pose_fixture(offset=(0.04, 0.0, 0.0002))
        self.assertEqual(self.ticks(1)["alignment_stable_seconds"], 0.0)
        self.pose_fixture()
        self.assertFalse(self.ticks(20)["task_success"])
        self.assertTrue(self.ticks(20)["task_success"])

    def test_aligned_pose_with_overhang_cannot_succeed(self):
        self.c.target_position[0] = 0.35
        self.pose_fixture()
        result = self.ticks(40)
        self.assertFalse(result["task_success"])
        self.assertFalse(result["final_conditions"]["inside_table"])

    def test_dropped_object_uses_shared_failure_and_cannot_succeed(self):
        self.pose_fixture(offset=(0.6, 0.0, -0.12))
        result = self.ticks(1)
        self.assertFalse(result["task_success"])
        self.assertTrue(result["terminal"])
        self.assertEqual(result["reason"], "object_drop")
        self.assertEqual(self.c.task_state, TaskState.FAIL)
