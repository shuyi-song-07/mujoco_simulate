"""A contact-deflected push arm must be able to retreat without raising limits."""
import unittest
from unittest.mock import patch

import numpy as np
import mujoco

from tasks.registry import get
from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
from tests.task3.rate_fixture import send


class PushRetreatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.controllers = {task: DualArmTaskController(get(task).config()) for task in
                           ('h_frame_push_align', 'push_r', 'tray_to_platform')}

    def blocked(self, task='h_frame_push_align', both=False):
        c = self.controllers[task]
        c.reset(0)
        # Reproduce the measured failure condition without modifying physics:
        # a previously accepted target is now >40 mm ahead of the actual arm.
        for arm in c.arms if both else (c.right,):
            arm.command_position = arm.get_actual_pose()[0] + [.06, 0., 0.]
        return c

    def test_retreat_can_reduce_existing_error_in_multiple_steps(self):
        for task in ('h_frame_push_align', 'push_r'):
            c = self.blocked(task)
            before = c.data.qpos.copy()
            for _ in range(6):
                old = np.linalg.norm(c.right.command_position-c.right.get_actual_pose()[0])
                result = send(c, ('HOLD', 'XY_HOLD'), delta=([0, 0, 0], [-.004, 0, 0]))
                self.assertTrue(result['accepted'], result['reason'])
                self.assertLess(np.linalg.norm(c.right.command_position-c.right.get_actual_pose()[0]), old)
            self.assertLess(np.linalg.norm(c.right.command_position-c.right.get_actual_pose()[0]), .04)
            np.testing.assert_array_equal(c.data.qpos, before)  # No pose teleport.

    def test_held_blocked_arm_does_not_veto_other_arms_retreat(self):
        c = self.blocked(both=True)
        held = c.left.command_position.copy()
        result = send(c, ('HOLD', 'XY_HOLD'), delta=([0, 0, 0], [-.004, 0, 0]))
        self.assertTrue(result['accepted'], result['reason'])
        np.testing.assert_allclose(c.left.command_position, held, rtol=0, atol=1e-9)

    def test_further_pressure_is_rejected_transactionally(self):
        c = self.blocked()
        before, ctrl = c.action_vector_m().copy(), c.data.ctrl.copy()
        result = send(c, ('XY_HOLD', 'XY_HOLD'), delta=([-.002, 0, 0], [.004, 0, 0]))
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'tracking_lead_limit')
        np.testing.assert_array_equal(c.action_vector_m(), before)
        np.testing.assert_array_equal(c.data.ctrl, ctrl)

    def test_retreat_still_runs_ik_and_collision_checks(self):
        for failure in ('ik_unreachable', 'predicted_robot_robot_collision'):
            c = self.blocked()
            before = c.action_vector_m().copy()
            if failure == 'ik_unreachable':
                context = patch.object(c.right, '_ik_iteration', return_value=None)
            else:
                context = patch.object(c.rigid_control, 'collision_reason', return_value=failure)
            with context:
                result = send(c, ('HOLD', 'XY_HOLD'), delta=([0, 0, 0], [-.004, 0, 0]))
            self.assertFalse(result['accepted'])
            self.assertEqual(result['reason'], failure)
            np.testing.assert_array_equal(c.action_vector_m(), before)

    def test_workspace_and_joint_guards_remain_active(self):
        c = self.blocked()
        arm = c.right
        old = arm.workspace.copy()
        try:
            arm.workspace[0, 0] = arm.world_to_task(arm.command_position)[0]+.01
            result = send(c, ('HOLD', 'XY_HOLD'), delta=([0, 0, 0], [-.004, 0, 0]))
            self.assertEqual(result['reason'], 'workspace_limit')
        finally:
            arm.workspace[:] = old
        c = self.blocked()
        iterate = c.right._ik_iteration
        def at_limit(*args):
            iterate(*args)
            c.right.ik_data.qpos[c.right.qpos_indices[0]] = c.model.jnt_range[c.right.joint_ids[0], 0]+.001
        with patch.object(c.right, '_ik_iteration', side_effect=at_limit):
            result = send(c, ('HOLD', 'XY_HOLD'), delta=([0, 0, 0], [-.004, 0, 0]))
        self.assertFalse(result['accepted'])
        self.assertIn(result['reason'], ('joint_limit', 'ik_unreachable'))

    def test_tray_does_not_enter_push_recovery(self):
        c = self.blocked('tray_to_platform')
        result = send(c, ('XY_HOLD', 'XY_HOLD'), delta=([0, 0, 0], [-.004, 0, 0]))
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'tracking_lead_limit')

    def at_joint_buffer(self):
        c = self.controllers['h_frame_push_align']
        c.reset(0)
        # Seven robot joint angles from the first real blocked round after
        # unloading contact; joint 2 is inside the 5 mrad planning buffer.
        q = [-1.54561947, 1.76179288, 1.37889649, -2.41138929,
             -1.8401772, 1.55423369, -1.47427053]
        c.data.qpos[c.right.qpos_indices] = q
        c.data.ctrl[c.right.actuator_ids] = q
        mujoco.mj_forward(c.model, c.data)
        c.right.command_position = c.right.get_actual_pose()[0]
        return c

    def test_can_leave_joint_buffer_but_not_push_further_into_it(self):
        c = self.at_joint_buffer()
        previous = c.data.ctrl[c.right.actuator_ids].copy()
        result = send(c, ('HOLD', 'Z_CLOSE'), delta=([0, 0, 0], [0, 0, .004]))
        self.assertTrue(result['accepted'], result['reason'])
        self.assertLess(c.right.ik_data.qpos[c.right.qpos_indices[1]], previous[1])
        c = self.at_joint_buffer()
        result = send(c, ('HOLD', 'Z_CLOSE'), delta=([0, 0, 0], [0, 0, -.004]))
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'joint_limit')

    def test_held_arm_in_joint_buffer_does_not_block_partner(self):
        c = self.at_joint_buffer()
        held = c.right.command_position.copy()
        result = send(c, ('Z_CLOSE', 'HOLD'), delta=([0, 0, .004], [0, 0, 0]))
        self.assertTrue(result['accepted'], result['reason'])
        np.testing.assert_allclose(c.right.command_position, held, rtol=0, atol=1e-9)

    def test_coupled_reference_keeps_existing_lead_guard(self):
        c = self.blocked(both=True)
        obj = c.data.xpos[c.object_id].copy()
        rot = c.data.xmat[c.object_id].reshape(3, 3).copy()
        c.contact_control.reference = [(rot.T@(a.command_position-obj), rot.T@a.command_rotation)
                                       for a in c.arms]
        c.rigid_control.reference_id = c.contact_control.reference
        c.rigid_control.object_position = obj
        c.rigid_control.object_rotation = rot
        c.rigid_control.handoff_start = c.contact_control.reference
        action = c.action_vector_m()
        action[[0, 7]] -= .004
        self.assertEqual(c.rigid_control.move(action), 'tracking_lead_limit')


if __name__ == '__main__':
    unittest.main()
