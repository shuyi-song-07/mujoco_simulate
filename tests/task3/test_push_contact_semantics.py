"""Non-grasp contact must not freeze pushing or enter rigid grasp handoff."""
import unittest
from unittest.mock import patch

import numpy as np

from tasks.registry import get
from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
from simulation.mujoco.dual_arm.panda_arm_controller import rotation_angle
from simulation.mujoco.dual_arm.rigid_control import rz
from rollout.common.dual_executor import PolicyExecutor
from tests.task3.rate_fixture import send


class PushContactSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.controllers = {task: DualArmTaskController(get(task).config()) for task in
                           ('h_frame_push_align', 'push_r', 'tray_to_platform')}

    def contacted(self, task, sides=('left', 'right'), policy=False):
        c = self.controllers[task]
        c.reset(0)
        executor = PolicyExecutor(c) if policy else None
        cc = c.contact_control
        # Measured contact qualification fixture, never a physical-success claim.
        for side in sides:
            c.contacts[side] = True
            c.contacts[side + '_fingers'] = [True, True]
            c.contacts[side + '_finger_grip_forces_n'] = [10., 10.]
            cc.grasp_close_qualified[side] = True
            cc.since[side] = -1.
        with patch.object(cc, 'rate_grasp_contact', side_effect=lambda side: side in sides):
            cc.update()
        return c, executor

    def test_dual_push_contact_does_not_create_grasp_reference(self):
        for task in ('h_frame_push_align', 'push_r'):
            for policy in (False, True):
                c, _ = self.contacted(task, policy=policy)
                self.assertIsNone(c.contact_control.reference)
                self.assertFalse(c.contact_control.ever_dual)
                self.assertFalse(any(c.contact_control.stable.values()))
                self.assertTrue(c.contacts['left'])  # Contact facts are retained.
                self.assertTrue(c.contacts['right'])

    def test_single_contact_does_not_silently_zero_requested_motion(self):
        for task in ('h_frame_push_align', 'push_r'):
            c, _ = self.contacted(task, ('right',))
            before = c.right.command_position.copy()
            result = send(c, ('HOLD', 'XY_HOLD'), delta=([0, 0, 0], [.001, 0, 0]))
            self.assertTrue(result['accepted'], result['reason'])
            self.assertGreater(np.linalg.norm(c.right.command_position-before), .0009)
            c, executor = self.contacted(task, ('right',), policy=True)
            action = c.action_vector_m().astype(float)
            action[7] += .001
            projected, report, reason = executor._project_motion(action)
            self.assertIsNone(reason)
            self.assertEqual(report['partial_hold_sides'], [])
            self.assertAlmostEqual(projected[7], action[7])

    def test_tray_keeps_real_grasp_coupling_and_partial_hold(self):
        c, _ = self.contacted('tray_to_platform')
        self.assertIsNotNone(c.contact_control.reference)
        self.assertTrue(c.contact_control.ever_dual)
        c, _ = self.contacted('tray_to_platform', ('right',))
        before = c.action_vector_m().copy()
        send(c, ('HOLD', 'XY_HOLD'), delta=([0, 0, 0], [.001, 0, 0]))
        np.testing.assert_allclose(c.action_vector_m(), before, atol=1e-7)

    def test_push_opening_pause_holds_width_without_creating_grasp_lock(self):
        for task in ('h_frame_push_align', 'push_r'):
            c, _ = self.contacted(task, ('right',))
            opened = send(c, ('HOLD', 'Z_OPEN'))
            self.assertTrue(opened['accepted'])
            self.assertTrue(c.contact_control.release_opening['right'])
            c.contact_control.cancel_rate_control('hand_lost')
            self.assertEqual(c.right.gripper_goal_width_m, c.right.gripper_command_width_m)
            self.assertIsNone(c.contact_control.hold_width_m['right'])
            before = c.right.command_position.copy()
            result = send(c, ('HOLD', 'XY_HOLD'), delta=([0, 0, 0], [.001, 0, 0]))
            self.assertTrue(result['accepted'], result['reason'])
            self.assertGreater(np.linalg.norm(c.right.command_position-before), .0009)

    def test_grasp_handoff_does_not_import_measured_roll_pitch_into_targets(self):
        c = self.controllers['tray_to_platform']
        c.reset(0)
        obj = c.data.xpos[c.object_id].copy()
        rot = c.data.xmat[c.object_id].reshape(3, 3).copy()
        angle = .04
        rx = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                       [0, np.sin(angle), np.cos(angle)]])
        refs = [(rot.T @ (a.command_position-obj), rot.T @ rx @ a.command_rotation) for a in c.arms]
        c.contact_control.reference = refs
        physical_reference = [r[1].copy() for r in refs]
        # Isolate target generation; measured finger deflection is allowed,
        # unsupported user roll/pitch and the physical reference stay unchanged.
        with patch.object(c.rigid_control, 'feasible', return_value=None), \
                patch.object(c.rigid_control, 'collision_reason', return_value=None):
            for _ in range(30):
                reason = c.rigid_control.move(c.action_vector_m())
                self.assertIsNone(reason)
                for a in c.arms:
                    self.assertLess(rotation_angle(a.command_rotation, a.fixed_rotation), .0001)
        for ref, original in zip(refs, physical_reference):
            np.testing.assert_array_equal(ref[1], original)


if __name__ == '__main__':
    unittest.main()
