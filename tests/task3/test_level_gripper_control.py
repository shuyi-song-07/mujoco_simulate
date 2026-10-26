"""Current-mode palm/fist strategy through real MuJoCo and shared execution."""
import unittest
import importlib.util
from unittest.mock import patch

import numpy as np

from simulation.mujoco.dual_arm.config import load_config
from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
from simulation.mujoco.dual_arm.task_scenes import scenario_config
from test_gesture_action_control import tick, pump
from test_recoverable_contact import settle


def level_controller(at_handles=False):
    config = scenario_config(load_config('config/collection.json'), 'S1')
    config['scene']['randomization'] = dict(object_xy_range_m=[0., 0.], target_xy_range_m=[0., 0.], object_yaw_range_rad=0.)
    if at_handles:
        config['scene'].update(left_home_ee=[0., -.22, .372], right_home_ee=[0., .22, .372])
    c = DualArmTaskController(config)
    c.set_episode_timeout_enabled(False)
    return c


def real_grasp():
    c = level_controller(True)
    settle(c, .3)
    pump(c, ('Z_CLOSE', 'Z_CLOSE'), 75)
    assert all(c.contact_control.stable.values()), c.get_status()
    return c


class LevelGripperTests(unittest.TestCase):
    def tail_controller(self, opening=True):
        from simulation.mujoco.dual_arm.task_states import TaskState
        c = level_controller()
        c.config['recording']['post_success_seconds'] = .5
        for arm in c.arms:
            arm._write_gripper_width(.04)
            arm.set_gripper_width(.08 if opening else 0.)
        def complete(): c.task_state = TaskState.DONE
        with patch.object(c.contact_control, 'update', side_effect=complete):
            c._update_task()
        return c

    def test_success_tail_continues_only_opening_with_frozen_pose(self):
        c = self.tail_controller()
        pose = c.action_vector_m().copy()
        deadline = c.success_tail['end_time']
        for _ in range(30):
            c.pause('task_done')
            c.pause('command_timeout')
            c.step(1/60)
        self.assertAlmostEqual(c.data.time, deadline, places=8)
        for arm in c.arms:
            self.assertAlmostEqual(arm.gripper_command_width_m, .08, places=7)
        np.testing.assert_allclose(c.action_vector_m()[[0,1,2,3,4,5,7,8,9,10,11,12]],
                                   pose[[0,1,2,3,4,5,7,8,9,10,11,12]])
        c.step(1/60)
        self.assertFalse(c.success_tail['active'])
        self.assertTrue(c.paused)

    def test_success_tail_never_continues_closing_and_obeys_manual_pause(self):
        for reason in ('user_pause', 'robot_robot_collision'):
            c = self.tail_controller()
            c.step(1/60)
            c.pause(reason, fail=reason == 'robot_robot_collision')
            width = [a.gripper_command_width_m for a in c.arms]
            c.step(.1)
            self.assertEqual(width, [a.gripper_command_width_m for a in c.arms])
            self.assertFalse(c.success_tail['active'])
        c = self.tail_controller(opening=False)
        c.step(.1)
        np.testing.assert_allclose([a.gripper_command_width_m for a in c.arms], [.04,.04])

    @unittest.skipUnless(importlib.util.find_spec('flask'), 'Recorder requires the collection environment')
    def test_recording_gate_accepts_empty_partial_jaws_but_not_grasp_history(self):
        from simulation.mujoco.dual_arm.teleop_service import validate_recording_initial_state
        c = level_controller()
        pump(c, ('Z_CLOSE', 'Z_CLOSE'), 8)
        c.pause('hand_lost')
        status = c.get_status()
        self.assertLess(status['grippers']['right']['command_width_m'], .079)
        validate_recording_initial_state(status, True)
        for change in ('contact', 'history', 'phase'):
            s = c.get_status()
            if change == 'contact': s['contacts']['left_fingers'] = [True, False]
            if change == 'history': s['physical_contact']['ever_dual'] = True
            if change == 'phase': s['task_state'] = 'PARTIAL_CONTACT'
            with self.assertRaises(ValueError): validate_recording_initial_state(s, True)
        status['physical_contact']['gripper_trigger_policy'] = 'palm_fist_edge_v2'
        with self.assertRaises(ValueError): validate_recording_initial_state(status, True)

    def test_done_tick_freezes_without_browser_and_rejects_later_controls(self):
        from simulation.mujoco.dual_arm.task_states import TaskState
        c = level_controller()
        pump(c, ('Z_CLOSE', 'Z_CLOSE'), 4)
        def complete(): c.task_state = TaskState.DONE
        # Simulate only the task monitor's decision; exercise real completion
        # handling, actuator holding and subsequent command rejection.
        with patch.object(c.contact_control, 'update', side_effect=complete):
            c._update_task()
        self.assertTrue(c.paused)
        self.assertEqual(c.pause_reason, 'task_done')
        frozen = c.action_vector_m().copy()
        self.assertFalse(tick(c, ('Z_CLOSE', 'Z_CLOSE'))['accepted'])
        self.assertFalse(tick(c, ('Z_OPEN', 'Z_OPEN'))['accepted'])
        np.testing.assert_allclose(c.action_vector_m(), frozen)

    @unittest.skipUnless(importlib.util.find_spec('flask'), 'Recorder requires the collection environment')
    def test_recorder_start_requires_binding_but_not_visible_hands(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from simulation.mujoco.dual_arm import teleop_service
        c = level_controller()
        pump(c, ('Z_CLOSE', 'Z_CLOSE'), 8)
        c.pause('hand_lost')
        rec = teleop_service.DualArmRecorder.__new__(teleop_service.DualArmRecorder)
        rec.controller, rec.config = c, c.config
        rec.recording, rec.paper_recording = False, True
        rec.args = SimpleNamespace(input_kind='human_teleop')
        rec.telemetry = {'calibrated': True, 'hands': {},
            'aperture_calibration': {'phase': 'NOT_REQUIRED', 'reason': 'categorical_rate_control'}}
        rec.session, rec.attempt = 'solo-test', 0
        class ReadyToRecord(Exception): pass
        # Stop exactly at the metadata write, after the actual recording gates.
        # No mocked binding or hand guard: only storage/worker startup is omitted.
        with tempfile.TemporaryDirectory() as directory, patch.object(teleop_service, 'ROOT', Path(directory)), \
                patch.object(rec, 'metadata', side_effect=ReadyToRecord):
            with self.assertRaises(ReadyToRecord): rec.start_recording()
            rec.telemetry['calibrated'] = False
            with self.assertRaisesRegex(ValueError, 'Bind both hands'): rec.start_recording()

    def test_first_fist_closes_and_same_shape_after_xy_resumes_without_edge(self):
        c = level_controller()
        self.assertTrue(c.contact_control.level_control)
        self.assertEqual(c.action_contract()['gesture_parameters']['mode_confirm_ms'], 100)
        before = c.left.gripper_command_width_m
        self.assertTrue(tick(c, ('Z_CLOSE', 'HOLD'))['accepted'])
        self.assertAlmostEqual(before-c.left.gripper_command_width_m, .08*.02, places=6)
        pump(c, ('Z_CLOSE', 'HOLD'), 5)
        widths = c.action_vector_m()[[6, 13]].copy()
        pump(c, ('XY_HOLD', 'XY_HOLD'), 10)
        np.testing.assert_allclose(c.action_vector_m()[[6, 13]], widths)
        tick(c, ('Z_CLOSE', 'HOLD'))
        self.assertLess(c.left.gripper_command_width_m, widths[0])
        self.assertAlmostEqual(c.right.gripper_command_width_m, widths[1])

    def test_open_and_close_allow_z_but_hold_freezes_both_outputs(self):
        c = level_controller()
        pump(c, ('Z_CLOSE', 'Z_CLOSE'), 8)
        before = c.action_vector_m().copy()
        pump(c, ('Z_OPEN', 'Z_CLOSE'), 5, delta=([0, 0, .0005], [0, 0, .0005]))
        self.assertGreater(c.left.gripper_command_width_m, before[6])
        self.assertLess(c.right.gripper_command_width_m, before[13])
        self.assertGreater(c.action_vector_m()[2], before[2])
        self.assertGreater(c.action_vector_m()[9], before[9])
        frozen = c.action_vector_m().copy()
        pump(c, ('HOLD', 'HOLD'), 8, delta=([.002, 0, .002], [.002, 0, .002]), widths=(0., 0.))
        np.testing.assert_allclose(c.action_vector_m(), frozen, atol=1e-7)

    def test_real_dual_grasp_one_palm_opens_immediately_without_other_palm(self):
        c = real_grasp()
        before = c.action_vector_m()[[6, 13]].copy()
        result = tick(c, ('Z_OPEN', 'Z_CLOSE'))
        self.assertTrue(result['accepted'], result)
        self.assertTrue(c.contact_control.release_executed['left'])
        self.assertFalse(c.contact_control.release_executed['right'])
        self.assertEqual(c.contact_control.release_progress_s['left'], 0.)
        self.assertGreater(c.left.gripper_command_width_m, before[0])
        self.assertAlmostEqual(c.right.gripper_command_width_m, before[1])
        self.assertIsNone(c.contact_control.reference)
        self.assertEqual(c.contact_control.release_count, 1)
        # Observe past the 0.3 s contact-loss grace period as well.
        pump(c, ('Z_OPEN', 'Z_CLOSE'), 30)
        self.assertEqual(c.contact_control.release_count, 1)
        self.assertIsNone(c.get_status()['failure_reason'])
        self.assertIsNone(c.contact_control.reference)
        self.assertAlmostEqual(c.right.gripper_command_width_m, before[1])

    def test_airborne_both_palms_no_support_or_stationary_wait(self):
        c = real_grasp()
        pump(c, ('Z_CLOSE', 'Z_CLOSE'), 65, delta=([0., 0., .0008], [0., 0., .0008]))
        self.assertFalse(c.contact_control.support_ready)
        before = c.action_vector_m()[[6, 13]].copy()
        result = tick(c, ('Z_OPEN', 'Z_OPEN'), delta=([0., 0., .0005], [0., 0., .0005]))
        self.assertTrue(result['accepted'], result)
        self.assertTrue(all(c.contact_control.release_executed.values()))
        self.assertTrue(np.all(c.action_vector_m()[[6, 13]] > before))

    def test_loss_and_manual_pause_stop_travel_then_fresh_mode_can_resume(self):
        c = level_controller()
        pump(c, ('Z_CLOSE', 'Z_CLOSE'), 5)
        settle(c, .4)
        frozen = c.action_vector_m()[[6, 13]].copy()
        settle(c, .2)
        np.testing.assert_allclose(c.action_vector_m()[[6, 13]], frozen)
        tick(c, ('Z_CLOSE', 'Z_CLOSE'))
        self.assertTrue(np.all(c.action_vector_m()[[6, 13]] < frozen))
        c.pause('operator_pause')
        frozen = c.action_vector_m()[[6, 13]].copy()
        settle(c, .2)
        np.testing.assert_allclose(c.action_vector_m()[[6, 13]], frozen)
        tick(c, ('Z_OPEN', 'Z_OPEN'))
        self.assertTrue(np.all(c.action_vector_m()[[6, 13]] > frozen))


if __name__ == '__main__':
    unittest.main()
