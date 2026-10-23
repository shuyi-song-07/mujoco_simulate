"""Manual control, camera priority, and repeated entry token regressions."""
import unittest
import queue
import time
import numpy as np
import mujoco
from test_mujoco_gesture_controls import controller

class ManualPolicyTest(unittest.TestCase):
    def test_can_reopen_and_close_without_holding_red_or_reaching_plate(self):
        state = controller()
        state.gripper_orientation_finalized = True
        state.is_holding_cube = lambda: False
        state.cube_over_target = False
        for index, command in enumerate(("close", "open", "close", "open")):
            state.handle_gesture({"command": command, "controlToken": "test", "sentAt": index + 1})
            self.assertEqual(state.gripper_closed, command == "close")
            self.assertEqual(state.data.ctrl[7], 0 if command == "close" else 255)

    def test_diagonal_hand_loss_keeps_gripper_closed_and_rejects_late_open(self):
        state = controller()
        state.gripper_orientation_finalized = True
        state.handle_gesture({"command": "close", "controlToken": "test", "sentAt": 1})
        state.handle_gesture({"command": "gesture_state", "gesture": "ILoveYou", "handVisible": True, "sentAt": 2})
        state.handle_gesture({"command": "gesture_state", "gesture": "None", "handVisible": False, "sentAt": 3})
        state.handle_gesture({"command": "pause"})
        state.handle_gesture({"command": "open", "controlToken": "test", "sentAt": 2.5})
        self.assertTrue(state.control_paused)
        self.assertTrue(state.gripper_closed)
        self.assertEqual(state.data.ctrl[7], 0)

    def test_same_tick_fist_closes_and_descends_in_either_order(self):
        for commands in (("down", "close"), ("close", "down")):
            with self.subTest(commands=commands):
                state = controller()
                state.command_queue = queue.Queue()
                state.args.max_motion_command_age_ms = 150
                state.gripper_orientation_finalized = True
                state.is_holding_cube = lambda: False
                before_z = state._commanded_hand_position()[2]
                now = time.time() * 1000
                for index, command in enumerate(commands):
                    state.command_queue.put({"command": command, "controlToken": "test", "sentAt": now + index * .001})
                state.process_queued_commands()
                self.assertTrue(state.gripper_closed)
                self.assertEqual(state.data.ctrl[7], 0)
                self.assertLess(state._commanded_hand_position()[2], before_z - .003)

    def test_close_does_not_revive_expired_or_invalid_token_descent(self):
        for age_ms, token in ((1000, "test"), (0, "old-token")):
            with self.subTest(age_ms=age_ms, token=token):
                state = controller()
                state.command_queue = queue.Queue()
                state.args.max_motion_command_age_ms = 150
                state.gripper_orientation_finalized = True
                before = state.data.ctrl[:7].copy()
                now = time.time() * 1000
                state.command_queue.put({"command": "down", "controlToken": token, "sentAt": now - age_ms})
                state.command_queue.put({"command": "close", "controlToken": "test", "sentAt": now})
                state.process_queued_commands()
                self.assertTrue(state.gripper_closed)
                np.testing.assert_array_equal(state.data.ctrl[:7], before)

    def test_empty_diagonal_can_use_top_view_after_a_completed_delivery(self):
        state = controller()
        state.target_view_reached_with_cube = True
        state.gripper_above_cube_view = False
        state.transport_view_active = False
        state.z_locked = False
        state.latest_gesture = "ILoveYou"
        state.hand_visible = True
        state.last_gesture_update_at = time.perf_counter()
        state.is_holding_cube = lambda: False
        self.assertTrue(state.operator_uses_topdown())
        state.gripper_above_cube_view = True
        self.assertFalse(state.operator_uses_topdown())

    def test_close_does_not_block_explicit_vertical_motion(self):
        state = controller()
        state.gripper_orientation_finalized = True
        state.is_holding_cube = lambda: False
        before = state._commanded_hand_position()
        state.handle_gesture({'command': 'close', 'controlToken': 'test', 'sentAt': 1})
        targets = state.data.ctrl[:7].copy()
        state.handle_gesture({'command': 'up', 'controlToken': 'test', 'sentAt': 2})
        self.assertFalse(np.array_equal(targets, state.data.ctrl[:7]))
        self.assertGreater(state._commanded_hand_position()[2], before[2] + .003)

    def test_only_carried_arrival_switches_view_after_150ms(self):
        state = controller()
        state.args.target_dwell_s = .15
        state.args.target_exit_margin = .02
        state.target_entered_sim_time = None
        state.target_view_reached_with_cube = False
        state.carried_target_since = None
        state.transport_view_active = True
        state.update_gripper_approach_view = lambda: None
        state.is_holding_cube = lambda: False
        joint = state.model.body_jntadr[state.cube_id]
        address = state.model.jnt_qposadr[joint]
        state.data.qpos[address:address + 2] = state.model.body_pos[state.target_plate_id][:2]
        mujoco.mj_forward(state.model, state.data)
        state.z_locked = True
        state.update_target_state()
        state.data.time = .2
        state.update_target_state()
        self.assertTrue(state.cube_over_target)
        self.assertFalse(state.target_view_reached_with_cube)
        self.assertTrue(state.z_locked)
        state.is_holding_cube = lambda: True
        state.update_target_state()
        state.data.time = .34
        state.update_target_state()
        self.assertFalse(state.target_view_reached_with_cube)
        state.data.time = .36
        state.update_target_state()
        self.assertTrue(state.target_view_reached_with_cube)
        self.assertFalse(state.z_locked)

    def test_reentry_rejects_stale_tokens_but_accepts_same_gesture(self):
        state = controller()
        state.session = 'test-session'
        state.control_epoch = 0
        state.gripper_orientation_finalized = True
        for cycle in range(20):
            stamp = 1000 + cycle * 1000
            state.handle_gesture({'command': 'gesture_state', 'gesture': 'None', 'handVisible': False, 'sentAt': stamp})
            self.assertTrue(state.control_paused)
            token = f'entry-{cycle}'
            state.handle_gesture({'command': 'resume', 'controlContext': 'test-session:0', 'controlToken': token, 'sentAt': stamp + 10})
            # A delayed old lost-hand report must not undo the new resume.
            state.handle_gesture({'command': 'gesture_state', 'gesture': 'None', 'handVisible': False, 'sentAt': stamp})
            self.assertFalse(state.control_paused)
            state.handle_gesture({'command': 'gesture_state', 'gesture': 'ILoveYou', 'handVisible': True, 'sentAt': stamp + 350})
            before = state.data.ctrl[:7].copy()
            state.handle_gesture({'command': 'move_xy', 'dx': .2, 'dy': .2, 'controlToken': 'old'})
            np.testing.assert_array_equal(before, state.data.ctrl[:7])
            state.handle_gesture({'command': 'move_xy', 'dx': .2, 'dy': .2, 'controlToken': token})
            self.assertFalse(np.array_equal(before, state.data.ctrl[:7]))
