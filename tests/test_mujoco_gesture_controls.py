"""Regression checks for Panda gripper yaw and diagonal-to-vertical control."""

import queue
import time
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from simulation.mujoco.record_mujoco_panda import (
    DEFAULT_MODEL_PATH,
    INITIAL_ARM_QPOS,
    MujocoPandaRecorder,
    parse_args,
)


def controller():
    state = object.__new__(MujocoPandaRecorder)
    state.model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH))
    state.data = mujoco.MjData(state.model)
    state.action_data = mujoco.MjData(state.model)
    state.data.qpos[:7] = INITIAL_ARM_QPOS
    state.data.ctrl[:7] = INITIAL_ARM_QPOS
    state.data.ctrl[7] = 255
    mujoco.mj_forward(state.model, state.data)
    state.args = SimpleNamespace(
        task_mode="pick_place", vertical_step=0.008, table_top_z=0.4,
        target_top_z=0.416, target_radius=0.05, safe_reach_radius=0.72,
        max_joint_step=0.08, max_joint_target_lead=0.12,
        max_planar_step=0.015,
        grasped_min_width=0.02, grasped_max_width=0.11,
        grasped_max_distance=0.18,
    )
    state.base_id = state.model.body("link0").id
    state.hand_id = state.model.body("hand").id
    state.cube_id = state.model.body("cube").id
    state.target_plate_id = state.model.body("target_plate").id
    state.cube_geom_id = int(state.model.body_geomadr[state.cube_id])
    state.distractor_ids = [state.model.body(name).id for name in ("distractor_blue", "distractor_green", "distractor_yellow")]
    state.randomized_geom_ids = [state.cube_geom_id] + [int(state.model.body_geomadr[body]) for body in state.distractor_ids]
    state.gripper_yaw_joint_id = state.model.joint("gripper_yaw_joint").id
    state.gripper_yaw_qpos_address = int(state.model.jnt_qposadr[state.gripper_yaw_joint_id])
    state.gripper_yaw_dof_address = int(state.model.jnt_dofadr[state.gripper_yaw_joint_id])
    state.gripper_yaw_actuator_id = state.model.actuator("gripper_yaw_actuator").id
    state.finger_joint_ids = [state.model.joint(name).id for name in ("finger_joint1", "finger_joint2")]
    state.finger_qpos_addresses = [int(state.model.jnt_qposadr[joint_id]) for joint_id in state.finger_joint_ids]
    state.data.qpos[state.finger_qpos_addresses] = [0.05, 0.05]
    mujoco.mj_forward(state.model, state.data)
    state.finger_body_ids = (state.model.body("left_finger").id, state.model.body("right_finger").id)
    state.last_bilateral_cube_contact_at = float("-inf")
    state.initial_hand_rotation_in_base = state.data.xmat[state.hand_id].reshape(3, 3).copy()
    state.target_hand_rotation_in_base = state.initial_hand_rotation_in_base.copy()
    state.jacp = np.zeros((3, state.model.nv))
    state.jacr = np.zeros((3, state.model.nv))
    state.latest_gesture = "None"
    state.last_gesture_command_at = float("-inf")
    state.last_resume_command_at = float("-inf")
    state.last_gesture_update_at = time.perf_counter()
    state.hand_visible = False
    state.control_paused = False
    state.control_token = "test"
    state.gripper_yaw = 0.0
    state.gripper_orientation_locked = False
    state.gripper_orientation_finalized = False
    state.pointing_arm_qpos = None
    state.gripper_closed = False
    state.grasp_close_started_at = float("-inf")
    state.grasp_support_z = state.args.table_top_z
    state.pending_vertical_steps = 0
    state.last_deferred_vertical_at = float("-inf")
    state.motion_limit_reason = None
    state.last_gripper_command_at = float("-inf")
    state.last_motion_command_at = float("-inf")
    state.z_lock_used_for_current_grasp = False
    state.z_locked = False
    state.locked_z = None
    state.cube_over_target = False
    state.target_view_reached_with_cube = False
    state.transport_view_active = False
    state.gripper_above_cube_view = False
    state.vertical_anchor_xy = None
    state.side_preview_jpeg = None
    return state


class GestureControlTest(unittest.TestCase):
    def test_hand_motion_defaults_increase_planar_diagonal_and_vertical_steps(self):
        args = parse_args(["--headless-test"])
        self.assertAlmostEqual(args.max_planar_step, 0.0216)
        self.assertAlmostEqual(args.vertical_step, 0.01152)

        state = controller()
        state.args.max_planar_step = args.max_planar_step
        state.args.vertical_step = args.vertical_step
        requested = []
        state.apply_web_motion = lambda dx, dy, dz: requested.append((dx, dy, dz))

        state.handle_gesture({"command": "move_xy", "dx": 0.5, "dy": 0.5, "controlToken": "test"})
        np.testing.assert_allclose(requested.pop(), [0.0108, 0.0108, 0.0])

        state.handle_gesture({"command": "up", "controlToken": "test"})
        np.testing.assert_allclose(requested.pop(), [0.0, 0.0, 0.01152])

    def test_first_and_repeated_open_move_fingers_through_actuator(self):
        state = controller()
        finger_dof_addresses = [
            int(state.model.jnt_dofadr[joint_id]) for joint_id in state.finger_joint_ids
        ]
        state.data.qpos[state.finger_qpos_addresses] = [0.012, 0.012]
        state.data.qvel[finger_dof_addresses] = [0.005, 0.005]
        mujoco.mj_forward(state.model, state.data)
        state.gripper_closed = True
        state.data.ctrl[7] = 0.0
        finger_positions = state.data.qpos[state.finger_qpos_addresses].copy()
        finger_velocities = state.data.qvel[finger_dof_addresses].copy()
        state.handle_gesture({"command": "open", "controlToken": "test", "sentAt": 1})
        self.assertFalse(state.gripper_closed)
        np.testing.assert_array_equal(
            state.data.qpos[state.finger_qpos_addresses], finger_positions
        )
        np.testing.assert_array_equal(
            state.data.qvel[finger_dof_addresses], finger_velocities
        )
        self.assertEqual(state.data.ctrl[7], 255)

        state.handle_gesture({"command": "open", "controlToken": "test", "sentAt": 2})
        np.testing.assert_array_equal(
            state.data.qpos[state.finger_qpos_addresses], finger_positions
        )
        self.assertEqual(state.data.ctrl[7], 255)

        for _ in range(100):
            mujoco.mj_step(state.model, state.data)
        self.assertGreater(
            sum(state.data.qpos[address] for address in state.finger_qpos_addresses),
            sum(finger_positions),
        )

    def test_missing_gesture_updates_do_not_pause_control(self):
        state = controller()
        state.command_queue = queue.Queue()
        state.last_gesture_update_at = time.perf_counter() - 10.0
        state.process_queued_commands()
        self.assertFalse(state.control_paused)
        self.assertEqual(state.control_token, "test")

    def test_every_layout_box_is_free_and_cannot_remain_below_table(self):
        state = controller()
        for body_id, geom_id in zip([state.cube_id, *state.distractor_ids], state.randomized_geom_ids):
            joint_id = int(state.model.body_jntadr[body_id])
            self.assertEqual(state.model.jnt_type[joint_id], mujoco.mjtJoint.mjJNT_FREE)
            qpos_address = int(state.model.jnt_qposadr[joint_id])
            state.data.qpos[qpos_address:qpos_address + 3] = [0.52, 0.3, 0.38]
            state.data.qpos[qpos_address + 3:qpos_address + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(state.model, state.data)
        before = state.data.qpos.copy()
        state.enforce_layout_support()
        for body_id in [state.cube_id, *state.distractor_ids]:
            joint_id = int(state.model.body_jntadr[body_id])
            address = int(state.model.jnt_qposadr[joint_id])
            self.assertLessEqual(state.data.qpos[address + 2] - before[address + 2], 0.0010001)
        for _ in range(100):
            state.enforce_layout_support()
        for body_id, geom_id in zip([state.cube_id, *state.distractor_ids], state.randomized_geom_ids):
            half = float(state.model.geom_size[geom_id][2])
            bottom = float(state.data.xpos[body_id][2] - half)
            self.assertGreaterEqual(bottom, state.args.table_top_z - 0.00101)

    def test_grasp_contact_does_not_press_cube_into_table(self):
        state = controller()
        half = float(state.model.geom_size[state.cube_geom_id][2])
        hand = state.data.xpos[state.hand_id].copy()
        cube_joint = int(state.model.body_jntadr[state.cube_id])
        cube_address = int(state.model.jnt_qposadr[cube_joint])
        state.data.qpos[cube_address:cube_address + 3] = [
            hand[0], hand[1], state.args.table_top_z + half + 0.002,
        ]
        state.data.qpos[cube_address + 3:cube_address + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(state.model, state.data)

        for _ in range(22):
            state.handle_gesture({"command": "down", "controlToken": "test"})
            for _ in range(35):
                mujoco.mj_step(state.model, state.data)
        state.handle_gesture({"command": "close", "controlToken": "test", "sentAt": 1})
        minimum_bottom = float("inf")
        for step in range(1200):
            if step % 80 == 0:
                state.handle_gesture({"command": "down", "controlToken": "test"})
            mujoco.mj_step(state.model, state.data)
            cube_bottom = float(state.data.xpos[state.cube_id][2] - half)
            minimum_bottom = min(minimum_bottom, cube_bottom)

        self.assertGreaterEqual(minimum_bottom, state.args.table_top_z - 0.00075)

    def test_one_sided_contact_does_not_count_as_holding(self):
        # Recreate the old hard-coded joint trajectory. Its hand mesh can
        # push the cube upward while only one finger touches it, so height
        # alone must never count as a grasp. Actual two-finger pickup is
        # exercised through the controller in test_mujoco_grasp_physics.py.
        start_arm = np.array([
            -0.1474426389, 0.1088735610, -0.1136709526,
            -1.6651794910, 0.0113847712, 1.7583295107, 0.5371947289,
        ])
        lifted_arm = np.array([
            -0.14957999, 0.12961961, -0.11484645, -1.48672652,
            0.01367702, 1.6012094, 0.53597713,
        ])
        cube_half_size = 0.03254157298
        cube_yaw = 2.8595095229
        original_xy = np.array([0.562760706, -0.155161473])

        def arrange(centered, half_size=cube_half_size, yaw=cube_yaw):
            state = controller()
            state._set_box_half_size(state.cube_id, state.cube_geom_id, half_size)
            state.model.body_pos[state.target_plate_id][:2] = [0.525178094, 0.033120809]
            state.data.qpos[:7] = start_arm
            state.data.ctrl[:7] = start_arm
            state.gripper_yaw = -0.1667463034
            state.gripper_orientation_locked = True
            state.data.qpos[state.gripper_yaw_qpos_address] = state.gripper_yaw
            state.data.ctrl[state.gripper_yaw_actuator_id] = state.gripper_yaw
            mujoco.mj_forward(state.model, state.data)
            cube_joint = int(state.model.body_jntadr[state.cube_id])
            cube_address = int(state.model.jnt_qposadr[cube_joint])
            xy = state.data.xpos[state.hand_id][:2].copy() if centered else original_xy
            state.data.qpos[cube_address:cube_address + 3] = [
                *xy, state.args.table_top_z + half_size + 0.002,
            ]
            state.data.qpos[cube_address + 3:cube_address + 7] = [
                np.cos(yaw / 2), 0, 0, np.sin(yaw / 2),
            ]
            state.data.qpos[state.finger_qpos_addresses] = [0.05, 0.05]
            state.data.ctrl[7] = 0.0
            state.gripper_closed = True
            mujoco.mj_forward(state.model, state.data)
            return state

        def step(state, index):
            mujoco.mj_step(state.model, state.data)
            if index % 17 == 16:
                state.hard_lock_gripper_yaw()
                state.enforce_layout_support()

        off_center = arrange(centered=False)
        widths = []
        for index in range(4000):
            step(off_center, index)
            if index >= 3000 and index % 10 == 0:
                widths.append(sum(off_center.data.qpos[off_center.finger_qpos_addresses]))
        self.assertLess(max(widths) - min(widths), 0.001)
        self.assertFalse(off_center.is_holding_cube())

        centered = arrange(centered=True)
        initial_cube_z = float(centered.data.xpos[centered.cube_id][2])
        minimum_bottom = float("inf")
        hold_cube_z = []
        for index in range(5000):
            elapsed = index * centered.model.opt.timestep
            if 1.0 <= elapsed < 2.5:
                centered.data.ctrl[:7] = start_arm + (lifted_arm - start_arm) * ((elapsed - 1.0) / 1.5)
            elif elapsed >= 2.5:
                centered.data.ctrl[:7] = lifted_arm
            step(centered, index)
            if index % 10 == 0:
                rotation = centered.data.xmat[centered.cube_id].reshape(3, 3)
                vertical_extent = float(np.abs(rotation[2]) @ centered.model.geom_size[centered.cube_geom_id])
                cube_z = float(centered.data.xpos[centered.cube_id][2])
                minimum_bottom = min(minimum_bottom, cube_z - vertical_extent)
                if elapsed >= 3.5:
                    hold_cube_z.append(cube_z)
        self.assertGreater(min(hold_cube_z) - initial_cube_z, 0.04)
        self.assertGreaterEqual(minimum_bottom, centered.args.table_top_z - 0.001)
        centered_contacts = {
            int(centered.model.geom_bodyid[
                contact.geom2 if contact.geom1 == centered.cube_geom_id else contact.geom1
            ])
            for contact in centered.data.contact[:centered.data.ncon]
            if centered.cube_geom_id in (contact.geom1, contact.geom2)
        }
        self.assertIn(centered.hand_id, centered_contacts)
        self.assertEqual(len(centered_contacts.intersection(centered.finger_body_ids)), 1)
        self.assertFalse(centered.is_holding_cube())

        # A wide aperture is likewise insufficient without both contacts.
        wide = arrange(centered=True, half_size=0.034, yaw=cube_yaw - np.pi / 6)
        for index in range(4000):
            elapsed = index * wide.model.opt.timestep
            if 1.0 <= elapsed < 2.5:
                wide.data.ctrl[:7] = start_arm + (lifted_arm - start_arm) * ((elapsed - 1.0) / 1.5)
            elif elapsed >= 2.5:
                wide.data.ctrl[:7] = lifted_arm
            step(wide, index)
        wide_width = sum(wide.data.qpos[wide.finger_qpos_addresses])
        self.assertGreater(wide_width, 0.075)
        self.assertLess(wide_width, 0.11)
        wide_contacts = {
            int(wide.model.geom_bodyid[
                contact.geom2 if contact.geom1 == wide.cube_geom_id else contact.geom1
            ])
            for contact in wide.data.contact[:wide.data.ncon]
            if wide.cube_geom_id in (contact.geom1, contact.geom2)
        }
        self.assertIn(wide.hand_id, wide_contacts)
        self.assertEqual(len(wide_contacts.intersection(wide.finger_body_ids)), 1)
        self.assertFalse(wide.is_holding_cube())
        wide.gripper_closed = False
        self.assertFalse(wide.is_holding_cube())

    def test_pointing_orientation_stays_fixed_during_diagonal_motion(self):
        state = controller()
        angle = 0.65
        state.handle_gesture({"command": "gesture_state", "gesture": "Pointing_Up", "handVisible": True, "sentAt": 1})
        state.handle_gesture({"command": "set_gripper_yaw", "angle": angle, "controlToken": "test"})
        for _ in range(240):
            mujoco.mj_step(state.model, state.data)
            state.hold_pointing_arm()
        held_rotation = state.target_hand_rotation_in_base.copy()
        start_position = state.data.xpos[state.hand_id].copy()
        state.handle_gesture({"command": "finalize_gripper_yaw", "angle": angle, "controlToken": "test"})
        state.handle_gesture({"command": "gesture_state", "gesture": "ILoveYou", "handVisible": True, "sentAt": 2})

        for _ in range(24):
            state.handle_gesture({"command": "move_xy", "dx": 0.5, "dy": 0.5, "controlToken": "test"})
            state.maintain_gripper_orientation()
            for _ in range(8):
                mujoco.mj_step(state.model, state.data)
            state.hard_lock_gripper_yaw()

        state.action_data.qpos[:] = state.data.qpos
        state.action_data.qpos[:7] = state.data.ctrl[:7]
        state.action_data.qpos[state.gripper_yaw_qpos_address] = state.data.ctrl[state.gripper_yaw_actuator_id]
        mujoco.mj_forward(state.model, state.action_data)
        base_rotation = state.action_data.xmat[state.base_id].reshape(3, 3)
        commanded_rotation = base_rotation.T @ state.action_data.xmat[state.hand_id].reshape(3, 3)
        orientation_error = 0.5 * sum(
            np.cross(commanded_rotation[:, axis], held_rotation[:, axis])
            for axis in range(3)
        )
        self.assertLess(np.linalg.norm(orientation_error), 1e-4)
        self.assertGreater(
            np.linalg.norm(state.action_data.xpos[state.hand_id][:2] - start_position[:2]),
            0.05,
        )
        self.assertAlmostEqual(state.data.ctrl[state.gripper_yaw_actuator_id], angle)
        self.assertAlmostEqual(state.data.qpos[state.gripper_yaw_qpos_address], angle)
        self.assertAlmostEqual(state.data.qvel[state.gripper_yaw_dof_address], 0.0)
        np.testing.assert_allclose(state.target_hand_rotation_in_base, held_rotation, atol=1e-12)




    def test_pointing_up_moves_only_dedicated_yaw_joint(self):
        for angle in (0.5, -0.5):
            with self.subTest(angle=angle):
                state = controller()
                initial_arm = state.data.qpos[:7].copy()
                initial_hand = state.data.xpos[state.hand_id].copy()
                state.handle_gesture({"command": "gesture_state", "gesture": "Pointing_Up", "handVisible": True, "sentAt": 1})
                state.handle_gesture({"command": "set_gripper_yaw", "angle": angle, "controlToken": "test"})
                np.testing.assert_array_equal(state.data.ctrl[:7], initial_arm)
                self.assertAlmostEqual(state.data.ctrl[state.gripper_yaw_actuator_id], angle)
                self.assertAlmostEqual(state.data.qpos[state.gripper_yaw_qpos_address], 0.0)
                for _ in range(300):
                    mujoco.mj_step(state.model, state.data)
                    state.hold_pointing_arm()
                np.testing.assert_array_equal(state.data.qpos[:7], initial_arm)
                np.testing.assert_allclose(state.data.xpos[state.hand_id], initial_hand, atol=1e-9)
                self.assertGreater(state.data.qpos[state.gripper_yaw_qpos_address] * np.sign(angle), 0.4)

    def test_closed_fist_releases_transport_lock_and_can_descend(self):
        state = controller()
        state.z_locked = True
        state.locked_z = float(state.data.xpos[state.hand_id][2])
        state.transport_view_active = True
        state.latest_gesture = "Closed_Fist"
        initial_hand_z = float(state.data.xpos[state.hand_id][2])
        state.handle_gesture({"command": "start_vertical", "controlToken": "test"})
        self.assertFalse(state.z_locked)
        self.assertTrue(state.operator_uses_topdown())
        state.handle_gesture({"command": "down", "controlToken": "test"})
        state.action_data.qpos[:] = state.data.qpos
        state.action_data.qpos[:7] = state.data.ctrl[:7]
        mujoco.mj_forward(state.model, state.action_data)
        self.assertLess(state.action_data.xpos[state.hand_id][2], initial_hand_z)

    def test_down_command_unlocks_even_if_start_vertical_was_delayed(self):
        state = controller()
        state.z_locked = True
        state.locked_z = float(state.data.xpos[state.hand_id][2])
        state.latest_gesture = "Closed_Fist"
        state.handle_gesture({"command": "down", "controlToken": "test"})
        self.assertFalse(state.z_locked)

    def test_carried_cube_does_not_trigger_empty_gripper_approach_clamp(self):
        state = controller()
        hand = state.data.xpos[state.hand_id].copy()
        cube_joint = int(state.model.body_jntadr[state.cube_id])
        cube_address = int(state.model.jnt_qposadr[cube_joint])
        state.data.qpos[cube_address:cube_address + 3] = [hand[0], hand[1], hand[2] - 0.06]
        mujoco.mj_forward(state.model, state.data)
        state.is_holding_cube = lambda: True
        state.gripper_closed = True
        state.latest_gesture = "Closed_Fist"
        state.handle_gesture({"command": "down", "controlToken": "test"})
        state.action_data.qpos[:] = state.data.qpos
        state.action_data.qpos[:7] = state.data.ctrl[:7]
        mujoco.mj_forward(state.model, state.action_data)
        self.assertLess(state.action_data.xpos[state.hand_id][2], hand[2])

    def test_pointing_locks_arm_until_next_gesture_finalizes_yaw(self):
        state = controller()
        state.handle_gesture({"command": "gesture_state", "gesture": "Pointing_Up", "pointingControlActive": True, "handVisible": True, "sentAt": 1})
        self.assertIsNotNone(state.pointing_arm_qpos)
        held_targets = state.data.ctrl[:7].copy()
        state.handle_gesture({"command": "gesture_state", "gesture": "Closed_Fist", "handVisible": True, "sentAt": 2})
        state.handle_gesture({"command": "down", "controlToken": "test"})
        state.handle_gesture({"command": "close", "controlToken": "test", "sentAt": 2})
        np.testing.assert_array_equal(state.data.ctrl[:7], held_targets)
        self.assertEqual(state.data.ctrl[7], 255)
        self.assertIsNotNone(state.pointing_arm_qpos)
        state.handle_gesture({"command": "finalize_gripper_yaw", "angle": 0.4, "controlToken": "test"})
        self.assertIsNone(state.pointing_arm_qpos)
        self.assertTrue(state.gripper_orientation_finalized)
        state.handle_gesture({"command": "gesture_state", "gesture": "Pointing_Up", "pointingControlActive": True, "handVisible": True, "sentAt": 3})
        self.assertIsNone(state.pointing_arm_qpos)
        state.handle_gesture({"command": "set_gripper_yaw", "angle": -0.4, "controlToken": "test"})
        self.assertAlmostEqual(state.gripper_yaw, 0.4)
        state.gripper_ready_to_grasp = lambda: True
        state.handle_gesture({"command": "close", "controlToken": "test", "sentAt": 3})
        state.handle_gesture({"command": "down", "controlToken": "test"})
        state.grasp_close_started_at = float(state.data.time) - 0.4
        self.assertFalse(np.array_equal(state.data.ctrl[:7], held_targets))

    def test_none_cannot_start_pointing_or_rotate_a_closed_grasp(self):
        state = controller()
        state.handle_gesture({"command": "gesture_state", "gesture": "None", "pointingControlActive": True, "handVisible": True, "sentAt": 1})
        self.assertIsNone(state.pointing_arm_qpos)
        state.gripper_ready_to_grasp = lambda: True
        state.handle_gesture({"command": "close", "controlToken": "test", "sentAt": 2})
        self.assertTrue(state.gripper_closed)
        self.assertTrue(state.gripper_orientation_finalized)
        initial_yaw = state.data.qpos[state.gripper_yaw_qpos_address]
        state.handle_gesture({"command": "gesture_state", "gesture": "Pointing_Up", "pointingControlActive": True, "handVisible": True, "sentAt": 3})
        state.handle_gesture({"command": "set_gripper_yaw", "angle": 0.54, "controlToken": "test"})
        self.assertIsNone(state.pointing_arm_qpos)
        self.assertAlmostEqual(state.data.qpos[state.gripper_yaw_qpos_address], initial_yaw)
        self.assertAlmostEqual(state.data.ctrl[state.gripper_yaw_actuator_id], initial_yaw)

    def test_first_motion_freezes_default_orientation_before_grasp(self):
        state = controller()
        state.handle_gesture({"command": "start_vertical", "controlToken": "test"})
        self.assertTrue(state.gripper_orientation_finalized)
        self.assertTrue(state.gripper_orientation_locked)
        state.handle_gesture({"command": "gesture_state", "gesture": "Pointing_Up", "pointingControlActive": True, "handVisible": True, "sentAt": 1})
        state.handle_gesture({"command": "set_gripper_yaw", "angle": 0.54, "controlToken": "test"})
        self.assertIsNone(state.pointing_arm_qpos)
        self.assertAlmostEqual(state.gripper_yaw, 0.0)

    def test_explicit_diagonal_move_works_below_transport_height(self):
        for send_entry in (False, True):
            with self.subTest(send_entry=send_entry):
                state = controller()
                state.is_holding_cube = lambda: True
                state._transport_cube_height = lambda: 10.0
                state.gripper_closed = True
                before = state._commanded_hand_position()
                state.vertical_anchor_xy = before[:2].copy()
                if send_entry:
                    state.handle_gesture({"command": "start_planar", "controlToken": "test"})
                state.handle_gesture({
                    "command": "move_xy", "dx": 0.5, "dy": 0.5,
                    "lockZ": True, "controlToken": "test",
                })
                after = state._commanded_hand_position()
                self.assertTrue(state.z_locked)
                self.assertIsNone(state.vertical_anchor_xy)
                self.assertGreater(np.linalg.norm(after[:2] - before[:2]), 0.002)
                self.assertLess(abs(after[2] - before[2]), 0.002)


if __name__ == "__main__":
    unittest.main()
