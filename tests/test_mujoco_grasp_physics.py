"""End-to-end grasp checks against MuJoCo contacts and the HTTP close gate."""

import math
import queue
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from simulation.mujoco.record_mujoco_panda import (
    INITIAL_ARM_QPOS,
    GestureServer,
    MujocoPandaRecorder,
    parse_args,
)


def arm_pose_for_hand(state, destination):
    """Solve the same fixed-orientation approach used by the live controller."""
    trial = mujoco.MjData(state.model)
    q = INITIAL_ARM_QPOS.copy()
    jacp = np.zeros((3, state.model.nv))
    jacr = np.zeros_like(jacp)
    limits = state.model.jnt_range[:7]
    desired_rotation = state.initial_hand_rotation_in_base
    for _ in range(220):
        trial.qpos[:7] = q
        mujoco.mj_forward(state.model, trial)
        position_error = destination - trial.xpos[state.hand_id]
        rotation = trial.xmat[state.hand_id].reshape(3, 3)
        rotation_error = 0.5 * sum(
            np.cross(rotation[:, axis], desired_rotation[:, axis])
            for axis in range(3)
        )
        if np.linalg.norm(position_error) < 0.0003 and np.linalg.norm(rotation_error) < 0.004:
            return q
        mujoco.mj_jacBody(state.model, trial, jacp, jacr, state.hand_id)
        jacobian = np.vstack((jacp[:, :7], jacr[:, :7]))
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 0.002 * np.eye(6),
            np.r_[position_error, rotation_error],
        )
        q = np.clip(q + np.clip(delta, -0.06, 0.06), limits[:, 0] + 0.03, limits[:, 1] - 0.03)
    raise AssertionError("The grasp pose must be reachable")


def grasp_case(half, yaw, offset, *, hand_z=None, through_http=False,
               descending=False, manual_descend=False):
    with patch.object(sys, "argv", ["grasp-physics-test"]):
        args = parse_args()
    args.random_seed = 1
    with patch.object(MujocoPandaRecorder, "_start_render_workers", lambda self: None):
        state = MujocoPandaRecorder(args)

    model, data = state.model, state.data
    mujoco.mj_resetData(model, data)
    state._set_box_half_size(state.cube_id, state.cube_geom_id, half)
    cube_address = state.cube_qpos_address
    data.qpos[cube_address:cube_address + 3] = [0.5, 0.0, 0.4 + half + 0.002]
    data.qpos[cube_address + 3:cube_address + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    for body_id, xy in zip(state.distractor_ids, [(0.38, 0.23), (0.68, 0.22), (0.68, -0.22)]):
        geom_id = int(model.body_geomadr[body_id])
        other_half = float(model.geom_size[geom_id, 2])
        address = int(model.jnt_qposadr[model.body_jntadr[body_id]])
        data.qpos[address:address + 3] = [*xy, 0.4 + other_half + 0.002]
        data.qpos[address + 3:address + 7] = [1, 0, 0, 0]
    model.body_pos[state.target_plate_id][:2] = [0.38, -0.23]
    mujoco.mj_forward(model, data)
    if hand_z is None:
        hand_z = state._grasp_target_hand_z() + 0.08
    q = arm_pose_for_hand(state, np.array([0.5, offset, hand_z]))
    data.qpos[:7] = q
    data.qpos[state.gripper_yaw_qpos_address] = 0
    data.qpos[state.finger_qpos_addresses] = [0.05, 0.05]
    data.ctrl[:7] = q
    data.ctrl[7] = 255
    data.ctrl[state.gripper_yaw_actuator_id] = 0
    mujoco.mj_forward(model, data)
    state.control_paused = False
    state.control_token = "test"
    state.gripper_orientation_locked = True
    state.gripper_orientation_finalized = True
    state.gripper_yaw = 0
    state.pointing_arm_qpos = None
    state.z_locked = False
    state.locked_z = None
    state.gripper_closed = False
    state.pending_vertical_steps = 0
    for _ in range(300):
        mujoco.mj_step(model, data)
    state.enforce_layout_support()
    mujoco.mj_forward(model, data)
    initial_hand = data.xpos[state.hand_id].copy()
    initial_cube_before_descent = data.xpos[state.cube_id].copy()
    max_manual_descent_jump = 0.0
    if manual_descend:
        state.handle_gesture({"command": "start_vertical", "controlToken": "test"})
        for _ in range(15):
            state.handle_gesture({"command": "down", "controlToken": "test"})
            for _ in range(60):
                mujoco.mj_step(model, data)
            state.hard_lock_gripper_yaw()
            state.enforce_layout_support()
            max_manual_descent_jump = max(
                max_manual_descent_jump,
                float(data.xpos[state.cube_id][2] - initial_cube_before_descent[2]),
            )
        for _ in range(300):
            mujoco.mj_step(model, data)
            state.enforce_layout_support()
    if descending:
        for _ in range(5):
            state.move_hand(0, 0, -0.008)
            for _ in range(10):
                mujoco.mj_step(model, data)
            state.hard_lock_gripper_yaw()
            state.enforce_layout_support()

    hand_before_close = data.xpos[state.hand_id].copy()
    status = state.get_runtime_status()
    arm_target_before_close = data.ctrl[:7].copy()
    if through_http:
        commands = queue.Queue()
        server = GestureServer(
            commands, 0, lambda: None, lambda: None, lambda: None,
            state.get_runtime_status,
        )
        try:
            response = server.server.app.test_client().post(
                "/control", json={"command": "close", "controlToken": "test", "sentAt": 1.0},
            )
            assert response.status_code == 200, response.get_json()
            state.handle_gesture(commands.get_nowait())
        finally:
            server.server.server_close()
    else:
        state.handle_gesture({"command": "close", "controlToken": "test", "sentAt": 1.0})

    initial_assist_phase = None
    initial_gripper_ctrl = float(data.ctrl[7])
    arm_target_after_close = data.ctrl[:7].copy()
    initial_cube = data.xpos[state.cube_id].copy()
    max_preclose_jump = 0.0
    max_preclose_drift = 0.0
    minimum_bottom = float("inf")
    bilateral_contact = False
    close_finished = False
    up_count = 0
    for tick in range(450):
        if close_finished and tick % 6 == 0 and up_count < 12:
            state.handle_gesture({"command": "up", "controlToken": "test", "sentAt": 2.0 + tick * 0.02})
            up_count += 1
        state.maintain_gripper_orientation()
        for _ in range(16):
            mujoco.mj_step(model, data)
        state.hard_lock_gripper_yaw()
        state.enforce_layout_support()
        cube_pos = data.xpos[state.cube_id]
        rotation = data.xmat[state.cube_id].reshape(3, 3)
        vertical_extent = float(np.abs(rotation[2]) @ model.geom_size[state.cube_geom_id])
        minimum_bottom = min(minimum_bottom, float(cube_pos[2] - vertical_extent))
        if data.ctrl[7] == 255:
            max_preclose_jump = max(max_preclose_jump, float(cube_pos[2] - initial_cube[2]))
            max_preclose_drift = max(max_preclose_drift, float(np.linalg.norm(cube_pos[:2] - initial_cube[:2])))
        if state.gripper_closed and float(data.time) - state.grasp_close_started_at >= 0.35:
            close_finished = True
        touching_fingers = set()
        force = np.zeros(6)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if state.cube_geom_id == contact.geom1:
                other = contact.geom2
            elif state.cube_geom_id == contact.geom2:
                other = contact.geom1
            else:
                continue
            body_id = int(model.geom_bodyid[other])
            if body_id in state.finger_body_ids:
                mujoco.mj_contactForce(model, data, contact_index, force)
                if force[0] > 0.01:
                    touching_fingers.add(body_id)
        bilateral_contact |= len(touching_fingers) == 2
        if close_finished and up_count == 12 and data.time > 5:
            break
    return {
        "status": status,
        "state": state,
        "initial_assist_phase": initial_assist_phase,
        "initial_gripper_ctrl": initial_gripper_ctrl,
        "arm_target_before_close": arm_target_before_close,
        "arm_target_after_close": arm_target_after_close,
        "initial_cube": initial_cube,
        "initial_hand": initial_hand,
        "hand_before_close": hand_before_close,
        "max_manual_descent_jump": max_manual_descent_jump,
        "max_preclose_jump": max_preclose_jump,
        "max_preclose_drift": max_preclose_drift,
        "minimum_bottom": minimum_bottom,
        "bilateral_contact": bilateral_contact,
        "close_finished": close_finished,
    }


def stacked_grasp_case(seed, *, approach_offset=0.08, manual_descend=False, red_below=False):
    with patch.object(sys, "argv", ["stacked-grasp-test"]):
        args = parse_args()
    args.random_seed = seed
    args.stack_probability = 1.0
    with patch.object(MujocoPandaRecorder, "_start_render_workers", lambda self: None):
        state = MujocoPandaRecorder(args)
    model, data = state.model, state.data
    for _ in range(30):
        if bool(state.current_layout[0]["stacked_on"]) != red_below:
            break
        state.reset_simulation()
    base_name = state.current_layout[0]["stacked_on"]
    if red_below:
        base_name = next(entry["name"] for entry in state.current_layout if entry.get("stacked_on") == "cube")
    assert base_name is not None
    base_id = state._body_id(base_name)
    for _ in range(1200):
        mujoco.mj_step(model, data)
        state.enforce_layout_support()
    cube = data.xpos[state.cube_id].copy()
    initial_support = state._cube_support_height(cube[:2])
    approach_z = state._grasp_target_hand_z() + approach_offset
    q = arm_pose_for_hand(state, np.array([*cube[:2], approach_z]))
    data.qpos[:7] = q
    data.ctrl[:7] = q
    data.qpos[state.finger_qpos_addresses] = [0.05, 0.05]
    data.ctrl[7] = 255
    data.qpos[state.gripper_yaw_qpos_address] = 0
    data.ctrl[state.gripper_yaw_actuator_id] = 0
    mujoco.mj_forward(model, data)
    state.control_paused = False
    state.control_token = "test"
    state.gripper_orientation_locked = True
    state.gripper_orientation_finalized = True
    state.gripper_yaw = 0
    state.pointing_arm_qpos = None
    state.z_locked = False
    state.locked_z = None
    state.gripper_closed = False
    state.pending_vertical_steps = 0
    for _ in range(300):
        mujoco.mj_step(model, data)
        state.enforce_layout_support()
    mujoco.mj_forward(model, data)
    initial_cube = data.xpos[state.cube_id].copy()
    initial_base = data.xpos[base_id].copy()
    initial_hand = data.xpos[state.hand_id].copy()
    max_manual_descent_jump = 0.0
    if manual_descend:
        state.handle_gesture({"command": "start_vertical", "controlToken": "test"})
        for tick in range(15):
            state.handle_gesture({"command": "down", "controlToken": "test"})
            for _ in range(60):
                mujoco.mj_step(model, data)
            state.hard_lock_gripper_yaw()
            state.enforce_layout_support()
            max_manual_descent_jump = max(
                max_manual_descent_jump,
                float(data.xpos[state.cube_id][2] - initial_cube[2]),
            )
        for _ in range(300):
            mujoco.mj_step(model, data)
            state.enforce_layout_support()
    hand_before_close = data.xpos[state.hand_id].copy()
    arm_target_before_close = data.ctrl[:7].copy()
    state.handle_gesture({"command": "close", "controlToken": "test", "sentAt": 1})
    arm_target_after_close = data.ctrl[:7].copy()
    initial_phase = None
    max_preclose_cube_jump = 0.0
    max_preclose_cube_drift = 0.0
    max_base_displacement = 0.0
    bilateral_contact = False
    locked_before_lift = False
    close_finished = False
    up_count = 0
    for tick in range(550):
        if close_finished and tick % 6 == 0 and up_count < 18:
            state.handle_gesture({"command": "up", "controlToken": "test", "sentAt": 2 + tick * 0.02})
            up_count += 1
        state.maintain_gripper_orientation()
        for _ in range(16):
            mujoco.mj_step(model, data)
        state.hard_lock_gripper_yaw()
        state.enforce_layout_support()
        state.update_transport_state()
        cube_now = data.xpos[state.cube_id]
        if data.ctrl[7] == 255:
            max_preclose_cube_jump = max(max_preclose_cube_jump, float(cube_now[2] - initial_cube[2]))
            max_preclose_cube_drift = max(max_preclose_cube_drift, float(np.linalg.norm(cube_now[:2] - initial_cube[:2])))
        max_base_displacement = max(max_base_displacement, float(np.linalg.norm(data.xpos[base_id] - initial_base)))
        if state.z_locked and cube_now[2] < initial_cube[2] + 0.045:
            locked_before_lift = True
        if state.gripper_closed and float(data.time) - state.grasp_close_started_at >= 0.35:
            close_finished = True
        touching_fingers = set()
        force = np.zeros(6)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if state.cube_geom_id == contact.geom1:
                other = contact.geom2
            elif state.cube_geom_id == contact.geom2:
                other = contact.geom1
            else:
                continue
            body_id = int(model.geom_bodyid[other])
            if body_id in state.finger_body_ids:
                mujoco.mj_contactForce(model, data, contact_index, force)
                if force[0] > 0.01:
                    touching_fingers.add(body_id)
        bilateral_contact |= len(touching_fingers) == 2
        if close_finished and up_count == 18 and data.time > 7:
            break
    return {
        "state": state,
        "base_id": base_id,
        "initial_cube": initial_cube,
        "initial_hand": initial_hand,
        "initial_support": initial_support,
        "initial_phase": initial_phase,
        "hand_before_close": hand_before_close,
        "max_manual_descent_jump": max_manual_descent_jump,
        "arm_target_before_close": arm_target_before_close,
        "arm_target_after_close": arm_target_after_close,
        "max_preclose_cube_jump": max_preclose_cube_jump,
        "max_preclose_cube_drift": max_preclose_cube_drift,
        "max_base_displacement": max_base_displacement,
        "bilateral_contact": bilateral_contact,
        "locked_before_lift": locked_before_lift,
        "close_finished": close_finished,
    }


class GraspPhysicsTest(unittest.TestCase):
    def test_red_below_larger_block_can_be_manually_grasped(self):
        for seed in (0, 4, 8):
            with self.subTest(seed=seed):
                result = stacked_grasp_case(seed, manual_descend=True, red_below=True)
                state = result["state"]
                self.assertEqual(state.layout_reachability["mode"], "red_below")
                upper = next(e for e in state.current_layout if e.get("stacked_on") == "cube")
                self.assertGreaterEqual(upper["half_size_m"], 0.014)
                self.assertLessEqual(upper["half_size_m"], 0.016)
                self.assertAlmostEqual(result["initial_support"], state.args.table_top_z)
                self.assertTrue(result["bilateral_contact"])
                self.assertTrue(state.is_holding_cube())
                self.assertGreater(state.data.xpos[state.cube_id][2] - result["initial_cube"][2], 0.05)
                np.testing.assert_array_equal(result["arm_target_before_close"], result["arm_target_after_close"])

    def test_manual_close_grasps_centered_table_and_stacked_boxes(self):
        table_hand_z = 0.4 + 0.025 + 0.1034 + 0.001
        table = grasp_case(0.025, 0, 0, hand_z=table_hand_z + 0.08,
                           through_http=True, manual_descend=True)
        self.assertTrue(table["status"]["grasp_ready"])
        self.assertEqual(table["initial_gripper_ctrl"], 0)
        np.testing.assert_array_equal(
            table["arm_target_before_close"], table["arm_target_after_close"]
        )
        self.assertTrue(table["bilateral_contact"])
        self.assertTrue(table["state"].is_holding_cube())
        self.assertLess(table["hand_before_close"][2] - table["initial_hand"][2], -0.05)
        self.assertLess(table["max_manual_descent_jump"], 0.005)
        self.assertGreater(
            table["state"].data.xpos[table["state"].cube_id][2] - table["initial_cube"][2],
            0.06,
        )

        for seed in (0, 3):
            with self.subTest(seed=seed):
                stacked = stacked_grasp_case(
                    seed, approach_offset=0.08, manual_descend=True
                )
                state = stacked["state"]
                np.testing.assert_array_equal(
                    stacked["arm_target_before_close"], stacked["arm_target_after_close"]
                )
                self.assertTrue(stacked["bilateral_contact"])
                self.assertTrue(state.is_holding_cube())
                self.assertLess(
                    stacked["hand_before_close"][2] - stacked["initial_hand"][2], -0.05
                )
                self.assertLess(stacked["max_manual_descent_jump"], 0.005)
                self.assertGreater(
                    state.data.xpos[state.cube_id][2] - stacked["initial_cube"][2], 0.08
                )
                self.assertLess(stacked["max_base_displacement"], 0.002)

    def test_manual_offset_commands_do_not_raise_or_recenter_the_hand(self):
        with patch.object(sys, "argv", ["manual-grasp-test"]):
            args = parse_args()
        args.random_seed = 1
        with patch.object(MujocoPandaRecorder, "_start_render_workers", lambda self: None):
            state = MujocoPandaRecorder(args)
        model, data = state.model, state.data
        cube_address = state.cube_qpos_address
        data.qpos[cube_address:cube_address + 3] = [0.5, 0, 0.425]
        state._set_box_half_size(state.cube_id, state.cube_geom_id, 0.025)
        hand_xy = np.array([0.5, 0.0256])
        q = arm_pose_for_hand(state, np.array([*hand_xy, 0.51]))
        data.qpos[:7] = q
        data.ctrl[:7] = q
        mujoco.mj_forward(model, data)
        state.control_paused = False
        state.control_token = "test"
        state.gripper_orientation_locked = True
        state.gripper_orientation_finalized = True
        before = state._commanded_hand_position()

        state.move_hand(0, 0, 0)
        after_idle = state._commanded_hand_position()
        self.assertLess(abs(float(after_idle[2] - before[2])), 0.001)
        self.assertLess(float(np.linalg.norm(after_idle[:2] - before[:2])), 0.001)
        state.move_hand(0.008, 0, 0)
        after_planar = state._commanded_hand_position()
        self.assertLess(abs(float(after_planar[2] - after_idle[2])), 0.001)
        self.assertLess(float(np.linalg.norm(after_planar[:2] - (before[:2] + [0.008, 0]))), 0.005)
        self.assertIsNone(state.motion_limit_reason)

        arm_before_close = data.ctrl[:7].copy()
        commands = queue.Queue()
        server = GestureServer(
            commands, 0, lambda: None, lambda: None, lambda: None,
            state.get_runtime_status,
        )
        try:
            status = server.server.app.test_client().get("/health").get_json()
            self.assertTrue(status["grasp_ready"])
            for field in ("recording", "saving_episode", "holding_cube", "gripper_yaw_finalized", "paused"):
                self.assertIsInstance(status[field], bool)
            response = server.server.app.test_client().post(
                "/control", json={"command": "close", "controlToken": "test", "sentAt": 1},
            )
            self.assertEqual(response.status_code, 200)
            state.handle_gesture(commands.get_nowait())
        finally:
            server.server.server_close()
        self.assertEqual(data.ctrl[7], 0)
        np.testing.assert_array_equal(data.ctrl[:7], arm_before_close)
        state.handle_gesture({"command": "down", "controlToken": "test", "sentAt": 1.1})
        self.assertEqual(state.pending_vertical_steps, 0)
        self.assertIsNotNone(state.motion_limit_reason)
        np.testing.assert_allclose(data.ctrl[:7], arm_before_close, atol=2e-5)

        state.handle_gesture({"command": "open", "controlToken": "test", "sentAt": 2})
        state.move_hand(0, 0, -0.008)
        self.assertIsNotNone(state.motion_limit_reason)
        self.assertEqual(
            state.get_runtime_status()["motion_limit_reason"], state.motion_limit_reason
        )
        self.assertLessEqual(
            float(state._commanded_hand_position()[2]), float(after_planar[2]) + 0.001
        )


    def test_enlarged_cube_keeps_two_finger_contacts_at_multiple_angles(self):
        for yaw in (0, math.pi / 4, math.pi / 2):
            with self.subTest(yaw=yaw):
                result = grasp_case(0.034, yaw, 0, manual_descend=True)
                state = result["state"]
                cube_node = int(state.model.body_bvhadr[state.cube_id])
                np.testing.assert_allclose(state.model.bvh_aabb[cube_node, 3:], 0.034)
                self.assertTrue(result["close_finished"])
                self.assertTrue(result["bilateral_contact"])
                self.assertGreater(state.data.xpos[state.cube_id][2] - result["initial_cube"][2], 0.05)
                self.assertTrue(state.is_holding_cube())




    def test_stacked_cube_can_be_carried_and_released_on_target_plate(self):
        result = stacked_grasp_case(2, manual_descend=True)
        state = result["state"]
        model, data = state.model, state.data
        base_start = data.xpos[result["base_id"]].copy()
        self.assertTrue(state.is_holding_cube())
        state.handle_gesture({"command": "start_planar", "controlToken": "test", "sentAt": 30})
        self.assertTrue(state.z_locked)

        for tick in range(140):
            direction = state.current_target_xy - data.xpos[state.cube_id][:2]
            distance = float(np.linalg.norm(direction))
            if distance < 0.027:
                break
            step = direction / distance * min(0.9, distance / 0.04)
            state.handle_gesture({
                "command": "move_xy", "controlToken": "test", "sentAt": 31 + tick * 0.02,
                "dx": float(step[0]), "dy": float(step[1]), "lockZ": True,
            })
            state.maintain_locked_plane()
            for _ in range(16):
                mujoco.mj_step(model, data)
            state.hard_lock_gripper_yaw()
            state.enforce_layout_support()
            state.update_target_state()
            self.assertTrue(state.is_holding_cube())
            self.assertGreater(
                float(data.xpos[state.cube_id][2]), state._transport_cube_height() - 0.02
            )
        self.assertLess(distance, 0.027)

        for _ in range(300):
            state.maintain_locked_plane()
            for _ in range(10):
                mujoco.mj_step(model, data)
            state.hard_lock_gripper_yaw()
            state.enforce_layout_support()
            state.update_target_state()
        self.assertTrue(state.cube_over_target)
        self.assertFalse(state.z_locked)
        state.handle_gesture({"command": "open", "controlToken": "test", "sentAt": 100})
        for _ in range(1000):
            state.maintain_gripper_orientation()
            for _ in range(5):
                mujoco.mj_step(model, data)
            state.hard_lock_gripper_yaw()
            state.enforce_layout_support()
            state.update_target_state()

        self.assertLess(
            float(np.linalg.norm(state.current_target_xy - data.xpos[state.cube_id][:2])),
            state.args.target_radius,
        )
        half = float(model.geom_size[state.cube_geom_id][2])
        self.assertAlmostEqual(
            float(data.xpos[state.cube_id][2]), state.args.target_top_z + half, delta=0.003
        )
        self.assertLess(
            float(np.linalg.norm(data.xpos[result["base_id"]] - base_start)), 0.002
        )


if __name__ == "__main__":
    unittest.main()
