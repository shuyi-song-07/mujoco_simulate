"""Headless physics and safety acceptance for Task 3 (unittest, no new deps)."""
from __future__ import annotations

import unittest

import mujoco
import numpy as np

from simulation.mujoco.dual_arm.dual_arm_task_controller import DualArmTaskController, scripted_demo
from simulation.mujoco.dual_arm.config import config_hash, load_config


class StopAtGrasp(Exception):
    pass


def verified_grasp(controller):
    def stop(c):
        if c.task_state == "DUAL_GRASPED":
            raise StopAtGrasp
    try:
        scripted_demo(controller, on_step=stop)
    except StopAtGrasp:
        pass
    if controller.task_state != "DUAL_GRASPED":
        raise AssertionError("Physics pilot did not form a dual grasp")


class DualArmControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = DualArmTaskController()

    def test_namespaced_model_and_collision_free_home(self):
        c = self.controller
        for side in ("left", "right"):
            for index in range(1, 8):
                self.assertGreaterEqual(c.model.joint(f"{side}_joint{index}").id, 0)
            for index in range(1, 9):
                self.assertGreaterEqual(c.model.actuator(f"{side}_actuator{index}").id, 0)
            self.assertGreaterEqual(c.model.site(f"{side}_ee").id, 0)
        for camera in ("top", "front", "side"):
            self.assertGreaterEqual(c.model.camera(camera).id, 0)
        self.assertEqual(c.model.nu, 16)
        self.assertFalse(c._robot_robot_collision())
        self.assertFalse(any(c.model.eq_type == mujoco.mjtEq.mjEQ_WELD))
        c.step(2.0)
        self.assertEqual(c.metrics.collision_count, 0)
        self.assertEqual(c.state_vector().shape, (16,))
        self.assertEqual(c.action_vector().shape, (14,))

    def test_actuators_are_independent(self):
        c = self.controller
        right_before = c.data.ctrl[c.right.actuator_ids].copy()
        left_before = c.data.ctrl[c.left.actuator_ids].copy()
        c.left.move_delta_task_frame(0.003, 0, 0)
        np.testing.assert_array_equal(c.data.ctrl[c.right.actuator_ids], right_before)
        self.assertGreater(np.linalg.norm(c.data.ctrl[c.left.actuator_ids]-left_before), 1e-6)
        left_after = c.data.ctrl[c.left.actuator_ids].copy()
        c.right.move_delta_task_frame(0, 0.003, 0)
        np.testing.assert_array_equal(c.data.ctrl[c.left.actuator_ids], left_after)

    def test_stale_motion_and_stale_gripper_are_rejected(self):
        c = self.controller
        before = c.data.ctrl.copy()
        motion = {"command": "dual_motion", "mode": "xy", "sentAt": 0, "left": {"dx": 1}, "right": {"dx": 1}}
        result = c.apply_command(motion, now_ms=1000)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "stale_command")
        result = c.apply_command({"command": "dual_gripper", "action": "close", "sentAt": 0}, now_ms=1000)
        self.assertFalse(result["accepted"])
        np.testing.assert_array_equal(before, c.data.ctrl)

    def test_atomic_gripper_one_tick_latch_and_duplicate_event(self):
        c = self.controller
        moment = c.data.time
        event = {"command": "dual_gripper", "action": "close", "sentAt": 0, "eventId": "test-close"}
        self.assertTrue(c.apply_command(event, now_ms=0)["accepted"])
        self.assertEqual(c.data.time, moment)
        self.assertEqual(c.data.ctrl[c.left.gripper_actuator_id], 0)
        self.assertEqual(c.data.ctrl[c.right.gripper_actuator_id], 0)
        self.assertTrue(c.apply_command(event, now_ms=0)["accepted"])
        self.assertEqual(c.metrics.gripper_event_count, 1)
        c.apply_command({"command": "pause", "sentAt": 1}, now_ms=1)
        self.assertEqual(c.gripper_latch, "CLOSED")
        self.assertEqual(c.data.ctrl[c.left.gripper_actuator_id], 0)

    def test_xy_z_modes_are_reversible_and_do_not_change_gripper(self):
        c = self.controller
        before = c.left.command_position.copy()
        c.apply_command({"command": "dual_motion", "mode": "xy", "sentAt": 0, "left": {"dx": 0.5, "dz": 1}, "right": {"dx": 0.5, "dz": 1}}, now_ms=0)
        self.assertAlmostEqual(c.left.command_position[2], before[2])
        c.apply_command({"command": "dual_motion", "mode": "z", "sentAt": 1, "left": {"dx": 1, "dz": -0.5}, "right": {"dx": 1, "dz": -0.5}}, now_ms=1)
        self.assertLess(c.left.command_position[2], before[2])
        self.assertEqual(c.control_mode, "Z")
        self.assertEqual(c.gripper_latch, "OPEN")

    def test_watchdog_freezes_motion_without_releasing(self):
        c = self.controller
        c.apply_command({"command": "dual_gripper", "action": "close", "sentAt": 0}, now_ms=0)
        c.apply_command({"command": "pause", "sentAt": 1, "reason": "hand_lost"}, now_ms=1)
        c.step(0.3)
        self.assertEqual(c.pause_reason, "hand_lost")
        self.assertEqual(c.gripper_latch, "CLOSED")
        c.reset()
        c.apply_command({"command": "dual_motion", "mode": "xy", "sentAt": 0, "left": {}, "right": {}}, now_ms=0)
        c.step(0.6)
        c.step()
        self.assertTrue(c.paused)
        self.assertEqual(c.pause_reason, "command_watchdog")
        self.assertEqual(c.gripper_latch, "OPEN")

    def test_wrong_grasp_fails_and_open_reset_recover(self):
        c = self.controller
        c.apply_command({"command": "dual_gripper", "action": "close", "sentAt": 0}, now_ms=0)
        c.step(1.4)
        self.assertEqual(c.task_state, "FAIL")
        self.assertEqual(c.failure_reason, "grasp_failure")
        self.assertTrue(c.get_status()["release_ready"])
        self.assertTrue(all(np.sum(c.data.qpos[arm.finger_qpos_indices]) < 0.01 for arm in c.arms))
        self.assertTrue(c.apply_command({"command": "dual_gripper", "action": "open", "sentAt": 0}, now_ms=0)["accepted"])
        self.assertEqual(c.gripper_latch, "OPEN")
        c.step(0.4)
        self.assertTrue(all(np.sum(c.data.qpos[arm.finger_qpos_indices]) > 0.07 for arm in c.arms))
        self.assertEqual(c.task_state, "FAIL")
        self.assertEqual(c.failure_reason, "grasp_failure")
        self.assertTrue(c.paused)
        c.reset()
        self.assertEqual(c.task_state, "PREGRASP")
        self.assertIsNone(c.failure_reason)

    def test_preview_idle_does_not_timeout_and_control_clock_does_not_drift(self):
        c = self.controller
        c.step(c.config["evaluation"]["max_episode_seconds"]+1)
        self.assertEqual(c.task_state, "PREGRASP")
        self.assertFalse(c.get_status()["task_timer_started"])
        c.reset()
        for _ in range(60):
            c.step()
        self.assertAlmostEqual(c.data.time, 1.0, places=6)
        c.start_task_timer()
        c.step(c.config["evaluation"]["max_episode_seconds"]+0.01)
        self.assertEqual(c.failure_reason, "timeout")

    def test_stationary_telemetry_keeps_mode_and_confirmation_holds_latch(self):
        c = self.controller
        hands = {side: {"visible": True} for side in ("left", "right")}
        for _ in range(40):
            c.update_telemetry({"control_mode": "XY", "calibrated": True, "hands": hands})
            c.step(0.04)
        self.assertEqual(c.control_mode, "XY")
        self.assertFalse(c.paused)
        self.assertEqual(c.gripper_latch, "OPEN")
        c.update_telemetry({"control_mode": "GRASP_CONFIRM", "calibrated": True, "hands": hands})
        self.assertEqual(c.control_mode, "GRASP_CONFIRM")
        self.assertTrue(c.paused)
        self.assertFalse(c.training_recordable)
        self.assertEqual(c.gripper_latch, "OPEN")
        c.update_telemetry({"control_mode": "Z", "calibrated": True, "hands": hands})
        self.assertFalse(c.paused)
        c.update_telemetry({"control_mode": "XY", "calibrated": False, "hands": hands})
        self.assertTrue(c.paused)
        self.assertEqual(c.gripper_latch, "OPEN")

    def test_practice_motion_and_pause_do_not_use_episode_deadline(self):
        c = self.controller
        c.set_episode_timeout_enabled(False)
        c.apply_command({"command": "dual_motion", "mode": "xy", "sentAt": 0,
                         "left": {"dx": 0.1}, "right": {"dx": 0.1}}, now_ms=0)
        self.assertTrue(c.get_status()["task_timer_started"])
        c.pause("hand_lost")
        c.step(c.config["evaluation"]["max_episode_seconds"] + 0.1)
        self.assertEqual(c.task_state, "PREGRASP")
        self.assertIsNone(c.failure_reason)
        self.assertEqual(c.gripper_latch, "OPEN")
        self.assertTrue(c.get_status()["practice_mode"])
        self.assertFalse(c.get_status()["episode_timeout_enabled"])
        c.reset()
        self.assertFalse(c.episode_timeout_enabled)

    def test_recording_deadline_starts_fresh_and_counts_paused_time(self):
        c = self.controller
        self.assertTrue(c.episode_timeout_enabled)
        c.set_episode_timeout_enabled(False)
        c.start_task_timer()
        c.data.time += 120  # Older practice time cannot consume the next trial.
        qpos, ctrl = c.data.qpos.copy(), c.data.ctrl.copy()
        c.set_episode_timeout_enabled(True)
        self.assertFalse(c.get_status()["task_timer_started"])
        np.testing.assert_array_equal(c.data.qpos, qpos)
        np.testing.assert_array_equal(c.data.ctrl, ctrl)
        c.start_task_timer()
        self.assertEqual(c.task_elapsed_seconds, 0)
        c.pause("operator_pause")
        c.step(c.config["evaluation"]["max_episode_seconds"] - 0.1)
        self.assertEqual(c.task_state, "PREGRASP")
        c.step(0.2)
        self.assertEqual(c.task_state, "FAIL")
        self.assertEqual(c.failure_reason, "timeout")
        self.assertEqual(c.gripper_latch, "OPEN")
        c.reset()
        self.assertTrue(c.episode_timeout_enabled)

    def test_practice_keeps_physical_grasp_failure_protection(self):
        c = self.controller
        c.set_episode_timeout_enabled(False)
        c.apply_command({"command": "dual_gripper", "action": "close", "sentAt": 0}, now_ms=0)
        c.step(1.4)
        self.assertEqual(c.task_state, "FAIL")
        self.assertEqual(c.failure_reason, "grasp_failure")

    def test_frozen_checkpoint_config_does_not_gain_new_ui_defaults(self):
        frozen = load_config()
        frozen["control"].pop("evidence_decay")
        frozen["control"].pop("gripper_ack_timeout_ms")
        frozen_hash = config_hash(frozen)
        controller = DualArmTaskController(config=frozen)
        self.assertEqual(controller.config_hash, frozen_hash)
        self.assertNotIn("evidence_decay", controller.config["control"])
        partial = load_config({"recording": {"inspection_videos": False}})
        self.assertEqual(partial["control"]["evidence_decay"], 1.5)
        self.assertFalse(partial["recording"]["inspection_videos"])

    def test_gesture_threshold_configuration_and_frozen_legacy_fallback(self):
        current = load_config()
        self.assertEqual(current["control"]["gripper_open_confidence_min"], 0.55)
        self.assertEqual(current["control"]["gripper_close_confidence_min"], 0.80)
        old = load_config()
        old["control"].pop("gripper_open_confidence_min")
        old["control"].pop("gripper_close_confidence_min")
        original_hash = config_hash(old)
        frozen = load_config(old)
        self.assertEqual(config_hash(frozen), original_hash)
        self.assertNotIn("gripper_open_confidence_min", frozen["control"])
        for key in ("gripper_open_confidence_min", "gripper_close_confidence_min"):
            for invalid in (0, -0.1, 60, 1.1, float("nan"), float("inf"), True, "0.6"):
                with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                    load_config({"control": {key: invalid}})

    def test_real_robot_collision_freezes_both(self):
        c = self.controller
        # External fault injection overlaps the two physical base meshes.
        c.model.body_pos[c.right.base_id] = c.model.body_pos[c.left.base_id]
        mujoco.mj_forward(c.model, c.data)
        self.assertTrue(c._robot_robot_collision())
        c.step()
        self.assertEqual(c.failure_reason, "robot_robot_collision")
        self.assertTrue(c.paused)
        self.assertEqual(c.metrics.collision_count, 1)

    def test_real_contact_loss_freezes_with_closed_grippers(self):
        c = self.controller
        verified_grasp(c)
        self.assertTrue(c.contacts["left"] and c.contacts["right"])
        # Move object away only in this deliberate safety fault injection.
        c.data.qpos[c.object_qpos_adr] += 0.3
        mujoco.mj_forward(c.model, c.data)
        c.step(0.04)
        c.step(0.4)
        self.assertEqual(c.failure_reason, "contact_loss")
        self.assertEqual(c.metrics.contact_loss_count, 1)
        self.assertEqual(c.gripper_latch, "CLOSED")

    def test_high_speed_release_is_rejected(self):
        c = self.controller
        verified_grasp(c)
        c.data.qvel[c.object_dof_adr] = 0.3
        result = c.apply_command({"command": "dual_gripper", "action": "open", "sentAt": 0}, now_ms=0)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "release_rejected_stop_before_opening")
        self.assertEqual(c.gripper_latch, "CLOSED")

    def test_common_translation_and_relative_safety(self):
        c = self.controller
        verified_grasp(c)
        before = [arm.command_position.copy() for arm in c.arms]
        result = c.apply_command({"command": "dual_motion", "mode": "xy", "sentAt": 0, "left": {"dx": 1}, "right": {"dx": 0}}, now_ms=0)
        self.assertTrue(result["accepted"])
        changes = [arm.command_position-target for arm, target in zip(c.arms, before)]
        self.assertAlmostEqual((changes[0][0]+changes[1][0])/2, 0.002, places=5)
        self.assertGreater(c.input_disagreement_m, 0)
        c.model.body_pos[c.right.base_id, 0] += 0.08
        mujoco.mj_forward(c.model, c.data)
        c.step()
        self.assertEqual(c.failure_reason, "relative_pose_violation")
        self.assertEqual(c.gripper_latch, "CLOSED")

    def test_failed_established_grasp_release_waits_for_physical_stop(self):
        c = self.controller
        verified_grasp(c)
        # Inject a terminal failure and speed independently of contact dynamics.
        c.pause("relative_pose_violation", fail=True)
        c.data.qvel[c.object_dof_adr] = 0.3
        self.assertFalse(c.get_status()["release_ready"])
        command = {"command": "dual_gripper", "action": "open", "sentAt": 0}
        self.assertEqual(c.apply_command(command, now_ms=0)["reason"], "release_rejected_stop_before_opening")
        self.assertEqual(c.gripper_latch, "CLOSED")
        c.data.qvel[c.object_dof_adr:c.object_dof_adr+6] = 0
        c.ee_speeds = [0, 0]
        self.assertTrue(c.get_status()["release_ready"])
        self.assertTrue(c.apply_command(command, now_ms=0)["accepted"])
        self.assertEqual(c.gripper_latch, "OPEN")
        self.assertEqual(c.task_state, "FAIL")
        self.assertEqual(c.failure_reason, "relative_pose_violation")
        self.assertTrue(c.paused)
        self.assertFalse(c.apply_command({**command, "action": "close"}, now_ms=0)["accepted"])

    def test_scripted_physical_pick_carry_release_success(self):
        c = self.controller
        result = scripted_demo(c)
        self.assertTrue(result["success"])
        self.assertEqual(result["task_state"], "DONE")
        self.assertEqual(result["metrics"]["grasp_verified_count"], 1)
        self.assertEqual(result["metrics"]["gripper_event_count"], 2)
        self.assertGreater(result["metrics"]["peak_object_height_m"], c.config["scene"]["table_top_z"]+0.10)
        self.assertEqual(result["metrics"]["collision_count"], 0)
        self.assertEqual(result["metrics"]["contact_loss_count"], 0)
        self.assertLess(np.linalg.norm(np.asarray(result["object_position"])-result["target_position"]), 0.02)


if __name__ == "__main__":
    unittest.main()
