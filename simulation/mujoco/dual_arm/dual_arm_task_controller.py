"""Real-contact, synchronized Task 3 controller, independent of Task 1/2."""
from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .config import DEFAULT_SCENE_PATH, config_hash, load_config
from .dual_arm_fsm import ControlMode, GripperLatch, TaskState
from .dual_arm_metrics import DualArmMetrics
from .panda_arm_controller import PandaArmController, rotation_angle, rpy_to_matrix


def _numbers(values):
    return " ".join(str(float(value)) for value in values)


def create_model(config=None) -> mujoco.MjModel:
    """Compile the scene with config-controlled geometry, without temp files."""
    config = load_config(config)
    scene = config["scene"]
    root = ET.parse(DEFAULT_SCENE_PATH).getroot()
    # MuJoCo's native Windows file opening cannot reliably handle a Chinese
    # absolute workspace path. VFS mesh bytes retain one original asset copy
    # and also make compilation independent of the caller's current directory.
    mesh_directory = DEFAULT_SCENE_PATH.parent.parent / "franka_emika_panda" / "assets"
    mesh_assets = {node.get("file"): (mesh_directory / node.get("file")).read_bytes() for node in root.findall("asset/mesh")}
    root.find("compiler").set("meshdir", "")
    root.find("option").set("timestep", str(scene["timestep"]))
    root.find("visual/global").set("offwidth", str(config["recording"]["width"]))
    root.find("visual/global").set("offheight", str(config["recording"]["height"]))
    for side in ("left", "right"):
        body = root.find(f".//body[@name='{side}_link0']")
        body.set("pos", _numbers(scene[f"{side}_base_position"]))
        body.set("quat", _numbers(scene[f"{side}_base_quaternion_wxyz"]))
        for robot_body in body.iter("body"):
            robot_body.set("gravcomp", str(scene["gravity_compensation"]))
        gripper_actuator = root.find(f".//actuator/general[@name='{side}_actuator8']")
        stiffness, damping = scene["gripper_stiffness_npm"], scene["gripper_damping_nspm"]
        gripper_actuator.set("gainprm", f"{stiffness * 0.04 / 255} 0 0")
        gripper_actuator.set("biasprm", f"0 {-stiffness} {-damping}")
        gripper_actuator.set("forcerange", f"{-scene['gripper_max_force_n']} {scene['gripper_max_force_n']}")
        for geom in root.findall(".//geom"):
            if geom.get("name", "").startswith(side) and (geom.get("class", "").startswith("fingertip") or geom.get("mesh") == "finger_0" and geom.get("class") == "collision"):
                geom.set("friction", _numbers(scene["finger_friction"]))
                geom.set("solref", "0.005 1")
                geom.set("solimp", "0.95 0.99 0.001")
    table = root.find(".//body[@name='table']")
    table.set("pos", f"0 0 {scene['table_top_z'] / 2}")
    table.find("geom").set("size", _numbers([*scene.get("table_half_size_xy", [0.5, 0.55]), scene['table_top_z'] / 2]))
    object_geom = root.find(".//geom[@name='object_geom']")
    object_geom.set("size", _numbers(scene["object_half_size"]))
    object_geom.set("mass", str(scene["object_mass_kg"]))
    object_geom.set("friction", _numbers(scene["object_friction"]))
    root.find(".//body[@name='cooperative_object']").set("pos", _numbers(scene["object_position"]))
    for index, side in enumerate(("left", "right")):
        root.find(f".//site[@name='{side}_grasp_site']").set("pos", _numbers(scene["grasp_offsets"][index]))
    for name, camera in scene["cameras"].items():
        element = root.find(f".//camera[@name='{name}']")
        element.set("pos", _numbers(camera["position"]))
        # Frozen older configurations explicitly retain perspective even when
        # the generated default scene uses orthographic cameras.
        element.set("projection", camera.get("projection", "perspective"))
        element.set("fovy", str(camera["fovy"]))
        if "xyaxes" in camera:
            element.set("xyaxes", _numbers(camera["xyaxes"]))
        if "target" in camera:
            root.find(f".//body[@name='{name}_camera_target']").set("pos", _numbers(camera["target"]))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets=mesh_assets)


class DualArmTaskController:
    def __init__(self, config=None, model=None, data=None):
        self.config = load_config(config)
        self.config_hash = config_hash(self.config)
        self.model = create_model(self.config) if model is None else model
        self.data = mujoco.MjData(self.model) if data is None else data
        self.left = PandaArmController(self.model, self.data, "left", self.config)
        self.right = PandaArmController(self.model, self.data, "right", self.config)
        self.arms = (self.left, self.right)
        # Policies and scripted evaluations keep the configured deadline by
        # default. Only the interactive recorder explicitly opts into practice.
        self.episode_timeout_enabled = True
        self.object_id = self.model.body("cooperative_object").id
        self.object_geom_id = self.model.geom("object_geom").id
        self.target_id = self.model.body("target").id
        self.object_joint_id = self.model.joint("object_freejoint").id
        self.object_qpos_adr = int(self.model.jnt_qposadr[self.object_joint_id])
        self.object_dof_adr = int(self.model.jnt_dofadr[self.object_joint_id])
        self.robot_geoms = {}
        self.finger_geoms = {}
        for side in ("left", "right"):
            self.robot_geoms[side] = {index for index in range(self.model.ngeom) if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith(f"{side}_")}
            self.finger_geoms[side] = []
            for finger in ("left_finger", "right_finger"):
                body_id = self.model.body(f"{side}_{finger}").id
                self.finger_geoms[side].append(set(np.flatnonzero(self.model.geom_bodyid == body_id)))
        self.reset()

    def reset(self, seed=0):
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        mujoco.mj_resetData(self.model, self.data)
        scene = self.config["scene"]
        randomization = scene["randomization"]
        object_position = np.asarray(scene["object_position"], dtype=float).copy()
        object_position[:2] += rng.uniform(-np.asarray(randomization["object_xy_range_m"]), np.asarray(randomization["object_xy_range_m"]))
        object_position[2] = scene["table_top_z"] + scene["object_half_size"][2] + 0.0002
        yaw = float(rng.uniform(-randomization["object_yaw_range_rad"], randomization["object_yaw_range_rad"]))
        self.data.qpos[self.object_qpos_adr:self.object_qpos_adr+7] = np.r_[object_position, math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
        self.target_position = np.asarray(scene["target_position"], dtype=float).copy()
        self.target_position[:2] += rng.uniform(-np.asarray(randomization["target_xy_range_m"]), np.asarray(randomization["target_xy_range_m"]))
        self.target_position[2] = scene["table_top_z"] + scene["object_half_size"][2]
        self.model.body_pos[self.target_id] = [*self.target_position[:2], scene["table_top_z"]+0.0005]
        for arm in self.arms:
            arm.reset()
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.task_state = TaskState.PREGRASP
        self.control_mode = ControlMode.IDLE
        self.gripper_latch = GripperLatch.OPEN
        self.paused = False
        self.pause_reason = None
        self.failure_reason = None
        self.metrics = DualArmMetrics()
        self.contacts = {"left": False, "right": False, "left_fingers": [False, False], "right_fingers": [False, False]}
        self.gripper_event = False
        self.last_gripper_event = None
        self._event_serial = 0
        self._seen_event_ids = set()
        self._last_sent_at = -math.inf
        self._last_command_sim_time = None
        self._task_started_at = None
        self._physics_remainder = 0.0
        self._grasp_started = None
        self._contact_stable_since = None
        self._contact_lost_since = None
        self._success_stable_since = None
        self._relative_reference = None
        self._relative_rotation_reference = None
        self.relative_pose_error_m = 0.0
        self.relative_orientation_error_rad = 0.0
        self.user_delta = {"left": [0.0, 0.0, 0.0], "right": [0.0, 0.0, 0.0]}
        self.common_delta = [0.0, 0.0, 0.0]
        self.input_disagreement_m = 0.0
        self.hand_state = {}
        self._previous_object_position = self.data.xpos[self.object_id].copy()
        self._previous_ee_positions = [arm.get_actual_pose()[0] for arm in self.arms]
        self.ee_speeds = [0.0, 0.0]
        self.initial_state = {"seed": self.seed, "object_position": object_position.tolist(), "object_quaternion_wxyz": self.data.qpos[self.object_qpos_adr+3:self.object_qpos_adr+7].tolist(), "target_position": self.target_position.tolist(), "base_transforms": {side: {"position": scene[f"{side}_base_position"], "quaternion_wxyz": scene[f"{side}_base_quaternion_wxyz"]} for side in ("left", "right")}, "task_frame": {"position": scene["task_frame_position"], "quaternion_wxyz": scene["task_frame_quaternion_wxyz"]}, "config_hash": self.config_hash}
        return self.get_status()

    def start_task_timer(self):
        """Start once, on recording/first meaningful action, not on app launch."""
        if self._task_started_at is None:
            self._task_started_at = float(self.data.time)
        return self._task_started_at

    def set_episode_timeout_enabled(self, enabled):
        """Switch practice/evaluation timing without changing physical state."""
        enabled = bool(enabled)
        if self.episode_timeout_enabled != enabled:
            self.episode_timeout_enabled = enabled
            self._task_started_at = None
        return self.episode_timeout_enabled

    @property
    def task_elapsed_seconds(self):
        return 0.0 if self._task_started_at is None else float(self.data.time-self._task_started_at)

    def _result(self, accepted, reason=None):
        if not accepted:
            self.metrics.rejected_command_count += 1
        return {"accepted": bool(accepted), "reason": reason, "task_state": str(self.task_state), "control_mode": str(self.control_mode), "gripper_latch": str(self.gripper_latch)}

    def pause(self, reason="operator_pause", fail=False):
        if not self.paused:
            self.metrics.pause_count += 1
            for arm in self.arms:
                arm.freeze()
        self.paused = True
        self.pause_reason = str(reason)
        self.control_mode = ControlMode.PAUSE
        if fail:
            self.task_state = TaskState.FAIL
            self.failure_reason = str(reason)

    def update_telemetry(self, telemetry):
        """Keep mode/heartbeat current even when the wrists are stationary.

        The HTTP owner validates packet age before calling this method. Robot
        task progress and physical gripper verification stay authoritative.
        """
        if not isinstance(telemetry, dict):
            return False
        self._last_command_sim_time = self.data.time
        if isinstance(telemetry.get("hands"), dict):
            self.hand_state = telemetry["hands"]
        if self.task_state in {TaskState.FAIL, TaskState.DONE}:
            return True
        try:
            mode = ControlMode(str(telemetry.get("control_mode", "IDLE")).upper())
        except ValueError:
            self.pause("invalid_telemetry_mode")
            return False
        hands_lost = bool(self.hand_state) and any(not self.hand_state.get(side, {}).get("visible", False) for side in ("left", "right"))
        if telemetry.get("calibrated") is False or mode == ControlMode.PAUSE or hands_lost:
            self.pause(telemetry.get("pause_reason") or ("hand_lost" if hands_lost else "calibration_required"))
            return True
        if self.task_state in {TaskState.GRASP_VERIFY, TaskState.RELEASE_VERIFY}:
            return True
        if mode in {ControlMode.GRASP_CONFIRM, ControlMode.RELEASE_CONFIRM}:
            if not self.paused:
                for arm in self.arms:
                    arm.freeze()
            self.control_mode = mode
            self.paused = True
            self.pause_reason = "confirmation"
        else:
            self.control_mode = mode
            self.paused = False
            self.pause_reason = None
        return True

    def apply_command(self, payload, now_ms=None):
        if not isinstance(payload, dict):
            return self._result(False, "invalid_payload")
        now_ms = time.time()*1000 if now_ms is None else float(now_ms)
        command = payload.get("command")
        if command in {"reset", "reset_scene"}:
            self.reset(seed=payload.get("seed", self.seed))
            return self._result(True)
        sent_at = payload.get("sentAt", now_ms)
        if not isinstance(sent_at, (int, float)) or not np.isfinite(sent_at):
            return self._result(False, "invalid_timestamp")
        age = now_ms - sent_at
        if age > self.config["control"]["max_command_age_ms"] or age < -self.config["control"]["max_future_command_ms"]:
            self.metrics.stale_command_count += 1
            return self._result(False, "stale_command")
        if sent_at < self._last_sent_at:
            self.metrics.stale_command_count += 1
            return self._result(False, "out_of_order_command")
        self._last_sent_at = sent_at
        self._last_command_sim_time = self.data.time
        if "hands" in payload:
            self.hand_state = payload["hands"]
        if command in {"pause", "dual_pause", "clutch"}:
            self.pause(payload.get("reason", "operator_pause"))
            return self._result(True)
        if command in {"dual_mode", "mode", "confirmation"}:
            try:
                mode = ControlMode(str(payload.get("mode", "IDLE")).upper())
            except ValueError:
                return self._result(False, "unknown_mode")
            if self.task_state == TaskState.FAIL:
                return self._result(False, "episode_failed_reset_required")
            if mode == ControlMode.PAUSE:
                self.pause(payload.get("reason", "operator_pause"))
            else:
                self.control_mode = mode
                self.paused = mode in {ControlMode.GRASP_CONFIRM, ControlMode.RELEASE_CONFIRM}
                self.pause_reason = "confirmation" if self.paused else None
            return self._result(True)
        if command == "dual_gripper":
            event_id = payload.get("eventId")
            if event_id is not None:
                event_id = str(event_id)
                if event_id in self._seen_event_ids:
                    return self._result(True, "duplicate_event")
            result = self._set_dual_gripper(payload.get("action"))
            if result["accepted"] and event_id is not None:
                self._seen_event_ids.add(event_id)
            return result
        if command != "dual_motion":
            return self._result(False, "unknown_command")
        if self.task_state in {TaskState.FAIL, TaskState.DONE, TaskState.GRASP_VERIFY, TaskState.RELEASE_VERIFY}:
            return self._result(False, "task_motion_frozen")
        try:
            mode = ControlMode(str(payload.get("mode", "")).upper())
            if mode not in {ControlMode.XY, ControlMode.Z}:
                return self._result(False, "invalid_motion_mode")
            deltas = [np.array([payload[side].get(axis, 0) for axis in ("dx", "dy", "dz")], dtype=float) for side in ("left", "right")]
            if not all(np.all(np.isfinite(delta)) for delta in deltas):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            return self._result(False, "invalid_motion")
        deltas = [np.clip(delta, -1, 1)*self.config["control"]["max_ee_step_m"] for delta in deltas]
        if mode == ControlMode.XY:
            for delta in deltas:
                delta[2] = 0
        else:
            for delta in deltas:
                delta[:2] = 0
        max_step = self.config["control"]["max_ee_step_m"]
        deltas = [delta*min(1.0, max_step/max(float(np.linalg.norm(delta)), 1e-12)) for delta in deltas]
        self.control_mode, self.paused, self.pause_reason = mode, False, None
        self._move_deltas(deltas)
        return self._result(True)

    def _set_dual_gripper(self, action):
        if action not in {"open", "close"}:
            return self._result(False, "invalid_gripper_action")
        target = GripperLatch.OPEN if action == "open" else GripperLatch.CLOSED
        if target == self.gripper_latch:
            return self._result(True, "unchanged_latch")
        if action == "close" and self.task_state in {TaskState.FAIL, TaskState.DONE}:
            return self._result(False, "episode_ended_reset_required")
        if action == "open" and not self.release_ready:
            return self._result(False, "release_rejected_stop_before_opening")
        for arm in self.arms:
            arm.set_gripper(action)
        if action == "close":
            self.start_task_timer()
        self.gripper_latch = target
        self._event_serial += 1
        self.gripper_event = True
        self.last_gripper_event = {"serial": self._event_serial, "action": action, "simulation_time": float(self.data.time)}
        self.metrics.gripper_event_count += 1
        self.paused, self.pause_reason = False, None
        self.control_mode = ControlMode.IDLE
        if self.task_state != TaskState.FAIL:
            if action == "close":
                self.task_state = TaskState.GRASP_VERIFY
                self._grasp_started = self.data.time
                self._contact_stable_since = None
            else:
                self.task_state = TaskState.RELEASE_VERIFY if self._relative_reference is not None else TaskState.PREGRASP
                self._success_stable_since = None
        else:
            self.paused = True
            self.control_mode = ControlMode.PAUSE
        return self._result(True)

    def _move_deltas(self, deltas):
        if any(np.linalg.norm(delta) > 1e-10 for delta in deltas):
            self.start_task_timer()
        self.user_delta = {side: delta.tolist() for side, delta in zip(("left", "right"), deltas)}
        self.input_disagreement_m = float(np.linalg.norm(deltas[0]-deltas[1]))
        self.metrics.max_input_disagreement_m = max(self.metrics.max_input_disagreement_m, self.input_disagreement_m)
        if self.task_state == TaskState.DUAL_GRASPED:
            common = (deltas[0]+deltas[1])/2
            self.common_delta = common.tolist()
            left_pos, _ = self.left.get_actual_pose()
            right_pos, _ = self.right.get_actual_pose()
            error = (right_pos-left_pos)-self._relative_reference
            correction = self.config["cooperation"]["relative_pose_soft_gain"]*error/2
            cap = self.config["control"]["max_ee_step_m"] / 2
            correction = np.clip(correction, -cap, cap)
            task_correction = self.left.task_rotation.T @ correction
            deltas = [common+task_correction, common-task_correction]
            # Clamp common motion jointly so one workspace limit cannot stretch the object.
            feasible = np.ones(3)
            for arm, delta in zip(self.arms, deltas):
                current = arm.world_to_task(arm.command_position)
                for axis in range(3):
                    if delta[axis] > 0:
                        feasible[axis] = min(feasible[axis], max(0, (arm.workspace[axis, 1]-current[axis])/delta[axis]))
                    elif delta[axis] < 0:
                        feasible[axis] = min(feasible[axis], max(0, (arm.workspace[axis, 0]-current[axis])/delta[axis]))
            deltas = [delta*feasible for delta in deltas]
        else:
            self.common_delta = [0.0, 0.0, 0.0]
        common_scale = min(1.0, self.config["control"]["max_ee_step_m"]/max(max(float(np.linalg.norm(delta)) for delta in deltas), 1e-12))
        deltas = [delta*common_scale for delta in deltas]
        for arm, delta in zip(self.arms, deltas):
            arm.move_delta_task_frame(*delta)
            if arm.workspace_clamped:
                self.metrics.workspace_clamp_count += 1

    def apply_policy_action(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (14,) or not np.all(np.isfinite(action)):
            return self._result(False, "invalid_policy_action")
        if self.task_state in {TaskState.FAIL, TaskState.DONE}:
            return self._result(False, "episode_ended")
        target_grippers = ["open" if action[index] > 127.5 else "close" for index in (6, 13)]
        if target_grippers[0] != target_grippers[1]:
            return self._result(False, "asymmetric_gripper_action")
        target_rotation = np.asarray(self.config["scene"]["ee_rotation_matrix"])
        for offset in (0, 7):
            if rotation_angle(rpy_to_matrix(action[offset+3:offset+6]), target_rotation) > self.config["control"]["policy_orientation_tolerance_rad"]:
                return self._result(False, "policy_orientation_outside_v1_envelope")
        self._last_command_sim_time = self.data.time
        self.paused, self.pause_reason = False, None
        if self.task_state not in {TaskState.GRASP_VERIFY, TaskState.RELEASE_VERIFY}:
            deltas = [action[offset:offset+3]-arm.world_to_task(arm.command_position) for arm, offset in zip(self.arms, (0, 7))]
            max_step = self.config["control"]["max_ee_step_m"]
            deltas = [delta * min(1.0, max_step/max(float(np.linalg.norm(delta)), 1e-12)) for delta in deltas]
            self._move_deltas(deltas)
            self.control_mode = ControlMode.IDLE
        return self._set_dual_gripper(target_grippers[0])

    def _contact_state(self):
        contacts = {"left_fingers": [False, False], "right_fingers": [False, False]}
        force = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if self.object_geom_id not in {g1, g2}:
                continue
            other = g2 if g1 == self.object_geom_id else g1
            mujoco.mj_contactForce(self.model, self.data, index, force)
            if abs(force[0]) < self.config["cooperation"]["minimum_contact_force_n"]:
                continue
            for side in ("left", "right"):
                for finger, geom_ids in enumerate(self.finger_geoms[side]):
                    if other in geom_ids:
                        contacts[f"{side}_fingers"][finger] = True
        contacts["left"] = all(contacts["left_fingers"])
        contacts["right"] = all(contacts["right_fingers"])
        return contacts

    def _robot_robot_collision(self):
        for contact in self.data.contact[:self.data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if (g1 in self.robot_geoms["left"] and g2 in self.robot_geoms["right"]) or (g2 in self.robot_geoms["left"] and g1 in self.robot_geoms["right"]):
                return True
        return False

    def _update_task(self):
        self.contacts = self._contact_state()
        object_position = self.data.xpos[self.object_id]
        velocity = self.data.qvel[self.object_dof_adr:self.object_dof_adr+6]
        if self.task_state == TaskState.GRASP_VERIFY:
            if self.contacts["left"] and self.contacts["right"]:
                if self._contact_stable_since is None:
                    self._contact_stable_since = self.data.time
                if self.data.time-self._contact_stable_since >= self.config["cooperation"]["grasp_stable_seconds"]:
                    self.task_state = TaskState.DUAL_GRASPED
                    left_pos, left_rot = self.left.get_actual_pose()
                    right_pos, right_rot = self.right.get_actual_pose()
                    self._relative_reference = right_pos-left_pos
                    self._relative_rotation_reference = left_rot.T @ right_rot
                    self.metrics.grasp_verified_count += 1
            else:
                self._contact_stable_since = None
            if self.task_state == TaskState.GRASP_VERIFY and self.data.time-self._grasp_started > self.config["cooperation"]["grasp_verify_timeout_seconds"]:
                self.pause("grasp_failure", fail=True)
        if self.task_state == TaskState.DUAL_GRASPED:
            left_pos, left_rot = self.left.get_actual_pose()
            right_pos, right_rot = self.right.get_actual_pose()
            self.relative_pose_error_m = float(np.linalg.norm((right_pos-left_pos)-self._relative_reference))
            self.relative_orientation_error_rad = rotation_angle(left_rot.T @ right_rot, self._relative_rotation_reference)
            self.metrics.max_relative_pose_error_m = max(self.metrics.max_relative_pose_error_m, self.relative_pose_error_m)
            self.metrics.max_relative_orientation_error_rad = max(self.metrics.max_relative_orientation_error_rad, self.relative_orientation_error_rad)
            cooperation = self.config["cooperation"]
            if self.relative_pose_error_m > cooperation["relative_pose_hard_limit_m"] or self.relative_orientation_error_rad > cooperation["relative_orientation_hard_limit_rad"]:
                self.pause("relative_pose_violation", fail=True)
            if not (self.contacts["left"] and self.contacts["right"]):
                if self._contact_lost_since is None:
                    self._contact_lost_since = self.data.time
                if self.data.time-self._contact_lost_since > cooperation["contact_loss_grace_seconds"]:
                    self.metrics.contact_loss_count += 1
                    self.pause("contact_loss", fail=True)
            else:
                self._contact_lost_since = None
            if object_position[2] < self.config["scene"]["table_top_z"]-self.config["evaluation"]["object_drop_margin_m"]:
                self.pause("object_drop", fail=True)
        if self.task_state == TaskState.RELEASE_VERIFY:
            evaluation = self.config["evaluation"]
            object_rotation = self.data.xmat[self.object_id].reshape(3, 3)
            within_target = np.linalg.norm(object_position[:2]-self.target_position[:2]) <= evaluation["target_position_tolerance_m"] and abs(object_position[2]-self.target_position[2]) <= evaluation["target_height_tolerance_m"]
            stable = np.linalg.norm(velocity[:3]) <= evaluation["stable_linear_speed_mps"] and np.linalg.norm(velocity[3:]) <= evaluation["stable_angular_speed_radps"]
            orientation_ok = rotation_angle(object_rotation, np.eye(3)) <= evaluation["target_orientation_tolerance_rad"]
            if within_target and stable and orientation_ok and self.gripper_latch == GripperLatch.OPEN and not self.metrics.collision_count:
                if self._success_stable_since is None:
                    self._success_stable_since = self.data.time
                if self.data.time-self._success_stable_since >= evaluation["stable_success_seconds"]:
                    self.task_state, self.control_mode = TaskState.DONE, ControlMode.IDLE
                    self.metrics.completion_time_seconds = self.task_elapsed_seconds
            else:
                self._success_stable_since = None
        if self.episode_timeout_enabled and self.task_state not in {TaskState.DONE, TaskState.FAIL} and self._task_started_at is not None and self.task_elapsed_seconds >= self.config["evaluation"]["max_episode_seconds"]:
            self.pause("timeout", fail=True)

    def step(self, dt=None):
        dt = 1/self.config["scene"]["control_hz"] if dt is None else float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("step dt must be a positive finite number")
        if dt < self.model.opt.timestep:
            raise ValueError("step dt must cover at least one MuJoCo physics timestep")
        if self._last_command_sim_time is not None and self.data.time-self._last_command_sim_time > self.config["control"]["command_watchdog_seconds"] and not self.paused and self.task_state not in {TaskState.DONE, TaskState.FAIL, TaskState.GRASP_VERIFY, TaskState.RELEASE_VERIFY}:
            self.pause("command_watchdog")
        for arm in self.arms:
            if not self.paused:
                arm.track_target()
            health = arm.check_health()
            if not health["finite"]:
                self.pause("joint_or_ik_failure", fail=True)
            if health["workspace_violation_count"] >= self.config["control"]["workspace_violation_limit"]:
                self.pause("workspace_violation", fail=True)
        previous_time = self.data.time
        physics_steps = int((dt+self._physics_remainder)/self.model.opt.timestep+1e-9)
        self._physics_remainder = dt+self._physics_remainder-physics_steps*self.model.opt.timestep
        for _ in range(physics_steps):
            mujoco.mj_step(self.model, self.data)
            if self.task_state != TaskState.FAIL and self._robot_robot_collision():
                self.metrics.collision_count += 1
                self.pause("robot_robot_collision", fail=True)
        actual_dt = self.data.time-previous_time
        object_position = self.data.xpos[self.object_id].copy()
        self.metrics.object_path_length_m += float(np.linalg.norm(object_position-self._previous_object_position))
        self.metrics.peak_object_height_m = max(self.metrics.peak_object_height_m, float(object_position[2]))
        self._previous_object_position = object_position
        for index, arm in enumerate(self.arms):
            position = arm.get_actual_pose()[0]
            self.ee_speeds[index] = float(np.linalg.norm(position-self._previous_ee_positions[index])/actual_dt)
            self._previous_ee_positions[index] = position
        self._update_task()
        return self.get_status()

    @property
    def training_recordable(self):
        return self.gripper_event or self.control_mode not in {ControlMode.GRASP_CONFIRM, ControlMode.RELEASE_CONFIRM}

    def consume_gripper_event(self):
        event = self.last_gripper_event if self.gripper_event else None
        self.gripper_event = False
        return event

    def state_vector(self):
        return np.concatenate([arm.state_vector() for arm in self.arms]).astype(np.float32)

    def action_vector(self):
        return np.concatenate([arm.action_vector() for arm in self.arms]).astype(np.float32)

    @property
    def release_ready(self):
        """A failed episode can release, but an established grasp must settle."""
        if self._relative_reference is None and self.task_state != TaskState.DUAL_GRASPED:
            return True
        velocity = self.data.qvel[self.object_dof_adr:self.object_dof_adr+6]
        limits = self.config["cooperation"]
        return bool(np.linalg.norm(velocity[:3]) <= limits["release_max_object_speed_mps"]
                    and np.linalg.norm(velocity[3:]) <= limits["release_max_angular_speed_radps"]
                    and max(self.ee_speeds) <= limits["release_max_ee_speed_mps"])

    def get_status(self):
        velocity = self.data.qvel[self.object_dof_adr:self.object_dof_adr+6]
        metrics = self.metrics.as_dict()
        metrics.update(relative_pose_error_m=self.relative_pose_error_m, relative_orientation_error_rad=self.relative_orientation_error_rad, input_disagreement_m=self.input_disagreement_m, object_linear_velocity_mps=velocity[:3].tolist(), object_angular_velocity_radps=velocity[3:].tolist())
        return {
            "task_mode": "dual_arm",
            "task": self.config["task"],
            "task_state": str(self.task_state),
            "control_mode": str(self.control_mode),
            "gripper_latch": str(self.gripper_latch),
            "gripper_closed": self.gripper_latch == GripperLatch.CLOSED,
            "release_ready": self.release_ready,
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "failure_reason": self.failure_reason,
            "success": self.task_state == TaskState.DONE,
            "simulation_time": float(self.data.time),
            "task_elapsed_seconds": self.task_elapsed_seconds,
            "task_timer_started": self._task_started_at is not None,
            "episode_timeout_enabled": self.episode_timeout_enabled,
            "practice_mode": not self.episode_timeout_enabled,
            "object_position": self.data.xpos[self.object_id].tolist(),
            "target_position": self.target_position.tolist(),
            "object_linear_velocity_mps": velocity[:3].tolist(),
            "object_angular_velocity_radps": velocity[3:].tolist(),
            "contacts": self.contacts,
            "relative_pose_error_m": self.relative_pose_error_m,
            "relative_orientation_error_rad": self.relative_orientation_error_rad,
            "collision_count": self.metrics.collision_count,
            "contact_loss_count": self.metrics.contact_loss_count,
            "ee_speeds_mps": self.ee_speeds,
            "metrics": metrics,
            "training_recordable": self.training_recordable,
            "gripper_event": self.gripper_event,
            "last_gripper_event": self.last_gripper_event,
            "user_delta": self.user_delta,
            "common_delta": self.common_delta,
            "input_disagreement_m": self.input_disagreement_m,
            "hands": self.hand_state,
            "arm_health": {arm.side: arm.check_health() for arm in self.arms},
            "actual_ee": {arm.side: {"position": arm.get_actual_pose()[0].tolist(),
                                     "rotation": arm.get_actual_pose()[1].tolist()} for arm in self.arms},
            "commanded_ee": {arm.side: {"position": arm.command_position.tolist(),
                                        "rotation": arm.command_rotation.tolist()} for arm in self.arms},
            "config_hash": self.config_hash,
            "seed": self.seed,
        }

    def snapshot(self):
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(), "ctrl": self.data.ctrl.copy(), "body_pos": self.model.body_pos.copy(), "time": float(self.data.time), "state": self.state_vector(), "action": self.action_vector(), "status": self.get_status(), "initial_state": self.initial_state}


def scripted_demo(controller: DualArmTaskController, on_step=None, reset=False, seed=0):
    """Physics-only pilot trajectory for pipeline checks; not human demonstrations.

    The callback receives controller after each control step. No qpos editing,
    object attachment, weld, or success override occurs after optional reset.
    """
    if reset:
        controller.reset(seed)

    def tick():
        controller.step(0.04)
        if on_step is not None:
            on_step(controller)
        controller.consume_gripper_event()
        if controller.task_state == TaskState.FAIL:
            raise RuntimeError(f"Scripted pilot failed: {controller.failure_reason}")

    def settle(seconds):
        for _ in range(math.ceil(seconds/0.04)):
            controller._last_command_sim_time = controller.data.time
            tick()

    def move_to(positions, mode, tolerance=0.0008, max_steps=250):
        axes = (0, 1) if mode == "xy" else (2,)
        for _ in range(max_steps):
            deltas = [np.asarray(position)-arm.world_to_task(arm.command_position) for position, arm in zip(positions, controller.arms)]
            error = float(np.linalg.norm(((deltas[0]+deltas[1])/2)[list(axes)])) if controller.task_state == TaskState.DUAL_GRASPED else max(float(np.linalg.norm(delta[list(axes)])) for delta in deltas)
            if error < tolerance:
                break
            payload = {"command": "dual_motion", "mode": mode, "sentAt": 0}
            for side, delta in zip(("left", "right"), deltas):
                normalized = delta/controller.config["control"]["max_ee_step_m"]
                payload[side] = dict(zip(("dx", "dy", "dz"), normalized.tolist()))
            result = controller.apply_command(payload, now_ms=0)
            if not result["accepted"]:
                raise RuntimeError(result["reason"])
            tick()
        else:
            raise RuntimeError("Scripted pilot target did not converge")
        settle(0.3)

    settle(0.4)
    object_position = controller.data.xpos[controller.object_id].copy()
    object_rotation = controller.data.xmat[controller.object_id].reshape(3, 3)
    grasp_world = [object_position + object_rotation @ np.asarray(offset) for offset in controller.config["scene"]["grasp_offsets"]]
    grasp = [controller.left.world_to_task(position) for position in grasp_world]
    move_to(grasp, "xy")
    move_to(grasp, "z")
    result = controller.apply_command({"command": "dual_gripper", "action": "close", "sentAt": 0}, now_ms=0)
    if not result["accepted"]:
        raise RuntimeError(result["reason"])
    settle(1.0)
    if controller.task_state != TaskState.DUAL_GRASPED:
        raise RuntimeError(f"Scripted pilot did not form a verified dual grasp: {controller.contacts}")
    lift = [arm.world_to_task(arm.command_position) for arm in controller.arms]
    for position in lift:
        position[2] += 0.12
    move_to(lift, "z")
    displacement_world = controller.target_position-controller.data.xpos[controller.object_id]
    displacement_world[2] = 0
    displacement = controller.left.task_rotation.T @ displacement_world
    carry = [arm.world_to_task(arm.command_position)+displacement for arm in controller.arms]
    move_to(carry, "xy")
    landing = [arm.world_to_task(arm.command_position) for arm in controller.arms]
    lower_delta = controller.target_position[2]-controller.data.xpos[controller.object_id][2]+0.002
    for position in landing:
        position[2] += lower_delta
    move_to(landing, "z")
    settle(0.5)
    result = controller.apply_command({"command": "dual_gripper", "action": "open", "sentAt": 0}, now_ms=0)
    if not result["accepted"]:
        raise RuntimeError(result["reason"])
    settle(1.0)
    if controller.task_state != TaskState.DONE:
        raise RuntimeError(f"Scripted pilot placement did not pass automatic success: {controller.get_status()}")
    return controller.get_status()
