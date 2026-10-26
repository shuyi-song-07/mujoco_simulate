"""Real-contact, synchronized Task 3 controller, independent of Task 1/2."""
from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .config import DEFAULT_SCENE_PATH, config_hash, load_config
from .task_states import ControlMode, GripperLatch, TaskState
from .task_metrics import DualArmMetrics
from .panda_arm_controller import PandaArmController, rotation_angle, rpy_to_matrix
from .action_contract import (V1, V2, GESTURE_GRASP_RELEASE_V2, GESTURE_RATE_GRIPPER_V3,
    INTERACTION_MODES, RATE_INTERACTION_MODES, describe,
    validate_action, validate_release_intent, validate_interaction_modes, legacy_to_metric, vector)
from .contact_control import ContactControl
from .rigid_control import RigidControl, rz
from .task_scenes import configure_paper_scene, reset_paper_obstacles, target_surface_z


def _numbers(values):
    return " ".join(str(float(value)) for value in values)


def create_model(config=None) -> mujoco.MjModel:
    """Compile the scene with config-controlled geometry, without temp files."""
    config = load_config(config)
    scene = config["scene"]
    from tasks.registry import local, hook
    binding = config.get('task_binding')
    scene_path = local(binding['model_xml']) if binding else DEFAULT_SCENE_PATH
    root = ET.parse(scene_path).getroot()
    # MuJoCo's native Windows file opening cannot reliably handle a Chinese
    # absolute workspace path. VFS mesh bytes retain one original asset copy
    # and also make compilation independent of the caller's current directory.
    from .config import PROJECT_ROOT
    mesh_directory = PROJECT_ROOT / binding['mesh_directory'] if binding else DEFAULT_SCENE_PATH.parent.parent / "franka_emika_panda" / "assets"
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
    hook(config, "configure_xml")(root, config) if binding else configure_paper_scene(root, scene)
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
        self.continuous_gripper = self.left.continuous_gripper
        self.continuous_task = bool(self.config.get("contact_control", {}).get("enabled", False))
        # Policies and scripted evaluations keep the configured deadline by
        # default. Only the interactive recorder explicitly opts into practice.
        self.episode_timeout_enabled = True
        self.object_id = self.model.body("cooperative_object").id
        self.object_geom_id = self.model.geom("object_geom").id
        self.object_geoms = set(np.flatnonzero(self.model.geom_bodyid == self.object_id))
        self.obstacle_geoms = {i for i in range(self.model.ngeom) if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("blocker_")}
        if 'placement_platform' in self.config['scene']:
            self.obstacle_geoms.add(self.model.geom('placement_platform_geom').id)
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
        self.success_tail = None
        rng = np.random.default_rng(self.seed)
        mujoco.mj_resetData(self.model, self.data)
        scene = self.config["scene"]
        randomization = scene["randomization"]
        object_position = np.asarray(scene["object_position"], dtype=float).copy()
        object_position[:2] += rng.uniform(-np.asarray(randomization["object_xy_range_m"]), np.asarray(randomization["object_xy_range_m"]))
        object_position[2] = scene["table_top_z"] + scene["object_half_size"][2] + 0.0002
        yaw = float(rng.uniform(-randomization["object_yaw_range_rad"], randomization["object_yaw_range_rad"]))
        if self.config.get('task_binding'):
            from tasks.registry import hook
            sampled = hook(self.config, 'initial_state')(self.config, self.seed)
            object_position = np.asarray(sampled['object_position'], dtype=float)
            yaw = float(sampled['object_yaw_rad'])
        self.data.qpos[self.object_qpos_adr:self.object_qpos_adr+7] = np.r_[object_position, math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
        self.target_position = np.asarray(scene["target_position"], dtype=float).copy()
        self.target_position[:2] += rng.uniform(-np.asarray(randomization["target_xy_range_m"]), np.asarray(randomization["target_xy_range_m"]))
        self.target_position[2] = target_surface_z(scene) + scene["object_half_size"][2]
        if self.config.get('task_binding'):
            self.target_position = np.asarray(sampled['target_position'], dtype=float)
        self.model.body_pos[self.target_id] = [*self.target_position[:2], target_surface_z(scene)+0.0005]
        self.target_yaw = float(sampled["target_yaw_rad"] if self.config.get("task_binding") else scene.get("target_yaw_rad", 0))
        self.model.body_quat[self.target_id] = [math.cos(self.target_yaw/2), 0, 0, math.sin(self.target_yaw/2)]
        if 'placement_platform' in scene:
            platform_id = self.model.body('placement_platform').id
            self.model.body_pos[platform_id] = [*self.target_position[:2], scene['table_top_z']+scene['placement_platform']['height_m']/2]
            self.model.body_quat[platform_id] = self.model.body_quat[self.target_id]
        from tasks.registry import hook
        initial_obstacles = (hook(self.config, 'reset')(self, object_position, yaw) if self.config.get('task_binding')
                             else reset_paper_obstacles(self.model, scene, object_position, yaw))
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
        self.last_action_result = None
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
        self.contact_control = ContactControl(self) if self.continuous_task else None
        self.rigid_control = RigidControl(self) if self.continuous_task and "rigid_control" in self.config else None
        self.initial_state = {"seed": self.seed, "object_position": object_position.tolist(), "object_quaternion_wxyz": self.data.qpos[self.object_qpos_adr+3:self.object_qpos_adr+7].tolist(), "target_position": self.target_position.tolist(), "base_transforms": {side: {"position": scene[f"{side}_base_position"], "quaternion_wxyz": scene[f"{side}_base_quaternion_wxyz"]} for side in ("left", "right")}, "task_frame": {"position": scene["task_frame_position"], "quaternion_wxyz": scene["task_frame_quaternion_wxyz"]}, "config_hash": self.config_hash}
        self.initial_state.update(target_yaw_rad=self.target_yaw, initial_obstacles=initial_obstacles)
        if 'placement_platform' in scene:
            gid = self.model.geom('placement_platform_geom').id
            self.initial_state['placement_platform'] = {
                'body_position': self.model.body_pos[self.model.body('placement_platform').id].tolist(),
                'size_xyz_m': (2*self.model.geom_size[gid]).tolist(),
                'top_z_m': target_surface_z(scene), 'yaw_rad': self.target_yaw}
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

    def _result(self, accepted, reason=None, *, preserve_gripper_baseline=False):
        if not accepted:
            self.metrics.rejected_command_count += 1
            if getattr(self, "contact_control", None):
                if preserve_gripper_baseline:
                    self.contact_control.cancel_rate_control(reason, reset_edge_baseline=False)
                else:
                    self.contact_control.cancel_release(reason or "release_input_interrupted")
        return {"accepted": bool(accepted), "reason": reason, "task_state": str(self.task_state), "control_mode": str(self.control_mode), "gripper_latch": str(self.gripper_latch)}

    def pause(self, reason="operator_pause", fail=False):
        tail = getattr(self, 'success_tail', None)
        if tail and tail['active']:
            if not fail and reason in {'task_done', 'command_timeout'}:
                return  # Automatic DONE notifications must not cancel the bounded opening.
            tail['active'] = False
            tail['interrupted_reason'] = str(reason)
        if getattr(self, "contact_control", None):
            self.contact_control.cancel_release(str(reason))
        if self.continuous_gripper:
            for arm in self.arms:
                arm.hold_gripper()
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
        if command == "continuous_input":
            if not self.continuous_task:
                return self._result(False, "continuous_task_configuration_required")
            if payload.get("mode") not in {"XY", "Z"}:
                return self._result(False, "invalid_motion_mode")
            try:
                delta = np.array([payload["delta_task_m"][s] for s in ("left", "right")], dtype=float)
                widths = np.asarray(payload["widths_m"], dtype=float)
                if delta.shape != (2, 3) or widths.shape != (2,) or not np.isfinite(delta).all() or not np.isfinite(widths).all() or np.max(np.linalg.norm(delta, axis=1)) > .02:
                    raise ValueError()
            except (KeyError, ValueError, TypeError):
                return self._result(False, "invalid_continuous_input")
            action = self.action_vector_m()
            for index, offset in enumerate((0, 7)):
                action[offset:offset+3] += delta[index]
                action[offset+6] = widths[index]
            return self.apply_action(action, schema_version=V2, mode=ControlMode(payload.get("mode", "XY")),
                source="continuous_camera", event_id=payload.get("eventId"),
                release_intent=payload.get("release_intent", (False, False)),
                interaction_modes=payload.get("interaction_modes"), control_protocol=payload.get("control_protocol"))
        if command == "metric_action":
            if not self.continuous_gripper:
                return self._result(False, "metric_command_requires_continuous_configuration")
            return self.apply_action(payload.get("action"), schema_version=payload.get("schema_version"),
                components=payload.get("components", ("motion", "gripper")), source="metric_command",
                mode=payload.get("mode"), event_id=payload.get("eventId"),
                release_intent=payload.get("release_intent", (False, False)),
                interaction_modes=payload.get("interaction_modes"), control_protocol=payload.get("control_protocol"))
        if self.continuous_gripper and command in {"dual_motion", "dual_gripper"}:
            return self._result(False, "continuous_validation_requires_metric_action")
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
            if payload.get("action") not in {"open", "close"}:
                return self._result(False, "invalid_gripper_action")
            target = self.action_vector_m()
            target[[6, 13]] = self.gripper_width_limits if payload["action"] == "open" else 0
            result = self.apply_action(target, schema_version=V2, components=("gripper",),
                                       source="legacy_command", event_id=event_id)
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
        target = self.action_vector_m()
        for offset, delta in zip((0, 7), deltas):
            target[offset:offset+3] += delta
        return self.apply_action(target, schema_version=V2, components=("motion",), mode=mode,
                                 source="legacy_command", event_id=payload.get("eventId"))

    @property
    def gripper_width_limits(self):
        return np.array([arm.max_gripper_width_m for arm in self.arms])

    def action_vector_m(self):
        return legacy_to_metric(self.action_vector(), self.gripper_width_limits, version=V1)

    def action_contract(self):
        contract = describe(self.gripper_width_limits, continuous_gripper=self.continuous_gripper)
        if self.continuous_task:
            contract["execution_scope"] = "recoverable_contact_task"
            contract["capabilities"]["recoverable_contact"] = True
            contract["capabilities"]["limited_yaw"] = self.config.get("rigid_control", {}).get("max_yaw_rad", 0)
            if self.contact_control.intent_control:
                contract["control_protocol"] = self.contact_control.control_protocol
                contract["release_intent_contract"] = {"names": ["left.release_intent", "right.release_intent"],
                    "shape": [2], "type": "bool", "required_for_policy": True,
                    "meaning": "explicit request, not release execution", "default": [False, False]}
                contract["release_parameters"] = {key: self.config["contact_control"][key] for key in (
                    "release_intent_aperture_normalized", "release_confirm_seconds", "release_input_max_gap_seconds",
                    "release_stationary_target_tolerance_m", "release_stationary_rotation_tolerance_rad")
                    if key in self.config["contact_control"]}
                contract["policy_compatibility"] = "14D-only checkpoints cannot release; a separate 2D intent head is required"
                if self.contact_control.gesture_control:
                    contract["interaction_mode_contract"] = {"names": ["left.interaction_mode", "right.interaction_mode"],
                        "shape": [2], "type": "string", "values": list(RATE_INTERACTION_MODES if self.contact_control.rate_control else INTERACTION_MODES), "required_for_policy": True,
                        "meaning": "per-side requested operation permission; physical contact and servo phase remain separate"}
                    contract["gesture_parameters"] = dict(self.config.get("gesture_control", {}))
                    contract["release_parameters"].pop("release_intent_aperture_normalized", None)
                    contract["policy_compatibility"] = "14D-only checkpoints are incompatible; explicit 2D interaction modes and release intent are required"
                    contract["release_intent_contract"]["meaning"] = "fresh explicit RELEASE confirmation; camera requires complete-palm evidence; OPENING continuation may have false intent"
                    if self.contact_control.rate_control:
                        contract["release_intent_contract"]["meaning"] = "true exactly in Z_OPEN; free grippers open, physical grasp requires stable supported confirmation"
                        contract["policy_contract_version"] = "gesture-rate-mode-policy-v3"
                        contract["gripper_rate_contract"] = {"Z_OPEN": 1, "Z_CLOSE": -1, "XY_HOLD": 0, "HOLD": 0,
                            "meaning": "direction request; real contact and release guards may hold it", "aperture_supervision": False}
                        contract["gripper_trigger_policy"] = self.contact_control.gripper_trigger_policy
                        if self.contact_control.level_control:
                            contract['policy_contract_version'] = 'gesture-level-mode-policy-v2'
                            contract['gripper_rate_contract']['meaning'] = 'current qualified mode independently opens/closes each jaw; XY_HOLD/HOLD freezes issued width; physical grasp retains preload'
                            contract['release_intent_contract']['meaning'] = 'accepted Z_OPEN independently releases that jaw without another confirmation timer'
                            contract['release_parameters'].update(require_table_support=False, require_stationary_targets=False,
                                require_both_sides=False, release_confirm_seconds=0.)
                        if self.contact_control.edge_control:
                            contract["policy_contract_version"] = "gesture-edge-mode-policy-v2" if self.contact_control.edge_hold_resume else "gesture-edge-mode-policy-v1"
                            contract["gripper_rate_contract"]["meaning"] = "confirmed palm/fist transitions authorize one bounded rate ramp; repeated labels do not retrigger"
                            contract["release_intent_contract"]["meaning"] = "raw mode evidence (Z_OPEN); edge authorization and dual-grasp confirmation gate release, without table support or stationarity gating"
                            contract["release_parameters"]["require_table_support"] = False
                            contract["release_parameters"]["require_stationary_targets"] = False
        return contract

    def apply_action(self, action, *, schema_version, components=("motion", "gripper"), mode=None,
                     source="trusted_python", event_id=None, release_intent=(False, False),
                     interaction_modes=None, control_protocol=None):
        """Common execution boundary. Network context is checked by HTTP owner.

        M02 enables continuous widths in a fixed-arm validation configuration;
        task-level continuous contact transitions are reserved for M03.
        """
        requested = None
        projected = None
        checked_intent = [False, False]
        checked_modes = None
        mode_adapter = None
        gesture_snapshot = None
        preserve_gripper_baseline = False
        if self.contact_control and self.contact_control.gesture_control:
            contact = self.contact_control
            gesture_snapshot = {name: dict(getattr(contact, name)) for name in (
                "interaction_modes", "interaction_mode_event_id", "operation_hold_width_m",
                "release_armed", "_hold_generation_time", "release_reentry_pending")}
            gesture_snapshot["opening"] = any(contact.release_opening.values())
            if contact.rate_control:
                gesture_snapshot["gripper_edge"] = {s: dict(v) for s, v in contact.gripper_edge.items()}

        def answer(accepted, reason=None):
            accepted_target = self.action_vector_m()
            result = self._result(accepted, reason, preserve_gripper_baseline=preserve_gripper_baseline)
            if not accepted and gesture_snapshot is not None:
                contact = self.contact_control
                # A rejected source cannot become the accepted operation-mode
                # or lock provenance. Revoking an earlier opening is a safety
                # transition and deliberately retains its new command hold.
                contact.interaction_modes = dict(gesture_snapshot["interaction_modes"])
                contact.interaction_mode_event_id = dict(gesture_snapshot["interaction_mode_event_id"])
                opening_aborted = gesture_snapshot["opening"] and not any(contact.release_opening.values())
                if not opening_aborted:
                    for name in ("operation_hold_width_m", "release_armed", "_hold_generation_time", "release_reentry_pending"):
                        setattr(contact, name, dict(gesture_snapshot[name]))
                if contact.rate_control:
                    # Failed motion never owns a trigger. V2 retains the last
                    # accepted gesture baseline after a trusted motion veto so
                    # that the next valid palm can still release. Invalid or
                    # stale input retains the full baseline reset.
                    for side, edge in contact.gripper_edge.items():
                        fields = ("edge_count", "event_id", "baseline_mode", "baseline_source") if preserve_gripper_baseline else ("edge_count", "event_id")
                        for name in fields:
                            edge[name] = gesture_snapshot["gripper_edge"][side][name]
                        if preserve_gripper_baseline:
                            # Resume only the PREVIOUSLY accepted transaction.
                            # This rejected packet cannot authorize its edge.
                            # cancel_rate_control already froze the actuators
                            # and reset confirmation/opening, so an interrupted
                            # release must earn a new confirmation interval.
                            edge["authorized_mode"] = gesture_snapshot["gripper_edge"][side]["authorized_mode"]
                            if edge["authorized_mode"] is not None:
                                edge.update(state="HOLD", reason="motion_rejected_wait_resume")
            if not accepted and preserve_gripper_baseline:
                # Server-owned execution result, never copied from input. The
                # recorder must not cancel the same rejection a second time.
                result["gripper_rejection_handled"] = "trusted_motion_hold_v2"
            result.update(schema_version=V2, source=source, eventId=event_id,
                          requested_target=None if requested is None else requested.tolist(),
                          projected_target=None if projected is None else projected.tolist(),
                          release_intent=checked_intent,
                          accepted_target=accepted_target.tolist(), actual_state=self.state_vector().tolist(),
                          simulation_time=float(self.data.time))
            result["grippers"] = {arm.side: arm.gripper_status() for arm in self.arms}
            result["intervened_fields"] = [] if requested is None else [int(i) for i in np.flatnonzero(
                np.abs(requested-accepted_target) > 1e-7)]
            if self.contact_control:
                result["control_protocol"] = self.contact_control.control_protocol
                result["physical_contact"] = self.contact_control.status()
                if self.contact_control.gesture_control:
                    result["interaction_modes"] = checked_modes
                    result["interaction_mode_adapter"] = mode_adapter
                    result["requested_control_protocol"] = control_protocol
            self.last_action_result = result
            return result

        try:
            requested = validate_action(action, schema_version, self.gripper_width_limits)
            checked_intent = validate_release_intent(release_intent)
            projected = requested.copy()
        except (TypeError, ValueError) as error:
            return answer(False, str(error))
        if not isinstance(components, (list, tuple)) or not components or any(not isinstance(c, str) or c not in {"motion", "gripper"} for c in components):
            return answer(False, "invalid_action_components")
        wants_motion, wants_gripper = "motion" in components, "gripper" in components
        if self.continuous_task:
            if self.task_state in {TaskState.FAIL, TaskState.DONE}:
                return answer(False, "episode_ended_reset_required")
            if self.contact_control.gesture_control:
                rate_control = self.contact_control.rate_control
                required_protocol = self.contact_control.control_protocol
                trusted_adapter = not rate_control and source in {"trusted_python", "scripted_test"}
                if control_protocol != required_protocol and not (control_protocol is None and trusted_adapter):
                    return answer(False, "control_protocol_required" if control_protocol is None else "control_protocol_mismatch")
                try:
                    if interaction_modes is None and trusted_adapter:
                        checked_modes = []
                        movements = []
                        for arm, offset in zip(self.arms, (0, 7)):
                            delta = requested[offset:offset+3]-arm.world_to_task(arm.command_position)
                            turning = rotation_angle(arm.task_rotation @ rpy_to_matrix(requested[offset+3:offset+6]), arm.command_rotation) > 1e-6
                            movements.append((delta, turning, wants_motion and (np.linalg.norm(delta) > 1e-7 or turning)))
                        any_moving = any(item[2] for item in movements)
                        for (delta, turning, moving), active in zip(movements, checked_intent):
                            if active:
                                selected = "RELEASE"
                            elif moving:
                                selected = "Z_APPROACH" if not turning and np.linalg.norm(delta[:2]) <= 1e-7 else "XY_HOLD"
                            else:
                                selected = "PINCH" if wants_gripper and not any_moving else "HOLD"
                            checked_modes.append(selected)
                        mode_adapter = {"name": "trusted_python_gesture_mode_defaults_v1", "source": source,
                            "interaction_modes": list(checked_modes), "network_eligible": False}
                    else:
                        checked_modes = validate_interaction_modes(interaction_modes, protocol=required_protocol)
                except ValueError as error:
                    return answer(False, str(error))
                release_mode = "Z_OPEN" if rate_control else "RELEASE"
                if any(active != (selected == release_mode) for active, selected in zip(checked_intent, checked_modes)):
                    return answer(False, "release_intent_mode_mismatch")
                planes = {"XY" if selected == "XY_HOLD" else "Z" for selected in checked_modes
                          if selected in {"XY_HOLD", "Z_HOLD", "Z_APPROACH", "PINCH", "Z_OPEN", "Z_CLOSE"}}
                if len(planes) > 1:
                    return answer(False, "interaction_motion_plane_conflict")
                if mode is not None and (str(mode) not in {"XY", "Z"} or (planes and str(mode) not in planes)):
                    return answer(False, "interaction_motion_plane_mismatch")
            projected, release_permits, reason = self.contact_control.project_action(requested, checked_intent,
                event_id=event_id, mode=mode, wants_motion=wants_motion, wants_gripper=wants_gripper,
                interaction_modes=checked_modes)
            positions = [a.task_to_world(projected[o:o+3]) for a, o in zip(self.arms, (0, 7))]
            rotations = [a.task_rotation @ rpy_to_matrix(projected[o+3:o+6]) for a, o in zip(self.arms, (0, 7))]
            reason = reason or (self.contact_control.release_reason(projected[[6, 13]]) if wants_gripper else None)
            reason = reason or (self.contact_control.motion_reason(positions, rotations) if wants_motion else None)
            if reason:
                return answer(False, reason)
            if wants_motion:
                if self.rigid_control:
                    reason = self.rigid_control.move(projected,
                        hold_targets=bool(checked_modes and all(m == "HOLD" for m in checked_modes)))
                    if reason:
                        preserve_gripper_baseline = bool(
                            self.contact_control.rate_control and
                            self.contact_control.gripper_trigger_policy == "palm_fist_edge_v2" and
                            reason in {"cartesian_step_limit", "rotation_step_limit", "workspace_limit",
                                "tracking_lead_limit", "ik_unreachable", "joint_limit",
                                "cooperative_axial_conflict", "cooperative_orientation_conflict",
                                "predicted_robot_robot_collision", "predicted_robot_obstacle_collision"})
                        return answer(False, reason)
                else:
                    if any(rotation_angle(rpy_to_matrix(requested[o+3:o+6]), np.asarray(self.config["scene"]["ee_rotation_matrix"])) > .03 for o in (0, 7)):
                        return answer(False, "limited_rotation_requires_M04")
                    deltas = [a.task_rotation.T @ (p-a.command_position) for a, p in zip(self.arms, positions)]
                    self._move_deltas(deltas)
            if wants_gripper:
                self.contact_control.commit_release(release_permits, event_id)
                for arm, width, original in zip(self.arms, projected[[6, 13]], requested[[6, 13]]):
                    arm.set_gripper_width(float(np.clip(width, 0, arm.max_gripper_width_m)))
                    arm.gripper_requested_width_m = float(original)
                    self.contact_control.hold_interventions += int(abs(width-original) > 1e-9)
                self.gripper_latch = GripperLatch.CONTINUOUS
            # Poses now reflect the common IK/rigid-motion projection; widths
            # retain accepted goals before the physics-rate actuator ramp.
            pose_fields = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
            projected[pose_fields] = self.action_vector_m()[pose_fields]
            self._last_command_sim_time = self.data.time
            self.paused, self.pause_reason = False, None
            self.control_mode = mode or ControlMode.IDLE
            if self.contact_control.gesture_control:
                contact = self.contact_control
                contact.aperture_direct_enabled = {side: bool(wants_gripper and selected == "PINCH"
                    and contact.hold_width_m[side] is None and contact.operation_hold_width_m[side] is None
                    and not contact.release_reentry_pending[side] and not contact.release_opening[side])
                    for side, selected in zip(("left", "right"), checked_modes)}
                if contact.rate_control:
                    for arm, selected in zip(self.arms, checked_modes):
                        direction = 1 if selected == "Z_OPEN" else -1 if selected == "Z_CLOSE" else 0
                        enabled = bool(wants_gripper and direction and
                            direction*(arm.gripper_goal_width_m-arm.gripper_command_width_m) > 1e-9)
                        contact.gripper_rate_enabled[arm.side] = enabled
                        contact.gripper_rate_direction[arm.side] = direction if enabled else 0
            return answer(True)
        if self.continuous_gripper:
            current = self.action_vector_m()
            pose_fields = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
            if wants_motion and np.any(np.abs(requested[pose_fields]-current[pose_fields]) > 1e-7):
                return answer(False, "continuous_task_motion_requires_M03")
            if self.task_state != TaskState.PREGRASP or self._relative_reference is not None:
                return answer(False, "continuous_validation_requires_reset")
            if wants_gripper:
                for arm, width in zip(self.arms, requested[[6, 13]]):
                    arm.set_gripper_width(float(np.clip(width, 0, arm.max_gripper_width_m)))
                self.gripper_latch = GripperLatch.CONTINUOUS
                self.paused, self.pause_reason = False, None
                self.control_mode = ControlMode.IDLE
                self._last_command_sim_time = self.data.time
            return answer(True)
        if wants_motion and self.task_state in {TaskState.FAIL, TaskState.DONE}:
            return answer(False, "episode_ended")
        frozen = self.task_state in {TaskState.GRASP_VERIFY, TaskState.RELEASE_VERIFY}
        if wants_motion and frozen and not wants_gripper:
            return answer(False, "task_motion_frozen")
        if mode is not None and mode not in {ControlMode.XY, ControlMode.Z}:
            return answer(False, "invalid_motion_mode")
        rotation = np.asarray(self.config["scene"]["ee_rotation_matrix"])
        if wants_motion and any(rotation_angle(rpy_to_matrix(requested[o+3:o+6]), rotation) >
                self.config["control"]["policy_orientation_tolerance_rad"] for o in (0, 7)):
            return answer(False, "policy_orientation_outside_v1_envelope")
        grip = None
        if wants_gripper:
            widths = requested[[6, 13]]
            opened = np.isclose(widths, self.gripper_width_limits, rtol=0, atol=1e-7)
            closed = np.isclose(widths, 0., rtol=0, atol=1e-7)
            if not np.all(opened | closed):
                return answer(False, "continuous_gripper_requires_M02")
            if opened[0] != opened[1]:
                return answer(False, "asymmetric_gripper_action")
            grip = "open" if opened[0] else "close"
            changing = (grip == "open") != (self.gripper_latch == GripperLatch.OPEN)
            # Check release/episode gates before any motion to avoid a rejected
            # combined request leaving partial target changes behind.
            if changing and grip == "open" and not self.release_ready:
                return answer(False, "release_rejected_stop_before_opening")
            if changing and grip == "close" and self.task_state in {TaskState.FAIL, TaskState.DONE}:
                return answer(False, "episode_ended_reset_required")
        self._last_command_sim_time = self.data.time
        if wants_motion and not frozen:
            deltas = [requested[o:o+3]-arm.world_to_task(arm.command_position) for arm, o in zip(self.arms, (0, 7))]
            if mode == ControlMode.XY:
                for delta in deltas:
                    delta[2] = 0
            elif mode == ControlMode.Z:
                for delta in deltas:
                    delta[:2] = 0
            maximum = self.config["control"]["max_ee_step_m"]
            deltas = [delta*min(1., maximum/max(float(np.linalg.norm(delta)), 1e-12)) for delta in deltas]
            self.control_mode, self.paused, self.pause_reason = mode or ControlMode.IDLE, False, None
            self._move_deltas(deltas)
        reason = None
        if wants_gripper:
            result = self._set_dual_gripper(grip)
            if not result["accepted"]:
                raise RuntimeError("Prechecked gripper condition changed during single-threaded execution")
            reason = result.get("reason")
        return answer(True, reason)

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
        """Explicit v1 checkpoint adapter; new policies must use apply_action."""
        try:
            action = vector(action)
            # Preserve old v1 threshold semantics only in this named adapter.
            action[[6, 13]] = np.where(action[[6, 13]] > 127.5, 255., 0.)
            metric = legacy_to_metric(action, self.gripper_width_limits, version=V1)
        except ValueError:
            return self._result(False, "invalid_policy_action")
        return self.apply_action(metric, schema_version=V2, source="legacy_policy_v1")

    def _contact_state(self):
        contacts = {"left_fingers": [False, False], "right_fingers": [False, False]}
        finger_forces = {side: [0.0, 0.0] for side in ("left", "right")}
        grip_forces = {side: [0.0, 0.0] for side in ("left", "right")}
        # Positive finger-joint travel opens the two opposing Panda fingers.
        # A load on an inside pad reacts in that finger's opening direction;
        # a top/outer-surface collision need not do so. Preserve signed sums
        # and all raw normal forces separately: this is a grasp qualification
        # check, not a complete friction-cone or grasp-wrench analysis.
        opening_axes = {arm.side: [
            self.data.xmat[self.model.jnt_bodyid[jid]].reshape(3, 3) @ self.model.jnt_axis[jid]
            for jid in arm.finger_joint_ids] for arm in self.arms}
        force = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if not ({g1, g2} & self.object_geoms):
                continue
            other = g2 if g1 in self.object_geoms else g1
            mujoco.mj_contactForce(self.model, self.data, index, force)
            for side in ("left", "right"):
                for finger, geom_ids in enumerate(self.finger_geoms[side]):
                    if other in geom_ids:
                        finger_forces[side][finger] += max(0.0, float(force[0]))
                        # MuJoCo's contact normal points from geom1 to geom2;
                        # normal force acts on geom2 along this direction.
                        on_finger = contact.frame[:3] * max(0.0, float(force[0])) * (1 if other == g2 else -1)
                        grip_forces[side][finger] += float(np.dot(on_finger, opening_axes[side][finger]))
                        if abs(force[0]) >= self.config["cooperation"]["minimum_contact_force_n"]:
                            contacts[f"{side}_fingers"][finger] = True
        if self.continuous_task:
            for side in ("left", "right"):
                contacts[f"{side}_fingers"] = [f >= self.config["cooperation"]["minimum_contact_force_n"] for f in finger_forces[side]]
        contacts["left"] = all(contacts["left_fingers"])
        contacts["right"] = all(contacts["right_fingers"])
        contacts["finger_normal_force_n"] = finger_forces
        for side in ("left", "right"):
            contacts[f"{side}_grip_forces_n"] = grip_forces[side]
        return contacts

    def _robot_robot_collision(self):
        for contact in self.data.contact[:self.data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if (g1 in self.robot_geoms["left"] and g2 in self.robot_geoms["right"]) or (g2 in self.robot_geoms["left"] and g1 in self.robot_geoms["right"]):
                return True
        return False

    def _robot_obstacle_collision(self):
        robots = self.robot_geoms["left"] | self.robot_geoms["right"]
        return any((int(c.geom1) in self.obstacle_geoms and int(c.geom2) in robots)
                   or (int(c.geom2) in self.obstacle_geoms and int(c.geom1) in robots)
                   for c in self.data.contact[:self.data.ncon])

    def _update_task(self):
        self.contacts = self._contact_state()
        if self.continuous_task:
            previous_task_state = self.task_state
            self.contact_control.update()
            if self.task_state == TaskState.DONE and previous_task_state != TaskState.DONE:
                # Freeze Cartesian motion immediately; retain only already
                # accepted opening goals for the configured recording tail.
                goals = [max(a.gripper_command_width_m, a.gripper_goal_width_m) for a in self.arms]
                self.pause("task_done")
                seconds = float(self.config.get('recording', {}).get('post_success_seconds', 0.))
                self.success_tail = {'active': seconds > 0, 'success_time': float(self.data.time),
                    'end_time': float(self.data.time) + seconds, 'seconds': seconds,
                    'opening_goals_m': goals, 'source': 'accepted_opening_goal_at_done',
                    'interrupted_reason': None}
            if self.episode_timeout_enabled and self.task_state not in {TaskState.DONE, TaskState.FAIL} and self._task_started_at is not None and self.task_elapsed_seconds >= self.config["evaluation"]["max_episode_seconds"]:
                self.pause("timeout", fail=True)
            return
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
            if not self.paused and (not self.continuous_gripper or self.continuous_task):
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
            if self.continuous_gripper and not self.paused:
                if self._last_command_sim_time is not None and self.data.time-self._last_command_sim_time > self.config["control"]["command_watchdog_seconds"]:
                    self.pause("command_watchdog")
            if self.continuous_gripper and not self.paused:
                if self.contact_control and self.contact_control.gesture_control:
                    if self.contact_control.rate_control:
                        self.contact_control.before_rate_gripper_step()
                    else:
                        self.contact_control.before_gesture_gripper_step()
                for arm in self.arms:
                    arm.advance_gripper()
            tail = getattr(self, 'success_tail', None)
            if tail and tail['active']:
                if self.task_state != TaskState.DONE or self.data.time >= tail['end_time'] - 1e-9:
                    tail['active'] = False
                    for arm in self.arms:
                        arm.hold_gripper()
                else:
                    for arm, goal in zip(self.arms, tail['opening_goals_m']):
                        arm.set_gripper_width(goal)
                        arm.advance_gripper()
            mujoco.mj_step(self.model, self.data)
            if self.task_state != TaskState.FAIL and self._robot_robot_collision():
                self.metrics.collision_count += 1
                self.pause("robot_robot_collision", fail=True)
            if self.task_state != TaskState.FAIL and self._robot_obstacle_collision():
                self.metrics.collision_count += 1
                self.pause("robot_obstacle_collision", fail=True)
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
        if self.continuous_task:
            return self.contact_control.support_ready
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
            "grippers": {arm.side: arm.gripper_status() for arm in self.arms},
            "release_ready": self.release_ready,
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "failure_reason": self.failure_reason,
            "success": self.task_state == TaskState.DONE,
            "success_tail": getattr(self, 'success_tail', None),
            "simulation_time": float(self.data.time),
            "task_elapsed_seconds": self.task_elapsed_seconds,
            "task_timer_started": self._task_started_at is not None,
            "episode_timeout_enabled": self.episode_timeout_enabled,
            "practice_mode": not self.episode_timeout_enabled,
            "object_position": self.data.xpos[self.object_id].tolist(),
            "target_position": self.target_position.tolist(),
            "object_yaw_rad": float(np.arctan2(self.data.xmat[self.object_id].reshape(3,3)[1,0],self.data.xmat[self.object_id].reshape(3,3)[0,0])),
            "target_yaw_rad": self.target_yaw,
            "target_orientation_error_rad": rotation_angle(self.data.xmat[self.object_id].reshape(3,3),rz(self.target_yaw)),
            "object_linear_velocity_mps": velocity[:3].tolist(),
            "object_angular_velocity_radps": velocity[3:].tolist(),
            "contacts": self.contacts,
            "physical_contact": self.contact_control.status() if self.contact_control else None,
            "motion_projection": self.rigid_control.last if self.rigid_control else None,
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
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(), "ctrl": self.data.ctrl.copy(), "body_pos": self.model.body_pos.copy(), "body_quat": self.model.body_quat.copy(), "time": float(self.data.time), "state": self.state_vector(), "action": self.action_vector_m() if self.continuous_task else self.action_vector(), "status": self.get_status(), "initial_state": self.initial_state}


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
