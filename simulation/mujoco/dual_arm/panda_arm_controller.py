"""Namespaced Panda Cartesian DLS control with explicit task-frame targets."""
from __future__ import annotations

import numpy as np
import mujoco


def rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 0.5 * sum(np.cross(current[:, axis], target[:, axis]) for axis in range(3))


def rotation_angle(current: np.ndarray, target: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(current.T @ target) - 1) / 2, -1, 1)))


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    return np.array([np.arctan2(rotation[2, 1], rotation[2, 2]), np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0])), np.arctan2(rotation[1, 0], rotation[0, 0])])


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


class PandaArmController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, side: str, config: dict):
        if side not in {"left", "right"}:
            raise ValueError("Arm side must be left or right")
        self.model, self.data, self.side, self.config = model, data, side, config
        self.joint_ids = np.array([model.joint(f"{side}_joint{i}").id for i in range(1, 8)])
        self.qpos_indices = model.jnt_qposadr[self.joint_ids]
        self.dof_indices = model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.array([model.actuator(f"{side}_actuator{i}").id for i in range(1, 8)])
        self.gripper_actuator_id = model.actuator(f"{side}_actuator8").id
        self.finger_joint_ids = np.array([model.joint(f"{side}_finger_joint{i}").id for i in (1, 2)])
        self.finger_qpos_indices = model.jnt_qposadr[self.finger_joint_ids]
        self.site_id = model.site(f"{side}_ee").id
        self.base_id = model.body(f"{side}_link0").id
        self.ik_data = mujoco.MjData(model)
        self.jacp, self.jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        quat = np.asarray(config["scene"]["task_frame_quaternion_wxyz"], dtype=float)
        quat /= np.linalg.norm(quat)
        flat = np.zeros(9)
        mujoco.mju_quat2Mat(flat, quat)
        self.task_rotation = flat.reshape(3, 3)
        self.task_origin = np.asarray(config["scene"]["task_frame_position"], dtype=float)
        self.fixed_rotation = self.task_rotation @ np.asarray(config["scene"]["ee_rotation_matrix"], dtype=float)
        self.workspace = np.asarray(config["scene"][f"{side}_workspace"], dtype=float)
        self.command_position = np.zeros(3)
        self.command_rotation = self.fixed_rotation.copy()
        self.workspace_clamped = False
        self.workspace_violation_count = 0
        self.ik_position_error_m = 0.0
        self.ik_orientation_error_rad = 0.0
        self.gripper_target = 255.0
        self._ik_tick_time = None
        self._ik_tick_origin = None

    def task_to_world(self, point):
        return self.task_origin + self.task_rotation @ np.asarray(point, dtype=float)

    def world_to_task(self, point):
        return self.task_rotation.T @ (np.asarray(point) - self.task_origin)

    def get_actual_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.site_xpos[self.site_id].copy(), self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def get_command_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.command_position.copy(), self.command_rotation.copy()

    def _copy_commanded_configuration(self):
        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qpos[self.qpos_indices] = self.data.ctrl[self.actuator_ids]
        self.ik_data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.ik_data)

    def _ik_iteration(self, position, rotation, max_step):
        current_pos = self.ik_data.site_xpos[self.site_id]
        current_rot = self.ik_data.site_xmat[self.site_id].reshape(3, 3)
        mujoco.mj_jacSite(self.model, self.ik_data, self.jacp, self.jacr, self.site_id)
        jac = np.vstack((self.jacp[:, self.dof_indices], self.jacr[:, self.dof_indices]))
        delta = np.r_[position - current_pos, self.config["control"]["orientation_gain"] * rotation_error(current_rot, rotation)]
        damping = self.config["control"]["dls_damping"]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), delta)
        dq = np.clip(dq, -max_step, max_step)
        ranges = self.model.jnt_range[self.joint_ids]
        self.ik_data.qpos[self.qpos_indices] = np.clip(self.ik_data.qpos[self.qpos_indices] + dq, ranges[:, 0]+0.001, ranges[:, 1]-0.001)
        mujoco.mj_forward(self.model, self.ik_data)

    def reset(self, home_qpos=None, gripper="open"):
        nominal = self.config["scene"].get(f"{self.side}_home_qpos", self.config["scene"]["home_qpos"]) if home_qpos is None else home_qpos
        self.data.qpos[self.qpos_indices] = nominal
        self.data.ctrl[self.actuator_ids] = nominal
        self.data.qpos[self.finger_qpos_indices] = 0.04 if gripper == "open" else 0
        self.set_gripper(gripper)
        self.command_position = self.task_to_world(self.config["scene"][f"{self.side}_home_ee"])
        self.command_rotation = self.fixed_rotation.copy()
        self._copy_commanded_configuration()
        for _ in range(600):
            self._ik_iteration(self.command_position, self.fixed_rotation, 0.1)
            error = np.linalg.norm(self.ik_data.site_xpos[self.site_id]-self.command_position)
            angle = rotation_angle(self.ik_data.site_xmat[self.site_id].reshape(3, 3), self.fixed_rotation)
            if error < 1e-5 and angle < 1e-4:
                break
        if error > 0.005 or angle > 0.03:
            raise ValueError(f"Unreachable {self.side} home pose: position error={error:.4f}m, orientation={angle:.4f}rad")
        self.data.qpos[self.qpos_indices] = self.ik_data.qpos[self.qpos_indices]
        self.data.ctrl[self.actuator_ids] = self.ik_data.qpos[self.qpos_indices]
        self.workspace_clamped = False
        self.workspace_violation_count = 0
        self._ik_tick_time = None
        self._ik_tick_origin = None
        mujoco.mj_forward(self.model, self.data)

    def set_gripper(self, action):
        if action not in {"open", "close"}:
            raise ValueError("Gripper action must be open or close")
        self.gripper_target = 255.0 if action == "open" else 0.0
        self.data.ctrl[self.gripper_actuator_id] = self.gripper_target

    def apply_limits(self, target_world):
        task_target = self.world_to_task(target_world)
        bounded = np.clip(task_target, self.workspace[:, 0], self.workspace[:, 1])
        self.workspace_clamped = bool(np.linalg.norm(bounded-task_target) > 1e-8)
        if self.workspace_clamped:
            self.workspace_violation_count += 1
        else:
            self.workspace_violation_count = max(0, self.workspace_violation_count - 1)
        result = self.task_to_world(bounded)
        actual = self.data.site_xpos[self.site_id]
        lead = result-actual
        max_lead = self.config["control"]["max_cartesian_target_lead_m"]
        if np.linalg.norm(lead) > max_lead:
            result = actual + lead * max_lead / np.linalg.norm(lead)
        return result

    def move_delta_task_frame(self, dx, dy, dz):
        delta = np.asarray([dx, dy, dz], dtype=float)
        if not np.all(np.isfinite(delta)):
            raise ValueError("Nonfinite Cartesian input")
        self.command_position = self.apply_limits(self.command_position + self.task_rotation @ delta)
        self.track_target()

    def track_target(self):
        self._copy_commanded_configuration()
        before = self.data.ctrl[self.actuator_ids].copy()
        if self._ik_tick_time != self.data.time:
            self._ik_tick_time = float(self.data.time)
            self._ik_tick_origin = before.copy()
        max_step = self.config["control"]["max_joint_step_rad"]
        for _ in range(3):
            self._ik_iteration(self.command_position, self.command_rotation, max_step / 3)
        next_target = before + np.clip(self.ik_data.qpos[self.qpos_indices]-before, -max_step, max_step)
        # Several commands may arrive before one physics tick; their combined
        # actuator change still shares one joint-step budget.
        next_target = self._ik_tick_origin + np.clip(next_target-self._ik_tick_origin, -max_step, max_step)
        actual = self.data.qpos[self.qpos_indices]
        lead = next_target-actual
        max_lead = self.config["control"]["max_joint_target_lead_rad"]
        if np.max(np.abs(lead)) > max_lead:
            lead *= max_lead/np.max(np.abs(lead))
        limits = self.model.jnt_range[self.joint_ids]
        self.data.ctrl[self.actuator_ids] = np.clip(actual+lead, limits[:, 0]+0.001, limits[:, 1]-0.001)
        self.ik_position_error_m = float(np.linalg.norm(self.command_position-self.data.site_xpos[self.site_id]))
        self.ik_orientation_error_rad = rotation_angle(self.data.site_xmat[self.site_id].reshape(3, 3), self.command_rotation)

    def freeze(self):
        # Holding the current servo target avoids dropping a loaded gripper.
        self._copy_commanded_configuration()
        self.command_position = self.ik_data.site_xpos[self.site_id].copy()
        self.command_rotation = self.fixed_rotation.copy()

    def hold_xy(self):
        self.track_target()

    def hold_z(self):
        self.track_target()

    def check_health(self):
        finite = bool(np.all(np.isfinite(self.data.qpos[self.qpos_indices])) and np.all(np.isfinite(self.data.ctrl[self.actuator_ids])))
        return {"finite": finite, "workspace_clamped": self.workspace_clamped, "workspace_violation_count": self.workspace_violation_count, "position_error_m": self.ik_position_error_m, "orientation_error_rad": self.ik_orientation_error_rad}

    def state_vector(self):
        return np.r_[self.data.qpos[self.qpos_indices], np.sum(self.data.qpos[self.finger_qpos_indices])].astype(np.float32)

    def action_vector(self):
        return np.r_[self.world_to_task(self.command_position), matrix_to_rpy(self.task_rotation.T @ self.command_rotation), self.gripper_target].astype(np.float32)
