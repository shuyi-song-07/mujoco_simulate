"""Measure settled planar alignment from the live MuJoCo state."""
import math

import numpy as np

from simulation.mujoco.dual_arm.task_states import ControlMode, TaskState


CRITERION = "papera-h-frame-push-align-v1"


def alignment_errors(object_position, object_rotation, target_position, target_yaw):
    delta = np.subtract(object_position, target_position)
    yaw = math.atan2(object_rotation[1, 0], object_rotation[0, 0])
    return {
        "xy_m": float(np.linalg.norm(delta[:2])),
        "z_m": float(delta[2]),
        "orientation_rad": abs(math.remainder(yaw - target_yaw, math.tau)),
    }


class PushAlignmentMonitor:
    def __init__(self, config):
        self.limits = config["scene"]["push_alignment"]
        self.evaluation = config["evaluation"]
        half_x, half_y, half_z = config["scene"]["object_half_size"]
        self.corners = np.array([[x, y, z]
                                 for x in (-half_x, half_x)
                                 for y in (-half_y, half_y)
                                 for z in (-half_z, half_z)])
        self.stable_since = None
        self.last_time = None

    def update(self, controller):
        c = controller
        now = float(c.data.time)
        if self.last_time is not None and now < self.last_time:
            raise ValueError("PushAlignmentMonitor cannot be reused across resets")
        self.last_time = now
        position = c.data.xpos[c.object_id]
        rotation = c.data.xmat[c.object_id].reshape(3, 3)
        errors = alignment_errors(position, rotation, c.target_position, c.target_yaw)
        tilt = math.acos(float(np.clip(rotation[2, 2], -1, 1)))
        table = c.model.geom("table_geom").id
        table_rotation = c.data.geom_xmat[table].reshape(3, 3)
        world_corners = self.corners @ rotation.T + position
        table_corners = (world_corners - c.data.geom_xpos[table]) @ table_rotation
        inside = np.all(np.max(np.abs(table_corners[:, :2]), axis=0) <= c.model.geom_size[table, :2])
        support = c.contact_control.support_checks
        velocity = c.data.qvel[c.object_dof_adr:c.object_dof_adr + 6]
        failed = c.task_state == TaskState.FAIL
        conditions = {
            "position_xy": errors["xy_m"] <= self.limits["position_tolerance_m"],
            "yaw": errors["orientation_rad"] <= self.limits["yaw_tolerance_rad"],
            "upright": tilt <= self.limits["max_tilt_rad"],
            "inside_table": bool(inside),
            "table_support": all(support.get(key, False) for key in ("table_load", "clearance", "penetration")),
            "linear_speed": bool(np.linalg.norm(velocity[:3]) <= self.evaluation["stable_linear_speed_mps"]),
            "angular_speed": bool(np.linalg.norm(velocity[3:]) <= self.evaluation["stable_angular_speed_radps"]),
            "not_failed": not failed,
        }
        if not all(conditions.values()):
            self.stable_since = None
        elif self.stable_since is None:
            self.stable_since = now
        stable_seconds = 0.0 if self.stable_since is None else now - self.stable_since
        success = self.stable_since is not None and stable_seconds >= self.limits["stable_seconds"] - 1e-9
        return {
            "version": CRITERION,
            "success_criterion": CRITERION,
            "task_success_criterion": CRITERION,
            "task_success": success,
            "success": success,
            "physical_success": success,
            "terminal": success or failed,
            "reason": (c.failure_reason or "controller_failure") if failed else None,
            "target_error": errors,
            "tilt_rad": tilt,
            "alignment_stable_seconds": stable_seconds,
            "final_conditions": conditions,
        }


def collection_success(contact_control, controller, now, object_pos, object_rot, any_contact):
    # ContactControl is recreated on every scene reset.
    monitor = getattr(contact_control, "push_alignment_monitor", None)
    if monitor is None:
        monitor = contact_control.push_alignment_monitor = PushAlignmentMonitor(controller.config)
    result = monitor.update(controller)
    controller._success_stable_since = monitor.stable_since
    if result["task_success"]:
        controller.task_state = TaskState.DONE
        controller.control_mode = ControlMode.IDLE
        controller.metrics.completion_time_seconds = controller.task_elapsed_seconds


def rollout_monitor(config, initial_state, **options):
    return PushAlignmentMonitor(config)
