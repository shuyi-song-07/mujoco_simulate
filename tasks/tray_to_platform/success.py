"""Task-specific physical completion shared by collection and policy rollout.

No poses, forces, contacts or grasp history are synthesized here. Collection
only marks DONE after the same measured history and final conditions used by
rollout. The shared contact controller remains responsible for physical safety.
"""
from __future__ import annotations
import copy
import math
import numpy as np
from simulation.mujoco.dual_arm.task_states import TaskState, ControlMode

SIDES = ("left", "right")
CRITERION = "papera-highwall-platform-contained-v2"
OPENING_CRITERION = "papera-physical-half-open-v1"


def _finite(value):
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))


class PlatformSuccessMonitor:
    """Track independently evidenced dual grasp, lift, and platform release."""
    def __init__(self, config, initial_state, min_lift_m=None, *, minimum_open_fraction=None, success_criterion=None):
        self.config = copy.deepcopy(config)
        task = config["scene"]["platform_task"]
        self.min_lift_m = float(task["minimum_lift_m"])
        self.minimum_open_fraction = float(task["minimum_open_fraction"])
        self.lift_stable_seconds = float(task["lift_stable_seconds"])
        for supplied, required, name in ((min_lift_m, self.min_lift_m, "minimum_lift_m"),
                                        (minimum_open_fraction, self.minimum_open_fraction, "minimum_open_fraction")):
            if supplied is not None and (not _finite(supplied) or abs(float(supplied)-required) > 1e-12):
                raise ValueError(f"{name} must match the frozen platform task: {required}")
        if not (self.min_lift_m > 0 and self.lift_stable_seconds > 0 and 0 < self.minimum_open_fraction <= 1):
            raise ValueError("Invalid platform completion parameters")
        if success_criterion not in (None, OPENING_CRITERION, CRITERION):
            raise ValueError(f"Unsupported platform criterion: {success_criterion!r}")
        self.success_criterion = CRITERION
        self.requested_success_criterion = success_criterion
        self.initial_z = float(initial_state["object_position"][2])
        self.dual_since = self.lift_since = self.placement_since = None
        self.ever_dual = self.lift_verified = False
        self.first_dual_time_s = self.first_lift_time_s = None
        self.max_lift_m = 0.
        self.last_time = None
        self.last_result = None

    def update(self, controller):
        status = controller.get_status()
        now = float(status["simulation_time"])
        if self.last_time is not None and now < self.last_time - 1e-9:
            raise ValueError("PlatformSuccessMonitor cannot be reused across resets")
        self.last_time = now
        contacts = status["contacts"]
        physical = status.get("physical_contact") or {}
        cooperation = self.config["cooperation"]
        minimum_force = cooperation["minimum_contact_force_n"]
        opposed = {}
        for side in SIDES:
            forces = contacts.get(f"{side}_grip_forces_n", [])
            fingers = contacts.get(f"{side}_fingers", [])
            gripper = status["grippers"][side]
            opposed[side] = bool(contacts.get(side) and len(fingers) == 2 and all(fingers)
                and len(forces) == 2 and all(_finite(f) and f >= minimum_force for f in forces)
                and gripper["command_width_m"] < gripper["max_width_m"]-1e-6)
        dual = all(opposed.values())
        self.dual_since = (now if self.dual_since is None else self.dual_since) if dual else None
        stable_dual = dual and now-self.dual_since >= cooperation["grasp_stable_seconds"]-1e-9
        if stable_dual and not self.ever_dual:
            self.ever_dual, self.first_dual_time_s = True, now
        lift = float(status["object_position"][2])-self.initial_z
        self.max_lift_m = max(self.max_lift_m, lift)
        bottom_relative = physical.get("bottom_height_m")
        surface = physical.get("support_surface")
        # The shared support measurement uses the actual lowest collision-box
        # corner relative to the currently selected support surface.
        surface_height = {"table": self.config["scene"]["table_top_z"],
            "placement_platform": self.config["scene"]["table_top_z"] + self.config["scene"]["placement_platform"]["height_m"]}.get(surface)
        bottom_clearance = (float(bottom_relative)+surface_height-self.config["scene"]["table_top_z"]
            if _finite(bottom_relative) and surface_height is not None else None)
        lifting = bool(stable_dual and lift >= self.min_lift_m and bottom_clearance is not None
                       and bottom_clearance >= self.min_lift_m)
        self.lift_since = (now if self.lift_since is None else self.lift_since) if lifting else None
        if lifting and now-self.lift_since >= self.lift_stable_seconds-1e-9 and not self.lift_verified:
            self.lift_verified, self.first_lift_time_s = True, now
        evaluation = self.config["evaluation"]
        position, target = np.asarray(status["object_position"]), np.asarray(status["target_position"])
        no_contact = all(len(contacts.get(f"{s}_fingers", [])) == 2 and not any(contacts[f"{s}_fingers"]) for s in SIDES)
        open_thresholds = {s: status["grippers"][s]["max_width_m"]*self.minimum_open_fraction for s in SIDES}
        load = physical.get("support_forces_n", {}).get("placement_platform")
        required_load = physical.get("support_requirements", {}).get("minimum_table_force_n")
        conditions = {
            "position_xy": float(np.linalg.norm(position[:2]-target[:2])) <= evaluation["target_position_tolerance_m"],
            "height": abs(float(position[2]-target[2])) <= evaluation["target_height_tolerance_m"],
            "linear_speed": float(np.linalg.norm(status["object_linear_velocity_mps"])) <= evaluation["stable_linear_speed_mps"],
            "angular_speed": float(np.linalg.norm(status["object_angular_velocity_radps"])) <= evaluation["stable_angular_speed_radps"],
            "target_platform_support": surface == "placement_platform" and physical.get("support_ready") is True,
            "measured_platform_load": _finite(load) and _finite(required_load) and required_load > 0 and load >= required_load,
            "tray_contained_on_platform": physical.get("placement", {}).get("contained") is True,
            "no_finger_contact": no_contact,
            "both_commanded_open": all(status["grippers"][s]["command_width_m"] >= open_thresholds[s] for s in SIDES),
            "both_actually_open": all(status["grippers"][s]["actual_width_m"] >= open_thresholds[s] for s in SIDES),
            "no_collision": status.get("collision_count", 0) == 0,
            "not_failed": status["task_state"] != "FAIL",
        }
        good = all(conditions.values())
        self.placement_since = (now if self.placement_since is None else self.placement_since) if good else None
        placement = good and now-self.placement_since >= evaluation["stable_success_seconds"]-1e-9
        physical_success = bool(placement and self.ever_dual and self.lift_verified)
        done = status["task_state"] == "DONE"
        success = physical_success and done
        terminal = success or status["task_state"] == "FAIL" or (done and not (self.ever_dual and self.lift_verified))
        reason = None
        if status["task_state"] == "FAIL":
            reason = status.get("failure_reason") or "controller_failure"
        elif done and not self.ever_dual:
            reason = "placement_without_verified_dual_grasp"
        elif done and not self.lift_verified:
            reason = "placement_without_verified_lift"
        elif done and not physical_success:
            reason = "awaiting_final_physical_settle"
        elif physical_success and not done:
            reason = "awaiting_task_done"
        self.last_result = {
            "version": CRITERION, "success_criterion": CRITERION,
            "task_success_criterion": CRITERION, "opening_criterion": OPENING_CRITERION,
            "orientation_required": False,
            "requested_success_criterion": self.requested_success_criterion,
            "minimum_open_fraction": self.minimum_open_fraction,
            "minimum_open_widths_m": open_thresholds,
            "task_success": success, "success": success, "physical_success": physical_success,
            "placement_success": bool(placement), "terminal": bool(terminal), "reason": reason,
            "ever_dual": self.ever_dual, "lift_verified": self.lift_verified,
            "current_opposed_grasps": opposed, "max_lift_m": self.max_lift_m,
            "lift_above_initial_m": lift, "min_lift_m": self.min_lift_m,
            "tray_bottom_clearance_above_table_m": bottom_clearance,
            "first_dual_time_s": self.first_dual_time_s, "first_lift_time_s": self.first_lift_time_s,
            "placement_stable_seconds": 0. if self.placement_since is None else now-self.placement_since,
            "target_error": {"xy_m": float(np.linalg.norm(position[:2]-target[:2])),
                "z_m": float(position[2]-target[2]), "orientation_rad": float(status["target_orientation_error_rad"])},
            "final_conditions": conditions,
        }
        return self.last_result


def collection_success(contact_control, controller, now, object_pos, object_rot, any_contact):
    monitor = getattr(contact_control, "platform_task_monitor", None)
    if monitor is None:
        monitor = contact_control.platform_task_monitor = PlatformSuccessMonitor(controller.config, controller.initial_state)
    result = monitor.update(controller)
    controller._success_stable_since = monitor.placement_since
    if result["physical_success"] and controller.task_state != TaskState.FAIL:
        controller.task_state, controller.control_mode = TaskState.DONE, ControlMode.IDLE
        controller.metrics.completion_time_seconds = controller.task_elapsed_seconds
        result.update(task_success=True, success=True, terminal=True, reason=None)


def rollout_monitor(config, initial_state, **options):
    return PlatformSuccessMonitor(config, initial_state, **options)
