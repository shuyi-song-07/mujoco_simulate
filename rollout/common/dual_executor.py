"""Direct metric policy targets with the existing physical Task3 safeguards.

This is an isolated deployment adapter, not a gesture impersonator. The model
supplies absolute left/right xyz, rpy and total finger widths in metres. It
never supplies modes, contact labels, success labels, or object poses.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from simulation.mujoco.dual_arm.contact_control import ContactControl
from simulation.mujoco.dual_arm.task_states import ControlMode, GripperLatch, TaskState
from simulation.mujoco.dual_arm.panda_arm_controller import matrix_to_rpy, rotation_angle, rpy_to_matrix
from simulation.mujoco.dual_arm.rigid_control import rz, yaw_of


EXECUTOR_VERSION = "papera-direct-metric-v1"
DEFAULT_EXECUTOR_CONFIG = {
    "version": EXECUTOR_VERSION,
    "release_deadband_m": 0.002,
    "control_hz": 60,
    "max_orientation_projection_rad": 0.35,
    "orientation_policy": "supported_world_yaw_projection",
    "coupled_policy": "absolute_two_position_rigid_fit",
}


class PolicyContactControl(ContactControl):
    """Retain force/support/success observation, replace human input leases.

The inherited observer requires opposing finger pressure, a preceding active
closure, and stable real contact. Its slip/drop/collision/settling checks stay
active. Support is evidence for *success*, never permission for model opening.
"""

    def __init__(self, controller, release_deadband_m):
        super().__init__(controller)
        self.release_deadband_m = float(release_deadband_m)
        self.control_protocol = EXECUTOR_VERSION
        # The old rate observer has the appropriate signed-force qualification.
        # Its camera and categorical-state hooks below are deliberately replaced.
        self.rate_control = self.gesture_control = self.intent_control = True
        self.edge_control = True

    def cancel_release(self, reason="policy_hold"):
        for arm in self.c.arms:
            arm.hold_gripper()

    def before_rate_gripper_step(self):
        # Panda.advance_gripper applies the shared 0.08 m/s servo ramp.
        pass

    def _refresh_release_anchors(self):
        self.unsupported_width_anchor = {s: None for s in self.since}

    def _update_holds(self):
        for arm in self.c.arms:
            side = arm.side
            if self.release_opening[side] and not any(self.c.contacts[f"{side}_fingers"]):
                self._released_contact_cleared[side] = True
            if self.release_opening[side] and self._released_contact_cleared[side]:
                self.release_opening[side] = False
            if (self.stable[side] and self.hold_width_m[side] is None
                    and self.grasp_close_qualified[side] and not self.release_opening[side]):
                self.hold_width_m[side] = arm.gripper_command_width_m
                arm.hold_gripper()
                self.release_executed[side] = False

    def proposed_widths(self, requested):
        widths, releases = [], []
        for arm, value in zip(self.c.arms, requested):
            held = self.hold_width_m[arm.side]
            width = float(np.clip(value, 0., arm.max_gripper_width_m))
            release = held is not None and width > held + self.release_deadband_m
            widths.append(held if held is not None and not release else width)
            if release:
                releases.append(arm.side)
        return np.asarray(widths), releases

    def commit_widths(self, requested, widths, releases):
        for arm, original, width in zip(self.c.arms, requested, widths):
            side = arm.side
            if side in releases:
                self.hold_width_m[side] = None
                self.stable[side], self.since[side] = False, None
                self.grasp_close_qualified[side] = False
                self.release_executed[side] = self.release_opening[side] = True
                self._released_contact_cleared[side] = False
                self.released = True
                self.reference = None
                self.c._relative_reference = None
                self.danger_since = None
                self.release_count += 1
            elif self.hold_width_m[side] is None:
                if width < arm.gripper_command_width_m - 1e-7:
                    self.grasp_close_qualified[side] = True
                    self.release_opening[side] = False
                    self.release_executed[side] = False
                elif width > arm.gripper_command_width_m + 1e-7:
                    # A fully open hand pushing down on a handle is not a grasp.
                    self.grasp_close_qualified[side] = False
            arm.set_gripper_width(float(width))
            arm.gripper_requested_width_m = float(original)
            self.hold_interventions += int(abs(width-original) > 1e-7)

    def status(self):
        result = super().status()
        result.update(control_protocol=EXECUTOR_VERSION, human_input_required=False,
                      release_deadband_m=self.release_deadband_m,
                      release_requires_support=False, release_requires_stationary=False)
        return result


class PolicyExecutor:
    """Attach after reset; apply once per 60 Hz control tick, then c.step(dt).

Applying a repeated absolute target moves toward that target; it does not replay
the old hand displacement. Resetting the controller invalidates this adapter.
"""

    def __init__(self, controller, config: dict | None = None):
        self.c = controller
        self.config = copy.deepcopy(DEFAULT_EXECUTOR_CONFIG)
        if config:
            unknown = set(config)-set(self.config)
            if unknown:
                raise ValueError(f"unknown_executor_parameters:{sorted(unknown)}")
            self.config.update(copy.deepcopy(config))
        if self.config["version"] != EXECUTOR_VERSION or self.config["control_hz"] != 60:
            raise ValueError("unsupported_policy_executor_contract")
        if self.config["orientation_policy"] != DEFAULT_EXECUTOR_CONFIG["orientation_policy"] or self.config["coupled_policy"] != DEFAULT_EXECUTOR_CONFIG["coupled_policy"]:
            raise ValueError("unsupported_policy_projection")
        deadband = self.config["release_deadband_m"]
        if isinstance(deadband, bool) or not isinstance(deadband, (int, float)) or not np.isfinite(deadband) or not 0 < deadband < .01:
            raise ValueError("release_deadband_m_must_be_between_zero_and_0.01")
        tolerance = self.config["max_orientation_projection_rad"]
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or not 0 < tolerance <= .5:
            raise ValueError("invalid_orientation_projection_tolerance")
        if not controller.continuous_task or controller.rigid_control is None:
            raise ValueError("continuous_contact_and_rigid_control_required")
        if controller.contact_control.ever_dual or any(controller.contact_control.stable.values()):
            raise ValueError("attach_policy_executor_after_reset_before_grasp")
        controller.contact_control = PolicyContactControl(controller, deadband)
        self.contact = controller.contact_control
        self.calls = self.intervention_count = self.rejection_count = 0

    def _project_motion(self, action):
        c, contact, rigid = self.c, self.contact, self.c.rigid_control
        result = action.copy()
        requested_p = [a.task_to_world(action[o:o+3]) for a, o in zip(c.arms, (0, 7))]
        requested_r = [a.task_rotation @ rpy_to_matrix(action[o+3:o+6]) for a, o in zip(c.arms, (0, 7))]
        report: dict[str, Any] = {"coupled": contact.reference is not None,
                                  "partial_hold_sides": [], "orientation_residual_rad": [],
                                  "rigid_position_residual_m": 0.}
        if contact.reference is not None:
            refs = contact.reference
            # Position-only least-squares XY yaw fit, then the full XYZ centre.
            # Human pose predictions cannot stretch the established rigid grasp.
            ref_axis = refs[1][0]-refs[0][0]
            predicted_axis = requested_p[1]-requested_p[0]
            if np.linalg.norm(predicted_axis[:2]) < 1e-6 or np.linalg.norm(ref_axis[:2]) < 1e-6:
                return result, report, "degenerate_rigid_prediction"
            raw_yaw = math.atan2(predicted_axis[1], predicted_axis[0])-math.atan2(ref_axis[1], ref_axis[0])
            raw_yaw = math.atan2(math.sin(raw_yaw), math.cos(raw_yaw))
            target_yaw = float(np.clip(raw_yaw, -rigid.limits["max_yaw_rad"], rigid.limits["max_yaw_rad"]))
            if rigid.reference_id is refs:
                current_p, current_r = rigid.object_position, rigid.object_rotation
            else:
                current_p = c.data.xpos[c.object_id].copy()
                current_r = c.data.xmat[c.object_id].reshape(3, 3).copy()
            target_r = rz(target_yaw-yaw_of(current_r)) @ current_r
            target_p = np.mean([p-target_r @ ref[0] for p, ref in zip(requested_p, refs)], axis=0)
            fitted_p = [target_p+target_r @ ref[0] for ref in refs]
            fitted_r = [target_r @ ref[1] for ref in refs]
            report.update(rigid_position_residual_m=float(max(np.linalg.norm(p-q) for p, q in zip(requested_p, fitted_p))),
                          desired_object_position=target_p.tolist(), desired_object_yaw_rad=target_yaw)
            report["orientation_residual_rad"] = [rotation_angle(p, q) for p, q in zip(requested_r, fitted_r)]
            # RigidControl.move consumes a COMMON object displacement plus yaw.
            # Recompute it from its accepted absolute object target every call.
            common = target_p-current_p
            yaw_delta = target_yaw-yaw_of(current_r)
            for arm, offset in zip(c.arms, (0, 7)):
                result[offset:offset+3] = arm.world_to_task(arm.command_position+common)
                result[offset+3:offset+6] = matrix_to_rpy(arm.task_rotation.T @ rz(yaw_delta) @ arm.command_rotation)
        else:
            for arm, offset, p, r in zip(c.arms, (0, 7), requested_p, requested_r):
                if contact.hold_width_m[arm.side] is not None:
                    result[offset:offset+3] = arm.world_to_task(arm.command_position)
                    accepted_r = arm.command_rotation
                    report["partial_hold_sides"].append(arm.side)
                else:
                    bounded = np.clip(arm.world_to_task(p), arm.workspace[:, 0], arm.workspace[:, 1])
                    result[offset:offset+3] = bounded
                    yaw = float(np.clip(yaw_of(r @ arm.fixed_rotation.T), -rigid.limits["max_yaw_rad"], rigid.limits["max_yaw_rad"]))
                    accepted_r = rz(yaw) @ arm.fixed_rotation
                result[offset+3:offset+6] = matrix_to_rpy(arm.task_rotation.T @ accepted_r)
                report["orientation_residual_rad"].append(rotation_angle(r, accepted_r))
        if max(report["orientation_residual_rad"], default=0.) > self.config["max_orientation_projection_rad"]:
            return result, report, "policy_orientation_residual_exceeded"
        return result, report, None

    def apply(self, action, dt=1/60):
        c = self.c
        self.calls += 1
        if c.contact_control is not self.contact:
            raise RuntimeError("controller_reset_requires_new_policy_executor")
        if not np.isfinite(dt) or not np.isclose(dt, 1/self.config["control_hz"], atol=1e-12):
            raise ValueError("policy_executor_requires_60_hz_control_ticks")
        requested, projected, projection = None, None, {}

        def answer(accepted, reason=None):
            target = c.action_vector_m().astype(float)
            # action_vector_m contains issued servo widths; also expose goals.
            goal = target.copy()
            goal[[6, 13]] = [a.gripper_goal_width_m for a in c.arms]
            difference = None if requested is None else requested-goal
            if difference is not None:
                # RPY is periodic. An equivalent +pi/-pi representation is not
                # an intervention, although raw requested/accepted vectors are
                # retained above for exact reproduction of the prediction.
                angles = [3, 4, 5, 10, 11, 12]
                difference[angles] = (difference[angles]+np.pi) % (2*np.pi)-np.pi
            changed = [] if difference is None else np.flatnonzero(np.abs(difference) > 1e-7).tolist()
            intervened = bool(changed or not accepted)
            self.intervention_count += int(intervened)
            self.rejection_count += int(not accepted)
            if not accepted:
                c.metrics.rejected_command_count += 1
            record = dict(accepted=bool(accepted), reason=reason, source="policy_rollout", executor_version=EXECUTOR_VERSION,
                          requested_target=None if requested is None else requested.tolist(),
                          projected_target=None if projected is None else projected.tolist(),
                          accepted_target=target.tolist(), accepted_goal=goal.tolist(),
                          actual_state=c.state_vector().tolist(), intervened=intervened,
                          intervened_fields=changed,
                          target_residual_wrapped=None if difference is None else difference.tolist(),
                          projection=projection,
                          physical_contact=self.contact.status(), simulation_time=float(c.data.time))
            c.last_action_result = record
            return record

        try:
            if isinstance(action, (str, bytes)):
                raise ValueError
            requested = np.asarray(action, dtype=float)
            if requested.shape != (14,) or not np.isfinite(requested).all():
                requested = None
                raise ValueError
        except (TypeError, ValueError):
            return answer(False, "invalid_finite_14d_policy_action")
        if c.task_state in {TaskState.FAIL, TaskState.DONE}:
            return answer(False, "episode_ended_reset_required")
        widths, releases = self.contact.proposed_widths(requested[[6, 13]])
        projected, projection, reason = self._project_motion(requested)
        projected[[6, 13]] = widths
        if reason:
            return answer(False, reason)
        if self.contact.danger_since is not None and not releases:
            return answer(False, "contact_recovery_required")
        # No release/gripper transaction is committed unless BOTH IK candidates
        # and swept robot collision checks pass. RigidControl commits atomically.
        reason = c.rigid_control.move(projected)
        projection["motion_guard"] = copy.deepcopy(c.rigid_control.last)
        if reason:
            return answer(False, reason)
        self.contact.commit_widths(requested[[6, 13]], widths, releases)
        projection["released_sides"] = releases
        c.gripper_latch = GripperLatch.CONTINUOUS
        c._last_command_sim_time = float(c.data.time)
        c.paused, c.pause_reason = False, None
        c.control_mode = ControlMode.IDLE
        c.start_task_timer()
        return answer(True)
