"""Physical grasp/release state independent of gesture labels and episode outcome."""
from __future__ import annotations

import mujoco
import numpy as np

from .task_states import TaskState, ControlMode
from .panda_arm_controller import rotation_angle, rpy_to_matrix
from .rigid_control import rz
from .action_contract import GRASP_HOLD_RELEASE_V1, GESTURE_GRASP_RELEASE_V2, GESTURE_RATE_GRIPPER_V3
from .gesture_control import GestureControlMixin
from .gesture_action_control import RateGestureControlMixin
from .task_scenes import target_surface_z, placement_geometry


class ContactControl(RateGestureControlMixin, GestureControlMixin):
    def __init__(self, controller):
        self.c = controller
        self.limits = controller.config["contact_control"]
        self.since = {s: None for s in ("left", "right")}
        self.stable = {s: False for s in self.since}
        self.support_since = None
        self.support_force_n = 0.
        self.support_surface = 'table'
        self.support_forces_n = {'table': 0.}
        self.bottom_height_m = 0.
        self.support_ready = False
        self.support_checks = {}
        self.support_blockers = []
        self.support_requirements = {}
        self.support_measurements = {}
        self._input_clock = getattr(controller, "_contact_input_clock", None)
        self.release_clock = getattr(controller, "_contact_input_clock_name", "simulation")
        self.danger_since = None
        self.ever_dual = False
        self.released = False
        self.reference = None
        self.state = "FREE"
        self.events = []
        # A small opening tolerance is a bounded allowance for noise, not a
        # per-packet allowance that can accumulate into an airborne release.
        self.unsupported_width_anchor = {s: None for s in self.since}
        self.release_guard_interventions = 0
        self.control_protocol = self.limits.get("control_protocol", "legacy-continuous-v1")
        self.rate_control = self.control_protocol == GESTURE_RATE_GRIPPER_V3
        self.gesture_control = self.control_protocol in {GESTURE_GRASP_RELEASE_V2, GESTURE_RATE_GRIPPER_V3}
        self.intent_control = self.gesture_control or self.control_protocol == GRASP_HOLD_RELEASE_V1
        if self.control_protocol not in {"legacy-continuous-v1", GRASP_HOLD_RELEASE_V1, GESTURE_GRASP_RELEASE_V2, GESTURE_RATE_GRIPPER_V3}:
            raise ValueError("unsupported_contact_control_protocol")
        if self.intent_control:
            keys = ("release_confirm_seconds", "release_input_max_gap_seconds",
                    "release_stationary_target_tolerance_m", "release_stationary_rotation_tolerance_rad")
            if not self.gesture_control:
                keys += ("release_intent_aperture_normalized",)
            for key in keys:
                value = self.limits.get(key)
                allow_zero = key == 'release_confirm_seconds' and controller.config.get('gesture_control', {}).get('gripper_trigger_policy') == 'palm_fist_level_v2'
                if type(value) not in {int, float} or not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
                    raise ValueError(f"contact_control.{key} must be positive and finite")
            if not self.gesture_control and self.limits["release_intent_aperture_normalized"] > 1:
                raise ValueError("release_intent_aperture_normalized must not exceed 1")
        self.hold_width_m = {s: None for s in self.since}
        self.release_intent = {s: False for s in self.since}
        self.release_intent_event_id = {s: None for s in self.since}
        self.release_armed = {s: False for s in self.since}
        self.release_progress_s = {s: 0. for s in self.since}
        self.release_reason_by_side = {s: None for s in self.since}
        self.release_executed = {s: False for s in self.since}
        self.release_opening = {s: False for s in self.since}
        self.release_event_id = {s: None for s in self.since}
        self._release_started = {s: None for s in self.since}
        self._released_contact_cleared = {s: False for s in self.since}
        self._last_release_input_time = None
        self._release_input_mode = None
        self._release_seen_events = set()
        self.hold_interventions = 0
        self.release_count = 0
        self.init_gesture_control()
        self.init_rate_control()

    def input_now(self):
        """Internal clock only: camera timestamps never authorize duration."""
        return float(self.c.data.time if self._input_clock is None else self._input_clock())

    def set_input_clock(self, clock, name):
        """Inject service monotonic time; preserve it across controller resets.

        Standalone physics drivers retain deterministic simulation time. A
        clock-domain switch revokes an existing input lease and confirmation.
        """
        if not self.rate_control or not callable(clock) or name not in {"simulation", "server_monotonic"}:
            raise ValueError("invalid_release_input_clock")
        if not np.isfinite(float(clock())):
            raise ValueError("invalid_release_input_clock")
        self.cancel_rate_control("release_clock_changed")
        self._last_release_input_time = None
        self._input_clock = self.c._contact_input_clock = clock
        self.release_clock = self.c._contact_input_clock_name = name

    def cancel_release(self, reason="release_input_interrupted"):
        """Cancel intent evidence, never the physical grasp/width latch."""
        if self.rate_control:
            return self.cancel_rate_control(reason)
        if self.gesture_control:
            return self.cancel_gesture_release(reason)
        for side in self.since:
            if self.release_opening[side]:
                arm = next(a for a in self.c.arms if a.side == side)
                arm.gripper_goal_width_m = arm.gripper_command_width_m
                # A paused release must earn a new intent window before its
                # remaining opening, while fingers still touch, can resume.
                if not self._released_contact_cleared[side]:
                    self.hold_width_m[side] = arm.gripper_command_width_m
                    self.release_armed[side] = False
                self.release_opening[side] = False
            self.release_intent[side] = False
            self.release_intent_event_id[side] = None
            self._release_started[side] = None
            self.release_progress_s[side] = 0.
            self.release_reason_by_side[side] = reason if self.hold_width_m[side] is not None else None

    def project_action(self, requested, intent, *, event_id, mode, wants_motion, wants_gripper, interaction_modes=None):
        """Prepare a width projection; release is committed only after IK passes.

        Intent is a separate action, not inferred from the accepted/rate-limited
        14D vector. Every source uses this simulation-time confirmation gate.
        """
        if self.rate_control:
            return self.project_rate_action(requested, intent, interaction_modes, event_id=event_id,
                mode=mode, wants_motion=wants_motion, wants_gripper=wants_gripper)
        if self.gesture_control:
            return self.project_gesture_action(requested, intent, interaction_modes, event_id=event_id,
                mode=mode, wants_motion=wants_motion, wants_gripper=wants_gripper)
        projected = requested.copy()
        permits = []
        if not self.intent_control:
            return projected, permits, None
        now = float(self.c.data.time)
        if event_id is not None:
            if not isinstance(event_id, str) or not event_id or len(event_id) > 256:
                self.cancel_release("invalid_release_event_id")
                return projected, permits, "invalid_release_event_id"
            if event_id in self._release_seen_events:
                self.cancel_release("release_duplicate_input")
                return projected, permits, "release_duplicate_input"
            if len(self._release_seen_events) >= 65536:
                self.cancel_release("release_event_history_full_reset_required")
                return projected, permits, "release_event_history_full_reset_required"
            self._release_seen_events.add(event_id)
        if self._last_release_input_time is not None:
            gap = now-self._last_release_input_time
            if gap <= 1e-9 and any(intent):
                self.cancel_release("release_duplicate_input")
                return projected, permits, "release_duplicate_input"
            if gap > self.limits["release_input_max_gap_seconds"]:
                self.cancel_release("release_input_interrupted")
        mode_name = None if mode is None else str(mode)
        if self._release_input_mode != mode_name:
            self.cancel_release("release_mode_changed")
        self._release_input_mode = mode_name
        self._last_release_input_time = now
        if any(self.release_opening[side] and not active for side, active in zip(self.since, intent)):
            self.cancel_release("release_intent_withdrawn")
        stationary = not wants_motion or all(
            np.linalg.norm(requested[o:o+3]-a.world_to_task(a.command_position))
            <= self.limits["release_stationary_target_tolerance_m"]
            and rotation_angle(a.task_rotation @ rpy_to_matrix(requested[o+3:o+6]),
                               a.command_rotation) <= self.limits["release_stationary_rotation_tolerance_rad"]
            for a, o in zip(self.c.arms, (0, 7)))
        both_held = all(v is not None for v in self.hold_width_m.values())
        for side, active in zip(self.since, intent):
            self.release_intent[side] = active
            self.release_intent_event_id[side] = event_id
            if self.hold_width_m[side] is None:
                self._release_started[side] = None
                self.release_progress_s[side] = 0.
                self.release_reason_by_side[side] = "release_completed" if self.release_executed[side] else None
                continue
            reason = None
            if not active:
                self.release_armed[side] = True
                reason = "grasp_hold"
            elif not self.release_armed[side]:
                reason = "release_needs_relaxed_intent"
            elif both_held and not all(intent):
                reason = "release_requires_both_sides"
            elif not wants_gripper:
                reason = "release_gripper_component_required"
            elif not self.support_ready:
                reason = "release_requires_stable_table_support"
            elif not stationary:
                reason = "release_requires_stationary_targets"
            if reason:
                self._release_started[side] = None
                self.release_progress_s[side] = 0.
                self.release_reason_by_side[side] = reason
                continue
            if self._release_started[side] is None:
                self._release_started[side] = now
            self.release_progress_s[side] = min(self.limits["release_confirm_seconds"],
                                                now-self._release_started[side])
            self.release_reason_by_side[side] = "release_confirmation"
            if now-self._release_started[side] + 1e-9 >= self.limits["release_confirm_seconds"]:
                permits.append(side)
        if both_held and len(permits) != 2:
            permits.clear()
        if wants_motion and stationary and any(self._release_started[s] is not None or self.release_opening[s]
                                                for s in self.since):
            # Tolerance absorbs hand tremor; it must not authorize a stream of
            # sub-threshold translations/rotations during a release window.
            fields = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
            projected[fields] = self.c.action_vector_m()[fields]
        if wants_gripper:
            for arm, side, offset in zip(self.c.arms, self.since, (6, 13)):
                held = self.hold_width_m[side]
                if side in permits or (held is None and self.release_executed[side] and
                                       (self.release_opening[side] or self.release_intent[side])):
                    projected[offset] = arm.max_gripper_width_m
                elif held is not None:
                    projected[offset] = held
        return projected, permits, None

    def commit_release(self, permits, event_id):
        if self.rate_control:
            return self.commit_rate_release(permits, event_id)
        if self.gesture_control:
            return self.commit_gesture_release(permits, event_id)
        if permits:
            self.release_count += 1
        for side in permits:
            self.hold_width_m[side] = None
            self.release_armed[side] = False
            self.release_executed[side] = True
            self.release_opening[side] = True
            self.release_event_id[side] = event_id
            self._released_contact_cleared[side] = False
            self.release_reason_by_side[side] = "release_completed"

    def _update_holds(self):
        if not self.intent_control:
            return
        if self.rate_control:
            return self.update_rate_holds()
        if self.gesture_control:
            return self.update_gesture_holds()
        now = float(self.c.data.time)
        if (self._last_release_input_time is not None and
                now-self._last_release_input_time > self.limits["release_input_max_gap_seconds"]):
            self.cancel_release("release_input_interrupted")
        if not self.support_ready and any(self._release_started[s] is not None or self.release_opening[s] for s in self.since):
            self.cancel_release("release_requires_stable_table_support")
        for arm in self.c.arms:
            side = arm.side
            if self.release_executed[side] and not any(self.c.contacts[f"{side}_fingers"]):
                self._released_contact_cleared[side] = True
            if self.release_opening[side] and self._released_contact_cleared[side] and arm.gripper_command_width_m >= arm.max_gripper_width_m-1e-9:
                self.release_opening[side] = False
            eligible = not self.release_executed[side] or (self._released_contact_cleared[side] and not self.release_opening[side])
            if self.stable[side] and self.hold_width_m[side] is None and eligible:
                # Preserve the actual servo preload, not measured finger width.
                self.hold_width_m[side] = arm.gripper_command_width_m
                arm.gripper_goal_width_m = arm.gripper_command_width_m
                self.release_armed[side] = False
                self.release_executed[side] = False
                self.release_event_id[side] = None
                self._released_contact_cleared[side] = False
                self._release_started[side] = None
                self.release_progress_s[side] = 0.
                self.release_reason_by_side[side] = "release_needs_relaxed_intent"

    def support(self):
        c = self.c
        scene = c.config['scene']
        surfaces = {'table': scene['table_top_z']}
        if 'placement_platform' in scene:
            surfaces['placement_platform'] = target_surface_z(scene)
        geometry = {name: set(np.flatnonzero(c.model.geom_bodyid == c.model.body(name).id)) for name in surfaces}
        loads = {name: 0. for name in surfaces}
        force = np.zeros(6)
        for i, con in enumerate(c.data.contact[:c.data.ncon]):
            g1, g2 = int(con.geom1), int(con.geom2)
            surface = next((name for name, ids in geometry.items() if
                (g1 in c.object_geoms and g2 in ids) or (g2 in c.object_geoms and g1 in ids)), None)
            if surface is not None:
                mujoco.mj_contactForce(c.model, c.data, i, force)
                # Contact frame normal points from geom1 to geom2. Include
                # friction components so the metric is actual vertical load.
                world_force = con.frame.reshape(3, 3).T @ force[:3]
                loads[surface] += float(world_force[2] * (1 if g2 in c.object_geoms else -1))
        bottoms = []
        for gid in c.object_geoms:
            rotation = c.data.geom_xmat[gid].reshape(3, 3)
            # Task scenes deliberately use box collision parts (M05).
            extent_z = np.abs(rotation[2]) @ c.model.geom_size[gid]
            bottoms.append(c.data.geom_xpos[gid, 2]-extent_z)
        self.support_forces_n = {name: max(0., load) for name, load in loads.items()}
        surface = max(self.support_forces_n, key=self.support_forces_n.get)
        if surface != self.support_surface:
            self.support_since = None
        self.support_surface = surface
        self.support_force_n = self.support_forces_n[surface]
        self.bottom_height_m = float(min(bottoms)-surfaces[surface])
        weight = c.model.body_mass[c.object_id] * np.linalg.norm(c.model.opt.gravity)
        velocity = c.data.qvel[c.object_dof_adr:c.object_dof_adr+6]
        self._update_support_checks(weight, velocity)

    def _update_support_checks(self, weight, velocity):
        """Evaluate measured support; geometry tolerance is not a force limit."""
        c = self.c
        limits = c.config["cooperation"]
        clearance = self.limits["support_height_tolerance_m"]
        # Only the new rate protocol opts into asymmetric soft-contact
        # tolerance. Historical protocols and missing-key configs stay exact.
        penetration = self.limits.get("support_penetration_tolerance_m", clearance) if self.rate_control else clearance
        linear, angular, ee = float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:])), float(max(c.ee_speeds))
        self.support_requirements = {
            "minimum_table_force_n": float(weight*self.limits["support_weight_fraction"]),
            "minimum_bottom_height_m": -penetration, "maximum_bottom_height_m": clearance,
            "max_object_linear_speed_mps": limits["release_max_object_speed_mps"],
            "max_object_angular_speed_radps": limits["release_max_angular_speed_radps"],
            "max_ee_speed_mps": limits["release_max_ee_speed_mps"],
            "stable_seconds": self.limits["support_stable_seconds"],
        }
        self.support_checks = {
            "table_load": self.support_force_n >= self.support_requirements["minimum_table_force_n"],
            "clearance": self.bottom_height_m <= clearance,
            "penetration": self.bottom_height_m >= -penetration,
            "object_linear_speed": linear <= limits["release_max_object_speed_mps"],
            "object_angular_speed": angular <= limits["release_max_angular_speed_radps"],
            "ee_speed": ee <= limits["release_max_ee_speed_mps"],
        }
        good = all(self.support_checks.values())
        if not good:
            self.support_since = None
        elif self.support_since is None:
            self.support_since = float(c.data.time)
        self.support_ready = bool(good and c.data.time-self.support_since >= self.limits["support_stable_seconds"])
        self.support_checks["stable_duration"] = self.support_ready
        reasons = {"table_load":"insufficient_table_support", "clearance":"object_above_table",
            "penetration":"excessive_table_penetration", "object_linear_speed":"object_linear_motion",
            "object_angular_speed":"object_angular_motion", "ee_speed":"end_effector_motion"}
        self.support_blockers = [reason for key, reason in reasons.items() if not self.support_checks[key]]
        if good and not self.support_ready:
            self.support_blockers.append("support_stabilizing")
        self.support_measurements = {"object_linear_speed_mps":linear, "object_angular_speed_radps":angular,
            "max_ee_speed_mps":ee,
            "stable_seconds":0. if self.support_since is None else float(c.data.time-self.support_since)}

    def _refresh_release_anchors(self):
        c = self.c
        if self.edge_control or self.level_control:
            self.unsupported_width_anchor = {s: None for s in self.since}
            return
        for arm in c.arms:
            engaged = self.stable[arm.side] or c.contacts.get(arm.side, False) or self.reference is not None
            if engaged and not self.support_ready:
                anchor = self.unsupported_width_anchor[arm.side]
                self.unsupported_width_anchor[arm.side] = (arm.gripper_command_width_m if anchor is None
                                                          else min(anchor, arm.gripper_command_width_m))
            else:
                self.unsupported_width_anchor[arm.side] = None

    def release_reason(self, widths):
        self._refresh_release_anchors()
        for arm, width in zip(self.c.arms, widths):
            anchor = self.unsupported_width_anchor[arm.side]
            if anchor is not None and width > anchor + self.limits["release_width_epsilon_m"] + 1e-9:
                return "release_requires_stable_table_support"
        return None

    def motion_reason(self, positions, rotations):
        if self.intent_control:
            opening = any(self.release_opening.values())
            changing = any(np.linalg.norm(p-a.command_position) > 1e-6 or rotation_angle(r, a.command_rotation) > 1e-6
                           for a, p, r in zip(self.c.arms, positions, rotations))
            if opening and changing and not (self.edge_control or self.level_control):
                return "release_opening_in_progress"
        if self.reference is None:
            for arm, position, rotation in zip(self.c.arms, positions, rotations):
                changing = (np.linalg.norm(position-arm.command_position) > 1e-6
                            or rotation_angle(rotation, arm.command_rotation) > 1e-6)
                releasing = (self.edge_control and self.release_opening[arm.side]) or (self.level_control and self.interaction_modes[arm.side] == 'Z_OPEN')
                if self.stable[arm.side] and changing and not releasing:
                    return "partial_grasp_holds_engaged_arm"
        if self.danger_since is not None:
            return "contact_recovery_required"
        return None

    def update(self):
        c = self.c
        self.support()
        self._refresh_release_anchors()
        # Support can disappear while a previously accepted, rate-limited
        # opening is still running. Cancel its unsupported remainder as well.
        for arm in c.arms:
            anchor = self.unsupported_width_anchor[arm.side]
            if anchor is not None:
                ceiling = anchor + self.limits["release_width_epsilon_m"]
                if arm.gripper_goal_width_m > ceiling + 1e-9:
                    arm.gripper_goal_width_m = ceiling
                    self.release_guard_interventions += 1
        if c.task_state in {TaskState.DONE, TaskState.FAIL}:
            return
        now = float(c.data.time)
        for side in self.since:
            if not (self.rate_grasp_contact(side) if self.rate_control else c.contacts[side]):
                self.since[side] = None
            elif self.since[side] is None:
                self.since[side] = now
            self.stable[side] = self.since[side] is not None and now-self.since[side] >= c.config["cooperation"]["grasp_stable_seconds"]
        self._update_holds()
        dual = all(self.stable.values())
        any_contact = any(any(c.contacts[f"{s}_fingers"]) for s in self.since)
        object_rot = c.data.xmat[c.object_id].reshape(3, 3)
        object_pos = c.data.xpos[c.object_id]
        previous = self.state
        if dual and self.reference is None:
            self.reference = [(object_rot.T @ (a.get_actual_pose()[0]-object_pos), object_rot.T @ a.get_actual_pose()[1]) for a in c.arms]
            c._relative_reference = c.right.get_actual_pose()[0]-c.left.get_actual_pose()[0]
            self.ever_dual, self.released = True, False
            c.metrics.grasp_verified_count += 1
            c.start_task_timer()
        danger = None
        if self.reference is not None:
            errors = [np.linalg.norm(a.get_actual_pose()[0]-(object_pos+object_rot @ ref[0])) for a, ref in zip(c.arms, self.reference)]
            angles = [rotation_angle(a.get_actual_pose()[1], object_rot @ ref[1]) for a, ref in zip(c.arms, self.reference)]
            c.relative_pose_error_m = float(max(errors))
            c.relative_orientation_error_rad = float(max(angles))
            c.metrics.max_relative_pose_error_m = max(c.metrics.max_relative_pose_error_m, c.relative_pose_error_m)
            c.metrics.max_relative_orientation_error_rad = max(c.metrics.max_relative_orientation_error_rad, c.relative_orientation_error_rad)
            tilt = float(np.arccos(np.clip(object_rot[2, 2], -1, 1)))
            if not all(c.contacts[s] for s in self.since):
                danger = "contact_loss"
            elif max(errors) > c.config["cooperation"]["relative_pose_hard_limit_m"] or max(angles) > c.config["cooperation"]["relative_orientation_hard_limit_rad"]:
                danger = "grasp_drift"
            elif tilt > self.limits["max_tilt_rad"]:
                danger = "object_tilt"
            # A supported release or table-top slip can be retried. Clear the
            # reference only with confirmed real support, never in mid-air.
            if danger and self.support_ready:
                self.reference = None
                c._relative_reference = None
                self.released = True
                danger = None
            if danger:
                if self.danger_since is None:
                    self.danger_since = now
                    c.metrics.contact_loss_count += int(danger == "contact_loss")
                c.pause(danger, fail=now-self.danger_since >= c.config["cooperation"]["contact_loss_grace_seconds"])
            elif dual or self.support_ready:
                self.danger_since = None
        if object_pos[2] < c.config["scene"]["table_top_z"]-c.config["evaluation"]["object_drop_margin_m"]:
            c.pause("object_drop", fail=True)
        self.state = "DUAL" if self.reference is not None and dual else "PARTIAL" if any_contact else "FREE"
        if c.task_state != TaskState.FAIL:
            c.task_state = TaskState.DUAL_GRASPED if self.state == "DUAL" else TaskState.PARTIAL_CONTACT if self.state == "PARTIAL" else TaskState.PREGRASP
        if self.state != previous:
            self.events.append({"time": now, "from": previous, "to": self.state})
            self.events = self.events[-128:]
        from tasks.registry import hook
        if c.config.get('task_binding'):
            hook(c.config, "collection_success")(self, c, now, object_pos, object_rot, any_contact)
        else:
            from tasks.adapters.tray import collection_success
            collection_success(self, c, now, object_pos, object_rot, any_contact)

    def status(self):
        result = {"state": self.state, "stable_grasps": dict(self.stable), "support_ready": self.support_ready,
                "support_checks": dict(self.support_checks), "support_blockers": list(self.support_blockers),
                "support_requirements": dict(self.support_requirements), "support_measurements": dict(self.support_measurements),
                "table_support_force_n": self.support_force_n, "bottom_height_m": self.bottom_height_m,
                "danger_since": self.danger_since, "ever_dual": self.ever_dual, "released": self.released,
                "unsupported_width_anchor_m": dict(self.unsupported_width_anchor),
                "release_guard_interventions": self.release_guard_interventions,
                "control_protocol": self.control_protocol,
                "release_clock": self.release_clock,
                "hold_width_m": dict(self.hold_width_m), "release_intent": dict(self.release_intent),
                "release_intent_event_id": dict(self.release_intent_event_id),
                "release_armed": dict(self.release_armed), "release_progress_s": dict(self.release_progress_s),
                "release_reason": dict(self.release_reason_by_side),
                "release_executed": dict(self.release_executed), "release_event_id": dict(self.release_event_id),
                "release_opening": dict(self.release_opening),
                "release_confirm_seconds": self.limits.get("release_confirm_seconds", .7),
                "hold_interventions": self.hold_interventions, "release_count": self.release_count}
        if 'placement_platform' in self.c.config['scene']:
            result.update(support_surface=self.support_surface, support_force_n=self.support_force_n,
                          support_forces_n=dict(self.support_forces_n), placement=placement_geometry(self.c),
                          table_support_force_n=self.support_forces_n.get('table', 0.))
        if self.gesture_control:
            result.update(operation_hold_width_m=dict(self.operation_hold_width_m),
                aperture_direct_enabled=dict(self.aperture_direct_enabled),
                interaction_modes=dict(self.interaction_modes), interaction_mode_event_id=dict(self.interaction_mode_event_id),
                release_cycle_state=dict(self.release_cycle_state), release_reentry_pending=dict(self.release_reentry_pending))
        if self.rate_control:
            result.update(gripper_rate_enabled=dict(self.gripper_rate_enabled),
                          gripper_rate_direction=dict(self.gripper_rate_direction),
                          rate_interrupted_release_hold=dict(self.rate_interrupted_release_hold),
                          gripper_trigger_policy=self.gripper_trigger_policy,
                          gripper_edge={s: dict(v) for s, v in self.gripper_edge.items()},
                          grasp_close_qualified=dict(self.grasp_close_qualified))
        return result
