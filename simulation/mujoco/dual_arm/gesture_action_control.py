"""Categorical and palm/fist edge gripper ramps with real grasp qualification.

Modes authorize a direction; arbitrary numeric width requests never override
it. Every accepted fresh sample refreshes the rate lease. Stable real contact
holds the current issued preload, independently of the camera's fist label.
"""
from __future__ import annotations

import numpy as np

from .panda_arm_controller import rotation_angle, rpy_to_matrix


class RateGestureControlMixin:
    def init_rate_control(self):
        self.gripper_trigger_policy = self.c.config.get("gesture_control", {}).get("gripper_trigger_policy", "level_v1")
        if self.gripper_trigger_policy not in {"level_v1", "palm_fist_edge_v1", "palm_fist_edge_v2", "palm_fist_level_v2"}:
            raise ValueError("unsupported_gripper_trigger_policy")
        self.edge_control = self.rate_control and self.gripper_trigger_policy in {"palm_fist_edge_v1", "palm_fist_edge_v2"}
        self.edge_hold_resume = self.edge_control and self.gripper_trigger_policy == "palm_fist_edge_v2"
        self.level_control = self.rate_control and self.gripper_trigger_policy == 'palm_fist_level_v2'
        self.gripper_rate_enabled = {s: False for s in self.since}
        self.gripper_rate_direction = {s: 0 for s in self.since}
        self.rate_interrupted_release_hold = {s: False for s in self.since}
        self._rate_regrasp_pending = set()
        self.grasp_close_qualified = {s: False for s in self.since}
        self.gripper_edge = {}
        self._edge_last_qualified = {s: None for s in self.since}
        for arm in self.c.arms:
            initial_open = arm.gripper_command_width_m >= arm.max_gripper_width_m-1e-9
            self.gripper_edge[arm.side] = {
                "baseline_mode": "Z_OPEN" if initial_open else None,
                "baseline_source": "initial_open_state" if initial_open else None,
                "authorized_mode": None, "event_id": None, "edge_count": 0,
                "state": "READY" if initial_open else "WAIT_BASELINE", "reason": None}

    def rate_grasp_contact(self, side):
        """Raw fingertip pressure is not evidence of a grasp at full opening.

        A commanded closure must precede a new rate-protocol grasp. Existing
        verified grasps remain protected across pauses or camera loss.
        """
        arm = next(a for a in self.c.arms if a.side == side)
        forces = self.c.contacts.get(f"{side}_grip_forces_n", ())
        opposed = len(forces) == 2 and all(
            force >= self.c.config["cooperation"]["minimum_contact_force_n"] for force in forces)
        qualified = (self.hold_width_m[side] is not None or self.reference is not None or
            (self.grasp_close_qualified[side] and
             arm.gripper_command_width_m < arm.max_gripper_width_m-1e-6 and opposed))
        return bool(self.c.contacts[side] and qualified)

    def _cancel_edge_side(self, side, reason, *, reset_baseline=False):
        edge = self.gripper_edge[side]
        if self.release_opening[side]:
            arm = next(a for a in self.c.arms if a.side == side)
            arm.hold_gripper()
            self.hold_width_m[side] = arm.gripper_command_width_m
            self.rate_interrupted_release_hold[side] = True
            self.release_opening[side] = False
            self._release_started[side] = None
            self.release_progress_s[side] = 0.
            self.release_cycle_state[side] = "ABORTED"
        edge["authorized_mode"] = None
        edge["reason"] = reason
        edge["state"] = "ABORTED"
        if reset_baseline:
            edge["baseline_mode"] = edge["baseline_source"] = None
            self._edge_last_qualified[side] = None
            edge["state"] = "WAIT_BASELINE"

    def _prepare_gripper_edges(self, modes, event_id, wants_gripper):
        if not self.edge_control:
            return
        now = self.input_now()
        for side, selected in zip(self.since, modes):
            edge = self.gripper_edge[side]
            last = self._edge_last_qualified[side]
            if (not self.edge_hold_resume and edge["authorized_mode"] is not None and last is not None and
                    now-last > self.limits["release_input_max_gap_seconds"]):
                self._cancel_edge_side(side, "gripper_edge_lease_expired")
            if selected == "XY_HOLD":
                self._cancel_edge_side(side, "gripper_edge_xy_hold")
            elif selected == "HOLD":
                if edge["authorized_mode"] is not None:
                    # A fresh, trusted classification HOLD freezes this side's
                    # servo in project_rate_action. V2 retains its pending edge
                    # through reconfirmation; the separate input watchdog still
                    # revokes it on actual stale/lost input or an explicit pause.
                    edge.update(state="HOLD", reason="gripper_edge_transition_hold")
            else:
                previous = edge["baseline_mode"]
                if previous is None:
                    edge.update(state="READY", reason="gripper_edge_baseline_established")
                elif selected != previous and wants_gripper:
                    edge.update(authorized_mode=selected, event_id=event_id,
                        edge_count=edge["edge_count"]+1,
                        state="OPENING" if selected == "Z_OPEN" else "CLOSING", reason=None)
                edge["baseline_mode"] = selected
                edge["baseline_source"] = "accepted_mode"
                self._edge_last_qualified[side] = now

    def _edge_authorized(self, side, selected):
        return not self.edge_control or self.gripper_edge[side]["authorized_mode"] == selected

    def _update_edge_status(self, side, reason):
        if not self.edge_control:
            return
        edge = self.gripper_edge[side]
        if reason == "grasp_hold":
            edge.update(state="HOLD", reason=reason)
        elif edge["authorized_mode"] is not None and self.interaction_modes[side] != "HOLD":
            edge["reason"] = reason
            edge["state"] = {"release_requires_stable_table_support": "WAIT_SUPPORT",
                "release_requires_stationary_targets": "WAIT_SUPPORT",
                "release_requires_both_sides": "WAIT_BOTH", "release_confirmation": "CONFIRMING",
                "release_opening": "OPENING"}.get(reason,
                    "OPENING" if edge["authorized_mode"] == "Z_OPEN" else "CLOSING")

    def cancel_rate_control(self, reason, *, reset_edge_baseline=True):
        self._rate_regrasp_pending.clear()
        for arm in self.c.arms:
            side = arm.side
            active = self.release_opening[side] or self._release_started[side] is not None
            if self.release_opening[side]:
                # An interrupted supported release needs fresh confirmation,
                # including when contact just cleared during the opening.
                self.hold_width_m[side] = arm.gripper_command_width_m
                self.rate_interrupted_release_hold[side] = True
            arm.hold_gripper()
            self.gripper_rate_enabled[side] = False
            self.gripper_rate_direction[side] = 0
            self.aperture_direct_enabled[side] = False
            self.operation_hold_width_m[side] = None
            self.release_opening[side] = False
            self.release_intent[side] = False
            self.release_intent_event_id[side] = None
            self._release_started[side] = None
            self.release_progress_s[side] = 0.
            self.release_reentry_pending[side] = False
            if active:
                self.release_cycle_state[side] = "ABORTED"
            self.release_reason_by_side[side] = reason
            if self.edge_control:
                self._cancel_edge_side(side, reason, reset_baseline=reset_edge_baseline)

    def project_rate_action(self, requested, intent, modes, *, event_id, mode,
                            wants_motion, wants_gripper):
        if self.level_control:
            return self._project_level_action(requested, intent, modes, event_id=event_id,
                                              wants_gripper=wants_gripper)
        projected = requested.copy()
        self._rate_regrasp_pending.clear()
        current = self.c.action_vector_m()
        reason = self._check_gesture_input(event_id, intent, modes)
        if reason:
            self.cancel_rate_control(reason)
            return projected, [], reason
        now = self.input_now()
        stationary = all(not wants_motion or (
            np.linalg.norm(requested[o:o+3]-a.world_to_task(a.command_position))
                <= self.limits["release_stationary_target_tolerance_m"]
            and rotation_angle(a.task_rotation @ rpy_to_matrix(requested[o+3:o+6]), a.command_rotation)
                <= self.limits["release_stationary_rotation_tolerance_rad"])
            for a, o in zip(self.c.arms, (0, 7)))
        dual = self.reference is not None or all(v is not None for v in self.hold_width_m.values())
        if (any(self.release_opening.values()) and
                ((not self.edge_control and (not stationary or not self.support_ready)) or
                 any(self.release_opening[s] and m != "Z_OPEN" and not (self.edge_control and m == "HOLD")
                     for s, m in zip(self.since, modes)))):
            self.cancel_rate_control("release_mode_cancelled" if self.edge_control else
                "release_motion_cancelled" if not stationary else
                "release_requires_stable_table_support" if not self.support_ready else "release_mode_cancelled",
                reset_edge_baseline=False)
        self._prepare_gripper_edges(modes, event_id, wants_gripper)
        permits = []
        for arm, selected, offset in zip(self.c.arms, modes, (0, 7)):
            side = arm.side
            authorized = wants_gripper and self._edge_authorized(side, selected)
            self.interaction_modes[side] = selected
            self.interaction_mode_event_id[side] = event_id
            self.release_intent[side] = selected == "Z_OPEN"
            self.release_intent_event_id[side] = event_id
            self.aperture_direct_enabled[side] = False
            self.operation_hold_width_m[side] = None
            self.release_reentry_pending[side] = False
            if selected == "HOLD":
                projected[offset:offset+6] = current[offset:offset+6]
            elif selected == "XY_HOLD":
                projected[offset+2] = current[offset+2]
            else:
                projected[offset:offset+2] = current[offset:offset+2]
                projected[offset+3:offset+6] = current[offset+3:offset+6]
            held = self.hold_width_m[side]
            if (authorized and selected == "Z_CLOSE" and self.rate_interrupted_release_hold[side]
                    and (self.edge_control or self.support_ready) and not any(self.c.contacts[f"{side}_fingers"])):
                # A release-interruption hold is not a new physical grasp.
                # Permit supported re-closing once the fingers are clear; the
                # latch is removed only after the whole IK transaction passes.
                self._rate_regrasp_pending.add(side)
                held = None
            protected = held is not None or self._physically_engaged(side)
            if (self.edge_control and self.reference is None
                    and (held is not None or self.stable[side]) and not self.release_opening[side]):
                # Partial grasp constrains only the held pose, in every mode.
                # Rejecting the shared packet for held-wrist jitter would also
                # erase the free hand's palm/fist baseline and abort its ramp.
                # Keep valid free-side movement/edges and held-side release;
                # joint/workspace, contact-loss and input checks still apply.
                projected[offset:offset+6] = current[offset:offset+6]
            why = None
            if self.release_opening[side]:
                if self.edge_control and selected == "HOLD":
                    arm.hold_gripper()
                    projected[offset+6] = arm.gripper_command_width_m
                    self.release_reason_by_side[side] = "gripper_edge_transition_hold"
                else:
                    projected[offset+6] = arm.max_gripper_width_m
                    self.release_reason_by_side[side] = "release_opening"
                    self._update_edge_status(side, "release_opening")
                continue
            if selected == "Z_OPEN" and protected and authorized:
                if dual and not all(m == "Z_OPEN" and self._edge_authorized(s, m) for s, m in zip(self.since, modes)):
                    why = "release_requires_both_sides"
                elif not wants_gripper:
                    why = "release_gripper_component_required"
                elif not self.edge_control and not self.support_ready:
                    why = "release_requires_stable_table_support"
                elif not self.edge_control and not stationary:
                    why = "release_requires_stationary_targets"
                else:
                    if self._release_started[side] is None:
                        self._release_started[side] = now
                    self._release_requires_support[side] = not self.edge_control
                    self.release_progress_s[side] = min(self.limits["release_confirm_seconds"], now-self._release_started[side])
                    self.release_cycle_state[side] = "CONFIRMING"
                    why = "release_confirmation"
                    if now-self._release_started[side]+1e-9 >= self.limits["release_confirm_seconds"]:
                        permits.append(side)
            elif held is not None:
                why = "grasp_hold"
            if why != "release_confirmation":
                self._release_started[side] = None
                self.release_progress_s[side] = 0.
                if self.release_cycle_state[side] == "CONFIRMING":
                    self.release_cycle_state[side] = "IDLE"
            self.release_reason_by_side[side] = why
            if authorized and selected == "Z_CLOSE" and held is None:
                projected[offset+6] = 0.
            elif authorized and selected == "Z_OPEN" and not protected:
                projected[offset+6] = arm.max_gripper_width_m
            else:
                arm.hold_gripper()
                projected[offset+6] = arm.gripper_command_width_m if held is None else held
            self._update_edge_status(side, why)
        if dual and len(permits) != 2:
            permits.clear()
        # Small target jitter must not accumulate through the confirmation or
        # the authorized opening. Above-tolerance Z still lowers a held object.
        if not self.edge_control and stationary and any(self._release_started[s] is not None or self.release_opening[s] for s in self.since):
            for offset in (0, 7):
                projected[offset:offset+6] = current[offset:offset+6]
        for arm, offset in zip(self.c.arms, (6, 13)):
            if arm.side in permits:
                projected[offset] = arm.max_gripper_width_m
        return projected, permits, None

    def _project_level_action(self, requested, intent, modes, *, event_id, wants_gripper):
        """Current qualified mode owns each jaw; no edge or second release timer.

        Browser mode qualification applies the configured 100 ms gate.
        This executor receives accepted modes and keeps ordinary input/physics
        checks. A release is committed only after the enclosing IK transaction.
        """
        projected, permits = requested.copy(), []
        self._rate_regrasp_pending.clear()
        reason = self._check_gesture_input(event_id, intent, modes)
        if reason:
            self.cancel_rate_control(reason)
            return projected, permits, reason
        current = self.c.action_vector_m()
        for arm, selected, offset in zip(self.c.arms, modes, (0, 7)):
            side = arm.side
            self.interaction_modes[side] = selected
            self.interaction_mode_event_id[side] = event_id
            self.release_intent[side] = selected == 'Z_OPEN'
            self.release_intent_event_id[side] = event_id
            self.aperture_direct_enabled[side] = False
            self.operation_hold_width_m[side] = None
            self.release_reentry_pending[side] = False
            self._release_started[side] = None
            self.release_progress_s[side] = 0.
            if selected == 'HOLD':
                projected[offset:offset+6] = current[offset:offset+6]
            elif selected == 'XY_HOLD':
                projected[offset+2] = current[offset+2]
            else:
                projected[offset:offset+2] = current[offset:offset+2]
                projected[offset+3:offset+6] = current[offset+3:offset+6]
            if self.release_opening[side] and (selected != 'Z_OPEN' or not wants_gripper):
                arm.hold_gripper()
                self.release_opening[side] = False
                self.release_cycle_state[side] = 'ABORTED'
            held = self.hold_width_m[side]
            if (selected == 'Z_CLOSE' and wants_gripper and self.rate_interrupted_release_hold[side]
                    and not any(self.c.contacts[f'{side}_fingers'])):
                self._rate_regrasp_pending.add(side)
                held = None
            # Retain the existing partial-grasp position protection, while an
            # explicit palm is free to open that side and move in Z.
            if self.reference is None and (held is not None or self.stable[side]) and selected != 'Z_OPEN':
                projected[offset:offset+6] = current[offset:offset+6]
            why = None
            if wants_gripper and selected == 'Z_OPEN':
                projected[offset+6] = arm.max_gripper_width_m
                if self._physically_engaged(side) and not self.release_opening[side]:
                    permits.append(side)
                why = 'release_opening' if self.release_opening[side] or side in permits else None
            elif wants_gripper and selected == 'Z_CLOSE' and held is None:
                projected[offset+6] = 0.
            else:
                arm.hold_gripper()
                projected[offset+6] = arm.gripper_command_width_m
                why = 'grasp_hold' if held is not None else None
            self.release_reason_by_side[side] = why
        return projected, permits, None

    def commit_rate_release(self, permits, event_id):
        for side, selected in self.interaction_modes.items():
            if selected == "Z_CLOSE" and self._edge_authorized(side, selected):
                self.grasp_close_qualified[side] = True
                if self.level_control:
                    self.release_executed[side] = False
        for side in self._rate_regrasp_pending:
            self.hold_width_m[side] = None
            self.rate_interrupted_release_hold[side] = False
            self._released_contact_cleared[side] = True
            self.release_cycle_state[side] = "IDLE"
        self._rate_regrasp_pending.clear()
        if permits:
            self.release_count += 1
        if (self.edge_control and len(permits) == 2) or (self.level_control and permits):
            # Explicit user-selected release may occur in motion or mid-air.
            # It ends the rigid grasp; subsequent gravity/drop remains physics.
            self.reference = None
            self.c._relative_reference = None
            self.released = True
        for side in permits:
            self.grasp_close_qualified[side] = False
            if self.edge_control:
                self.gripper_edge[side].update(state="OPENING", reason="release_opening")
            self.hold_width_m[side] = None
            self.rate_interrupted_release_hold[side] = False
            self.release_armed[side] = False
            self.release_executed[side] = True
            self.release_opening[side] = True
            self.release_event_id[side] = event_id
            self._released_contact_cleared[side] = False
            self.release_cycle_state[side] = "OPENING"
            self.release_reason_by_side[side] = "release_opening"
            self._release_started[side] = None

    def update_rate_holds(self):
        now = self.input_now()
        if (self._last_release_input_time is not None and
                now-self._last_release_input_time > self.limits["release_input_max_gap_seconds"]):
            self.cancel_rate_control("release_input_interrupted")
        if (not (self.edge_control or self.level_control) and not self.support_ready and
                any(self.release_opening[s] or self._release_started[s] is not None for s in self.since)):
            self.cancel_rate_control("release_requires_stable_table_support")
        for arm in self.c.arms:
            side = arm.side
            if self.edge_control:
                edge = self.gripper_edge[side]
                last = self._edge_last_qualified[side]
                if (not self.edge_hold_resume and edge["authorized_mode"] is not None and last is not None
                        and now-last > self.limits["release_input_max_gap_seconds"]):
                    self._cancel_edge_side(side, "gripper_edge_lease_expired")
                    arm.hold_gripper()
            if self.release_executed[side] and not any(self.c.contacts[f"{side}_fingers"]):
                self._released_contact_cleared[side] = True
            if (self.release_opening[side] and self._released_contact_cleared[side]
                    and arm.gripper_command_width_m >= arm.max_gripper_width_m-1e-9):
                self.release_opening[side] = False
                self.release_cycle_state[side] = "COMPLETE"
                self.release_reason_by_side[side] = "release_completed"
                self.release_progress_s[side] = 0.
            eligible = not self.release_executed[side] or (self._released_contact_cleared[side]
                and not self.release_opening[side])
            if self.stable[side] and self.hold_width_m[side] is None and eligible:
                self.hold_width_m[side] = arm.gripper_command_width_m
                self.rate_interrupted_release_hold[side] = False
                arm.hold_gripper()
                self.gripper_rate_enabled[side] = False
                self.gripper_rate_direction[side] = 0
                self.release_executed[side] = False
                self.release_event_id[side] = None
                self._released_contact_cleared[side] = False
                self._release_started[side] = None
                self.release_progress_s[side] = 0.
                self.release_cycle_state[side] = "IDLE"
                self.release_reason_by_side[side] = "grasp_hold"
                if self.edge_control:
                    self.gripper_edge[side].update(authorized_mode=None, state="HOLD", reason="grasp_hold")
            if abs(arm.gripper_goal_width_m-arm.gripper_command_width_m) <= 1e-9:
                self.gripper_rate_enabled[side] = False
                self.gripper_rate_direction[side] = 0
                if self.edge_control:
                    edge = self.gripper_edge[side]
                    terminal = ((edge["authorized_mode"] == "Z_CLOSE" and arm.gripper_command_width_m <= 1e-9)
                        or (edge["authorized_mode"] == "Z_OPEN" and arm.gripper_command_width_m >= arm.max_gripper_width_m-1e-9))
                    if terminal:
                        edge.update(authorized_mode=None, state="COMPLETE", reason="gripper_edge_completed")
            if (arm.gripper_command_width_m >= arm.max_gripper_width_m-1e-9 and
                    arm.gripper_goal_width_m >= arm.max_gripper_width_m-1e-9):
                self.grasp_close_qualified[side] = False

    def before_rate_gripper_step(self):
        self.c.contacts = self.c._contact_state()
        now = float(self.c.data.time)
        for side in self.since:
            if not self.rate_grasp_contact(side):
                self.since[side] = None
            elif self.since[side] is None:
                self.since[side] = now
            self.stable[side] = self.since[side] is not None and now-self.since[side] >= self.c.config["cooperation"]["grasp_stable_seconds"]
        if any(self.release_opening.values()) or any(v is not None for v in self._release_started.values()):
            self.support()
        self.update_rate_holds()
