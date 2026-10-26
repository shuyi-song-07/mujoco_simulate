"""Gesture-v2 operation holds and once-authorized, interruptible opening.

This mixin belongs to the shared physical executor. It accepts validated modes,
not camera landmarks; network gesture evidence is checked before execution.
"""
from __future__ import annotations

import numpy as np

from .panda_arm_controller import rotation_angle, rpy_to_matrix


class GestureControlMixin:
    def init_gesture_control(self):
        self.operation_hold_width_m = {s: None for s in self.since}
        self.aperture_direct_enabled = {s: False for s in self.since}
        self.interaction_modes = {s: "HOLD" for s in self.since}
        self.interaction_mode_event_id = {s: None for s in self.since}
        self.release_cycle_state = {s: "IDLE" for s in self.since}
        self.release_reentry_pending = {s: False for s in self.since}
        self._hold_generation_time = {s: None for s in self.since}
        self._release_requires_support = {s: False for s in self.since}

    def _physically_engaged(self, side):
        return (self.hold_width_m[side] is not None or self.stable[side]
                or any(self.c.contacts.get(f"{side}_fingers", (False, False)))
                or self.reference is not None)

    def cancel_gesture_release(self, reason):
        """Retain servo preload, stop the remaining ramp and revoke permission."""
        now = float(self.c.data.time)
        for arm in self.c.arms:
            side = arm.side
            self.aperture_direct_enabled[side] = False
            active = self.release_opening[side] or self._release_started[side] is not None
            if self.release_opening[side]:
                arm.hold_gripper()
                # Even after physical contacts have cleared, an interrupted
                # opening cannot fall back to a live PINCH width channel.
                self.operation_hold_width_m[side] = arm.gripper_command_width_m
                if self._physically_engaged(side) and not self._released_contact_cleared[side]:
                    self.hold_width_m[side] = arm.gripper_command_width_m
                self.release_armed[side] = False
                self._hold_generation_time[side] = now
                self.release_reentry_pending[side] = True
            self.release_opening[side] = False
            self.release_intent[side] = False
            self.release_intent_event_id[side] = None
            self._release_started[side] = None
            self.release_progress_s[side] = 0.
            if active:
                self.release_cycle_state[side] = "ABORTED"
            if active or self.hold_width_m[side] is not None or self.operation_hold_width_m[side] is not None:
                self.release_reason_by_side[side] = reason

    def _check_gesture_input(self, event_id, intent, modes):
        now = self.input_now()
        if event_id is not None:
            if not isinstance(event_id, str) or not event_id or len(event_id) > 256:
                return "invalid_release_event_id"
            if event_id in self._release_seen_events:
                return "release_duplicate_input"
            if len(self._release_seen_events) >= 65536:
                return "release_event_history_full_reset_required"
            self._release_seen_events.add(event_id)
        if self._last_release_input_time is not None:
            gap = now-self._last_release_input_time
            if gap <= 1e-9 and (any(intent) or any(self.release_opening.values())):
                return "release_duplicate_input"
            if gap > self.limits["release_input_max_gap_seconds"]:
                self.cancel_release("release_input_interrupted")
        self._last_release_input_time = now
        return None

    def project_gesture_action(self, requested, intent, modes, *, event_id, mode,
                               wants_motion, wants_gripper):
        projected = requested.copy()
        permits = []
        reason = self._check_gesture_input(event_id, intent, modes)
        if reason:
            self.cancel_gesture_release(reason)
            return projected, permits, reason
        now = float(self.c.data.time)
        current = self.c.action_vector_m()
        stationary_by_side = [not wants_motion or (
            np.linalg.norm(requested[o:o+3]-a.world_to_task(a.command_position))
            <= self.limits["release_stationary_target_tolerance_m"]
            and rotation_angle(a.task_rotation @ rpy_to_matrix(requested[o+3:o+6]), a.command_rotation)
            <= self.limits["release_stationary_rotation_tolerance_rad"])
            for a, o in zip(self.c.arms, (0, 7))]
        stationary = all(stationary_by_side)
        opening_before = any(self.release_opening.values())
        previous_modes = dict(self.interaction_modes)
        for side, selected in zip(self.since, modes):
            self.interaction_modes[side] = selected
            self.interaction_mode_event_id[side] = event_id
        if opening_before and not stationary:
            self.cancel_gesture_release("release_motion_cancelled")
            return current.copy(), [], "release_motion_cancelled"
        cancelled_opening = False
        if any(self.release_opening[s] and selected not in {"RELEASE", "OPENING"}
               for s, selected in zip(self.since, modes)):
            self.cancel_gesture_release("release_mode_cancelled")
            cancelled_opening = True
        elif any(self._release_started[s] is not None and previous_modes[s] != selected
                 and not self.release_opening[s] for s, selected in zip(self.since, modes)):
            self.cancel_gesture_release("release_mode_changed")
        mode_name = None if mode is None else str(mode)
        if self._release_input_mode != mode_name and any(self._release_started[s] is not None
                                                        for s in self.since):
            self.cancel_gesture_release("release_mode_changed")
        self._release_input_mode = mode_name

        dual_release = self.reference is not None or all(v is not None for v in self.hold_width_m.values())
        for index, (arm, selected, active, offset) in enumerate(zip(self.c.arms, modes, intent, (0, 7))):
            side = arm.side
            new_operation_hold = selected == "Z_HOLD" and self.operation_hold_width_m[side] is None
            if new_operation_hold:
                self.operation_hold_width_m[side] = arm.gripper_command_width_m
                self._hold_generation_time[side] = now
                self.release_armed[side] = False
            held = self.hold_width_m[side]
            if held is None:
                held = self.operation_hold_width_m[side]
            locked = held is not None
            self.release_intent[side] = active
            self.release_intent_event_id[side] = event_id

            # PINCH has no pose channel. Motion modes hold their inactive axes;
            # neutral/release modes keep all pose fields at the accepted target.
            if selected in {"PINCH", "RELEASE", "OPENING", "HOLD"} or cancelled_opening:
                projected[offset:offset+6] = current[offset:offset+6]
            elif selected == "XY_HOLD":
                projected[offset+2] = current[offset+2]
            else:
                projected[offset:offset+2] = current[offset:offset+2]
                projected[offset+3:offset+6] = current[offset+3:offset+6]
            if selected == "PINCH" and not stationary_by_side[index]:
                self.cancel_gesture_release("pinch_requires_stationary_targets")
                return projected, [], "pinch_requires_stationary_targets"

            if selected not in {"RELEASE", "OPENING", "PINCH"}:
                # This only exits a completed cycle. An aborted ramp stays
                # operation-locked until it earns another release permission.
                if self.release_cycle_state[side] == "COMPLETE":
                    self.release_reentry_pending[side] = False
                if locked and not new_operation_hold and (self._hold_generation_time[side] is None
                                                         or now > self._hold_generation_time[side]+1e-9):
                    self.release_armed[side] = True
            elif selected == "PINCH" and locked and not active and (self._hold_generation_time[side] is None
                                                                  or now > self._hold_generation_time[side]+1e-9):
                self.release_armed[side] = True

            if self.release_opening[side]:
                if self._release_requires_support[side] and not self.support_ready:
                    self.cancel_gesture_release("release_requires_stable_table_support")
                    return current.copy(), [], "release_requires_stable_table_support"
                projected[offset+6] = arm.max_gripper_width_m
                self.release_reason_by_side[side] = "release_opening"
                continue
            if selected == "OPENING" and self.release_cycle_state[side] != "COMPLETE":
                self.cancel_gesture_release("opening_requires_release_authorization")
                return projected, [], "opening_requires_release_authorization"

            reason = None
            if selected == "RELEASE" and locked:
                if not self.release_armed[side]:
                    reason = "release_needs_relaxed_intent"
                elif dual_release and not all(v == "RELEASE" and flag for v, flag in zip(modes, intent)):
                    reason = "release_requires_both_sides"
                elif not wants_gripper:
                    reason = "release_gripper_component_required"
                elif self._physically_engaged(side) and not self.support_ready:
                    reason = "release_requires_stable_table_support"
                elif not stationary:
                    reason = "release_requires_stationary_targets"
                if reason is None:
                    if self._release_started[side] is None:
                        self._release_started[side] = now
                    self._release_requires_support[side] = self._physically_engaged(side)
                    self.release_progress_s[side] = min(self.limits["release_confirm_seconds"], now-self._release_started[side])
                    self.release_cycle_state[side] = "CONFIRMING"
                    reason = "release_confirmation"
                    if now-self._release_started[side]+1e-9 >= self.limits["release_confirm_seconds"]:
                        permits.append(side)
            elif selected == "RELEASE":
                reason = "release_completed" if self.release_cycle_state[side] == "COMPLETE" else "release_requires_hold"
            elif locked:
                reason = "grasp_hold" if self.hold_width_m[side] is not None else "operation_hold"
            elif self.release_reentry_pending[side]:
                reason = "release_completed_reenter_required"
            if reason != "release_confirmation":
                self._release_started[side] = None
                self.release_progress_s[side] = 0.
                if self.release_cycle_state[side] == "CONFIRMING":
                    self.release_cycle_state[side] = "IDLE"
            self.release_reason_by_side[side] = reason

            can_pinch = wants_gripper and selected == "PINCH" and not locked and not self.release_reentry_pending[side]
            if not can_pinch:
                arm.hold_gripper()
                projected[offset+6] = arm.gripper_command_width_m if held is None else held
        if dual_release and len(permits) != 2:
            permits.clear()
        for arm, offset in zip(self.c.arms, (6, 13)):
            if arm.side in permits:
                projected[offset] = arm.max_gripper_width_m
        return projected, permits, None

    def commit_gesture_release(self, permits, event_id):
        if permits:
            self.release_count += 1
        for side in permits:
            self.aperture_direct_enabled[side] = False
            self.hold_width_m[side] = None
            self.operation_hold_width_m[side] = None
            self.release_armed[side] = False
            self.release_executed[side] = True
            self.release_opening[side] = True
            self.release_event_id[side] = event_id
            self._released_contact_cleared[side] = False
            self.release_cycle_state[side] = "OPENING"
            self.release_reentry_pending[side] = True
            self.release_reason_by_side[side] = "release_opening"
            self._release_started[side] = None

    def update_gesture_holds(self):
        now = float(self.c.data.time)
        active = any(self.release_opening.values()) or any(v is not None for v in self._release_started.values())
        if active and self._last_release_input_time is not None and now-self._last_release_input_time > self.limits["release_input_max_gap_seconds"]:
            self.cancel_gesture_release("release_input_interrupted")
        if active and not self.support_ready and any(self._release_requires_support[s]
                and (self.release_opening[s] or self._release_started[s] is not None) for s in self.since):
            self.cancel_gesture_release("release_requires_stable_table_support")
        for arm in self.c.arms:
            side = arm.side
            if self.release_executed[side] and not any(self.c.contacts[f"{side}_fingers"]):
                self._released_contact_cleared[side] = True
            if (self.release_opening[side] and self._released_contact_cleared[side]
                    and arm.gripper_command_width_m >= arm.max_gripper_width_m-1e-9):
                self.release_opening[side] = False
                self.release_cycle_state[side] = "COMPLETE"
                self.release_reason_by_side[side] = "release_completed"
                self.release_progress_s[side] = 0.
            eligible = not self.release_executed[side] or (self._released_contact_cleared[side]
                and not self.release_opening[side] and not self.release_reentry_pending[side])
            if self.stable[side] and self.hold_width_m[side] is None and eligible:
                self.aperture_direct_enabled[side] = False
                self.hold_width_m[side] = arm.gripper_command_width_m
                arm.hold_gripper()
                self.release_armed[side] = False
                self._hold_generation_time[side] = now
                self.release_executed[side] = False
                self.release_event_id[side] = None
                self._released_contact_cleared[side] = False
                self._release_started[side] = None
                self.release_progress_s[side] = 0.
                self.release_cycle_state[side] = "IDLE"
                self.release_reason_by_side[side] = "release_needs_relaxed_intent"

    def before_gesture_gripper_step(self):
        """Check actual contact/support and freshness before another servo step."""
        self.c.contacts = self.c._contact_state()
        now = float(self.c.data.time)
        for side in self.since:
            if not self.c.contacts[side]:
                self.since[side] = None
            elif self.since[side] is None:
                self.since[side] = now
            self.stable[side] = (self.since[side] is not None
                and now-self.since[side] >= self.c.config["cooperation"]["grasp_stable_seconds"])
        if any(self.release_opening.values()) or any(v is not None for v in self._release_started.values()):
            self.support()
        self.update_gesture_holds()
