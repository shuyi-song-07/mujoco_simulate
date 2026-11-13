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


class SinglePushPolicyExecutor:
    """Project A7 onto the declared fixed-orientation, closed-gripper push task.

    No paired grasp state is invented. The controller remains responsible for
    velocity limits, workspace, IK and physical command acceptance.
    """
    version = 'papera-single-push-metric-v1'

    def __init__(self, controller, config=None):
        if tuple(getattr(controller, 'arm_order', ())) != ('left',):
            raise ValueError('single_push_executor_requires_one_real_arm')
        self.c = controller
        self.config = copy.deepcopy(DEFAULT_EXECUTOR_CONFIG)
        if config:
            if set(config) - {'max_orientation_projection_rad', 'control_hz'}:
                raise ValueError('unsupported_single_push_executor_parameters')
            self.config.update(config)
        tolerance = self.config['max_orientation_projection_rad']
        if self.config['control_hz'] != 60 or not np.isfinite(tolerance) or not 0 < tolerance <= .5:
            raise ValueError('invalid_single_push_executor_limits')
        self.monitor = controller.success_monitor
        self.calls = self.intervention_count = self.rejection_count = 0

    def apply(self, action, dt=1/60):
        c = self.c
        if c.success_monitor is not self.monitor:
            raise RuntimeError('controller_reset_requires_new_policy_executor')
        if not np.isfinite(dt) or not np.isclose(dt, 1/60, atol=1e-12):
            raise ValueError('policy_executor_requires_60_hz_control_ticks')
        self.calls += 1
        requested = projected = None
        report = {'fixed_closed_gripper': True, 'fixed_orientation': True}
        try:
            if isinstance(action, (str, bytes)):
                raise ValueError
            requested = np.asarray(action, dtype=float)
            if requested.shape != (7,) or not np.isfinite(requested).all():
                raise ValueError
        except (ValueError, TypeError):
            requested = None
            result = {'accepted': False, 'reason': 'invalid_finite_7d_policy_action'}
        else:
            projected = requested.copy()
            wanted = c.left.task_rotation @ rpy_to_matrix(requested[3:6])
            residual = rotation_angle(wanted, c.left.fixed_rotation)
            projected[3:6] = matrix_to_rpy(c.left.task_rotation.T @ c.left.fixed_rotation)
            projected[6] = 0.
            report['orientation_residual_rad'] = [float(residual)]
            if residual > self.config['max_orientation_projection_rad']:
                result = {'accepted': False, 'reason': 'policy_orientation_residual_exceeded'}
            else:
                result = c.apply_action(projected, schema_version='single-panda-push-v1', source='policy_rollout')
        target = c.action_vector_m().astype(float)
        difference = None if requested is None else requested-target
        if difference is not None:
            difference[3:6] = (difference[3:6]+np.pi) % (2*np.pi)-np.pi
        changed = [] if difference is None else np.flatnonzero(np.abs(difference) > 1e-7).tolist()
        accepted = bool(result['accepted'])
        intervened = bool(changed or not accepted)
        self.intervention_count += int(intervened)
        self.rejection_count += int(not accepted)
        result.update(source='policy_rollout', executor_version=self.version,
                      requested_target=None if requested is None else requested.tolist(),
                      projected_target=None if projected is None else projected.tolist(),
                      accepted_target=target.tolist(), accepted_goal=target.tolist(),
                      actual_state=c.state_vector().tolist(), intervened=intervened,
                      intervened_fields=changed,
                      target_residual_wrapped=None if difference is None else difference.tolist(),
                      projection=report, physical_contact=c.contact_control.status(),
                      simulation_time=float(c.data.time))
        c.last_action_result = result
        return result


