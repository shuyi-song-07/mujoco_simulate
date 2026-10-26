"""Transactional Cartesian feasibility and supported XYZ + world-Yaw control."""
import mujoco
import numpy as np
from .panda_arm_controller import rotation_angle, rpy_to_matrix


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def yaw_of(rotation):
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def blend_rotation(start, end, fraction):
    """Shortest-arc interpolation for the finite grasp-target handoff."""
    a, b = np.empty(4), np.empty(4)
    mujoco.mju_mat2Quat(a, np.asarray(start).reshape(9))
    mujoco.mju_mat2Quat(b, np.asarray(end).reshape(9))
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    if dot > .9995:
        q = (1-fraction)*a + fraction*b
    else:
        angle = np.arccos(np.clip(dot, -1., 1.))
        q = (np.sin((1-fraction)*angle)*a + np.sin(fraction*angle)*b)/np.sin(angle)
    q /= np.linalg.norm(q)
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, q)
    return result.reshape(3, 3)


class RigidControl:
    def __init__(self, controller):
        self.c = controller
        self.limits = controller.config["rigid_control"]
        self.reference_id = None
        self.object_position = None
        self.object_rotation = None
        self.handoff_progress = 1.
        self.handoff_start = None
        self.handoff_step = 1.
        self.handoff_gap = 0.
        self.last = {"scale": 1., "reason": None, "coupled": False}
        self.collision_data = mujoco.MjData(controller.model)

    def collision_reason(self):
        """Check both candidate arms together without changing live physics.

        This finite joint-space sampling supplements the live collision guard;
        it is not a proof that every point of the dynamic path is collision-free.
        Intended robot-object and object-obstacle contacts remain allowed.
        """
        c, data = self.c, self.collision_data
        start = c.data.qpos.copy()
        candidate = start.copy()
        for arm in c.arms:
            candidate[arm.qpos_indices] = arm.ik_data.qpos[arm.qpos_indices]
        robots = c.robot_geoms["left"] | c.robot_geoms["right"]
        for fraction in np.linspace(0., 1., 5):
            data.qpos[:] = start + fraction*(candidate-start)
            data.qvel[:] = 0
            # The unchanged five samples inspect contact geometry, not forces.
            # Position stage includes collision detection and constraints;
            # real contact force integration remains in the live mj_step.
            mujoco.mj_fwdPosition(c.model, data)
            for contact in data.contact[:data.ncon]:
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if ((g1 in c.robot_geoms["left"] and g2 in c.robot_geoms["right"])
                        or (g2 in c.robot_geoms["left"] and g1 in c.robot_geoms["right"])):
                    return "predicted_robot_robot_collision"
                if ((g1 in c.obstacle_geoms and g2 in robots)
                        or (g2 in c.obstacle_geoms and g1 in robots)):
                    return "predicted_robot_obstacle_collision"
        return None

    def feasible(self, arm, position, rotation):
        task = arm.world_to_task(position)
        if np.any(task < arm.workspace[:, 0]-1e-9) or np.any(task > arm.workspace[:, 1]+1e-9):
            return "workspace_limit"
        if np.linalg.norm(position-arm.get_actual_pose()[0]) > self.c.config["control"]["max_cartesian_target_lead_m"]:
            return "tracking_lead_limit"
        arm._copy_commanded_configuration()
        for _ in range(35):
            arm._ik_iteration(position, rotation, .04)
            if (np.linalg.norm(arm.ik_data.site_xpos[arm.site_id]-position) <= self.limits["ik_position_tolerance_m"]
                    and rotation_angle(arm.ik_data.site_xmat[arm.site_id].reshape(3, 3), rotation) <= self.limits["ik_angle_tolerance_rad"]):
                break
        else:
            return "ik_unreachable"
        q = arm.ik_data.qpos[arm.qpos_indices]
        limits = arm.model.jnt_range[arm.joint_ids]
        if np.any(q < limits[:, 0]+.005) or np.any(q > limits[:, 1]-.005):
            return "joint_limit"
        return None

    def move(self, action, *, hold_targets=False):
        c = self.c
        if c.contact_control.reference is None:
            self.reference_id = None
            self.handoff_progress, self.handoff_start, self.handoff_gap = 1., None, 0.
        if hold_targets:
            # A fresh but unassigned/low-confidence pair is a clutch, not
            # permission to reconcile a newly acquired grasp in the background.
            c.user_delta = {arm.side: [0., 0., 0.] for arm in c.arms}
            c.common_delta = [0., 0., 0.]
            c.input_disagreement_m = 0.
            self.last = {"scale": 1., "reason": "mode_hold", "coupled": c.contact_control.reference is not None,
                         "accepted_positions": [a.command_position.tolist() for a in c.arms],
                         "accepted_rotations": [a.command_rotation.tolist() for a in c.arms],
                         "grasp_handoff_active": bool(c.contact_control.reference is not None and
                             (c.contact_control.reference is not self.reference_id or self.handoff_progress < 1.)),
                         "grasp_handoff_progress": self.handoff_progress,
                         "grasp_handoff_remaining_m": self.handoff_gap*(1-self.handoff_progress)}
            return None
        requested_p = [a.task_to_world(action[o:o+3]) for a, o in zip(c.arms, (0, 7))]
        requested_r = [a.task_rotation @ rpy_to_matrix(action[o+3:o+6]) for a, o in zip(c.arms, (0, 7))]
        yaw_deltas = []
        for arm, rotation in zip(c.arms, requested_r):
            relative = rotation @ arm.fixed_rotation.T
            yaw = yaw_of(relative)
            if rotation_angle(relative, rz(yaw)) > .02:
                return "roll_pitch_not_supported"
            if abs(yaw) > self.limits["max_yaw_rad"]+1e-6:
                return "yaw_limit"
            yaw_deltas.append(yaw_of(rotation @ arm.command_rotation.T))
        deltas = [p-a.command_position for a, p in zip(c.arms, requested_p)]
        ref = c.contact_control.reference
        coupled = ref is not None
        if coupled and ref is not self.reference_id:
            self.reference_id = ref
            self.object_position = c.data.xpos[c.object_id].copy()
            self.object_rotation = c.data.xmat[c.object_id].reshape(3, 3).copy()
            # Contact records actual poses, whereas the servos can still be
            # several millimetres ahead of them. Keep that physical reference
            # intact, but approach it from the last accepted command targets.
            # Scaling only the user's delta cannot bound this initial offset.
            self.handoff_start = [(self.object_rotation.T @ (a.command_position-self.object_position),
                                   self.object_rotation.T @ a.command_rotation) for a in c.arms]
            self.handoff_gap = max(np.linalg.norm(start[0]-end[0])
                                   for start, end in zip(self.handoff_start, ref))
            angle = max(rotation_angle(start[1], end[1]) for start, end in zip(self.handoff_start, ref))
            self.handoff_step = min(1., c.config["control"]["max_ee_step_m"]/max(self.handoff_gap, 1e-12),
                                    self.limits["max_yaw_step_rad"]/max(angle, 1e-12))
            self.handoff_progress = 0.
        self.last = {"scale": 0., "reason": None, "coupled": coupled, "requested_positions": [p.tolist() for p in requested_p]}
        if coupled:
            axis = c.right.command_position-c.left.command_position
            length2 = float(axis[:2] @ axis[:2])
            difference = deltas[1]-deltas[0]
            axial = float(difference @ axis / max(np.linalg.norm(axis), 1e-9))
            self.last["axial_disagreement_m"] = axial
            if abs(axial) > self.limits["axial_conflict_m"]:
                self.last["reason"] = "cooperative_axial_conflict"
                return self.last["reason"]
            if abs(yaw_deltas[0]-yaw_deltas[1]) > .02:
                return "cooperative_orientation_conflict"
            common = .5*(deltas[0]+deltas[1])
            explicit = float(np.mean(yaw_deltas))
            positional = float((axis[0]*difference[1]-axis[1]*difference[0])/max(length2, 1e-9))
            yaw = explicit if abs(explicit) > 1e-6 else positional
            yaw = float(np.clip(yaw, -self.limits["max_yaw_step_rad"], self.limits["max_yaw_step_rad"]))
            total_yaw = yaw_of(self.object_rotation)
            yaw = float(np.clip(total_yaw+yaw, -self.limits["max_yaw_rad"], self.limits["max_yaw_rad"])-total_yaw)
            def candidate(scale):
                pos = self.object_position+common*scale
                rot = rz(yaw*scale) @ self.object_rotation
                progress = min(1., self.handoff_progress+self.handoff_step*scale)
                if progress < 1.:
                    refs = [((1-progress)*start[0]+progress*end[0], blend_rotation(start[1], end[1], progress))
                            for start, end in zip(self.handoff_start, ref)]
                else:
                    refs = ref
                return [pos+rot @ r[0] for r in refs], [rot @ r[1] for r in refs], (pos, rot, progress)
        else:
            self.reference_id = None
            self.handoff_progress, self.handoff_start, self.handoff_gap = 1., None, 0.
            yaw_deltas = np.clip(yaw_deltas, -self.limits["max_yaw_step_rad"], self.limits["max_yaw_step_rad"])
            def candidate(scale):
                return [a.command_position+d*scale for a, d in zip(c.arms, deltas)], [rz(y*scale) @ a.command_rotation for a, y in zip(c.arms, yaw_deltas)], None
        max_step = c.config["control"]["max_ee_step_m"]
        scale = 1.
        reason = None
        for _ in range(12):
            positions, rotations, object_target = candidate(scale)
            largest = max(np.linalg.norm(p-a.command_position) for a, p in zip(c.arms, positions))
            if largest > max_step+1e-9:
                scale *= min(.5, max_step/largest)
                reason = "cartesian_step_limit"
                continue
            largest_angle = max(rotation_angle(r, a.command_rotation) for a, r in zip(c.arms, rotations))
            if coupled and self.handoff_progress < 1. and largest_angle > self.limits["max_yaw_step_rad"]+1e-9:
                scale *= min(.5, self.limits["max_yaw_step_rad"]/largest_angle)
                reason = "rotation_step_limit"
                continue
            reasons = [self.feasible(a, p, r) for a, p, r in zip(c.arms, positions, rotations)]
            failure = next((r for r in reasons if r), None)
            if failure is None:
                failure = self.collision_reason()
            if failure:
                reason = failure; scale *= .5; continue
            # Both arms have passed before any real controller target changes.
            for arm, p, r in zip(c.arms, positions, rotations):
                arm.command_position, arm.command_rotation = p.copy(), r.copy()
            for arm in c.arms:
                arm.track_target()
            if object_target is not None:
                self.object_position, self.object_rotation, self.handoff_progress = object_target
            self.last.update(scale=scale, reason=reason, accepted_positions=[p.tolist() for p in positions], accepted_rotations=[r.tolist() for r in rotations])
            self.last.update(grasp_handoff_active=bool(coupled and self.handoff_progress < 1.),
                             grasp_handoff_progress=self.handoff_progress,
                             grasp_handoff_remaining_m=self.handoff_gap*(1-self.handoff_progress))
            c.user_delta = {a.side: d.tolist() for a, d in zip(c.arms, deltas)}
            c.common_delta = (.5*(deltas[0]+deltas[1])).tolist() if coupled else [0., 0., 0.]
            if largest > 1e-8:
                c.start_task_timer()
            if scale < 1:
                c.metrics.workspace_clamp_count += 1
            return None
        self.last["reason"] = reason or "ik_unreachable"
        return self.last["reason"]
