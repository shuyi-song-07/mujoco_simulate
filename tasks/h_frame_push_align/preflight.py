"""Task preflight with open-jaw push poses and the frozen renderer frusta."""
from __future__ import annotations

import copy

import mujoco
import numpy as np

from rollout.protocol import PROTOCOL_VERSION, _corners, camera_frustum_check, sample_initial_state
from simulation.mujoco.dual_arm.panda_arm_controller import rotation_angle
from simulation.mujoco.dual_arm.rigid_control import rz
from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
from simulation.mujoco.dual_arm.training_camera import make_training_camera


def push_contact_target(controller, arm, position, yaw, *, clearance_m=0.008, height_above_center_m=0.004):
    """Estimate a rear-rail target from the leading open fingertip pad."""
    index = 0 if arm.side == "left" else 1
    offset = np.asarray(controller.config["scene"]["grasp_offsets"][index])
    object_rotation = rz(yaw)
    normal = object_rotation[:, 0]
    face = np.asarray(position) + object_rotation @ offset
    # Aim slightly above the stock center to keep clearance to the table.
    face[2] += height_above_center_m
    ee_position, ee_rotation = arm.get_actual_pose()
    to_fixed_pose = arm.fixed_rotation @ ee_rotation.T
    candidates = []
    for geom in sorted(set.union(*controller.finger_geoms[arm.side])):
        if controller.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        center = to_fixed_pose @ (controller.data.geom_xpos[geom] - ee_position)
        rotation = to_fixed_pose @ controller.data.geom_xmat[geom].reshape(3, 3)
        projections = rotation.T @ normal
        signs = np.where(np.abs(projections) < 1e-8, 0., np.sign(projections))
        candidates.append(center + rotation @ (controller.model.geom_size[geom] * signs))
    tip = max(candidates, key=lambda point: float(point @ normal))
    return face - tip - clearance_m * normal


def _push_waypoint(controller, arm, position, yaw, label):
    # The collision mesh projects farther than the pad. A slightly higher
    # approach leaves clearance at both the rail and the table.
    target = push_contact_target(controller, arm, position, yaw, clearance_m=0.0105, height_above_center_m=0.008)
    task_position = arm.world_to_task(target)
    if "workspace_z_min_m" in controller.config["scene"]:
        task_position[2] = max(task_position[2], arm.workspace[2, 0])
        target = arm.task_to_world(task_position)
    in_workspace = np.all(task_position >= arm.workspace[:, 0]) and np.all(task_position <= arm.workspace[:, 1])
    arm._copy_commanded_configuration()
    for _ in range(400):
        arm._ik_iteration(target, arm.fixed_rotation, 0.1)
        position_error = float(np.linalg.norm(arm.ik_data.site_xpos[arm.site_id] - target))
        orientation_error = rotation_angle(arm.ik_data.site_xmat[arm.site_id].reshape(3, 3), arm.fixed_rotation)
        if position_error <= 0.001 and orientation_error <= 0.015:
            break
    qpos = arm.ik_data.qpos[arm.qpos_indices].copy()
    limits = arm.model.jnt_range[arm.joint_ids]
    margin = float(np.min(np.r_[qpos - limits[:, 0], limits[:, 1] - qpos]))
    return {
        "waypoint": label, "side": arm.side, "target_world_m": target.tolist(),
        "inside_workspace": bool(in_workspace), "ik_position_error_m": position_error,
        "ik_angle_error_rad": orientation_error, "joint_limit_margin_rad": margin,
        "static_ik_pass": bool(position_error <= 0.001 and orientation_error <= 0.015 and margin >= 0.005),
    }, qpos


def _geometry_corners(controller, ids):
    sizes = controller.model.geom_size[ids].copy()
    cylinders = controller.model.geom_type[ids] == mujoco.mjtGeom.mjGEOM_CYLINDER
    sizes[cylinders] = sizes[cylinders][:, [0, 0, 1]]
    return _corners(controller.data.geom_xpos[ids], controller.data.geom_xmat[ids].reshape(-1, 3, 3), sizes)


def _free_camera_frustum(camera, points, aspect):
    offset = points - camera.pos
    depth = offset @ camera.forward
    lateral = offset @ np.cross(camera.forward, camera.up)
    vertical = offset @ camera.up
    scale = np.maximum(depth, camera.frustum_near) / camera.frustum_near
    half_height = (camera.frustum_top - camera.frustum_bottom) / 2 * scale
    half_width = half_height * aspect
    horizontal_fraction = np.abs(lateral - camera.frustum_center * scale) / half_width
    vertical_center = (camera.frustum_top + camera.frustum_bottom) / 2 * scale
    vertical_fraction = np.abs(vertical - vertical_center) / half_height
    inside = ((depth >= camera.frustum_near) & (depth <= camera.frustum_far)
              & (horizontal_fraction <= 1) & (vertical_fraction <= 1))
    return {"all_corners_in_frustum": bool(inside.all()), "inside_corner_count": int(inside.sum()),
            "corner_count": len(points), "max_horizontal_fraction": float(horizontal_fraction.max()),
            "max_vertical_fraction": float(vertical_fraction.max()), "occlusion_checked": False}


def _penetrations(controller, data, geom_groups):
    pairs = []
    for contact in data.contact[:data.ncon]:
        first, second = int(contact.geom1), int(contact.geom2)
        if contact.dist < -1e-5 and any(
            (first in a and second in b) or (second in a and first in b) for a, b in geom_groups
        ):
            pairs.append({"geoms": [controller.model.geom(first).name, controller.model.geom(second).name],
                          "penetration_m": -float(contact.dist)})
    return pairs


def preflight(config, seeds, demonstration_seeds=(), engineering_seeds=()):
    """Validate every requested reset and static approach without filtering seeds."""
    seeds = list(seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Cases require unique nonnegative integer seeds")
    overlap = sorted(set(seeds) & (set(demonstration_seeds) | set(engineering_seeds)))
    if overlap:
        raise ValueError(f"Case seeds overlap source/engineering seeds: {overlap}")
    controller = DualArmTaskController(config)
    model = controller.model
    scratch = mujoco.MjData(model)
    visual = mujoco.MjvScene(model, maxgeom=model.ngeom + model.nsite + 64)
    mujoco.mjv_updateScene(model, controller.data, mujoco.MjvOption(), None,
                          make_training_camera(config), mujoco.mjtCatBit.mjCAT_ALL, visual)
    oblique = mujoco.mjv_averageCamera(*visual.camera)
    aspect = config["recording"]["width"] / config["recording"]["height"]
    cameras = [model.camera(i).name for i in range(model.ncam)]
    table_geom = model.geom("table_geom").id
    object_geoms = [g for g in sorted(controller.object_geoms) if model.geom_contype[g] or model.geom_conaffinity[g]]
    marker_geoms = np.flatnonzero(model.geom_bodyid == controller.target_id).tolist()
    robots = {side: ids - {model.geom(f"{side}_pedestal_geom").id} for side, ids in controller.robot_geoms.items()}
    robot_geoms = robots["left"] | robots["right"]
    checked_contacts = ((robots["left"], robots["right"]),
                        (robot_geoms, set(object_geoms) | {table_geom} | controller.obstacle_geoms))
    rows = []
    for seed in seeds:
        controller.reset(seed)
        predicted = sample_initial_state(config, seed)
        for name in ("object_position", "object_quaternion_wxyz", "target_position", "target_yaw_rad"):
            if not np.allclose(predicted[name], controller.initial_state[name], atol=1e-12, rtol=0):
                raise ValueError(f"Initial-state sampler and live reset differ: {name}")
        initial = np.asarray(predicted["object_position"])
        target = np.asarray(predicted["target_position"])
        yaw, target_yaw = predicted["object_yaw_rad"], predicted["target_yaw_rad"]
        points = _geometry_corners(controller, object_geoms)
        target_points = target + (points - initial) @ rz(yaw) @ rz(target_yaw).T
        markers = _geometry_corners(controller, marker_geoms)
        poses = {"initial": points, "target": target_points, "target_marker": markers}
        views = {name: {label: camera_frustum_check(model, controller.data, name, geometry, aspect)
                        for label, geometry in poses.items()}
                 for name in cameras if name in ("top", "front", "side")}
        views["oblique"] = {label: _free_camera_frustum(oblique, geometry, aspect) for label, geometry in poses.items()}
        table_pos = controller.data.geom_xpos[table_geom]
        table_rotation = controller.data.geom_xmat[table_geom].reshape(3, 3)
        table_size = model.geom_size[table_geom]
        initial_on_table = (points - table_pos) @ table_rotation
        target_on_table = (target_points - table_pos) @ table_rotation
        initial_clearance = float(initial_on_table[:, 2].min() - table_size[2])
        initial_contacts = _penetrations(controller, controller.data, checked_contacts)
        ik, approach_penetrations = [], {}
        for position, angle, label in ((initial, yaw, "initial_push_approach"), (target, target_yaw, "target_push_approach")):
            scratch.qpos[:] = controller.data.qpos
            address = controller.object_qpos_adr
            scratch.qpos[address:address+7] = np.r_[position, np.cos(angle/2), 0, 0, np.sin(angle/2)]
            for arm in controller.arms:
                row, qpos = _push_waypoint(controller, arm, position, angle, label)
                ik.append(row)
                scratch.qpos[arm.qpos_indices] = qpos
            mujoco.mj_forward(model, scratch)
            approach_penetrations[label] = _penetrations(controller, scratch, checked_contacts)
        checks = {
            "finite_reset": bool(np.isfinite(controller.data.qpos).all() and np.isfinite(controller.data.qvel).all()),
            "initial_object_on_table_xy": bool((np.abs(initial_on_table[:, :2]) <= table_size[:2]).all()),
            "target_on_table_xy": bool((np.abs(target_on_table[:, :2]) <= table_size[:2]).all()),
            "initial_object_near_table_top": -1e-5 <= initial_clearance <= 0.001,
            "no_initial_robot_collision": not initial_contacts,
            "all_waypoints_in_workspace": all(item["inside_workspace"] for item in ik),
            "all_waypoints_static_ik": all(item["static_ik_pass"] for item in ik),
            "no_static_approach_penetration": not any(approach_penetrations.values()),
            "all_object_waypoints_in_all_frusta": all(v["all_corners_in_frustum"] for camera in views.values() for v in camera.values()),
        }
        rows.append({"seed": seed, "initial_state": copy.deepcopy(controller.initial_state),
                     "checks": checks, "flags": [name for name, passed in checks.items() if not passed],
                     "initial_table_clearance_m": initial_clearance, "initial_penetrations": initial_contacts,
                     "static_ik": ik, "approach_penetrations": approach_penetrations, "views": views, "retained": True})
    return {"version": PROTOCOL_VERSION, "seeds": seeds, "cases": rows,
            "summary": {"case_count": len(rows), "retained_count": len(rows), "filtered_count": 0,
                        "cases_with_flags": sum(bool(row["flags"]) for row in rows),
                        "flag_counts": {key: sum(not row["checks"][key] for row in rows) for key in rows[0]["checks"]}},
            "limitations": ["Static open-jaw contact approaches do not prove a collision-free push trajectory.",
                            "Camera frusta do not account for robot occlusion.",
                            "Every requested seed is retained without policy-based selection."]}
