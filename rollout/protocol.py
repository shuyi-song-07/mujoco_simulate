"""Frozen source-scene verification and independent physical rollout scoring.

Preflight retains every specified seed. It checks finite reset states, table
containment, waypoint IK, and camera frusta, not dynamic task solvability or
occlusion. No policy weights, predictions, or outcomes enter case selection.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "papera-rollout-physical-success-v1"
FULL_OPEN_SUCCESS_CRITERION = "papera-physical-full-open-v1"
HALF_OPEN_SUCCESS_CRITERION = "papera-physical-half-open-v1"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sample_initial_state(config, seed):
    if config.get('task_binding'):
        from tasks.registry import hook
        return hook(config, 'initial_state')(config, seed)
    from tasks.adapters.tray import initial_state
    return initial_state(config, seed)


def episode_source_records(episode):
    """Use the pack's frozen provenance, not a scan of candidate raw data."""
    return episode.get("source_fingerprints", episode.get("raw_source_fingerprints", []))


def verify_oblique_camera(episode, metadata):
    """Resolve the recorded oblique renderer, including the audited rename proof.

    Original episode configs predate oblique. Derived RGB carries its own frozen
    renderer provenance; using today's collection camera alone is insufficient.
    """
    from simulation.mujoco.dual_arm.training_camera import training_camera_metadata
    from simulation.mujoco.dual_arm.render_identity import render_identity
    records = episode_source_records(episode)
    matches = [row for row in records if Path(row["path"]).name == "oblique_provenance.json"]
    if matches:
        raise ValueError('Historical derived camera packs belong to the original project')
    else:
        config = metadata["config"]
        identity = metadata.get("render_identity")
        if not identity or "training_camera" not in config or identity != render_identity(config):
            raise ValueError("Native oblique source lacks a matching recorded render identity")
        camera = identity["camera"]
        source_root = episode["source_ref"]["dataset_root"]
        proof = {"mode": "native_recorded_identity", "verified": True}
    if camera != training_camera_metadata(config):
        raise ValueError("Recorded oblique camera differs from the live renderer")
    return {"camera": camera, "render_identity": identity,
            "current_render_identity": current if matches else identity,
            "compatibility": proof, "source_dataset_root": str(source_root)}


def load_scene_provenance(pack_dir, raw_root=None):
    """Read/hash-check all source episode metadata, then freeze latest config.

    Full source configs differ in historical human gesture bookkeeping only;
    the physical scene, cameras, controller limits and evaluation must agree.
    All locally saved/rejected raw episode seeds are excluded from testing.
    """
    from simulation.mujoco.dual_arm.config import config_hash, load_config

    pack_dir = Path(pack_dir).resolve()
    manifest_path = pack_dir / "manifest.json"
    manifest = _read(manifest_path)
    if not manifest.get("complete") or not manifest.get("episodes"):
        raise ValueError("Training pack must be complete and nonempty")
    single_view = manifest.get("version") == "papera-single-view-training-pack-v2"
    entries, samples = [], []
    cameras, original_roots = {}, set()
    reference = None
    for episode in manifest["episodes"]:
        candidates = [item for item in episode_source_records(episode)
                      if Path(item["path"]).name == "episode_metadata.json"]
        if len(candidates) != 1:
            raise ValueError(f"Exactly one frozen episode metadata required: {episode['key']}")
        source = candidates[0]
        path = Path(source["path"])
        if not path.is_file() or _sha(path) != source["sha256"]:
            raise ValueError(f"Raw source metadata missing or changed: {path}")
        metadata = _read(path)
        if single_view:
            from simulation.mujoco.dual_arm.action_contract import STATE_NAMES, ACTION_NAMES_V2
            if (metadata.get("state_names") != STATE_NAMES or metadata.get("action_names") != ACTION_NAMES_V2
                    or metadata.get("action_units") != ["m", "m", "m", "rad", "rad", "rad", "m"]*2):
                raise ValueError("Frozen source state/action fields or units differ from the policy")
        config = load_config(metadata["config"])
        if config_hash(config) != metadata["config_hash"]:
            raise ValueError(f"Frozen config hash mismatch: {path}")
        from tasks.registry import validate_binding
        validate_binding(config.get('task_binding'), manifest.get('task_id'))
        if metadata.get('task_binding') != manifest.get('task_binding'):
            raise ValueError('Raw/Pack task binding mismatch')
        physical = {key: value for key, value in config.items()
                    if key not in {"version", "gesture_control"}}
        if reference is None:
            reference = physical
        elif physical != reference:
            raise ValueError("Training sources do not share one physical scene/controller/camera/evaluation contract")
        if single_view:
            # One proof per frozen source camera rather than recompiling it for
            # every episode in the same collection session.
            root = episode["source_ref"]["dataset_root"]
            if root not in cameras:
                cameras[root] = verify_oblique_camera(episode, metadata)
            original_roots.add(str(Path(cameras[root]["source_dataset_root"]).parent))
        elif episode.get("cameras") != config["scene"]["cameras"]:
            raise ValueError(f"Pack/source camera mismatch: {episode['key']}")
        entries.append({"episode": episode["key"], "path": str(path), "sha256": source["sha256"],
                        "config_hash": metadata["config_hash"], "seed": int(metadata["seed"]),
                        "controller_version": config["version"], "config": config})
        samples.append(sample_initial_state(config, int(metadata["seed"])))
    entries.sort(key=lambda entry: entry["episode"])
    frozen = copy.deepcopy(entries[-1]["config"])
    if raw_root is None:
        # .../datasets/session/research/episode_snapshots/episode_metadata.json
        if single_view and len(original_roots) != 1:
            raise ValueError("Single-view sources do not share one original dataset root")
        raw_root = next(iter(original_roots)) if single_view else Path(entries[-1]["path"]).parents[3]
    raw_root = Path(raw_root).resolve()
    paths = sorted(raw_root.rglob("episode_metadata.json"))
    all_raw = []
    for path in paths:
        metadata = _read(path)
        seed = metadata.get("seed")
        if type(seed) is not int or seed < 0:
            raise ValueError(f"Raw episode has invalid/missing seed: {path}")
        all_raw.append({"path": str(path), "sha256": _sha(path), "seed": seed,
                        "outcome": metadata.get("outcome")})
    if not set(item["seed"] for item in entries).issubset(item["seed"] for item in all_raw):
        raise ValueError("Raw root does not contain all training-source seeds")
    xyz = np.asarray([sample["object_position"] for sample in samples])
    yaw = np.asarray([sample["object_yaw_rad"] for sample in samples])
    return {"version": PROTOCOL_VERSION, "config": frozen, "config_hash": config_hash(frozen),
            "physical_contract_sha256": _digest(reference),
            "pack_manifest_path": str(manifest_path), "pack_manifest_sha256": _sha(manifest_path),
            "representative_source": {key: value for key, value in entries[-1].items() if key != "config"},
            "sources": [{key: value for key, value in entry.items() if key != "config"} for entry in entries],
            "training_cameras": ["oblique"] if single_view else ["top", "front", "side"],
            "oblique_render_sources": cameras,
            "raw_root": str(raw_root), "raw_seed_sources": all_raw,
            "demonstration_seeds": sorted(set(item["seed"] for item in all_raw)),
            "observed_pose_ranges": {"object_position_min_m": xyz.min(0).tolist(),
                                     "object_position_max_m": xyz.max(0).tolist(),
                                     "object_yaw_min_rad": float(yaw.min()),
                                     "object_yaw_max_rad": float(yaw.max())},
            "quality_review": manifest.get("quality_review"),
            "scope": "exploratory same-scene evaluation; new seeds do not imply held-out scene/generalization"}


def _corners(centers, rotations, half_sizes):
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
    return np.concatenate([center + (signs * size) @ rotation.T
                           for center, rotation, size in zip(centers, rotations, half_sizes)])


def camera_frustum_check(model, data, camera_name, points, aspect=640/480):
    """Geometric clipping only; does not certify visibility through robot parts."""
    import mujoco
    camera_id = model.camera(camera_name).id
    camera_rotation = data.cam_xmat[camera_id].reshape(3, 3)
    local = (np.asarray(points) - data.cam_xpos[camera_id]) @ camera_rotation
    depth = -local[:, 2]
    orthographic = int(model.cam_projection[camera_id]) == int(mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC)
    half_y = (np.full(len(local), model.cam_fovy[camera_id]/2) if orthographic
              else depth * np.tan(np.deg2rad(model.cam_fovy[camera_id])/2))
    half_x = half_y * aspect
    visible = (depth > 0) & (np.abs(local[:, 0]) <= half_x) & (np.abs(local[:, 1]) <= half_y)
    return {"all_corners_in_frustum": bool(visible.all()), "inside_corner_count": int(visible.sum()),
            "corner_count": len(visible), "max_horizontal_fraction": float(np.max(np.abs(local[:, 0])/half_x)),
            "max_vertical_fraction": float(np.max(np.abs(local[:, 1])/half_y)),
            "occlusion_checked": False}


def _waypoint_checks(controller, position, yaw, label):
    """Full-pose static IK in isolated ik_data, not single-step tracking limits."""
    from simulation.mujoco.dual_arm.panda_arm_controller import rotation_angle
    from simulation.mujoco.dual_arm.rigid_control import rz

    rows = []
    rotation = rz(yaw)
    for arm, offset in zip(controller.arms, controller.config["scene"]["grasp_offsets"]):
        target = position + rotation @ np.asarray(offset)
        target_rotation = rotation @ arm.fixed_rotation
        task = arm.world_to_task(target)
        workspace = bool(np.all(task >= arm.workspace[:, 0]) and np.all(task <= arm.workspace[:, 1]))
        arm._copy_commanded_configuration()
        for _ in range(400):
            arm._ik_iteration(target, target_rotation, .1)
            error = float(np.linalg.norm(arm.ik_data.site_xpos[arm.site_id] - target))
            angle = rotation_angle(arm.ik_data.site_xmat[arm.site_id].reshape(3, 3), target_rotation)
            if error <= .001 and angle <= .015:
                break
        ranges = arm.model.jnt_range[arm.joint_ids]
        q = arm.ik_data.qpos[arm.qpos_indices]
        margin = float(min(np.min(q-ranges[:, 0]), np.min(ranges[:, 1]-q)))
        rows.append({"waypoint": label, "side": arm.side, "target_world_m": target.tolist(),
                     "inside_workspace": workspace, "ik_position_error_m": error,
                     "ik_angle_error_rad": angle, "joint_limit_margin_rad": margin,
                     "static_ik_pass": bool(error <= .001 and angle <= .015 and margin >= .005)})
    return rows


def preflight_cases(config, seeds, demonstration_seeds=(), engineering_seeds=()):
    """Preserve specified order and every seed, reporting rather than filtering."""
    from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
    from simulation.mujoco.dual_arm.rigid_control import rz

    seeds = list(seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Cases require unique nonnegative integer seeds")
    overlap = sorted(set(seeds) & (set(demonstration_seeds) | set(engineering_seeds)))
    if overlap:
        raise ValueError(f"Case seeds overlap source/engineering seeds: {overlap}")
    controller = DualArmTaskController(config=config)
    table_half = np.asarray(config["scene"]["table_half_size_xy"])
    rows = []
    for seed in seeds:
        controller.reset(seed)
        predicted = sample_initial_state(config, seed)
        for name in ("object_position", "object_quaternion_wxyz", "target_position", "target_yaw_rad"):
            if not np.allclose(predicted[name], controller.initial_state[name], atol=1e-12):
                raise ValueError(f"Initial-state sampler and live reset differ: {name}")
        object_geoms = sorted(controller.object_geoms)
        centers = controller.data.geom_xpos[object_geoms].copy()
        rotations = controller.data.geom_xmat[object_geoms].reshape(-1, 3, 3).copy()
        sizes = controller.model.geom_size[object_geoms].copy()
        points = _corners(centers, rotations, sizes)
        initial_position = np.asarray(predicted["object_position"])
        target_position = np.asarray(predicted["target_position"])
        # A representative 8 cm carry waypoint, not an imposed policy trajectory.
        lift_position = initial_position + [0., 0., .08]
        target_lift_position = target_position + [0., 0., .08]
        yaw = predicted["object_yaw_rad"]
        target_yaw = predicted["target_yaw_rad"]
        relative = (points-initial_position) @ rz(yaw)
        target_points = target_position + relative @ rz(target_yaw).T
        target_geoms = np.flatnonzero(controller.model.geom_bodyid == controller.target_id)
        target_marker_points = _corners(controller.data.geom_xpos[target_geoms],
                                        controller.data.geom_xmat[target_geoms].reshape(-1, 3, 3),
                                        controller.model.geom_size[target_geoms])
        lift_points = points + [0., 0., .08]
        target_lift_points = target_points + [0., 0., .08]
        inside_table = bool(np.all(np.abs(points[:, :2]) <= table_half))
        target_inside_table = bool(np.all(np.abs(target_points[:, :2]) <= table_half))
        views = {name: {label: camera_frustum_check(controller.model, controller.data, name, geometry)
                        for label, geometry in (("initial", points), ("lift_8cm", lift_points),
                                                ("target", target_points), ("target_lift_8cm", target_lift_points),
                                                ("target_marker", target_marker_points))}
                 for name in ("top", "front", "side", "oblique") if name in [controller.model.camera(i).name for i in range(controller.model.ncam)]}
        ik = []
        for position, angle, label in ((initial_position, yaw, "initial_grasp"),
                                       (lift_position, yaw, "lift_8cm"),
                                       (target_lift_position, target_yaw, "target_lift_8cm"),
                                       (target_position, target_yaw, "target_place")):
            ik.extend(_waypoint_checks(controller, position, angle, label))
        checks = {"finite_reset": bool(np.isfinite(controller.data.qpos).all() and np.isfinite(controller.data.qvel).all()),
                  "initial_object_on_table_xy": inside_table, "target_on_table_xy": target_inside_table,
                  "no_initial_robot_collision": not controller._robot_robot_collision() and not controller._robot_obstacle_collision(),
                  "all_waypoints_in_workspace": all(item["inside_workspace"] for item in ik),
                  "all_waypoints_static_ik": all(item["static_ik_pass"] for item in ik),
                  "all_object_waypoints_in_all_frusta": all(v["all_corners_in_frustum"] for camera in views.values() for v in camera.values())}
        rows.append({"seed": seed, "initial_state": copy.deepcopy(controller.initial_state),
                     "checks": checks, "flags": [key for key, ok in checks.items() if not ok],
                     "static_ik": ik, "views": views, "retained": True})
    return {"version": PROTOCOL_VERSION, "seeds": seeds, "cases": rows,
            "summary": {"case_count": len(rows), "retained_count": len(rows), "filtered_count": 0,
                        "cases_with_flags": sum(bool(row["flags"]) for row in rows),
                        "flag_counts": {key: sum(not row["checks"][key] for row in rows) for key in rows[0]["checks"]}},
            "limitations": ["Static waypoints do not prove dynamic collision-free grasp/lift/carry/release.",
                            "Camera frustum checks do not prove freedom from robot occlusion.",
                            "No rejection sampling or selection based on policy results."]}


class SuccessMonitor:
    """Physical task evidence, independent of stale teleoperation FSM labels.

    Call at each 60 Hz control tick. A fully qualified dual grip must persist
    120 ms; while both grips remain qualified, elevation >= 2 cm must persist
    another 120 ms. Placement uses frozen target/support/settling tolerances.
    """
    def __init__(self, config, initial_state, min_lift_m=.02, *, minimum_open_fraction=1.0,
                 success_criterion=None):
        if not np.isfinite(min_lift_m) or min_lift_m <= 0:
            raise ValueError("min_lift_m must be positive and finite")
        if (isinstance(minimum_open_fraction, (bool, np.bool_))
                or not np.isfinite(minimum_open_fraction)
                or not 0 < minimum_open_fraction <= 1):
            raise ValueError("minimum_open_fraction must be in (0, 1]")
        self.config = copy.deepcopy(config)
        self.initial_z = float(initial_state["object_position"][2])
        self.min_lift_m = float(min_lift_m)
        self.minimum_open_fraction = float(minimum_open_fraction)
        self.success_criterion = success_criterion or (
            HALF_OPEN_SUCCESS_CRITERION if self.minimum_open_fraction < 1.0
            else FULL_OPEN_SUCCESS_CRITERION)
        self.dual_since = self.lift_since = self.placement_since = None
        self.ever_dual = self.lift_verified = False
        self.first_dual_time_s = self.first_lift_time_s = None
        self.max_lift_m = 0.
        self.last_time = None

    def update(self, controller):
        status = controller.get_status()
        now = float(status["simulation_time"])
        if self.last_time is not None and now < self.last_time-1e-9:
            raise ValueError("SuccessMonitor cannot be reused across resets")
        self.last_time = now
        contacts = status["contacts"]
        force_min = self.config["cooperation"]["minimum_contact_force_n"]
        hold_seconds = self.config["cooperation"]["grasp_stable_seconds"]
        # Opposing closing-axis load excludes full-open fingers pressing down
        # on the tray. No gesture label is needed to judge a learned policy.
        opposed = {side: bool(contacts.get(side) and len(contacts.get(f"{side}_grip_forces_n", [])) == 2
                              and all(force >= force_min for force in contacts[f"{side}_grip_forces_n"])
                              and status["grippers"][side]["command_width_m"] < status["grippers"][side]["max_width_m"]-1e-6)
                   for side in ("left", "right")}
        dual = all(opposed.values())
        self.dual_since = (now if self.dual_since is None else self.dual_since) if dual else None
        stable_dual = dual and now-self.dual_since >= hold_seconds-1e-9
        if stable_dual and not self.ever_dual:
            self.ever_dual, self.first_dual_time_s = True, now
        lift = float(status["object_position"][2])-self.initial_z
        self.max_lift_m = max(self.max_lift_m, lift)
        lifting = stable_dual and lift >= self.min_lift_m
        self.lift_since = (now if self.lift_since is None else self.lift_since) if lifting else None
        if lifting and now-self.lift_since >= hold_seconds-1e-9 and not self.lift_verified:
            self.lift_verified, self.first_lift_time_s = True, now
        evaluation = self.config["evaluation"]
        position, target = np.asarray(status["object_position"]), np.asarray(status["target_position"])
        physical = status.get("physical_contact") or {}
        any_contact = any(any(contacts.get(f"{side}_fingers", [])) for side in ("left", "right"))
        # Preserve the historical full-open criterion (1 mm servo tolerance).
        # Half-open is a separate criterion: exactly 40 mm for an 80 mm jaw.
        open_thresholds = {side: (float(gripper["max_width_m"]) - .001
                                  if self.minimum_open_fraction == 1.0 else
                                  float(gripper["max_width_m"]) * self.minimum_open_fraction)
                           for side, gripper in status["grippers"].items()}
        conditions = {
            "position_xy": float(np.linalg.norm(position[:2]-target[:2])) <= evaluation["target_position_tolerance_m"],
            "height": abs(float(position[2]-target[2])) <= evaluation["target_height_tolerance_m"],
            "orientation": status["target_orientation_error_rad"] <= evaluation["target_orientation_tolerance_rad"],
            "linear_speed": float(np.linalg.norm(status["object_linear_velocity_mps"])) <= evaluation["stable_linear_speed_mps"],
            "angular_speed": float(np.linalg.norm(status["object_angular_velocity_radps"])) <= evaluation["stable_angular_speed_radps"],
            "table_support": bool(physical.get("support_ready")),
            "no_finger_contact": not any_contact,
            "both_commanded_open": all(status["grippers"][side]["command_width_m"] >= threshold
                                        for side, threshold in open_thresholds.items()),
            "both_actually_open": all(status["grippers"][side]["actual_width_m"] >= threshold
                                      for side, threshold in open_thresholds.items()),
            "no_collision": status.get("collision_count", 0) == 0,
            "not_failed": status["task_state"] != "FAIL",
        }
        if 'placement_platform' in self.config['scene']:
            conditions['target_platform_support'] = physical.get('support_surface') == 'placement_platform'
            conditions['tray_contained_on_platform'] = physical.get('placement', {}).get('contained') is True
        good = all(conditions.values())
        self.placement_since = (now if self.placement_since is None else self.placement_since) if good else None
        placement = good and now-self.placement_since >= evaluation["stable_success_seconds"]-1e-9
        success = bool(placement and self.ever_dual and self.lift_verified)
        done = status["task_state"] == "DONE"
        verified = self.ever_dual and self.lift_verified
        # The legacy FSM times commanded-open, while this scorer additionally
        # checks measured opening. The runner may hold goals and settle for a
        # bounded grace interval after DONE; it must not issue more actions.
        terminal = success or status["task_state"] == "FAIL" or (done and not verified)
        reason = None
        if terminal and not success:
            if status["task_state"] == "FAIL":
                reason = status.get("failure_reason") or "controller_failure"
            elif not self.ever_dual:
                reason = "placement_without_verified_dual_grasp"
            elif not self.lift_verified:
                reason = "placement_without_verified_lift"
            else:
                reason = "placement_physical_conditions_not_met"
        elif done and not success:
            reason = "awaiting_final_physical_settle"
        return {"version": PROTOCOL_VERSION, "success_criterion": self.success_criterion,
                "minimum_open_fraction": self.minimum_open_fraction,
                "minimum_open_widths_m": open_thresholds,
                "task_success": success, "success": success,
                "placement_success": bool(placement), "terminal": bool(terminal), "reason": reason,
                "ever_dual": self.ever_dual, "lift_verified": self.lift_verified,
                "current_opposed_grasps": opposed, "max_lift_m": self.max_lift_m,
                "lift_above_initial_m": lift, "min_lift_m": self.min_lift_m,
                "first_dual_time_s": self.first_dual_time_s, "first_lift_time_s": self.first_lift_time_s,
                "placement_stable_seconds": 0. if self.placement_since is None else now-self.placement_since,
                "target_error": {"xy_m": float(np.linalg.norm(position[:2]-target[:2])),
                                 "z_m": float(position[2]-target[2]),
                                 "orientation_rad": float(status["target_orientation_error_rad"])},
                "final_conditions": conditions}
