#!/usr/bin/env python
"""Record gesture-controlled MuJoCo Panda demonstrations as a LeRobot dataset.

The script reuses the Panda scene from the user's existing MuJoCo project while
running with LeRobot's Python 3.12 environment. The old Python 3.11 environment
is never modified.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.serving import WSGIRequestHandler, make_server


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "assets" / "franka_emika_panda" / "scene.xml"
)
INITIAL_ARM_QPOS = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
# With closed fingers, this pose puts the fingertip contact-pad centres at
# Z=0.415 m: the 0.400 m table top plus half of the 30 mm Push-T height.
# The hand-body origin is Z=0.5234 m because the pads sit 0.1084 m below it.
PUSH_T_INITIAL_ARM_QPOS = np.array(
    [0.0, -0.548481772, 0.0, -2.36977552, 0.0, 1.82129375, 0.8]
)

# Push-T footprint in the body's local XY frame.  The two non-overlapping
# rectangles form one rigid body: a 100x30 mm top bar and a 70x30 mm stem.
PUSH_T_OBJECT_BAR_X = (0.020, 0.050)
PUSH_T_OBJECT_BAR_Y_HALF = 0.050
PUSH_T_OBJECT_STEM_X = (-0.050, 0.020)
PUSH_T_OBJECT_STEM_Y_HALF = 0.015
PUSH_T_TARGET_PADDING = 0.005
GESTURE_CODES = {
    "None": 0, "Closed_Fist": 1, "Open_Palm": 2, "Pointing_Up": 3,
    "Thumb_Down": 4, "Thumb_Up": 5, "Victory": 6, "ILoveYou": 7,
}
DISTRACTOR_NAMES = ("distractor_blue", "distractor_green", "distractor_yellow")
DEFAULT_STACK_PROBABILITY = 0.4
WEB_CONTROL_COMMANDS = {
    "open", "close", "left", "right", "forward", "backward", "up", "down",
    "start_vertical", "start_planar", "move_xy", "set_gripper_yaw", "finalize_gripper_yaw",
}
GRIPPER_CUBE_XY_MARGIN = 0.045
GRIPPER_HAND_ABOVE_CUBE_TOP = 0.075
GRIPPER_PAD_CENTER_ABOVE_CUBE_CENTER = 0.1034
GRIPPER_LOWEST_POINT_BELOW_HAND = 0.115
LAYOUT_APPROACH_CLEARANCE = 0.050
GRASP_HEIGHT_MARGIN = 0.001
GRIPPER_CUBE_VIEW_DWELL_S = 0.05


def gripper_centered_over_cube(
    hand_pos: np.ndarray,
    cube_pos: np.ndarray,
    cube_half_size: float,
    *,
    was_above: bool = False,
) -> bool:
    """Use a narrow, stable grasp approach area for the operator camera."""
    cube_top = float(cube_pos[2] + cube_half_size)
    xy_radius = min(0.014, cube_half_size * 0.45) + (0.008 if was_above else 0.0)
    height_gap = float(hand_pos[2] - cube_top)
    return (
        float(np.linalg.norm(hand_pos[:2] - cube_pos[:2])) <= xy_radius
        and 0.025 <= height_gap <= (0.16 if was_above else 0.13)
    )


def make_push_t_sample_points(resolution: int = 81) -> np.ndarray:
    """Return uniform local-XY samples covering the physical Push-T footprint."""
    axis = np.linspace(-0.05, 0.05, resolution)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    in_bar = (
        (xx >= PUSH_T_OBJECT_BAR_X[0])
        & (xx <= PUSH_T_OBJECT_BAR_X[1])
        & (np.abs(yy) <= PUSH_T_OBJECT_BAR_Y_HALF)
    )
    in_stem = (
        (xx >= PUSH_T_OBJECT_STEM_X[0])
        & (xx <= PUSH_T_OBJECT_STEM_X[1])
        & (np.abs(yy) <= PUSH_T_OBJECT_STEM_Y_HALF)
    )
    return np.column_stack((xx[in_bar | in_stem], yy[in_bar | in_stem]))


def push_t_coverage(
    object_xy: np.ndarray,
    object_rotation_xy: np.ndarray,
    target_xy: np.ndarray,
    target_rotation_xy: np.ndarray,
    sample_points: np.ndarray,
) -> float:
    """Approximate the fraction of the physical T footprint inside its goal."""
    world_points = sample_points @ object_rotation_xy.T + object_xy
    target_points = (world_points - target_xy) @ target_rotation_xy
    padding = PUSH_T_TARGET_PADDING
    in_bar = (
        (target_points[:, 0] >= PUSH_T_OBJECT_BAR_X[0] - padding)
        & (target_points[:, 0] <= PUSH_T_OBJECT_BAR_X[1] + padding)
        & (np.abs(target_points[:, 1]) <= PUSH_T_OBJECT_BAR_Y_HALF + padding)
    )
    in_stem = (
        (target_points[:, 0] >= PUSH_T_OBJECT_STEM_X[0] - padding)
        & (target_points[:, 0] <= PUSH_T_OBJECT_STEM_X[1] + padding)
        & (np.abs(target_points[:, 1]) <= PUSH_T_OBJECT_STEM_Y_HALF + padding)
    )
    return float(np.mean(in_bar | in_stem))


def training_render_process(*args) -> None:
    """Load video dependencies only inside the training-render process."""
    from render_workers import training_render_worker

    training_render_worker(*args)


def auxiliary_render_process(*args) -> None:
    """Load rendering dependencies only inside the preview process."""
    from render_workers import auxiliary_render_worker

    auxiliary_render_worker(*args)


def rotation_matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to roll, pitch, yaw (ZYX convention)."""
    horizontal = np.hypot(matrix[0, 0], matrix[1, 0])
    if horizontal > 1e-8:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        pitch = np.arctan2(-matrix[2, 0], horizontal)
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = np.arctan2(-matrix[1, 2], matrix[1, 1])
        pitch = np.arctan2(-matrix[2, 0], horizontal)
        yaw = 0.0
    return np.array([roll, pitch, yaw])


class PreviewRequestHandler(WSGIRequestHandler):
    """Keep frequent side-preview polling from flooding the operator terminal."""

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        if self.path.startswith(("/top-preview", "/side-preview", "/front-preview")):
            return
        super().log_request(code, size)


class GestureServer(threading.Thread):
    """Small local HTTP server that receives commands from the existing UI."""

    def __init__(
        self,
        command_queue: queue.Queue[dict],
        port: int,
        top_preview_provider,
        side_preview_provider,
        front_preview_provider,
        status_provider,
    ) -> None:
        super().__init__(daemon=True)
        app = Flask(__name__)
        CORS(app)

        @app.post("/control")
        def control():
            payload = request.get_json(silent=True) or {}
            if payload.get("command") == "set_gripper_yaw":
                status = status_provider()
                if status.get("gripper_closed") or status.get("gripper_yaw_finalized"):
                    return jsonify({"ok": False, "reason": "朝向仅能在初始移动或闭爪前调整"}), 409
            if payload.get("command") == "reset":
                status = status_provider()
                if status["recording"] or status["saving_episode"]:
                    return jsonify({"ok": False, "reason": "请先保存当前轨迹再重置场景"}), 409
            if payload.get("command") == "resume":
                status = status_provider()
                expected_context = f'{status["session"]}:{status["control_epoch"]}'
                if payload.get("controlContext") != expected_context:
                    return jsonify({"ok": False, "reason": "场景已变化，请重新入镜"}), 409
            if payload.get("command"):
                command_queue.put(payload)
            return jsonify({"ok": True})

        @app.get("/health")
        def health():
            return jsonify(
                {
                    "ok": True,
                    "service": "mujoco-panda-recorder",
                    **status_provider(),
                }
            )

        @app.get("/side-preview")
        def side_preview():
            jpeg = side_preview_provider()
            if jpeg is None:
                return Response(status=204)
            return Response(
                jpeg,
                mimetype="image/jpeg",
                headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
            )

        @app.get("/top-preview")
        def top_preview():
            jpeg = top_preview_provider()
            if jpeg is None:
                return Response(status=204)
            return Response(
                jpeg,
                mimetype="image/jpeg",
                headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
            )

        @app.get("/front-preview")
        def front_preview():
            jpeg = front_preview_provider()
            if jpeg is None:
                return Response(status=204)
            return Response(
                jpeg,
                mimetype="image/jpeg",
                headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
            )

        self.server = make_server(
            "127.0.0.1",
            port,
            app,
            threaded=True,
            request_handler=PreviewRequestHandler,
        )

    def run(self) -> None:
        self.server.serve_forever()

    def stop(self) -> None:
        self.server.shutdown()


class MujocoPandaRecorder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = mujoco.MjModel.from_xml_path(str(args.model_path))
        self.data = mujoco.MjData(self.model)
        self.command_queue: queue.Queue[dict] = queue.Queue()
        self.server: GestureServer | None = None
        self.stop_requested = False
        self.session = uuid.uuid4().hex
        self.control_epoch = -1
        self.control_paused = True
        self.control_token: str | None = None
        self.recording = False
        self.saving_episode = False
        self.episode_frames = 0
        self.saved_episodes = 0
        self.failed_episodes = 0
        self.attempted_episodes = 0
        self.rng = np.random.default_rng(args.random_seed)

        self.base_id = self._body_id("link0")
        self.hand_id = self._body_id("hand")
        self.gripper_yaw_joint_id = self._joint_id("gripper_yaw_joint")
        self.gripper_yaw_qpos_address = int(self.model.jnt_qposadr[self.gripper_yaw_joint_id])
        self.gripper_yaw_dof_address = int(self.model.jnt_dofadr[self.gripper_yaw_joint_id])
        self.gripper_yaw_actuator_id = int(self.model.actuator("gripper_yaw_actuator").id)
        self.cube_id = self._body_id("cube")
        self.target_plate_id = self._body_id("target_plate")
        self.cube_joint_id = self.model.body_jntadr[self.cube_id]
        self.cube_qpos_address = self.model.jnt_qposadr[self.cube_joint_id]
        self.cube_geom_id = self._geom_id("cube") if self._has_geom("cube") else int(self.model.body_geomadr[self.cube_id])
        self.distractor_ids = [self._body_id(name) for name in DISTRACTOR_NAMES] if args.task_mode == "pick_place" else []
        self.randomized_geom_ids = [self.cube_geom_id] + [int(self.model.body_geomadr[body]) for body in self.distractor_ids]
        self.initial_cube_z = float(self.model.body_pos[self.cube_id][2])
        self.initial_target_z = float(self.model.body_pos[self.target_plate_id][2])
        self.current_cube_xy = np.zeros(2)
        self.current_target_xy = np.zeros(2)
        self.current_cube_yaw = 0.0
        self.current_target_yaw = 0.0
        self.current_layout = []
        self.latest_gesture = "None"
        self.latest_gesture_score = 0.0
        self.hand_visible = False
        self.last_gesture_update_at = float("-inf")
        self.last_gesture_command_at = float("-inf")
        self.last_resume_command_at = float("-inf")
        self.gripper_yaw = 0.0
        self.gripper_orientation_locked = False
        self.gripper_orientation_finalized = False
        self.pointing_arm_qpos: np.ndarray | None = None
        self.push_t_samples = (
            make_push_t_sample_points() if args.task_mode == "push_t" else None
        )
        self.current_push_t_coverage = 0.0
        # The normal operator camera remains in the control process. All
        # offscreen training and auxiliary cameras live in render workers.
        self.observation_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.observation_camera)
        self.observation_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.observation_camera.lookat[:] = np.array(
            [
                args.observation_lookat_x,
                args.observation_lookat_y,
                args.observation_lookat_z,
            ]
        )
        self.observation_camera.distance = args.observation_distance
        # Match the "overview" camera written into each recorded episode.
        self.observation_camera.azimuth = args.observation_azimuth
        self.observation_camera.elevation = args.observation_elevation
        self.topdown_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.topdown_camera)
        self.topdown_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.topdown_camera.lookat[:] = np.array(
            [
                args.topdown_lookat_x,
                args.topdown_lookat_y,
                args.topdown_lookat_z,
            ]
        )
        self.topdown_camera.distance = args.topdown_distance
        self.topdown_camera.azimuth = args.topdown_azimuth
        self.topdown_camera.elevation = -90.0
        self.finger_joint_ids = [
            self._joint_id("finger_joint1"),
            self._joint_id("finger_joint2"),
        ]
        self.finger_qpos_addresses = [
            self.model.jnt_qposadr[joint_id] for joint_id in self.finger_joint_ids
        ]
        self.finger_body_ids = (
            self._body_id("left_finger"),
            self._body_id("right_finger"),
        )
        self.last_bilateral_cube_contact_at = float("-inf")

        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.action_data = mujoco.MjData(self.model)
        self.target_hand_rotation_in_base = np.eye(3)
        self.action_rpy_reference = np.zeros(3)
        self.z_locked = False
        self.locked_z: float | None = None
        self.cube_over_target = False
        self.target_view_reached_with_cube = False
        self.carried_target_since = None
        self.transport_view_active = False
        self.gripper_above_cube_view = False
        self.gripper_above_cube_since: float | None = None
        self.target_entered_sim_time: float | None = None
        self.gripper_closed = False
        self.grasp_close_started_at = float("-inf")
        self.grasp_support_z = args.table_top_z
        self.pending_vertical_steps = 0
        self.last_deferred_vertical_at = float("-inf")
        self.motion_limit_reason: str | None = None
        self.z_lock_used_for_current_grasp = False
        self.vertical_anchor_xy: np.ndarray | None = None
        self.last_gripper_command_at = float("-inf")
        self.last_motion_command_at = float("-inf")
        self.hand_motion_token: str | None = None
        self.hand_motion_segment = -1
        self.hand_motion_anchor: np.ndarray | None = None
        self.hand_motion_last_sent_at = float("-inf")
        self.hand_motion_pending = np.zeros(2, dtype=np.float64)
        self.hand_motion_pending_samples = 0
        self.filtered_wrist_uv = np.zeros(2, dtype=np.float64)
        self.filtered_wrist_timestamp_ms = -1.0
        self.filtered_wrist_valid = False
        self.hand_trajectory_samples: list[dict] = []
        self.raw_wrist_uv = np.zeros(2, dtype=np.float64)
        self.raw_wrist_timestamp_ms = -1.0
        self.raw_wrist_valid = False
        self.mapped_control_pending = np.zeros(3, dtype=np.float64)
        self.mapped_control_pending_commands = 0
        self.side_preview_jpeg: bytes | None = None
        self.top_preview_jpeg: bytes | None = None
        self.front_preview_jpeg: bytes | None = None
        self.last_aux_preview_request_at = float("-inf")
        self.reset_simulation()
        self._start_render_workers()

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"MuJoCo body not found: {name}")
        return body_id

    def _joint_id(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {name}")
        return joint_id

    def _has_geom(self, name: str) -> bool:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0

    def _geom_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def _start_render_workers(self) -> None:
        context = mp.get_context("spawn")
        self.training_queue = context.Queue()
        self.training_reply_queue = context.Queue()
        self.training_process = context.Process(
            target=training_render_process,
            args=(self.args, self.training_queue, self.training_reply_queue),
            name="panda-training-renderer",
            # LeRobot's non-streaming video writer starts encoder subprocesses.
            daemon=False,
        )
        self.training_process.start()
        self._wait_training_reply("ready", timeout=120)

        self.aux_queue = context.Queue(maxsize=1)
        self.aux_result_queue = context.Queue(maxsize=4)
        self.aux_ready_queue = context.Queue()
        self.aux_process = context.Process(
            target=auxiliary_render_process,
            args=(self.args, self.aux_queue, self.aux_result_queue, self.aux_ready_queue),
            name="panda-aux-renderer",
            daemon=True,
        )
        self.aux_process.start()
        try:
            result = self.aux_ready_queue.get(timeout=120)
        except queue.Empty as error:
            raise RuntimeError("辅助视角渲染进程启动超时") from error
        if result.get("type") == "fatal":
            raise RuntimeError(
                f"辅助视角渲染进程启动失败：{result['error']}\n{result['traceback']}"
            )

    def _wait_training_reply(self, expected: str, timeout: float = 180) -> dict:
        try:
            result = self.training_reply_queue.get(timeout=timeout)
        except queue.Empty as error:
            raise RuntimeError(f"录制渲染进程等待 {expected} 超时") from error
        if result.get("type") == "fatal":
            raise RuntimeError(
                f"录制渲染进程失败：{result['error']}\n{result['traceback']}"
            )
        if result.get("type") != expected:
            raise RuntimeError(
                f"录制渲染进程返回 {result.get('type')}，预期 {expected}"
            )
        return result

    def _render_snapshot(self) -> dict:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": float(self.data.time),
            "target_plate_pos": self.model.body_pos[self.target_plate_id].copy(),
            "randomized_geom_sizes": self.model.geom_size[self.randomized_geom_ids].copy(),
        }

    def _submit_auxiliary_snapshot(self) -> None:
        snapshot = self._render_snapshot()
        try:
            self.aux_queue.put_nowait(snapshot)
        except queue.Full:
            try:
                self.aux_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.aux_queue.put_nowait(snapshot)
            except queue.Full:
                pass

    def _drain_auxiliary_results(self) -> None:
        while True:
            try:
                camera_name, jpeg = self.aux_result_queue.get_nowait()
            except queue.Empty:
                return
            if camera_name == "side":
                self.side_preview_jpeg = jpeg
            elif camera_name == "front":
                self.front_preview_jpeg = jpeg
            elif camera_name == "top":
                self.top_preview_jpeg = jpeg

    def reset_simulation(self) -> None:
        self.control_epoch += 1
        self.support_recovery_bodies = set()
        self.control_paused = True
        self.control_token = None
        self.reset_hand_motion()
        mujoco.mj_resetData(self.model, self.data)
        self.randomize_object_positions()
        initial_arm_qpos = (
            PUSH_T_INITIAL_ARM_QPOS
            if self.args.task_mode == "push_t"
            else INITIAL_ARM_QPOS
        )
        self.data.qpos[:7] = initial_arm_qpos
        self.data.qpos[self.gripper_yaw_qpos_address] = 0.0
        finger_position = 0.0 if self.args.task_mode == "push_t" else 0.05
        for address in self.finger_qpos_addresses:
            self.data.qpos[address] = finger_position
        self.data.ctrl[:7] = initial_arm_qpos
        self.data.ctrl[7] = 0.0 if self.args.task_mode == "push_t" else 255.0
        self.data.ctrl[self.gripper_yaw_actuator_id] = 0.0
        mujoco.mj_forward(self.model, self.data)
        base_rotation = self.data.xmat[self.base_id].reshape(3, 3)
        hand_rotation = self.data.xmat[self.hand_id].reshape(3, 3)
        self.target_hand_rotation_in_base = base_rotation.T @ hand_rotation
        self.initial_hand_rotation_in_base = self.target_hand_rotation_in_base.copy()
        self.gripper_yaw = 0.0
        self.gripper_orientation_locked = False
        self.gripper_orientation_finalized = False
        self.pointing_arm_qpos = None
        self.action_rpy_reference = rotation_matrix_to_rpy(
            self.target_hand_rotation_in_base
        )

        self.z_locked = self.args.task_mode == "push_t"
        self.locked_z = (
            float(self.data.xpos[self.hand_id][2])
            if self.z_locked
            else None
        )
        self.cube_over_target = False
        self.target_view_reached_with_cube = False
        self.carried_target_since = None
        self.transport_view_active = False
        self.gripper_above_cube_view = False
        self.gripper_above_cube_since = None
        self.target_entered_sim_time = None
        self.gripper_closed = self.args.task_mode == "push_t"
        self.grasp_close_started_at = float("-inf")
        self.grasp_support_z = self.args.table_top_z
        self.pending_vertical_steps = 0
        self.last_deferred_vertical_at = float("-inf")
        self.motion_limit_reason = None
        self.last_bilateral_cube_contact_at = float("-inf")
        self.z_lock_used_for_current_grasp = False
        self.vertical_anchor_xy = None
        self.last_gripper_command_at = float("-inf")
        self.side_preview_jpeg = None
        self.top_preview_jpeg = None
        self.front_preview_jpeg = None
        self.latest_gesture = "None"
        self.latest_gesture_score = 0.0
        self.hand_visible = False
        self.last_gesture_update_at = float("-inf")
        self.last_gesture_command_at = float("-inf")
        self.last_resume_command_at = float("-inf")
        self.hand_trajectory_samples = []
        self.raw_wrist_uv = np.zeros(2, dtype=np.float64)
        self.raw_wrist_timestamp_ms = -1.0
        self.raw_wrist_valid = False
        self.mapped_control_pending = np.zeros(3, dtype=np.float64)
        self.mapped_control_pending_commands = 0

    def randomize_object_positions(self, attempt: int = 0) -> None:
        """Sample reachable tabletop, red-above and red-below rigid stacks."""
        cube_low = np.array([self.args.cube_x_min, self.args.cube_y_min])
        cube_high = np.array([self.args.cube_x_max, self.args.cube_y_max])
        target_low = np.array([self.args.plate_x_min, self.args.plate_y_min])
        target_high = np.array([self.args.plate_x_max, self.args.plate_y_max])
        if self.args.task_mode == "pick_place":
            stack_base_name = None
            stack_top_name = None
            if self.rng.random() < getattr(self.args, "stack_probability", DEFAULT_STACK_PROBABILITY):
                partner = str(self.rng.choice(DISTRACTOR_NAMES))
                if self.rng.random() < 0.5:
                    stack_base_name = partner
                else:
                    stack_top_name = partner
            # The stacked target stays small enough to sit wholly on its
            # support while leaving clearance for both open fingers.
            red_half = float(self.rng.uniform(0.032 if stack_top_name else 0.025, 0.029 if stack_base_name else 0.034))
            cube_yaw = float(self.rng.uniform(-np.pi, np.pi))
            stack_base_yaw = (
                cube_yaw + float(self.rng.uniform(-np.pi / 12, np.pi / 12))
                if stack_base_name else None
            )
            self._set_box_half_size(self.cube_id, self.cube_geom_id, red_half)
            distractor_halves = {}
            for name, body_id in zip(DISTRACTOR_NAMES, self.distractor_ids):
                if name == stack_base_name:
                    half = float(self.rng.uniform(0.043, 0.046))
                elif name == stack_top_name:
                    half = float(self.rng.uniform(0.014, 0.016))
                else:
                    half = float(self.rng.uniform(0.018, 0.039))
                self._set_box_half_size(body_id, int(self.model.body_geomadr[body_id]), half)
                distractor_halves[name] = half
            base_half = distractor_halves[stack_base_name] if stack_base_name else 0.0
            red_z = (
                self.args.table_top_z + 2 * base_half + red_half + 0.003
                if stack_base_name else self.args.table_top_z + red_half + 0.002
            )
            red_grasp_hand_z = max(
                self.args.table_top_z + 2 * base_half + red_half + GRIPPER_PAD_CENTER_ABOVE_CUBE_CENTER,
                self.args.table_top_z + 2 * base_half + 2 * red_half + GRIPPER_HAND_ABOVE_CUBE_TOP,
            ) + GRASP_HEIGHT_MARGIN
            if stack_top_name:
                red_grasp_hand_z = max(red_grasp_hand_z,
                    red_z + red_half + 2 * distractor_halves[stack_top_name]
                    + GRIPPER_HAND_ABOVE_CUBE_TOP + 0.003)
            initial_gripper_xy = None
            if stack_base_name or stack_top_name:
                initial_pose = mujoco.MjData(self.model)
                initial_pose.qpos[:7] = INITIAL_ARM_QPOS
                mujoco.mj_forward(self.model, initial_pose)
                initial_gripper_xy = initial_pose.xpos[self.hand_id][:2].copy()
            for _ in range(2_000):
                cube_xy = self.rng.uniform(cube_low, cube_high)
                target_xy = self.rng.uniform(target_low, target_high)
                if self.args.task_mode == "pick_place" and getattr(self.args, "layout_sides", "any") == "split":
                    azimuth = np.deg2rad(self.args.observation_azimuth)
                    right = np.array([np.sin(azimuth), -np.cos(azimuth)])
                    lookat_xy = np.array([self.args.observation_lookat_x,
                                          self.args.observation_lookat_y])
                    # Signed horizontal offset in the fixed overview camera.
                    # Keep both objects away from the centre line while x/y
                    # still vary within their reachable random ranges.
                    if (np.dot(cube_xy - lookat_xy, right) >= -0.05
                            or np.dot(target_xy - lookat_xy, right) <= 0.05):
                        continue
                if not (
                    self._safe_xy(cube_xy, red_half)
                    and (not stack_base_name or self._safe_xy(cube_xy, base_half))
                    and self._safe_xy(target_xy, 0.09)
                ):
                    continue
                if (initial_gripper_xy is not None
                        and np.linalg.norm(cube_xy - initial_gripper_xy) < 0.10):
                    # A raised red block can otherwise start inside the
                    # open gripper before the operator makes any gesture.
                    continue
                cluster_radius = max(red_half, base_half) * np.sqrt(2)
                minimum_target_distance = max(
                    self.args.object_min_separation,
                    cluster_radius + 0.09 + 0.025,
                )
                if np.linalg.norm(cube_xy - target_xy) < minimum_target_distance:
                    continue
                if not (
                    self._pose_reachable(cube_xy, red_grasp_hand_z)
                    and (red_grasp_hand_z is None or self._pose_reachable(
                        cube_xy, red_grasp_hand_z + LAYOUT_APPROACH_CLEARANCE
                    ))
                    and (red_grasp_hand_z is None or self._stack_approach_stable(
                        cube_xy, red_grasp_hand_z + 0.08
                    ))
                    and self._pose_reachable(target_xy)
                ):
                    continue
                placements = [(cube_xy, cluster_radius), (target_xy, 0.09)]
                distractor_poses = {}
                if stack_base_name:
                    distractor_poses[stack_base_name] = (
                        cube_xy.copy(), base_half, stack_base_yaw
                    )
                if stack_top_name:
                    distractor_poses[stack_top_name] = (
                        cube_xy.copy(), distractor_halves[stack_top_name],
                        cube_yaw,  # Red below: both boxes share one random heading.
                    )
                for name in DISTRACTOR_NAMES:
                    if name in (stack_base_name, stack_top_name):
                        continue
                    half = distractor_halves[name]
                    for _ in range(250):
                        xy = self.rng.uniform(cube_low, cube_high)
                        radius = half * np.sqrt(2)
                        if self._safe_xy(xy, half) and all(np.linalg.norm(xy - other) > radius + other_radius + 0.025 for other, other_radius in placements):
                            placements.append((xy, radius))
                            distractor_poses[name] = (xy, half, float(self.rng.uniform(-np.pi, np.pi)))
                            break
                    else:
                        break
                if len(distractor_poses) == len(self.distractor_ids):
                    break
            else:
                raise RuntimeError("Safe workspace cannot fit the randomized objects and target.")
            if stack_base_name:
                local_offset = self.rng.uniform(-0.001, 0.001, size=2)
                cos_yaw, sin_yaw = np.cos(stack_base_yaw), np.sin(stack_base_yaw)
                red_xy = cube_xy + np.array([
                    cos_yaw * local_offset[0] - sin_yaw * local_offset[1],
                    sin_yaw * local_offset[0] + cos_yaw * local_offset[1],
                ])
                if self._safe_xy(red_xy, red_half):
                    cube_xy = red_xy
            self.current_layout = [{
                "name": "cube", "xy_m": cube_xy.tolist(),
                "z_m": red_z, "half_size_m": red_half, "yaw_rad": cube_yaw,
                "stacked_on": stack_base_name,
            }]
            for name in DISTRACTOR_NAMES:
                xy, half, yaw = distractor_poses[name]
                body_id = self._body_id(name)
                address = int(self.model.jnt_qposadr[self.model.body_jntadr[body_id]])
                z = (red_z + red_half + half + 0.001 if name == stack_top_name
                     else self.args.table_top_z + half + 0.002)
                self.data.qpos[address:address + 3] = [xy[0], xy[1], z]
                self.data.qpos[address + 3:address + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                self.current_layout.append({
                    "name": name, "xy_m": xy.tolist(), "z_m": z,
                    "half_size_m": half, "yaw_rad": yaw, "stacked_on": "cube" if name == stack_top_name else None,
                })
        else:
            for _ in range(1_000):
                cube_xy = self.rng.uniform(cube_low, cube_high)
                target_xy = self.rng.uniform(target_low, target_high)
                if np.linalg.norm(cube_xy - target_xy) >= self.args.object_min_separation:
                    break
            else:
                raise RuntimeError("Object randomization ranges cannot satisfy --object-min-separation.")
            cube_yaw = float(np.deg2rad(self.rng.uniform(self.args.cube_yaw_min_deg, self.args.cube_yaw_max_deg)))
            self.current_layout = []

        # The cube has a free joint, so its pose lives in qpos. The plate is a
        # fixed MuJoCo body, so its position lives in model.body_pos.
        self.data.qpos[self.cube_qpos_address : self.cube_qpos_address + 3] = [
            cube_xy[0],
            cube_xy[1],
            red_z if self.args.task_mode == "pick_place" else self.args.table_top_z + self.args.cube_half_size + 0.002,
        ]
        self.data.qpos[self.cube_qpos_address + 3 : self.cube_qpos_address + 7] = [
            np.cos(cube_yaw / 2.0),
            0.0,
            0.0,
            np.sin(cube_yaw / 2.0),
        ]
        self.model.body_pos[self.target_plate_id] = [
            target_xy[0],
            target_xy[1],
            self.initial_target_z,
        ]
        self.current_cube_xy = cube_xy
        self.current_target_xy = target_xy
        self.current_cube_yaw = float(cube_yaw)
        self.current_target_yaw = 0.0
        self.current_push_t_coverage = 0.0
        if self.args.task_mode == "pick_place":
            mujoco.mj_forward(self.model, self.data)
            checks = [
                np.array([*cube_xy, red_grasp_hand_z + height])
                for height in (0.08, 0.04, 0.0)
            ]
            checks.append(np.array([*target_xy, self.args.target_top_z + 0.15]))
            if not all(self._layout_pose_clear(position) for position in checks):
                if attempt >= 24:
                    raise RuntimeError("Could not sample a collision-free reachable grasp layout after 25 attempts.")
                self.randomize_object_positions(attempt + 1)
                return
            self.layout_reachability = {
                "checked": True, "mode": "red_above" if stack_base_name else "red_below" if stack_top_name else "table",
                "grasp_hand_z_m": red_grasp_hand_z,
                "checked_poses": [position.tolist() for position in checks],
                "position_tolerance_m": 0.002, "joint_margin_rad": 0.08,
            }

    def _layout_pose_clear(self, target: np.ndarray) -> bool:
        """IK plus open-gripper collision checks in this exact sampled scene."""
        trial = mujoco.MjData(self.model)
        trial.qpos[:] = self.data.qpos
        trial.qpos[:7] = INITIAL_ARM_QPOS
        for name in ("finger_joint1", "finger_joint2"):
            trial.qpos[self.model.jnt_qposadr[self._joint_id(name)]] = 0.05
        mujoco.mj_forward(self.model, trial)
        desired = trial.xmat[self.hand_id].reshape(3, 3).copy()
        jacp = np.zeros((3, self.model.nv)); jacr = np.zeros_like(jacp)
        limits = self.model.jnt_range[:7]
        for _ in range(240):
            mujoco.mj_forward(self.model, trial)
            error = target - trial.xpos[self.hand_id]
            rotation = trial.xmat[self.hand_id].reshape(3, 3)
            angular = 0.5 * sum(np.cross(rotation[:, i], desired[:, i]) for i in range(3))
            if np.linalg.norm(error) < 0.002 and np.linalg.norm(angular) < 0.02:
                robot_ids = {self._body_id(name) for name in (
                    "link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
                    "hand", "left_finger", "right_finger")}
                for contact in trial.contact:
                    a, b = (int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
                    if (a in robot_ids or b in robot_ids) and contact.dist < -0.0005:
                        return False
                return True
            mujoco.mj_jacBody(self.model, trial, jacp, jacr, self.hand_id)
            jac = np.vstack((jacp[:, :7], jacr[:, :7]))
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.002 * np.eye(6), np.r_[error, angular])
            trial.qpos[:7] = np.clip(trial.qpos[:7] + np.clip(dq, -0.06, 0.06), limits[:, 0] + 0.08, limits[:, 1] - 0.08)
        return False

    def _set_box_half_size(self, body_id: int, geom_id: int, half: float) -> None:
        self.model.geom_size[geom_id] = [half, half, half]
        # MuJoCo keeps both per-geom bounds and a separate BVH leaf. Updating
        # only geom_aabb still lets the BVH cull fingertip contacts on a box
        # enlarged beyond the XML size (notably the 68 mm cube).
        self.model.geom_aabb[geom_id] = [0.0, 0.0, 0.0, half, half, half]
        self.model.geom_rbound[geom_id] = np.sqrt(3.0) * half
        bvh_start = int(self.model.body_bvhadr[body_id])
        bvh_stop = bvh_start + int(self.model.body_bvhnum[body_id])
        for node in range(bvh_start, bvh_stop):
            if int(self.model.bvh_nodeid[node]) == geom_id:
                self.model.bvh_aabb[node] = self.model.geom_aabb[geom_id]
        mass = 0.1 * (half / 0.03) ** 3
        self.model.body_mass[body_id] = mass
        self.model.body_inertia[body_id] = [2 * mass * half * half / 3] * 3

    def enforce_layout_support(self) -> None:
        """Keep every free rigid layout box above its supporting hard surface."""
        if self.args.task_mode != "pick_place":
            return
        plate_center = self.data.xpos[self.target_plate_id][:2]
        plate_geom_id = int(self.model.body_geomadr[self.target_plate_id])
        plate_radius = float(self.model.geom_size[plate_geom_id][0])
        corrected = False
        for body_id, geom_id in zip([self.cube_id, *self.distractor_ids], self.randomized_geom_ids):
            joint_id = int(self.model.body_jntadr[body_id])
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            qvel_address = int(self.model.jnt_dofadr[joint_id])
            half_sizes = self.model.geom_size[geom_id]
            rotation = self.data.xmat[body_id].reshape(3, 3)
            vertical_extent = float(np.abs(rotation[2]) @ half_sizes)
            horizontal_extent = float(np.linalg.norm(half_sizes[:2]))
            fully_on_plate = body_id == self.cube_id and (
                np.linalg.norm(self.data.xpos[body_id][:2] - plate_center)
                + horizontal_extent <= plate_radius
            )
            support_z = self.args.target_top_z if fully_on_plate else self.args.table_top_z
            bottom_z = float(self.data.xpos[body_id][2]) - vertical_extent
            recovering = getattr(self, "support_recovery_bodies", set())
            self.support_recovery_bodies = recovering
            if bottom_z < support_z - 0.005:
                recovering.add(body_id)
            if bottom_z >= support_z - 0.001:
                recovering.discard(body_id)
            if body_id in recovering:
                # Emergency recovery only after >5mm penetration; finish to
                # 1mm, at most 1mm per tick. Ordinary support is all contact.
                self.data.qpos[qpos_address + 2] += min(0.001, support_z - bottom_z)
                self.data.qvel[qvel_address + 2] = max(0.0, self.data.qvel[qvel_address + 2])
                corrected = True
        if corrected:
            mujoco.mj_forward(self.model, self.data)

    def _safe_xy(self, xy: np.ndarray, margin: float) -> bool:
        x, y = xy
        radius = float(np.hypot(x, y))
        return (0.34 <= radius <= self.args.safe_reach_radius
                and 0.1 + margin <= x <= 0.9 - margin
                and abs(y) <= 0.5 - margin)

    def _pose_reachable(self, xy: np.ndarray, hand_z: float | None = None) -> bool:
        """Verify an above-object grasp pose using the actual Panda Jacobian and joint limits."""
        trial = mujoco.MjData(self.model)
        trial.qpos[:7] = INITIAL_ARM_QPOS
        target = np.array([
            xy[0], xy[1],
            self.args.table_top_z + 0.15 if hand_z is None else hand_z,
        ])
        mujoco.mj_forward(self.model, trial)
        desired_rotation = trial.xmat[self.hand_id].reshape(3, 3).copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros_like(jacp)
        limits = self.model.jnt_range[:7]
        for _ in range(100):
            mujoco.mj_forward(self.model, trial)
            position_error = target - trial.xpos[self.hand_id]
            rotation = trial.xmat[self.hand_id].reshape(3, 3)
            orientation_error = 0.5 * sum(np.cross(rotation[:, i], desired_rotation[:, i]) for i in range(3))
            if np.linalg.norm(position_error) < 0.018 and np.linalg.norm(orientation_error) < 0.12:
                return True
            mujoco.mj_jacBody(self.model, trial, jacp, jacr, self.hand_id)
            jacobian = np.vstack((jacp[:, :7], jacr[:, :7]))
            delta = np.concatenate((position_error, orientation_error))
            dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.0025 * np.eye(6), delta)
            trial.qpos[:7] = np.clip(trial.qpos[:7] + np.clip(dq, -0.08, 0.08), limits[:, 0] + 0.08, limits[:, 1] - 0.08)
        return False

    def _stack_approach_stable(self, xy: np.ndarray, hand_z: float) -> bool:
        """Reject high approaches that put an arm link into the table."""
        trial = mujoco.MjData(self.model)
        q = INITIAL_ARM_QPOS.copy()
        trial.qpos[:7] = q
        mujoco.mj_forward(self.model, trial)
        desired_rotation = trial.xmat[self.hand_id].reshape(3, 3).copy()
        target = np.array([xy[0], xy[1], hand_z])
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros_like(jacp)
        limits = self.model.jnt_range[:7]
        solved = False
        for _ in range(220):
            trial.qpos[:7] = q
            mujoco.mj_forward(self.model, trial)
            position_error = target - trial.xpos[self.hand_id]
            rotation = trial.xmat[self.hand_id].reshape(3, 3)
            rotation_error = 0.5 * sum(
                np.cross(rotation[:, axis], desired_rotation[:, axis])
                for axis in range(3)
            )
            if (np.linalg.norm(position_error) < 0.0003
                    and np.linalg.norm(rotation_error) < 0.004):
                solved = True
                break
            mujoco.mj_jacBody(self.model, trial, jacp, jacr, self.hand_id)
            jacobian = np.vstack((jacp[:, :7], jacr[:, :7]))
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 0.002 * np.eye(6),
                np.r_[position_error, rotation_error],
            )
            q = np.clip(
                q + np.clip(delta, -0.06, 0.06),
                limits[:, 0] + 0.03, limits[:, 1] - 0.03,
            )
        if not solved:
            return False

        table_geom = self._geom_id("table_surface")
        robot_bodies = {
            self._body_id(name) for name in (
                "link0", "link1", "link2", "link3", "link4", "link5",
                "link6", "link7", "hand", "left_finger", "right_finger",
            )
        }
        for contact in trial.contact:
            if table_geom == contact.geom1:
                other = contact.geom2
            elif table_geom == contact.geom2:
                other = contact.geom1
            else:
                continue
            if (int(self.model.geom_bodyid[other]) in robot_bodies
                    and contact.dist < -0.0005):
                return False

        trial.ctrl[:7] = q
        trial.ctrl[7] = 255.0
        for name in ("finger_joint1", "finger_joint2"):
            joint_id = self._joint_id(name)
            trial.qpos[self.model.jnt_qposadr[joint_id]] = 0.05
        mujoco.mj_forward(self.model, trial)
        for _ in range(150):
            mujoco.mj_step(self.model, trial)
            for contact in trial.contact:
                if table_geom == contact.geom1:
                    other = contact.geom2
                elif table_geom == contact.geom2:
                    other = contact.geom1
                else:
                    continue
                if (int(self.model.geom_bodyid[other]) in robot_bodies
                        and contact.dist < -0.0005):
                    return False
        return bool(
            np.all(np.isfinite(trial.qpos))
            and np.linalg.norm(trial.xpos[self.hand_id] - target) < 0.025
        )

    def move_hand(self, dx: float, dy: float, dz: float) -> None:
        # Pointing_Up owns the dedicated gripper-yaw target. Cartesian moves
        # may change the seven arm joints, but must never rewrite or relax it.
        self.data.ctrl[self.gripper_yaw_actuator_id] = self.gripper_yaw
        if abs(dx) + abs(dy) + abs(dz) > 1e-9:
            self.motion_limit_reason = None
        # Build the next Cartesian step from the commanded joint target, not
        # from the lagging physical arm. Otherwise the same orientation error
        # is added repeatedly while the servos catch up, which makes a stream
        # of diagonal commands twist the redundant 7-DoF arm.
        self.action_data.qpos[:] = self.data.qpos
        self.action_data.qpos[:7] = self.data.ctrl[:7]
        self.action_data.qpos[self.gripper_yaw_qpos_address] = self.data.ctrl[self.gripper_yaw_actuator_id]
        self.action_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.action_data)
        target_pos = self.action_data.xpos[self.hand_id].copy()

        if self.vertical_anchor_xy is not None:
            correction = np.clip(self.vertical_anchor_xy - target_pos[:2], -0.01, 0.01)
            dx, dy = float(correction[0]), float(correction[1])

        if self.z_locked and self.locked_z is not None:
            dz = float(self.locked_z - target_pos[2])
        elif (
            self.args.task_mode == "pick_place"
            and self.is_holding_cube()
            and dz < 0
        ):
            cube_z = float(self.data.xpos[self.cube_id][2])
            support_z = self._cube_support_height(self.data.xpos[self.cube_id][:2])
            minimum_cube_z = support_z + float(self.model.geom_size[self.cube_geom_id][2]) + 0.002
            bounded_dz = max(dz, min(0.0, minimum_cube_z - cube_z))
            if bounded_dz > dz + 1e-9:
                surface = "底座" if support_z > self.args.table_top_z + 0.03 else "桌面"
                self.motion_limit_reason = f"已到{surface}下降限位"
            dz = bounded_dz

        # Gesture translation is defined in the robot-base frame.  In the
        # current scene link0 is aligned with world, but this conversion keeps
        # the control definition correct if the robot is moved later.
        base_rotation = self.action_data.xmat[self.base_id].reshape(3, 3)
        world_delta = base_rotation @ np.array([dx, dy, dz], dtype=float)
        if self.args.task_mode == "pick_place":
            proposed = target_pos + world_delta
            proposed[0] = np.clip(proposed[0], 0.32, 0.74)
            proposed[1] = np.clip(proposed[1], -0.32, 0.32)
            # Bounds may stop a requested vertical step, but zero and planar
            # commands must not raise or lower the hand on their own.
            if dz < 0:
                support_z, support_name = self._hand_descent_support(proposed[:2])
                minimum_z = support_z + GRIPPER_LOWEST_POINT_BELOW_HAND
                bounded_z = max(proposed[2], min(minimum_z, target_pos[2]))
                if bounded_z > proposed[2] + 1e-9:
                    self.motion_limit_reason = f"已到{support_name}下降限位"
                proposed[2] = bounded_z
            elif dz > 0:
                proposed[2] = min(proposed[2], max(0.82, target_pos[2]))
            cube_xy = self.data.xpos[self.cube_id][:2]
            cube_half_height = float(self.model.geom_size[self.cube_geom_id][2])
            support_z = self._cube_support_height(cube_xy)
            approach_radius = float(self.model.geom_size[self.cube_geom_id][0]) + GRIPPER_CUBE_XY_MARGIN
            xy_error = float(np.linalg.norm(proposed[:2] - cube_xy))
            if not self.is_holding_cube() and xy_error <= approach_radius and dz < 0:
                # Use the nominal support plane instead of the current cube
                # Z: soft contact can be slightly penetrated already.
                supported_cube_top = support_z + 2.0 * cube_half_height + 0.001
                min_hand_z = max(
                    supported_cube_top + GRIPPER_HAND_ABOVE_CUBE_TOP,
                    self._red_stack_top_height() + GRIPPER_HAND_ABOVE_CUBE_TOP,
                    support_z + cube_half_height + GRIPPER_PAD_CENTER_ABOVE_CUBE_CENTER,
                )
                bounded_z = max(proposed[2], min(min_hand_z, target_pos[2]))
                if bounded_z > proposed[2] + 1e-9:
                    surface = "底座" if support_z > self.args.table_top_z + 0.03 else "桌面"
                    self.motion_limit_reason = f"已到{surface}上红块的安全下降高度"
                proposed[2] = bounded_z
            planar_radius = np.linalg.norm(proposed[:2])
            if planar_radius > self.args.safe_reach_radius + 0.04:
                proposed[:2] *= (self.args.safe_reach_radius + 0.04) / planar_radius
            world_delta = proposed - target_pos

        target_hand_rotation_world = (
            base_rotation @ self.target_hand_rotation_in_base
        )
        current_hand_rotation_world = self.action_data.xmat[self.hand_id].reshape(3, 3)
        orientation_error = 0.5 * sum(
            np.cross(
                current_hand_rotation_world[:, axis],
                target_hand_rotation_world[:, axis],
            )
            for axis in range(3)
        )

        mujoco.mj_jacBody(
            self.model,
            self.action_data,
            self.jacp,
            self.jacr,
            self.hand_id,
        )
        position_jacobian = self.jacp[:, :7]
        rotation_jacobian = self.jacr[:, :7]

        # Preserve the Pointing_Up orientation as the primary task. Planar
        # translation is solved only in the rotational nullspace, so a stream
        # of diagonal commands cannot slowly rotate the gripper.
        orientation_damping = 0.006
        rotation_inverse = rotation_jacobian.T @ np.linalg.solve(
            rotation_jacobian @ rotation_jacobian.T
            + (orientation_damping * orientation_damping) * np.eye(3),
            np.eye(3),
        )
        orientation_step = rotation_inverse @ orientation_error
        rotation_nullspace = np.eye(7) - rotation_inverse @ rotation_jacobian
        projected_position_jacobian = position_jacobian @ rotation_nullspace
        remaining_position = world_delta - position_jacobian @ orientation_step
        position_damping = 0.03
        position_step = rotation_nullspace @ projected_position_jacobian.T @ np.linalg.solve(
            projected_position_jacobian @ projected_position_jacobian.T
            + (position_damping * position_damping) * np.eye(3),
            remaining_position,
        )
        dq = orientation_step + position_step
        dq = np.clip(dq, -self.args.max_joint_step, self.args.max_joint_step)
        next_joint_target = self.data.ctrl[:7] + dq
        # Never let a command stream run far ahead of the physical arm. This
        # preserves the Cartesian path while preventing a delayed controller
        # from chasing a remote joint configuration with a violent twist.
        target_lead = next_joint_target - self.data.qpos[:7]
        largest_lead = float(np.max(np.abs(target_lead)))
        if largest_lead > self.args.max_joint_target_lead:
            target_lead *= self.args.max_joint_target_lead / largest_lead
        self.data.ctrl[:7] = self.data.qpos[:7] + target_lead
        for index in range(7):
            low, high = self.model.actuator_ctrlrange[index]
            self.data.ctrl[index] = np.clip(self.data.ctrl[index], low, high)
            joint_low, joint_high = self.model.jnt_range[index]
            self.data.ctrl[index] = np.clip(self.data.ctrl[index], joint_low + 0.06, joint_high - 0.06)

    def set_z_lock(self, locked: bool, reason: str | None = None) -> None:
        was_locked = self.z_locked
        self.z_locked = locked
        if locked and not was_locked:
            if self.args.task_mode == "pick_place" and self.is_holding_cube():
                self.transport_view_active = True
            # Capture the commanded height once. Repeated ILoveYou commands
            # must never overwrite it with the slightly sagging physical
            # height, otherwise the horizontal plane ratchets downward.
            self.action_data.qpos[:] = self.data.qpos
            self.action_data.qpos[:7] = self.data.ctrl[:7]
            self.action_data.qpos[self.gripper_yaw_qpos_address] = self.data.ctrl[self.gripper_yaw_actuator_id]
            self.action_data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.action_data)
            self.locked_z = float(self.action_data.xpos[self.hand_id][2])
        elif not locked:
            self.locked_z = None
        if locked and not was_locked:
            cube_z = float(self.data.xpos[self.cube_id][2])
            if reason == "left_target":
                print("[视角] 方块移出圆盘：重新锁定 Z 并切换到俯视视角。")
            else:
                print(
                    f"[视角] 当前夹持物体高度 {cube_z:.3f} m："
                    "锁定 Z 并切换到俯视视角。"
                )
        elif was_locked and not locked:
            self.side_preview_jpeg = None
            if reason == "target":
                self.transport_view_active = False
                print("[视角] 已进入圆盘区域：解除 Z 并恢复斜视角。")
            elif reason == "vertical":
                print("[控制] 已解除运输高度锁定，恢复升降控制。")
            else:
                print("[视角] 已解除 Z 并恢复斜视角。")

    def maintain_locked_plane(self) -> None:
        """Continuously keep the commanded end effector on the locked Z plane."""
        if (self.z_locked and self.locked_z is not None
                and self.pointing_arm_qpos is None):
            self.move_hand(0.0, 0.0, 0.0)

    def maintain_gripper_orientation(self) -> None:
        """Continuously hold the last Pointing_Up orientation during motion."""
        if (
            self.gripper_orientation_locked
            and self.pointing_arm_qpos is None
            and not self.z_locked
        ):
            self.move_hand(0.0, 0.0, 0.0)

    def hard_lock_gripper_yaw(self) -> None:
        """Prevent the dedicated yaw joint from drifting after Pointing_Up."""
        if not self.gripper_orientation_locked:
            return
        self.data.ctrl[self.gripper_yaw_actuator_id] = self.gripper_yaw
        if self.pointing_arm_qpos is not None and not self.gripper_orientation_finalized:
            # During the initial adjustment, let the yaw actuator rotate the
            # empty gripper. Teleporting qpos to each hand angle creates an
            # impulse that can force the fingers apart on contact.
            return
        self.data.qpos[self.gripper_yaw_qpos_address] = self.gripper_yaw
        self.data.qvel[self.gripper_yaw_dof_address] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def hold_pointing_arm(self) -> None:
        """Keep all seven arm joints fixed while the gripper yaw joint moves."""
        if self.pointing_arm_qpos is None:
            return
        self.data.qpos[:7] = self.pointing_arm_qpos
        self.data.qvel[:7] = 0.0
        self.data.ctrl[:7] = self.pointing_arm_qpos
        mujoco.mj_forward(self.model, self.data)

    def is_holding_cube(self) -> bool:
        finger_width = float(
            sum(self.data.qpos[address] for address in self.finger_qpos_addresses)
        )
        hand_cube_distance = float(
            np.linalg.norm(
                self.data.xpos[self.hand_id] - self.data.xpos[self.cube_id]
            )
        )
        if not (
            self.gripper_closed
            and self.args.grasped_min_width
            <= finger_width
            <= self.args.grasped_max_width
            and hand_cube_distance <= self.args.grasped_max_distance
        ):
            return False

        contacting_fingers = set()
        force = np.zeros(6)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if contact.geom1 == self.cube_geom_id:
                other_geom = contact.geom2
            elif contact.geom2 == self.cube_geom_id:
                other_geom = contact.geom1
            else:
                continue
            finger_body = int(self.model.geom_bodyid[other_geom])
            if finger_body not in self.finger_body_ids:
                continue
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            if force[0] > 0.01:
                contacting_fingers.add(finger_body)
                if len(contacting_fingers) == 2:
                    self.last_bilateral_cube_contact_at = float(self.data.time)
                    return True
        # Brief contact dropouts must not flicker the camera or transport lock.
        return float(self.data.time) - self.last_bilateral_cube_contact_at <= 0.05

    def gripper_ready_to_grasp(self) -> bool:
        """Closing is an explicit manual action at any position."""
        return True

    def _cube_support_height(self, cube_xy: np.ndarray) -> float:
        """Read the current surface under the red box, including a movable base."""
        target_xy = self.data.xpos[self.target_plate_id][:2]
        support_z = (
            self.args.target_top_z
            if np.linalg.norm(cube_xy - target_xy) <= self.args.target_radius
            else self.args.table_top_z
        )
        if self.args.task_mode != "pick_place":
            return support_z

        cube_half = float(self.model.geom_size[self.cube_geom_id][0])
        for body_id in self.distractor_ids:
            geom_id = int(self.model.body_geomadr[body_id])
            half_size = self.model.geom_size[geom_id]
            if self.data.xpos[body_id][2] > self.data.xpos[self.cube_id][2]:
                continue
            rotation = self.data.xmat[body_id].reshape(3, 3)
            local_xy = rotation[:2, :2].T @ (
                cube_xy - self.data.xpos[body_id][:2]
            )
            # Include a small margin so the box cannot be lowered through a
            # base while an edge still overlaps it during lateral motion.
            if np.all(np.abs(local_xy) <= half_size[:2] + cube_half + 0.003):
                top_z = float(
                    self.data.xpos[body_id][2]
                    + np.abs(rotation[2]) @ half_size
                )
                support_z = max(support_z, top_z)
        return support_z

    def _hand_descent_support(self, hand_xy: np.ndarray) -> tuple[float, str]:
        """Highest surface under the fingers during a requested descent."""
        support_z = self.args.table_top_z
        support_name = "桌面"
        plate_id = int(self.model.body_geomadr[self.target_plate_id])
        plate_radius = float(self.model.geom_size[plate_id][0])
        if np.linalg.norm(hand_xy - self.data.xpos[self.target_plate_id][:2]) <= plate_radius + 0.05:
            support_z = self.args.target_top_z
            support_name = "圆盘"
        for body_id in self.distractor_ids:
            geom_id = int(self.model.body_geomadr[body_id])
            half_size = self.model.geom_size[geom_id]
            red_half = float(self.model.geom_size[self.cube_geom_id][0])
            if (self.data.xpos[body_id][2] > self.data.xpos[self.cube_id][2]
                    and np.linalg.norm(self.data.xpos[body_id][:2] - self.data.xpos[self.cube_id][:2])
                    + np.linalg.norm(half_size[:2]) < red_half - 0.003):
                continue
            rotation = self.data.xmat[body_id].reshape(3, 3)
            local_xy = rotation[:2, :2].T @ (
                hand_xy - self.data.xpos[body_id][:2]
            )
            if np.all(np.abs(local_xy) <= half_size[:2] + 0.05):
                top_z = float(self.data.xpos[body_id][2] + np.abs(rotation[2]) @ half_size)
                if top_z > support_z:
                    support_z = top_z
                    support_name = "底座"
        return support_z, support_name

    def _red_stack_top_height(self) -> float:
        """Highest central obstruction above the red box, for palm clearance."""
        red_pos = self.data.xpos[self.cube_id]
        half = float(self.model.geom_size[self.cube_geom_id][0])
        top = float(red_pos[2] + half)
        for body_id in self.distractor_ids:
            geom_id = int(self.model.body_geomadr[body_id])
            size = self.model.geom_size[geom_id]
            if (self.data.xpos[body_id][2] > red_pos[2]
                    and np.linalg.norm(self.data.xpos[body_id][:2] - red_pos[:2]) < half):
                top = max(top, float(self.data.xpos[body_id][2] + np.abs(self.data.xmat[body_id].reshape(3, 3)[2]) @ size))
        return top

    def _grasp_target_hand_z(self) -> float:
        half = float(self.model.geom_size[self.cube_geom_id][2])
        cube_xy = self.data.xpos[self.cube_id][:2]
        support_z = self._cube_support_height(cube_xy)
        cube_top = support_z + 2 * half + 0.001
        return max(
            cube_top + GRIPPER_HAND_ABOVE_CUBE_TOP,
            self._red_stack_top_height() + GRIPPER_HAND_ABOVE_CUBE_TOP,
            support_z + half + GRIPPER_PAD_CENTER_ABOVE_CUBE_CENTER,
        ) + GRASP_HEIGHT_MARGIN


    def _commanded_hand_position(self) -> np.ndarray:
        """Read the arm's commanded pose without changing the live simulation."""
        self.action_data.qpos[:] = self.data.qpos
        self.action_data.qpos[:7] = self.data.ctrl[:7]
        self.action_data.qpos[self.gripper_yaw_qpos_address] = self.gripper_yaw
        self.action_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.action_data)
        return self.action_data.xpos[self.hand_id].copy()


    def gripper_approaching_cube(self) -> bool:
        """Expand web side/front views before the precise camera switch."""
        if self.args.task_mode != "pick_place" or self.is_holding_cube():
            return False
        cube_pos = self.data.xpos[self.cube_id]
        hand_pos = self.data.xpos[self.hand_id]
        cube_top = float(cube_pos[2] + self.model.geom_size[self.cube_geom_id][2])
        return (
            float(np.linalg.norm(hand_pos[:2] - cube_pos[:2])) <= 0.07
            and 0.02 <= float(hand_pos[2] - cube_top) <= 0.22
        )

    def update_transport_state(self) -> None:
        """Retain transport state once a held object has actually been lifted."""
        if self.args.task_mode == "push_t":
            return
        cube_z = float(self.data.xpos[self.cube_id][2])
        if (
            self.gripper_closed
            and self.is_holding_cube()
            and not self.z_locked
            and not self.z_lock_used_for_current_grasp
            and not self.cube_over_target
            and cube_z >= self._transport_cube_height()
        ):
            self.set_z_lock(True)
            self.z_lock_used_for_current_grasp = True

    def _transport_cube_height(self) -> float:
        """Require a real lift before transport when the red box starts stacked."""
        half = float(self.model.geom_size[self.cube_geom_id][2])
        return max(self.args.transport_cube_z, self.grasp_support_z + half + 0.05)

    def operator_uses_topdown(self) -> bool:
        # ILoveYou uses the top view even when the gripper is empty. A carried
        # cube keeps the existing top view through the locked transport phase.
        if (self.args.task_mode == "pick_place" and self.target_view_reached_with_cube
                and self.is_holding_cube()):
            return False
        if self.args.task_mode == "pick_place" and self.gripper_above_cube_view:
            return False
        return self.z_locked or self.transport_view_active or (
            self.args.task_mode == "pick_place"
            and self.latest_gesture == "ILoveYou"
            and self.hand_visible
            and time.perf_counter() - self.last_gesture_update_at < 0.5
        )

    def update_viewer_camera(self, viewer) -> None:
        if self.operator_uses_topdown():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = self.topdown_camera.lookat
            viewer.cam.distance = self.topdown_camera.distance
            viewer.cam.azimuth = self.topdown_camera.azimuth
            viewer.cam.elevation = self.topdown_camera.elevation
        else:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = self.observation_camera.lookat
            viewer.cam.distance = self.observation_camera.distance
            viewer.cam.azimuth = self.observation_camera.azimuth
            viewer.cam.elevation = self.observation_camera.elevation

    def get_runtime_status(self) -> dict:
        cube_half_size = float(self.model.geom_size[self.cube_geom_id][2])
        grasp_xy_error = float(np.linalg.norm(
            self.data.xpos[self.hand_id][:2] - self.data.xpos[self.cube_id][:2]
        ))
        grasp_xy_tolerance = min(0.07, cube_half_size + GRIPPER_CUBE_XY_MARGIN)
        grasp_vertical_velocity = float(self.data.cvel[self.hand_id][5])
        return {
            "task_mode": self.args.task_mode,
            "session": self.session,
            "control_epoch": self.control_epoch,
            "paused": self.control_paused,
            "control_token": self.control_token,
            "layout_reachability": getattr(self, "layout_reachability", None),
            "z_locked": self.z_locked,
            "operator_view": "topdown" if self.operator_uses_topdown() else "overview",
            "visual_observation": [
                "overview_145deg",
                "camera_2_025deg",
                "camera_3_265deg",
            ],
            "action_frame": "robot_base_link0",
            "action_axes": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
            "gripper_closed": self.gripper_closed,
            "recording": self.recording,
            "saving_episode": self.saving_episode,
            "holding_cube": self.is_holding_cube(),
            "gripper_yaw_finalized": self.gripper_orientation_finalized,
            "gripper_yaw_rad": self.gripper_yaw,
            "gripper_yaw_locked": self.gripper_orientation_locked,
            "pointing_arm_locked": self.pointing_arm_qpos is not None,
            "grasp_close_elapsed_s": max(0.0, float(self.data.time) - self.grasp_close_started_at) if self.gripper_closed else 0.0,
            "grasp_ready": self.gripper_ready_to_grasp(),
            "grasp_xy_error_m": round(grasp_xy_error, 4),
            "grasp_xy_tolerance_m": round(grasp_xy_tolerance, 4),
            "grasp_vertical_speed_m_s": round(abs(grasp_vertical_velocity), 4),
            "grasp_vertical_velocity_m_s": round(grasp_vertical_velocity, 4),
            "motion_limit_reason": self.motion_limit_reason,
            "gripper_approaching_cube": self.gripper_approaching_cube(),
            "cube_z_m": round(float(self.data.xpos[self.cube_id][2]), 4),
            "hand_z_m": round(float(self.data.xpos[self.hand_id][2]), 4),
            "locked_z_m": None if self.locked_z is None else round(self.locked_z, 4),
            "transport_cube_z_m": round(self._transport_cube_height(), 4),
            "push_t_coverage": (
                round(self.current_push_t_coverage, 4)
                if self.args.task_mode == "push_t"
                else None
            ),
        }

    def get_side_preview(self) -> bytes | None:
        self.last_aux_preview_request_at = time.perf_counter()
        return self.side_preview_jpeg

    def get_top_preview(self) -> bytes | None:
        self.last_aux_preview_request_at = time.perf_counter()
        return self.top_preview_jpeg

    def get_front_preview(self) -> bytes | None:
        self.last_aux_preview_request_at = time.perf_counter()
        return self.front_preview_jpeg

    def aux_preview_is_requested(self) -> bool:
        """Avoid rendering operator views while no browser is watching them."""
        return time.perf_counter() - self.last_aux_preview_request_at <= 1.0


    def update_target_state(self) -> None:
        if self.args.task_mode == "push_t":
            self.update_push_t_target_state()
            return
        self.update_gripper_approach_view()
        distance = np.linalg.norm(
            self.data.xpos[self.cube_id][:2] - self.data.xpos[self.target_plate_id][:2]
        )
        holding = self.is_holding_cube()
        if distance > self.args.target_radius + self.args.target_exit_margin:
            self.target_view_reached_with_cube = False
        if distance <= self.args.target_radius and holding:
            if getattr(self, "carried_target_since", None) is None:
                self.carried_target_since = float(self.data.time)
            if float(self.data.time) - self.carried_target_since >= self.args.target_dwell_s:
                self.target_view_reached_with_cube = True
                self.transport_view_active = False
                if self.z_locked:
                    self.set_z_lock(False, reason="target")
        else:
            self.carried_target_since = None
        if not self.cube_over_target:
            if distance <= self.args.target_radius:
                if self.target_entered_sim_time is None:
                    self.target_entered_sim_time = float(self.data.time)
                stable_time = float(self.data.time) - self.target_entered_sim_time
                if stable_time >= self.args.target_dwell_s:
                    self.cube_over_target = True
            else:
                self.target_entered_sim_time = None
            return

        exit_radius = self.args.target_radius + self.args.target_exit_margin
        if distance > exit_radius:
            self.cube_over_target = False
            self.target_entered_sim_time = None
            if self.args.task_mode == "push_t":
                return
            cube_high_enough = (
                float(self.data.xpos[self.cube_id][2])
                >= self.args.transport_cube_z - self.args.relock_height_margin
            )
            if self.is_holding_cube() and cube_high_enough and not self.z_locked:
                self.set_z_lock(True, reason="left_target")
                self.z_lock_used_for_current_grasp = True

    def update_gripper_approach_view(self) -> None:
        cube_pos = self.data.xpos[self.cube_id]
        hand_pos = self.data.xpos[self.hand_id]
        if self.is_holding_cube():
            self.gripper_above_cube_view = False
            self.gripper_above_cube_since = None
            return
        centered = gripper_centered_over_cube(
            hand_pos,
            cube_pos,
            float(self.model.geom_size[self.cube_geom_id][2]),
            was_above=self.gripper_above_cube_view,
        )
        if not centered:
            self.gripper_above_cube_view = False
            self.gripper_above_cube_since = None
            return
        if self.gripper_above_cube_since is None:
            self.gripper_above_cube_since = float(self.data.time)
        if float(self.data.time) - self.gripper_above_cube_since >= GRIPPER_CUBE_VIEW_DWELL_S:
            self.gripper_above_cube_view = True

    def update_push_t_target_state(self) -> None:
        """Track Push-T success using 2D object/goal footprint coverage."""
        if self.push_t_samples is None:
            return
        object_rotation = self.data.xmat[self.cube_id].reshape(3, 3)[:2, :2]
        target_rotation = self.data.xmat[self.target_plate_id].reshape(3, 3)[:2, :2]
        coverage = push_t_coverage(
            self.data.xpos[self.cube_id][:2],
            object_rotation,
            self.data.xpos[self.target_plate_id][:2],
            target_rotation,
            self.push_t_samples,
        )
        self.current_push_t_coverage = coverage
        if not self.cube_over_target:
            if coverage >= self.args.push_t_success_coverage:
                if self.target_entered_sim_time is None:
                    self.target_entered_sim_time = float(self.data.time)
                stable_time = float(self.data.time) - self.target_entered_sim_time
                if stable_time >= self.args.target_dwell_s:
                    self.cube_over_target = True
            else:
                self.target_entered_sim_time = None
            return

        if coverage < self.args.push_t_exit_coverage:
            self.cube_over_target = False
            self.target_entered_sim_time = None

    def reset_hand_motion(self) -> None:
        """Do not carry wrist displacement across a pause or re-entry."""
        self.hand_motion_token = None
        self.hand_motion_segment = -1
        self.hand_motion_anchor = None
        self.hand_motion_last_sent_at = float("-inf")
        self.hand_motion_pending = np.zeros(2, dtype=np.float64)
        self.hand_motion_pending_samples = 0
        self.filtered_wrist_uv = np.zeros(2, dtype=np.float64)
        self.filtered_wrist_timestamp_ms = -1.0
        self.filtered_wrist_valid = False

    def update_raw_wrist(self, payload: dict) -> None:
        """Keep the original MediaPipe wrist sample and capture time intact."""
        point = payload.get("wristUv")
        try:
            timestamp = float(payload.get("wristCapturedAt", -1.0))
            wrist = np.asarray(point, dtype=np.float64)
        except (TypeError, ValueError):
            wrist = np.empty(0)
            timestamp = -1.0
        self.raw_wrist_valid = bool(
            payload.get("handVisible") and wrist.shape == (2,)
            and np.all(np.isfinite(wrist)) and np.isfinite(timestamp) and timestamp > 0
        )
        if self.raw_wrist_valid:
            self.raw_wrist_uv = wrist.copy()
            self.raw_wrist_timestamp_ms = timestamp
        else:
            self.raw_wrist_uv = np.zeros(2, dtype=np.float64)
            self.raw_wrist_timestamp_ms = -1.0

    def apply_web_motion(self, dx: float, dy: float, dz: float) -> None:
        """Record a control request accepted by the backend in robot-base metres."""
        self.move_hand(dx, dy, dz)
        if not hasattr(self, "mapped_control_pending"):
            self.mapped_control_pending = np.zeros(3, dtype=np.float64)
            self.mapped_control_pending_commands = 0
        self.mapped_control_pending += np.array([dx, dy, dz], dtype=np.float64)
        self.mapped_control_pending_commands += 1

    def update_hand_motion(self, payload: dict) -> None:
        """Build a continuous filtered wrist path within each armed segment."""
        wrist = payload.get("wristUv")
        token = payload.get("handMotionToken")
        if (not payload.get("handMotionReady") or not self.hand_visible
                or self.control_paused or not isinstance(token, str)
                or token != self.control_token or not isinstance(wrist, list)
                or len(wrist) != 2):
            self.reset_hand_motion()
            return
        try:
            point = np.asarray(wrist, dtype=np.float64)
            segment = int(payload.get("handMotionSegment"))
            sent_at = float(payload.get("sentAt"))
        except (TypeError, ValueError):
            self.reset_hand_motion()
            return
        if (not np.all(np.isfinite(point)) or not np.isfinite(sent_at)
                or segment < 0 or np.any(point < -0.05) or np.any(point > 1.05)):
            self.reset_hand_motion()
            return
        if (self.hand_motion_token != token or self.hand_motion_segment != segment
                or sent_at - self.hand_motion_last_sent_at > 150.0):
            self.reset_hand_motion()
            self.hand_motion_token = token
            self.hand_motion_segment = segment
            self.hand_motion_anchor = point
            self.hand_motion_last_sent_at = sent_at
            self.filtered_wrist_uv = point.copy()
            self.filtered_wrist_timestamp_ms = getattr(self, "raw_wrist_timestamp_ms", sent_at)
            self.filtered_wrist_valid = True
            return
        if sent_at <= self.hand_motion_last_sent_at:
            return
        self.hand_motion_last_sent_at = sent_at
        self.filtered_wrist_timestamp_ms = getattr(self, "raw_wrist_timestamp_ms", sent_at)
        delta = point - self.hand_motion_anchor
        distance = float(np.linalg.norm(delta))
        if distance > 0.10:
            # A tracking jump starts a new local reference, never a label.
            self.hand_motion_anchor = point
            self.hand_motion_pending.fill(0.0)
            self.hand_motion_pending_samples = 0
            return
        if distance >= 0.0015:
            self.hand_motion_pending += delta
            self.filtered_wrist_uv += delta
            self.hand_motion_anchor = point
        self.hand_motion_pending_samples += 1

    def append_hand_trajectory_sample(self, payload: dict) -> None:
        """Preserve every received sample, including gaps, at camera report rate."""
        if not getattr(self, "recording", False):
            return
        self.hand_trajectory_samples.append({
            "sample_index": len(self.hand_trajectory_samples),
            "frame_index_at_receive": self.episode_frames,
            "raw_wrist_uv": self.raw_wrist_uv.tolist() if self.raw_wrist_valid else None,
            "raw_timestamp_ms": self.raw_wrist_timestamp_ms,
            "filtered_wrist_uv": (self.filtered_wrist_uv.tolist()
                                  if self.filtered_wrist_valid else None),
            "filtered_timestamp_ms": self.filtered_wrist_timestamp_ms,
            "hand_motion_segment": self.hand_motion_segment if self.filtered_wrist_valid else -1,
            "sent_at_ms": float(payload.get("sentAt", -1.0)),
        })

    def handle_gesture(self, payload: dict) -> None:
        command = payload.get("command")
        if command == "pause":
            self.control_paused = True
            self.control_token = None
            self.pointing_arm_qpos = None
            self.pending_vertical_steps = 0
            self.reset_hand_motion()
            self.mapped_control_pending.fill(0.0)
            self.mapped_control_pending_commands = 0
            return
        if command == "resume":
            token = payload.get("controlToken")
            if (isinstance(token, str) and token
                    and payload.get("controlContext") == f"{self.session}:{self.control_epoch}"):
                self.control_token = token
                self.control_paused = False
                self.last_resume_command_at = float(payload.get("sentAt", 0.0))
                self.last_gesture_update_at = time.perf_counter()
            return
        if command == "reset":
            if not self.recording and not self.saving_episode:
                self.reset_simulation()
            return
        if command in WEB_CONTROL_COMMANDS and (
            self.control_paused or payload.get("controlToken") != self.control_token
        ):
            return
        if command == "gesture_state":
            sent_at = float(payload.get("sentAt", 0.0))
            if sent_at < self.last_gesture_command_at:
                return
            self.last_gesture_command_at = sent_at
            gesture = payload.get("gesture", "None")
            self.latest_gesture = gesture if gesture in GESTURE_CODES else "None"
            self.latest_gesture_score = float(np.clip(payload.get("gestureScore", 0.0), 0, 1))
            self.hand_visible = bool(payload.get("handVisible", False))
            self.update_raw_wrist(payload)
            self.update_hand_motion(payload)
            self.append_hand_trajectory_sample(payload)
            pointing_control = payload.get(
                "pointingControlActive", self.latest_gesture == "Pointing_Up"
            ) is True
            if (pointing_control and self.hand_visible and not self.gripper_closed
                    and not self.gripper_orientation_finalized
                    and (self.latest_gesture == "Pointing_Up" or self.gripper_orientation_locked)
                    and self.pointing_arm_qpos is None):
                # Stop any pending Cartesian trajectory. Pointing_Up controls
                # only the gripper's dedicated yaw actuator. Raw None may
                # resume an existing adjustment after hand re-entry, but may
                # not start an adjustment from the default pose.
                self.data.ctrl[:7] = self.data.qpos[:7]
                self.pointing_arm_qpos = self.data.qpos[:7].copy()
            elif self.gripper_closed or self.gripper_orientation_finalized or not self.hand_visible:
                self.pointing_arm_qpos = None
            self.last_gesture_update_at = time.perf_counter()
            if not self.hand_visible and sent_at >= self.last_resume_command_at:
                self.control_paused = True
                self.control_token = None
                self.reset_hand_motion()
            return
        if command == "record_start":
            self.start_episode()
            return
        if command == "record_save":
            self.save_episode()
            return
        if command in {"record_failure", "record_discard"}:
            self.save_failed_episode()
            return
        if command == "record_stop":
            self.stop_requested = True
            return

        if self.args.task_mode == "push_t":
            self.handle_pushing_gesture(payload)
            return

        if self.pointing_arm_qpos is not None and command in {
            "left", "right", "forward", "backward", "up", "down",
            "move_xy", "start_vertical", "start_planar", "open", "close",
        }:
            return

        if command in {
            "left", "right", "forward", "backward", "up", "down",
            "move_xy", "start_vertical", "start_planar", "open", "close",
        } and not self.gripper_orientation_finalized:
            # Pointing_Up is an initial, one-time adjustment. The first arm or
            # gripper command freezes even the default zero-yaw orientation.
            self.gripper_orientation_finalized = True
            self.gripper_orientation_locked = True
            self.hard_lock_gripper_yaw()

        if command in {"set_gripper_yaw", "finalize_gripper_yaw"}:
            if self.gripper_closed or self.gripper_orientation_finalized:
                return
            if command == "set_gripper_yaw" and self.pointing_arm_qpos is None:
                return
            angle = float(payload.get("angle", self.gripper_yaw))
            if np.isfinite(angle):
                self.gripper_yaw = float(np.clip(angle, -np.pi / 2, np.pi / 2))
                c, s = np.cos(self.gripper_yaw), np.sin(self.gripper_yaw)
                local_yaw_rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
                self.target_hand_rotation_in_base = self.initial_hand_rotation_in_base @ local_yaw_rotation
                self.data.ctrl[self.gripper_yaw_actuator_id] = self.gripper_yaw
                self.gripper_orientation_locked = True
                self.hard_lock_gripper_yaw()
                if command == "finalize_gripper_yaw":
                    self.gripper_orientation_finalized = True
                    self.pointing_arm_qpos = None
            return

        if command in {"close", "open"}:
            sent_at = float(payload.get("sentAt", 0.0))
            if sent_at < self.last_gripper_command_at:
                return
            self.last_gripper_command_at = sent_at

        if command == "close":
            if self.gripper_closed:
                return
            if not self.gripper_ready_to_grasp():
                return
            self.z_lock_used_for_current_grasp = False
            self.pending_vertical_steps = 0
            self.grasp_support_z = self._cube_support_height(
                self.data.xpos[self.cube_id][:2]
            )
            self.last_bilateral_cube_contact_at = float("-inf")
            self.data.ctrl[7] = 0.0
            self.gripper_closed = True
            self.grasp_close_started_at = float(self.data.time)
        elif command == "open":
            self.pending_vertical_steps = 0
            self.last_deferred_vertical_at = float("-inf")
            self.data.ctrl[7] = 255.0
            self.gripper_closed = False
            self.grasp_close_started_at = float("-inf")
            self.last_bilateral_cube_contact_at = float("-inf")
            self.transport_view_active = False
            self.set_z_lock(False, reason="open")
        elif command == "left":
            self.apply_web_motion(0.0, self.args.step, 0.0)
        elif command == "right":
            self.apply_web_motion(0.0, -self.args.step, 0.0)
        elif command == "forward":
            self.apply_web_motion(self.args.step, 0.0, 0.0)
        elif command == "backward":
            self.apply_web_motion(-self.args.step, 0.0, 0.0)
        elif command == "up":
            if self.z_locked:
                self.set_z_lock(False, reason="vertical")
            self.apply_web_motion(0.0, 0.0, self.args.vertical_step)
        elif command == "down":
            if self.z_locked:
                self.set_z_lock(False, reason="vertical")
            self.apply_web_motion(0.0, 0.0, -self.args.vertical_step)
        elif command == "start_vertical":
            self.action_data.qpos[:] = self.data.qpos
            self.action_data.qpos[:7] = self.data.ctrl[:7]
            self.action_data.qpos[self.gripper_yaw_qpos_address] = self.data.ctrl[self.gripper_yaw_actuator_id]
            self.action_data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.action_data)
            self.vertical_anchor_xy = self.action_data.xpos[self.hand_id][:2].copy()
            if self.z_locked:
                self.set_z_lock(False, reason="vertical")
        elif command == "start_planar":
            self.vertical_anchor_xy = None
            # An explicit planar gesture owns the current height even before
            # the automatic transport-height threshold has been reached.
            if self.is_holding_cube() and not self.cube_over_target:
                self.set_z_lock(True, reason="planar")
                self.z_lock_used_for_current_grasp = True
        elif command == "move_xy":
            self.vertical_anchor_xy = None
            # Contact can become bilateral after start_planar was received.
            # Acquire the plane here too rather than silently dropping motion.
            if self.is_holding_cube() and not self.cube_over_target:
                self.set_z_lock(True, reason="planar")
                self.z_lock_used_for_current_grasp = True
            dx = np.clip(float(payload.get("dx", 0.0)), -1.0, 1.0)
            dy = np.clip(float(payload.get("dy", 0.0)), -1.0, 1.0)
            self.apply_web_motion(
                dx * self.args.max_planar_step,
                dy * self.args.max_planar_step,
                0.0,
            )

    def handle_pushing_gesture(self, payload: dict) -> None:
        """Apply pushing-only controls without changing pick-and-place rules."""
        command = payload.get("command")
        # Push-T uses one fixed, closed pushing surface for the whole episode.
        if command in {"open", "close"}:
            return
        if command == "start_vertical":
            return
        if command in {"up", "down"}:
            return
        if command == "start_planar":
            self.vertical_anchor_xy = None
            if not self.z_locked:
                self.set_z_lock(True, reason="pushing")
            return
        if command == "move_xy":
            self.vertical_anchor_xy = None
            if not self.z_locked:
                self.set_z_lock(True, reason="pushing")
            dx = np.clip(float(payload.get("dx", 0.0)), -1.0, 1.0)
            dy = np.clip(float(payload.get("dy", 0.0)), -1.0, 1.0)
            self.apply_web_motion(
                dx * self.args.max_planar_step,
                dy * self.args.max_planar_step,
                0.0,
            )
            return
        # Keep the keyboard-style single-axis commands usable from the web API.
        planar_deltas = {
            "left": (0.0, self.args.step, 0.0),
            "right": (0.0, -self.args.step, 0.0),
            "forward": (self.args.step, 0.0, 0.0),
            "backward": (-self.args.step, 0.0, 0.0),
        }
        if command in planar_deltas:
            if not self.z_locked:
                self.set_z_lock(True, reason="pushing")
            self.apply_web_motion(*planar_deltas[command])

    def handle_key(self, keycode: int) -> None:
        if self.args.task_mode == "push_t":
            if keycode in {ord("S"), ord("s")}:
                self.start_episode()
            elif keycode in {ord("N"), ord("n")}:
                self.save_episode()
            elif keycode in {ord("R"), ord("r")}:
                self.save_failed_episode()
            elif keycode in {ord("Q"), ord("q")}:
                self.stop_requested = True
            elif keycode in {265, 264, 263, 262}:
                if not self.z_locked:
                    self.set_z_lock(True, reason="pushing")
                deltas = {
                    265: (self.args.step, 0.0, 0.0),
                    264: (-self.args.step, 0.0, 0.0),
                    263: (0.0, self.args.step, 0.0),
                    262: (0.0, -self.args.step, 0.0),
                }
                self.move_hand(*deltas[keycode])
            return
        if keycode == 265:
            self.move_hand(self.args.step, 0.0, 0.0)
        elif keycode == 264:
            self.move_hand(-self.args.step, 0.0, 0.0)
        elif keycode == 263:
            self.move_hand(0.0, self.args.step, 0.0)
        elif keycode == 262:
            self.move_hand(0.0, -self.args.step, 0.0)
        elif keycode in {ord("I"), ord("i")}:
            self.move_hand(0.0, 0.0, self.args.vertical_step)
        elif keycode in {ord("K"), ord("k")}:
            self.move_hand(0.0, 0.0, -self.args.vertical_step)
        elif keycode in {ord("J"), ord("j")}:
            self.data.ctrl[7] = 255.0
            self.gripper_closed = False
            self.last_bilateral_cube_contact_at = float("-inf")
        elif keycode in {ord("L"), ord("l")}:
            self.data.ctrl[7] = 0.0
            self.gripper_closed = True
            self.last_bilateral_cube_contact_at = float("-inf")
        elif keycode in {ord("S"), ord("s")}:
            self.start_episode()
        elif keycode in {ord("N"), ord("n")}:
            self.save_episode()
        elif keycode in {ord("R"), ord("r")}:
            self.save_failed_episode()
        elif keycode in {ord("Q"), ord("q")}:
            self.stop_requested = True

    def process_queued_commands(self) -> None:
        """Process controls without replaying a stale backlog of motion commands."""
        latest_motion: dict | None = None
        motion_commands = {
            "left",
            "right",
            "forward",
            "backward",
            "up",
            "down",
            "move_xy",
            "set_gripper_yaw",
        }
        while not self.command_queue.empty():
            payload = self.command_queue.get_nowait()
            if payload.get("command") in motion_commands:
                try:
                    sent_at = float(payload.get("sentAt", time.time() * 1000.0))
                except (TypeError, ValueError):
                    continue
                if (
                    sent_at > self.last_motion_command_at
                    and (
                        latest_motion is None
                        or sent_at > float(latest_motion["sentAt"])
                    )
                ):
                    latest_motion = {**payload, "sentAt": sent_at}
            else:
                if (payload.get("command") == "close" and latest_motion is not None
                        and latest_motion["command"] != "down"):
                    # A fist explicitly requests descent and closure together.
                    # Preserve that descent; other earlier motion keeps its
                    # existing cancellation rule on the closing transition.
                    if float(latest_motion["sentAt"]) <= float(payload.get("sentAt", 0.0)):
                        latest_motion = None
                self.handle_gesture(payload)
        if latest_motion is not None:
            sent_at = float(latest_motion["sentAt"])
            command_age_ms = time.time() * 1000.0 - sent_at
            if command_age_ms <= self.args.max_motion_command_age_ms:
                self.last_motion_command_at = sent_at
                self.handle_gesture(latest_motion)

    def start_episode(self) -> None:
        if self.recording:
            print("[LeRobot] 当前 demonstration 已经在录制。")
            return
        self.training_queue.put(
            {
                "type": "start",
                "attempt_index": self.attempted_episodes,
                "cube_xy_m": self.current_cube_xy.tolist(),
                "target_plate_xy_m": self.current_target_xy.tolist(),
                "cube_yaw_rad": self.current_cube_yaw,
                "target_yaw_rad": self.current_target_yaw,
                "layout": self.current_layout,
                "layout_sides": self.args.layout_sides,
                "layout_reachability": getattr(self, "layout_reachability", None),
            }
        )
        self._wait_training_reply("started")
        self.reset_hand_motion()
        self.hand_trajectory_samples = []
        self.mapped_control_pending.fill(0.0)
        self.mapped_control_pending_commands = 0
        self.recording = True
        self.episode_frames = 0
        self.attempted_episodes += 1
        print(f"[LeRobot] 开始录制 demonstration 尝试 {self.attempted_episodes}。")
        print(
            "[随机初始位置] "
            f"方块=({self.current_cube_xy[0]:.3f}, {self.current_cube_xy[1]:.3f}) m，"
            f"目标=({self.current_target_xy[0]:.3f}, {self.current_target_xy[1]:.3f}) m，"
            f"方块 yaw={np.rad2deg(self.current_cube_yaw):.1f}°。"
        )

    def save_episode(self) -> None:
        if not self.recording or self.episode_frames == 0:
            print("[LeRobot] 没有正在录制的数据，先按 S 开始。")
            return
        episode_frames = self.episode_frames
        self.saving_episode = True
        try:
            self.training_queue.put(
                {
                    "type": "save",
                    "episode_index": self.saved_episodes,
                    "frames": episode_frames,
                    "hand_trajectory_samples": self.hand_trajectory_samples,
                }
            )
            result = self._wait_training_reply("saved")
        except Exception as error:
            self.saving_episode = False
            print(f"[LeRobot] 保存失败，场景未重置以保护当前数据：{error}")
            return
        if result.get("inspection_error"):
            print(
                "[检查视频] episode 核心数据已保存，但独立检查视频收尾失败："
                f"{result['inspection_error']}"
            )
        self.saved_episodes += 1
        print(
            f"[LeRobot] 已保存 demonstration {self.saved_episodes}，"
            f"共 {episode_frames} 帧。"
        )
        self.recording = False
        self.saving_episode = False
        self.episode_frames = 0
        self.reset_simulation()

    def save_failed_episode(self) -> None:
        if not self.recording or self.episode_frames == 0:
            print("[LeRobot] 没有正在录制的数据，先按 S 开始。")
            return
        episode_frames = self.episode_frames
        self.saving_episode = True
        try:
            self.training_queue.put(
                {
                    "type": "save_failure",
                    "episode_index": self.failed_episodes,
                    "frames": episode_frames,
                    "hand_trajectory_samples": self.hand_trajectory_samples,
                }
            )
            result = self._wait_training_reply("failure_saved")
        except Exception as error:
            self.saving_episode = False
            print(f"[LeRobot] 失败轨迹保存失败，场景未重置：{error}")
            return
        if result.get("inspection_error"):
            print(
                "[检查视频] 失败 episode 核心数据已保存，但独立检查视频收尾失败："
                f"{result['inspection_error']}"
            )
        self.failed_episodes += 1
        print(
            f"[LeRobot] 已将失败 demonstration {self.failed_episodes} 保存到失败数据集，"
            f"共 {episode_frames} 帧。"
        )
        self.recording = False
        self.saving_episode = False
        self.episode_frames = 0
        self.reset_simulation()

    def record_frame(self) -> None:
        gesture_fresh = time.perf_counter() - self.last_gesture_update_at < 0.5
        gesture = self.latest_gesture if gesture_fresh else "None"
        raw_wrist_valid = bool(self.raw_wrist_valid and gesture_fresh)
        raw_wrist_uv = (self.raw_wrist_uv.astype(np.float32).copy() if raw_wrist_valid
                        else np.zeros(2, dtype=np.float32))
        raw_wrist_timestamp_ms = self.raw_wrist_timestamp_ms if raw_wrist_valid else -1.0
        filtered_wrist_valid = bool(self.filtered_wrist_valid and gesture_fresh)
        filtered_wrist_uv = (self.filtered_wrist_uv.astype(np.float32).copy()
                             if filtered_wrist_valid else np.zeros(2, dtype=np.float32))
        filtered_wrist_timestamp_ms = (self.filtered_wrist_timestamp_ms
                                       if filtered_wrist_valid else -1.0)
        hand_delta_valid = bool(self.hand_motion_pending_samples and not self.control_paused
                                and time.time() * 1000.0 - self.hand_motion_last_sent_at <= 150.0)
        hand_delta_uv = self.hand_motion_pending.astype(np.float32).copy() if hand_delta_valid else np.zeros(2, dtype=np.float32)
        hand_motion_segment = self.hand_motion_segment if hand_delta_valid else -1
        self.hand_motion_pending.fill(0.0)
        self.hand_motion_pending_samples = 0
        mapped_control_valid = bool(self.mapped_control_pending_commands)
        mapped_control_delta = self.mapped_control_pending.astype(np.float32).copy()
        mapped_control_commands = self.mapped_control_pending_commands
        self.mapped_control_pending.fill(0.0)
        self.mapped_control_pending_commands = 0
        finger_width = float(
            sum(self.data.qpos[address] for address in self.finger_qpos_addresses)
        )
        state = np.concatenate(
            [self.data.qpos[:7], np.array([self.data.qpos[self.gripper_yaw_qpos_address], finger_width])]
        ).astype(np.float32)
        # Convert the commanded seven-joint target to a full Cartesian pose in
        # the robot-base frame: XYZ (metres), RPY (radians), and gripper target.
        self.action_data.qpos[:] = self.data.qpos
        self.action_data.qpos[:7] = self.data.ctrl[:7]
        self.action_data.qpos[self.gripper_yaw_qpos_address] = self.data.ctrl[self.gripper_yaw_actuator_id]
        mujoco.mj_forward(self.model, self.action_data)
        base_position = self.action_data.xpos[self.base_id]
        base_rotation = self.action_data.xmat[self.base_id].reshape(3, 3)
        target_in_base = base_rotation.T @ (
            self.action_data.xpos[self.hand_id] - base_position
        )
        hand_rotation = self.action_data.xmat[self.hand_id].reshape(3, 3)
        target_rotation_in_base = base_rotation.T @ hand_rotation
        target_rpy = rotation_matrix_to_rpy(target_rotation_in_base)
        target_rpy += 2.0 * np.pi * np.round(
            (self.action_rpy_reference - target_rpy) / (2.0 * np.pi)
        )
        action = np.concatenate(
            [target_in_base, target_rpy, np.array([self.data.ctrl[7]])]
        ).astype(np.float32)
        self.training_queue.put(
            {
                "type": "frame",
                "snapshot": self._render_snapshot(),
                "state": state,
                "action": action,
                "gesture": np.array([GESTURE_CODES[gesture]], dtype=np.int64),
                "gesture_score": np.array([self.latest_gesture_score if gesture_fresh else 0.0], dtype=np.float32),
                "hand_visible": np.array([int(self.hand_visible and gesture_fresh)], dtype=np.int64),
                "hand_delta_uv": hand_delta_uv,
                "hand_delta_valid": np.array([int(hand_delta_valid)], dtype=np.int64),
                "hand_motion_segment": np.array([hand_motion_segment], dtype=np.int64),
                "raw_wrist_uv": raw_wrist_uv,
                "raw_wrist_timestamp_ms": np.array([raw_wrist_timestamp_ms], dtype=np.float64),
                "raw_wrist_valid": np.array([int(raw_wrist_valid)], dtype=np.int64),
                "filtered_wrist_uv": filtered_wrist_uv,
                "filtered_wrist_timestamp_ms": np.array([filtered_wrist_timestamp_ms], dtype=np.float64),
                "filtered_wrist_valid": np.array([int(filtered_wrist_valid)], dtype=np.int64),
                "mapped_control_delta_xyz_m": mapped_control_delta,
                "mapped_control_valid": np.array([int(mapped_control_valid)], dtype=np.int64),
                "mapped_control_commands": np.array([mapped_control_commands], dtype=np.int64),
            }
        )
        self.episode_frames += 1

    def run_headless_test(self) -> None:
        steps_per_frame = max(1, round(1.0 / self.args.fps / self.model.opt.timestep))
        for _ in range(self.args.test_episodes):
            self.start_episode()
            for frame_index in range(self.args.test_frames):
                if self.args.test_motion:
                    quarter = max(1, self.args.test_frames // 4)
                    phase = min(3, frame_index // quarter)
                    test_deltas = (
                        (0.002, 0.0, 0.0),
                        (0.0, 0.002, 0.0),
                        (-0.002, 0.0, 0.0),
                        (0.0, -0.002, 0.0),
                    )
                    self.move_hand(*test_deltas[phase])
                for _ in range(steps_per_frame):
                    mujoco.mj_step(self.model, self.data)
                self.update_target_state()
                self.update_transport_state()
                self.record_frame()
            if self.args.test_failure:
                self.save_failed_episode()
            else:
                self.save_episode()

    def run_interactive(self) -> None:
        self.server = GestureServer(
            self.command_queue,
            self.args.port,
            self.get_top_preview,
            self.get_side_preview,
            self.get_front_preview,
            self.get_runtime_status,
        )
        self.server.start()
        print(f"[手势控制] 已监听 http://127.0.0.1:{self.args.port}/control")
        print("\n========== MuJoCo + LeRobot 录制 ==========")
        print("S：开始录制   N：成功并保存")
        print("R：失败并保存 Q：结束并写入数据集")
        print("方向键/I/K/J/L 仍可作为备用控制")
        print(f"数据集目录：{self.args.dataset_root}")
        print(f"失败数据集：{self.args.failure_dataset_root}")
        print("===========================================\n")

        ui_hz = 60
        substeps = max(1, round(1.0 / ui_hz / self.model.opt.timestep))
        record_interval = 1.0 / self.args.fps
        next_record_time = time.perf_counter()
        # Alternate three auxiliary cameras without blocking a simulation tick
        # on multiple renders.
        side_preview_interval = 1.0 / (3.0 * self.args.side_preview_fps)
        next_side_preview_time = time.perf_counter()

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.handle_key,
        ) as viewer:
            self.update_viewer_camera(viewer)
            while viewer.is_running() and not self.stop_requested:
                tick_started = time.perf_counter()
                self.process_queued_commands()
                self.maintain_locked_plane()
                self.maintain_gripper_orientation()
                for _ in range(substeps):
                    mujoco.mj_step(self.model, self.data)
                self.hold_pointing_arm()
                self.hard_lock_gripper_yaw()
                self.enforce_layout_support()
                self.update_target_state()
                self.update_transport_state()
                self.update_viewer_camera(viewer)
                self._drain_auxiliary_results()

                now = time.perf_counter()
                if now >= next_side_preview_time and self.aux_preview_is_requested():
                    self._submit_auxiliary_snapshot()
                    next_side_preview_time = now + side_preview_interval
                elif now >= next_side_preview_time:
                    next_side_preview_time = now + side_preview_interval

                if self.recording and now >= next_record_time:
                    self.record_frame()
                    next_record_time = now + record_interval
                elif not self.recording:
                    next_record_time = now

                viewer.sync()
                remaining = 1.0 / ui_hz - (time.perf_counter() - tick_started)
                if remaining > 0:
                    time.sleep(remaining)

    def close(self) -> None:
        if self.recording:
            self.training_queue.put({"type": "discard"})
            self._wait_training_reply("discarded")
            print("[LeRobot] 未完成的 demonstration 已丢弃。")
            self.recording = False
        self.training_queue.put({"type": "close"})
        self._wait_training_reply("closed")
        self.training_process.join(timeout=10)
        self.aux_queue.put(None)
        self.aux_process.join(timeout=10)
        if self.server is not None:
            self.server.stop()
        print(f"[LeRobot] 录制结束，共保存 {self.saved_episodes} 个 demonstrations。")
        print(f"[LeRobot] 失败数据集共保存 {self.failed_episodes} 个 demonstrations。")
        print(f"[LeRobot] 数据集：{self.args.dataset_root}")
        print(f"[LeRobot] 失败数据集：{self.args.failure_dataset_root}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Record MuJoCo Panda demonstrations in LeRobot format."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--task-mode",
        choices=("pick_place", "push_t"),
        default="pick_place",
        help="Select task-specific control rules without changing the other task.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "datasets" / f"mujoco_panda_pick_{timestamp}",
    )
    parser.add_argument(
        "--failure-dataset-root",
        type=Path,
        default=REPO_ROOT / "datasets" / f"mujoco_panda_pick_failures_{timestamp}",
    )
    parser.add_argument("--repo-id", default="local/mujoco_panda_pick")
    parser.add_argument(
        "--task",
        default="Pick up the red cube and place it on the white target plate",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--step", type=float, default=0.015)
    parser.add_argument("--vertical-step", type=float, default=0.008)
    parser.add_argument("--max-planar-step", type=float, default=0.015)
    parser.add_argument("--max-joint-step", type=float, default=0.08)
    parser.add_argument("--max-joint-target-lead", type=float, default=0.12)
    parser.add_argument("--max-motion-command-age-ms", type=float, default=150.0)
    parser.add_argument("--target-radius", type=float, default=0.05)
    parser.add_argument("--target-dwell-s", type=float, default=0.15)
    parser.add_argument("--target-exit-margin", type=float, default=0.02)
    parser.add_argument("--relock-height-margin", type=float, default=0.02)
    parser.add_argument("--transport-cube-z", type=float, default=0.50)
    parser.add_argument("--grasped-min-width", type=float, default=0.02)
    parser.add_argument("--grasped-max-width", type=float, default=0.11)
    parser.add_argument("--grasped-max-distance", type=float, default=0.18)
    parser.add_argument("--observation-lookat-x", type=float, default=0.48)
    parser.add_argument("--observation-lookat-y", type=float, default=0.0)
    parser.add_argument("--observation-lookat-z", type=float, default=0.48)
    parser.add_argument("--observation-distance", type=float, default=1.45)
    parser.add_argument("--observation-azimuth", type=float, default=145.0)
    parser.add_argument("--observation-camera-2-azimuth", type=float, default=25.0)
    parser.add_argument("--observation-camera-3-azimuth", type=float, default=265.0)
    parser.add_argument("--observation-elevation", type=float, default=-15.0)
    parser.add_argument("--topdown-lookat-x", type=float, default=0.5)
    parser.add_argument("--topdown-lookat-y", type=float, default=0.0)
    parser.add_argument("--topdown-lookat-z", type=float, default=0.4)
    parser.add_argument("--topdown-distance", type=float, default=1.2)
    # Azimuth 0 displays the previous top-down image rotated 90 degrees
    # counterclockwise.  This changes only the operator view, not robot axes.
    parser.add_argument("--topdown-azimuth", type=float, default=0.0)
    parser.add_argument("--side-lookat-x", type=float, default=0.5)
    parser.add_argument("--side-lookat-y", type=float, default=0.0)
    parser.add_argument("--side-lookat-z", type=float, default=0.5)
    parser.add_argument("--side-distance", type=float, default=0.78)
    parser.add_argument("--side-azimuth", type=float, default=90.0)
    parser.add_argument("--side-elevation", type=float, default=-12.0)
    parser.add_argument("--side-preview-fps", type=float, default=8.0)
    parser.add_argument("--aux-width", type=int, default=320)
    parser.add_argument("--aux-height", type=int, default=240)
    parser.add_argument("--front-lookat-x", type=float, default=0.50)
    parser.add_argument("--front-lookat-y", type=float, default=0.0)
    parser.add_argument("--front-lookat-z", type=float, default=0.48)
    parser.add_argument("--front-distance", type=float, default=0.78)
    parser.add_argument("--front-azimuth", type=float, default=180.0)
    parser.add_argument("--front-elevation", type=float, default=-12.0)
    parser.add_argument("--table-top-z", type=float, default=0.4)
    parser.add_argument("--target-top-z", type=float, default=0.416)
    parser.add_argument("--cube-half-size", type=float, default=0.030)
    parser.add_argument("--cube-x-min", type=float, default=0.37)
    parser.add_argument("--cube-x-max", type=float, default=0.68)
    parser.add_argument("--cube-y-min", type=float, default=-0.24)
    parser.add_argument("--cube-y-max", type=float, default=0.24)
    parser.add_argument("--plate-x-min", type=float, default=0.37)
    parser.add_argument("--plate-x-max", type=float, default=0.68)
    parser.add_argument("--plate-y-min", type=float, default=-0.24)
    parser.add_argument("--plate-y-max", type=float, default=0.24)
    parser.add_argument("--layout-sides", choices=("split", "any"), default="split",
                        help="Keep the cube left and target plate right in the overview camera.")
    parser.add_argument("--safe-reach-radius", type=float, default=0.72)
    parser.add_argument("--object-min-separation", type=float, default=0.16)
    parser.add_argument("--stack-probability", type=float, default=DEFAULT_STACK_PROBABILITY,
                        help="Fraction of pick/place resets with the red target above or below another rigid box.")
    parser.add_argument("--cube-yaw-min-deg", type=float, default=0.0)
    parser.add_argument("--cube-yaw-max-deg", type=float, default=0.0)
    parser.add_argument("--push-t-success-coverage", type=float, default=0.90)
    parser.add_argument("--push-t-exit-coverage", type=float, default=0.85)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument(
        "--streaming-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--headless-test", action="store_true")
    parser.add_argument("--test-frames", type=int, default=8)
    parser.add_argument("--test-episodes", type=int, default=1)
    parser.add_argument("--test-motion", action="store_true")
    parser.add_argument("--test-failure", action="store_true")
    args = parser.parse_args(argv)
    args.model_path = args.model_path.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.failure_dataset_root = args.failure_dataset_root.expanduser().resolve()
    if not args.model_path.is_file():
        parser.error(f"MuJoCo scene not found: {args.model_path}")
    if args.dataset_root.exists():
        parser.error(
            f"Dataset directory already exists: {args.dataset_root}. "
            "Choose a new --dataset-root so existing data is never overwritten."
        )
    if args.failure_dataset_root.exists():
        parser.error(
            f"Failure dataset directory already exists: {args.failure_dataset_root}. "
            "Choose a new --failure-dataset-root so existing data is never overwritten."
        )
    if args.failure_dataset_root == args.dataset_root:
        parser.error("Success and failure dataset roots must be different.")
    if args.side_preview_fps <= 0:
        parser.error("--side-preview-fps must be greater than zero.")
    if args.max_joint_step <= 0:
        parser.error("--max-joint-step must be greater than zero.")
    if args.max_joint_target_lead <= 0:
        parser.error("--max-joint-target-lead must be greater than zero.")
    if args.max_motion_command_age_ms <= 0:
        parser.error("--max-motion-command-age-ms must be greater than zero.")
    if args.aux_width <= 0 or args.aux_height <= 0:
        parser.error("--aux-width and --aux-height must be greater than zero.")
    if args.test_frames <= 0 or args.test_episodes <= 0:
        parser.error("--test-frames and --test-episodes must be greater than zero.")
    for name in ("cube", "plate"):
        if getattr(args, f"{name}_x_min") >= getattr(args, f"{name}_x_max"):
            parser.error(f"--{name}-x-min must be less than --{name}-x-max.")
        if getattr(args, f"{name}_y_min") >= getattr(args, f"{name}_y_max"):
            parser.error(f"--{name}-y-min must be less than --{name}-y-max.")
    if args.object_min_separation <= 0:
        parser.error("--object-min-separation must be greater than zero.")
    if not 0 <= args.stack_probability <= 1:
        parser.error("--stack-probability must be between 0 and 1.")
    if args.cube_yaw_min_deg > args.cube_yaw_max_deg:
        parser.error("--cube-yaw-min-deg must not exceed --cube-yaw-max-deg.")
    if not 0 < args.push_t_exit_coverage < args.push_t_success_coverage <= 1:
        parser.error(
            "Push-T coverage must satisfy 0 < exit < success <= 1."
        )
    return args


def main() -> None:
    args = parse_args()
    recorder = MujocoPandaRecorder(args)
    try:
        if args.headless_test:
            recorder.run_headless_test()
        else:
            recorder.run_interactive()
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
