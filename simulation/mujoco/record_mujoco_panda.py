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
        if self.path.startswith(("/side-preview", "/front-preview")):
            return
        super().log_request(code, size)


class GestureServer(threading.Thread):
    """Small local HTTP server that receives commands from the existing UI."""

    def __init__(
        self,
        command_queue: queue.Queue[dict],
        port: int,
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
        self.recording = False
        self.saving_episode = False
        self.episode_frames = 0
        self.saved_episodes = 0
        self.rng = np.random.default_rng(args.random_seed)

        self.base_id = self._body_id("link0")
        self.hand_id = self._body_id("hand")
        self.cube_id = self._body_id("cube")
        self.target_plate_id = self._body_id("target_plate")
        self.cube_joint_id = self.model.body_jntadr[self.cube_id]
        self.cube_qpos_address = self.model.jnt_qposadr[self.cube_joint_id]
        self.initial_cube_z = float(self.model.body_pos[self.cube_id][2])
        self.initial_target_z = float(self.model.body_pos[self.target_plate_id][2])
        self.current_cube_xy = np.zeros(2)
        self.current_target_xy = np.zeros(2)
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

        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.action_data = mujoco.MjData(self.model)
        self.target_hand_rotation_in_base = np.eye(3)
        self.action_rpy_reference = np.zeros(3)
        self.z_locked = False
        self.locked_z: float | None = None
        self.cube_over_target = False
        self.target_entered_sim_time: float | None = None
        self.gripper_closed = False
        self.z_lock_used_for_current_grasp = False
        self.vertical_anchor_xy: np.ndarray | None = None
        self.last_gripper_command_at = float("-inf")
        self.last_motion_command_at = float("-inf")
        self.side_preview_jpeg: bytes | None = None
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

    def _start_render_workers(self) -> None:
        context = mp.get_context("spawn")
        self.training_queue = context.Queue()
        self.training_reply_queue = context.Queue()
        self.training_process = context.Process(
            target=training_render_process,
            args=(self.args, self.training_queue, self.training_reply_queue),
            name="panda-training-renderer",
            daemon=True,
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
            else:
                self.front_preview_jpeg = jpeg

    def reset_simulation(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.randomize_object_positions()
        self.data.qpos[:7] = INITIAL_ARM_QPOS
        for address in self.finger_qpos_addresses:
            self.data.qpos[address] = 0.04
        self.data.ctrl[:7] = INITIAL_ARM_QPOS
        self.data.ctrl[7] = 255.0
        mujoco.mj_forward(self.model, self.data)
        base_rotation = self.data.xmat[self.base_id].reshape(3, 3)
        hand_rotation = self.data.xmat[self.hand_id].reshape(3, 3)
        self.target_hand_rotation_in_base = base_rotation.T @ hand_rotation
        self.action_rpy_reference = rotation_matrix_to_rpy(
            self.target_hand_rotation_in_base
        )

        self.z_locked = False
        self.locked_z = None
        self.cube_over_target = False
        self.target_entered_sim_time = None
        self.gripper_closed = False
        self.z_lock_used_for_current_grasp = False
        self.vertical_anchor_xy = None
        self.last_gripper_command_at = float("-inf")
        self.side_preview_jpeg = None
        self.front_preview_jpeg = None

    def randomize_object_positions(self) -> None:
        """Sample safe, reachable XY starts while keeping both objects upright."""
        cube_low = np.array([self.args.cube_x_min, self.args.cube_y_min])
        cube_high = np.array([self.args.cube_x_max, self.args.cube_y_max])
        target_low = np.array([self.args.plate_x_min, self.args.plate_y_min])
        target_high = np.array([self.args.plate_x_max, self.args.plate_y_max])

        for _ in range(1_000):
            cube_xy = self.rng.uniform(cube_low, cube_high)
            target_xy = self.rng.uniform(target_low, target_high)
            if np.linalg.norm(cube_xy - target_xy) >= self.args.object_min_separation:
                break
        else:
            raise RuntimeError(
                "Object randomization ranges cannot satisfy --object-min-separation."
            )

        # The cube has a free joint, so its pose lives in qpos. The plate is a
        # fixed MuJoCo body, so its position lives in model.body_pos.
        self.data.qpos[self.cube_qpos_address : self.cube_qpos_address + 3] = [
            cube_xy[0],
            cube_xy[1],
            self.initial_cube_z,
        ]
        self.data.qpos[self.cube_qpos_address + 3 : self.cube_qpos_address + 7] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        self.model.body_pos[self.target_plate_id] = [
            target_xy[0],
            target_xy[1],
            self.initial_target_z,
        ]
        self.current_cube_xy = cube_xy
        self.current_target_xy = target_xy

    def move_hand(self, dx: float, dy: float, dz: float) -> None:
        # Build the next Cartesian step from the commanded joint target, not
        # from the lagging physical arm. Otherwise the same orientation error
        # is added repeatedly while the servos catch up, which makes a stream
        # of diagonal commands twist the redundant 7-DoF arm.
        self.action_data.qpos[:] = self.data.qpos
        self.action_data.qpos[:7] = self.data.ctrl[:7]
        self.action_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.action_data)
        target_pos = self.action_data.xpos[self.hand_id].copy()

        if self.vertical_anchor_xy is not None:
            correction = np.clip(self.vertical_anchor_xy - target_pos[:2], -0.01, 0.01)
            dx, dy = float(correction[0]), float(correction[1])

        if self.z_locked and self.locked_z is not None:
            dz = float(self.locked_z - target_pos[2])
        elif self.gripper_closed and dz < 0:
            cube_z = float(self.data.xpos[self.cube_id][2])
            support_z = self.args.target_top_z if self.cube_over_target else self.args.table_top_z
            minimum_cube_z = support_z + self.args.cube_half_size + 0.002
            if cube_z <= minimum_cube_z:
                dz = 0.0

        # Gesture translation is defined in the robot-base frame.  In the
        # current scene link0 is aligned with world, but this conversion keeps
        # the control definition correct if the robot is moved later.
        base_rotation = self.action_data.xmat[self.base_id].reshape(3, 3)
        world_delta = base_rotation @ np.array([dx, dy, dz], dtype=float)

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
        task_jacobian = np.vstack((self.jacp[:, :7], self.jacr[:, :7]))
        task_delta = np.concatenate((world_delta, orientation_error))
        # Damped least squares keeps the solution continuous and avoids large
        # joint rotations when two Cartesian axes move together.
        damping = 0.03
        task_inverse = np.linalg.solve(
            task_jacobian @ task_jacobian.T
            + (damping * damping) * np.eye(task_jacobian.shape[0]),
            task_delta,
        )
        dq = task_jacobian.T @ task_inverse
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

    def set_z_lock(self, locked: bool, reason: str | None = None) -> None:
        was_locked = self.z_locked
        self.z_locked = locked
        if locked and not was_locked:
            # Capture the commanded height once. Repeated ILoveYou commands
            # must never overwrite it with the slightly sagging physical
            # height, otherwise the horizontal plane ratchets downward.
            self.action_data.qpos[:] = self.data.qpos
            self.action_data.qpos[:7] = self.data.ctrl[:7]
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
                    f"[视角] 方块已到运输高度 {cube_z:.3f} m："
                    "锁定 Z 并切换到俯视视角。"
                )
        elif was_locked and not locked:
            self.side_preview_jpeg = None
            if reason == "target":
                print("[视角] 已进入圆盘区域：解除 Z 并恢复斜视角。")
            else:
                print("[视角] 已解除 Z 并恢复斜视角。")

    def maintain_locked_plane(self) -> None:
        """Continuously keep the commanded end effector on the locked Z plane."""
        if self.z_locked and self.locked_z is not None:
            self.move_hand(0.0, 0.0, 0.0)

    def is_holding_cube(self) -> bool:
        finger_width = float(
            sum(self.data.qpos[address] for address in self.finger_qpos_addresses)
        )
        hand_cube_distance = float(
            np.linalg.norm(
                self.data.xpos[self.hand_id] - self.data.xpos[self.cube_id]
            )
        )
        return (
            self.gripper_closed
            and self.args.grasped_min_width
            <= finger_width
            <= self.args.grasped_max_width
            and hand_cube_distance <= self.args.grasped_max_distance
        )

    def update_transport_state(self) -> None:
        """Enter the top-down XY transport phase at a fixed absolute height."""
        cube_z = float(self.data.xpos[self.cube_id][2])
        if (
            self.gripper_closed
            and not self.z_locked
            and not self.z_lock_used_for_current_grasp
            and not self.cube_over_target
            and cube_z >= self.args.transport_cube_z
        ):
            self.set_z_lock(True)
            self.z_lock_used_for_current_grasp = True

    def update_viewer_camera(self, viewer) -> None:
        if self.z_locked:
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
        return {
            "z_locked": self.z_locked,
            "operator_view": "topdown" if self.z_locked else "overview",
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
            "cube_z_m": round(float(self.data.xpos[self.cube_id][2]), 4),
            "hand_z_m": round(float(self.data.xpos[self.hand_id][2]), 4),
            "locked_z_m": None if self.locked_z is None else round(self.locked_z, 4),
            "transport_cube_z_m": self.args.transport_cube_z,
        }

    def get_side_preview(self) -> bytes | None:
        self.last_aux_preview_request_at = time.perf_counter()
        return self.side_preview_jpeg

    def get_front_preview(self) -> bytes | None:
        self.last_aux_preview_request_at = time.perf_counter()
        return self.front_preview_jpeg

    def aux_preview_is_requested(self) -> bool:
        """Avoid rendering operator views while no browser is watching them."""
        return time.perf_counter() - self.last_aux_preview_request_at <= 1.0

    def open_gripper_immediately(self) -> None:
        self.data.ctrl[7] = 255.0
        for joint_id in self.finger_joint_ids:
            qpos_address = self.model.jnt_qposadr[joint_id]
            dof_address = self.model.jnt_dofadr[joint_id]
            self.data.qpos[qpos_address] = 0.04
            self.data.qvel[dof_address] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def stabilize_cube_on_target(self) -> None:
        if not self.cube_over_target:
            return
        resting_z = self.args.target_top_z + self.args.cube_half_size + 0.002
        cube_joint_id = self.model.body_jntadr[self.cube_id]
        cube_qpos_address = self.model.jnt_qposadr[cube_joint_id]
        cube_dof_address = self.model.jnt_dofadr[cube_joint_id]
        if self.data.xpos[self.cube_id][2] <= resting_z + 0.03:
            self.data.qpos[cube_qpos_address + 2] = max(
                self.data.qpos[cube_qpos_address + 2], resting_z
            )
            self.data.qvel[cube_dof_address : cube_dof_address + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def update_target_state(self) -> None:
        distance = np.linalg.norm(
            self.data.xpos[self.cube_id][:2] - self.data.xpos[self.target_plate_id][:2]
        )
        if not self.cube_over_target:
            if distance <= self.args.target_radius:
                if self.target_entered_sim_time is None:
                    self.target_entered_sim_time = float(self.data.time)
                stable_time = float(self.data.time) - self.target_entered_sim_time
                if stable_time >= self.args.target_dwell_s:
                    self.cube_over_target = True
                    if self.z_locked:
                        self.set_z_lock(False, reason="target")
            else:
                self.target_entered_sim_time = None
            return

        exit_radius = self.args.target_radius + self.args.target_exit_margin
        if distance > exit_radius:
            self.cube_over_target = False
            self.target_entered_sim_time = None
            cube_high_enough = (
                float(self.data.xpos[self.cube_id][2])
                >= self.args.transport_cube_z - self.args.relock_height_margin
            )
            if self.is_holding_cube() and cube_high_enough and not self.z_locked:
                self.set_z_lock(True, reason="left_target")
                self.z_lock_used_for_current_grasp = True

    def handle_gesture(self, payload: dict) -> None:
        command = payload.get("command")
        if command == "record_start":
            self.start_episode()
            return
        if command == "record_save":
            self.save_episode()
            return
        if command == "record_discard":
            self.discard_episode()
            return
        if command == "record_stop":
            self.stop_requested = True
            return

        if command in {"close", "open"}:
            sent_at = float(payload.get("sentAt", 0.0))
            if sent_at < self.last_gripper_command_at:
                return
            self.last_gripper_command_at = sent_at

        if command == "close":
            if not self.gripper_closed:
                self.z_lock_used_for_current_grasp = False
            self.data.ctrl[7] = 0.0
            self.gripper_closed = True
        elif command == "open":
            self.open_gripper_immediately()
            self.gripper_closed = False
            # Transport Z/top-down may only be released by update_target_state
            # after the cube has entered the target area. Opening elsewhere
            # must not bypass that task rule.
            if self.cube_over_target:
                self.set_z_lock(False)
            self.stabilize_cube_on_target()
        elif command == "left":
            self.move_hand(0.0, self.args.step, 0.0)
        elif command == "right":
            self.move_hand(0.0, -self.args.step, 0.0)
        elif command == "forward":
            self.move_hand(self.args.step, 0.0, 0.0)
        elif command == "backward":
            self.move_hand(-self.args.step, 0.0, 0.0)
        elif command == "up":
            self.move_hand(0.0, 0.0, self.args.vertical_step)
        elif command == "down":
            self.move_hand(0.0, 0.0, -self.args.vertical_step)
        elif command == "start_vertical":
            self.vertical_anchor_xy = self.data.xpos[self.hand_id][:2].copy()
        elif command == "start_planar":
            self.vertical_anchor_xy = None
            # Planar transport is always performed at the current absolute
            # height. Only update_target_state may release this lock after the
            # cube has remained inside the target radius for the dwell time.
            force_diagonal_lock = bool(payload.get("lockZ", False))
            if (
                (force_diagonal_lock or self.is_holding_cube())
                and not self.cube_over_target
            ):
                self.set_z_lock(True, reason="planar")
                self.z_lock_used_for_current_grasp = True
        elif command == "move_xy":
            self.vertical_anchor_xy = None
            if bool(payload.get("lockZ", False)) and not self.cube_over_target:
                self.set_z_lock(True, reason="planar")
                self.z_lock_used_for_current_grasp = True
            # Once the cube is grasped, planar motion is accepted only while Z
            # is locked (or after the target region has deliberately unlocked it).
            if self.is_holding_cube() and not self.z_locked and not self.cube_over_target:
                return
            dx = np.clip(float(payload.get("dx", 0.0)), -1.0, 1.0)
            dy = np.clip(float(payload.get("dy", 0.0)), -1.0, 1.0)
            self.move_hand(
                dx * self.args.max_planar_step,
                dy * self.args.max_planar_step,
                0.0,
            )

    def handle_key(self, keycode: int) -> None:
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
            self.open_gripper_immediately()
            self.gripper_closed = False
        elif keycode in {ord("L"), ord("l")}:
            self.data.ctrl[7] = 0.0
            self.gripper_closed = True
        elif keycode in {ord("S"), ord("s")}:
            self.start_episode()
        elif keycode in {ord("N"), ord("n")}:
            self.save_episode()
        elif keycode in {ord("R"), ord("r")}:
            self.discard_episode()
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
                "episode_index": self.saved_episodes,
                "cube_xy_m": self.current_cube_xy.tolist(),
                "target_plate_xy_m": self.current_target_xy.tolist(),
            }
        )
        self._wait_training_reply("started")
        self.recording = True
        self.episode_frames = 0
        print(f"[LeRobot] 开始录制 demonstration {self.saved_episodes + 1}。")
        print(
            "[随机初始位置] "
            f"方块=({self.current_cube_xy[0]:.3f}, {self.current_cube_xy[1]:.3f}) m，"
            f"圆盘=({self.current_target_xy[0]:.3f}, {self.current_target_xy[1]:.3f}) m。"
        )

    def save_episode(self) -> None:
        if not self.recording or self.episode_frames == 0:
            print("[LeRobot] 没有正在录制的数据，先按 S 开始。")
            return
        episode_frames = self.episode_frames
        self.saving_episode = True
        try:
            self.training_queue.put({"type": "save", "frames": episode_frames})
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

    def discard_episode(self) -> None:
        if self.recording:
            self.training_queue.put({"type": "discard"})
            self._wait_training_reply("discarded")
        self.recording = False
        self.episode_frames = 0
        self.reset_simulation()
        print("[LeRobot] 本次 demonstration 已丢弃，仿真已重置。")

    def record_frame(self) -> None:
        finger_width = float(
            sum(self.data.qpos[address] for address in self.finger_qpos_addresses)
        )
        state = np.concatenate(
            [self.data.qpos[:7], np.array([finger_width])]
        ).astype(np.float32)
        # Convert the commanded seven-joint target to a full Cartesian pose in
        # the robot-base frame: XYZ (metres), RPY (radians), and gripper target.
        self.action_data.qpos[:] = self.data.qpos
        self.action_data.qpos[:7] = self.data.ctrl[:7]
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
            self.save_episode()

    def run_interactive(self) -> None:
        self.server = GestureServer(
            self.command_queue,
            self.args.port,
            self.get_side_preview,
            self.get_front_preview,
            self.get_runtime_status,
        )
        self.server.start()
        print(f"[手势控制] 已监听 http://127.0.0.1:{self.args.port}/control")
        print("\n========== MuJoCo + LeRobot 录制 ==========")
        print("S：开始录制   N：成功并保存")
        print("R：失败并重录 Q：结束并写入数据集")
        print("方向键/I/K/J/L 仍可作为备用控制")
        print(f"数据集目录：{self.args.dataset_root}")
        print("===========================================\n")

        ui_hz = 60
        substeps = max(1, round(1.0 / ui_hz / self.model.opt.timestep))
        record_interval = 1.0 / self.args.fps
        next_record_time = time.perf_counter()
        # Alternate cameras so each stays at side_preview_fps without blocking
        # one simulation tick with two consecutive renders.
        side_preview_interval = 1.0 / (2.0 * self.args.side_preview_fps)
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
                for _ in range(substeps):
                    mujoco.mj_step(self.model, self.data)
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
        print(f"[LeRobot] 数据集：{self.args.dataset_root}")


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Record MuJoCo Panda demonstrations in LeRobot format."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "datasets" / f"mujoco_panda_pick_{timestamp}",
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
    parser.add_argument("--max-motion-command-age-ms", type=float, default=250.0)
    parser.add_argument("--target-radius", type=float, default=0.05)
    parser.add_argument("--target-dwell-s", type=float, default=0.25)
    parser.add_argument("--target-exit-margin", type=float, default=0.02)
    parser.add_argument("--relock-height-margin", type=float, default=0.02)
    parser.add_argument("--transport-cube-z", type=float, default=0.50)
    parser.add_argument("--grasped-min-width", type=float, default=0.02)
    parser.add_argument("--grasped-max-width", type=float, default=0.075)
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
    parser.add_argument("--cube-x-min", type=float, default=0.44)
    parser.add_argument("--cube-x-max", type=float, default=0.51)
    parser.add_argument("--cube-y-min", type=float, default=-0.07)
    parser.add_argument("--cube-y-max", type=float, default=0.07)
    parser.add_argument("--plate-x-min", type=float, default=0.54)
    parser.add_argument("--plate-x-max", type=float, default=0.62)
    parser.add_argument("--plate-y-min", type=float, default=-0.13)
    parser.add_argument("--plate-y-max", type=float, default=0.09)
    parser.add_argument("--object-min-separation", type=float, default=0.16)
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
    args = parser.parse_args()
    args.model_path = args.model_path.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    if not args.model_path.is_file():
        parser.error(f"MuJoCo scene not found: {args.model_path}")
    if args.dataset_root.exists():
        parser.error(
            f"Dataset directory already exists: {args.dataset_root}. "
            "Choose a new --dataset-root so existing data is never overwritten."
        )
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
