"""Off-process MuJoCo rendering for the Panda demonstration recorder."""

from __future__ import annotations

import io
import json
import queue
import shutil
import threading
import traceback
from pathlib import Path

import av
import mujoco
import numpy as np
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image


CAMERA_NAMES = ("overview", "camera_2", "camera_3")


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {name}")
    return body_id


def _free_camera(lookat, distance: float, azimuth: float, elevation: float):
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray(lookat, dtype=float)
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def _training_cameras(args):
    lookat = [
        args.observation_lookat_x,
        args.observation_lookat_y,
        args.observation_lookat_z,
    ]
    return {
        "overview": _free_camera(
            lookat,
            args.observation_distance,
            args.observation_azimuth,
            args.observation_elevation,
        ),
        "camera_2": _free_camera(
            lookat,
            args.observation_distance,
            args.observation_camera_2_azimuth,
            args.observation_elevation,
        ),
        "camera_3": _free_camera(
            lookat,
            args.observation_distance,
            args.observation_camera_3_azimuth,
            args.observation_elevation,
        ),
    }


def _auxiliary_cameras(args):
    return {
        "side": _free_camera(
            [args.side_lookat_x, args.side_lookat_y, args.side_lookat_z],
            args.side_distance,
            args.side_azimuth,
            args.side_elevation,
        ),
        "front": _free_camera(
            [args.front_lookat_x, args.front_lookat_y, args.front_lookat_z],
            args.front_distance,
            args.front_azimuth,
            args.front_elevation,
        ),
    }


def _apply_snapshot(model, data, target_plate_id: int, snapshot: dict) -> None:
    data.qpos[:] = snapshot["qpos"]
    data.qvel[:] = snapshot["qvel"]
    data.ctrl[:] = snapshot["ctrl"]
    data.time = snapshot["time"]
    model.body_pos[target_plate_id] = snapshot["target_plate_pos"]
    mujoco.mj_forward(model, data)


class EpisodeVideoRecorder:
    """Write one independently playable MP4 per camera without blocking rendering."""

    def __init__(self, directory: Path, fps: int, width: int, height: int) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.frame_queue: queue.Queue[dict[str, np.ndarray] | None] = queue.Queue(maxsize=60)
        self.error: Exception | None = None
        self.outputs = {}
        for camera_name in CAMERA_NAMES:
            container = av.open(str(directory / f"{camera_name}.mp4"), mode="w")
            stream = container.add_stream(
                "h264_videotoolbox",
                rate=fps,
                options={"q:v": "60"},
            )
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            self.outputs[camera_name] = (container, stream)
        self.worker = threading.Thread(target=self._encode_loop, daemon=True)
        self.worker.start()

    def add_frames(self, frames: dict[str, np.ndarray]) -> None:
        if self.error is not None:
            raise RuntimeError("Independent episode video encoder failed") from self.error
        self.frame_queue.put(frames)

    def _encode_loop(self) -> None:
        try:
            while True:
                frames = self.frame_queue.get()
                if frames is None:
                    break
                for camera_name, image in frames.items():
                    container, stream = self.outputs[camera_name]
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
        except Exception as error:
            self.error = error

    def close(self) -> None:
        self.frame_queue.put(None)
        self.worker.join(timeout=120)
        if self.worker.is_alive():
            raise RuntimeError("Independent episode video encoder did not stop")
        if self.error is not None:
            raise RuntimeError("Independent episode video encoder failed") from self.error
        for container, stream in self.outputs.values():
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        self.outputs.clear()


def _dataset_features(args) -> dict:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "axes": [
                    "joint1", "joint2", "joint3", "joint4",
                    "joint5", "joint6", "joint7", "finger_width_m",
                ]
            },
        },
        "observation.images.overview": {
            "dtype": "video",
            "shape": (args.height, args.width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera_2": {
            "dtype": "video",
            "shape": (args.height, args.width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera_3": {
            "dtype": "video",
            "shape": (args.height, args.width, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "axes": [
                    "end_effector_target_x_base_m",
                    "end_effector_target_y_base_m",
                    "end_effector_target_z_base_m",
                    "end_effector_target_rx_base_rad",
                    "end_effector_target_ry_base_rad",
                    "end_effector_target_rz_base_rad",
                    "gripper_target_0_255",
                ]
            },
        },
    }


def training_render_worker(args, command_queue, reply_queue) -> None:
    """Render and encode every training snapshot in order in a separate process."""
    dataset = None
    renderer = None
    inspection_recorder = None
    temporary_video_dir = None
    episode_info = None
    try:
        model = mujoco.MjModel.from_xml_path(str(args.model_path))
        data = mujoco.MjData(model)
        target_plate_id = _body_id(model, "target_plate")
        cameras = _training_cameras(args)
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            root=args.dataset_root,
            robot_type="mujoco_panda",
            features=_dataset_features(args),
            use_videos=True,
            # PyAV already supplies the FFmpeg libraries used by this recorder.
            # Pinning the read backend prevents LeRobot from probing TorchCodec,
            # which otherwise loads Homebrew FFmpeg into the same macOS process.
            video_backend="pyav",
            streaming_encoding=args.streaming_encoding,
            rgb_encoder=RGBEncoderConfig(vcodec="auto", crf=23),
            encoder_threads=2,
            image_writer_threads=4,
        )
        video_root = args.dataset_root / "episode_videos"
        manifest_path = video_root / "initial_positions.jsonl"
        reply_queue.put({"type": "ready"})

        while True:
            command = command_queue.get()
            kind = command["type"]
            if kind == "start":
                episode_info = command
                temporary_video_dir = video_root / f".episode_{command['episode_index']:06d}_recording"
                if temporary_video_dir.exists():
                    shutil.rmtree(temporary_video_dir)
                inspection_recorder = EpisodeVideoRecorder(
                    temporary_video_dir, args.fps, args.width, args.height
                )
                reply_queue.put({"type": "started"})
            elif kind == "frame":
                snapshot = command["snapshot"]
                _apply_snapshot(model, data, target_plate_id, snapshot)
                images = {}
                for camera_name, camera in cameras.items():
                    renderer.update_scene(data, camera=camera)
                    images[camera_name] = renderer.render().copy()
                if inspection_recorder is not None:
                    inspection_recorder.add_frames(images)
                dataset.add_frame(
                    {
                        "observation.state": command["state"],
                        "observation.images.overview": images["overview"],
                        "observation.images.camera_2": images["camera_2"],
                        "observation.images.camera_3": images["camera_3"],
                        "action": command["action"],
                        "task": args.task,
                    }
                )
            elif kind == "save":
                dataset.save_episode()
                inspection_error = None
                try:
                    if inspection_recorder is not None:
                        inspection_recorder.close()
                    final_dir = video_root / f"episode_{episode_info['episode_index']:06d}"
                    if temporary_video_dir is not None:
                        temporary_video_dir.rename(final_dir)
                except Exception as error:
                    inspection_error = str(error)
                video_root.mkdir(parents=True, exist_ok=True)
                with manifest_path.open("a", encoding="utf-8") as manifest:
                    manifest.write(
                        json.dumps(
                            {
                                "episode_index": episode_info["episode_index"],
                                "frames": command["frames"],
                                "cube_xy_m": episode_info["cube_xy_m"],
                                "target_plate_xy_m": episode_info["target_plate_xy_m"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                inspection_recorder = None
                temporary_video_dir = None
                episode_info = None
                reply_queue.put({"type": "saved", "inspection_error": inspection_error})
            elif kind == "discard":
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                if inspection_recorder is not None:
                    inspection_recorder.close()
                if temporary_video_dir is not None:
                    shutil.rmtree(temporary_video_dir, ignore_errors=True)
                inspection_recorder = None
                temporary_video_dir = None
                episode_info = None
                reply_queue.put({"type": "discarded"})
            elif kind == "close":
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                if inspection_recorder is not None:
                    inspection_recorder.close()
                if temporary_video_dir is not None:
                    shutil.rmtree(temporary_video_dir, ignore_errors=True)
                dataset.finalize()
                reply_queue.put({"type": "closed"})
                break
    except Exception as error:
        reply_queue.put(
            {"type": "fatal", "error": str(error), "traceback": traceback.format_exc()}
        )
    finally:
        if renderer is not None:
            renderer.close()


def auxiliary_render_worker(args, command_queue, result_queue, ready_queue) -> None:
    """Render only the newest side/front snapshot in a separate process."""
    renderer = None
    try:
        model = mujoco.MjModel.from_xml_path(str(args.model_path))
        data = mujoco.MjData(model)
        target_plate_id = _body_id(model, "target_plate")
        cameras = _auxiliary_cameras(args)
        renderer = mujoco.Renderer(model, height=args.aux_height, width=args.aux_width)
        next_camera = "side"
        ready_queue.put({"type": "ready"})
        while True:
            command = command_queue.get()
            if command is None:
                break
            _apply_snapshot(model, data, target_plate_id, command)
            renderer.update_scene(data, camera=cameras[next_camera])
            image = renderer.render()
            output = io.BytesIO()
            Image.fromarray(image).save(output, format="JPEG", quality=78)
            result = (next_camera, output.getvalue())
            try:
                result_queue.put_nowait(result)
            except queue.Full:
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
                result_queue.put_nowait(result)
            next_camera = "front" if next_camera == "side" else "side"
    except Exception as error:
        ready_queue.put(
            {"type": "fatal", "error": str(error), "traceback": traceback.format_exc()}
        )
    finally:
        if renderer is not None:
            renderer.close()
