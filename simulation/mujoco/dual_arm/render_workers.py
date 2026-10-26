"""Task3 snapshot recording and independent previews, without changing Task1/2.

Small physics snapshots are spooled during teleoperation. Configured views are
rendered together when saving, so capture does not contend with IK/physics and
frames are written only once to the chosen success/failure dataset.
"""
from __future__ import annotations

import io
import json
import os
import queue
import traceback
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from .training_camera import (OPERATOR_CAMERAS, TRAINING_CAMERA_NAME,
    make_training_camera, recording_camera_names, training_camera_metadata,
    training_camera_spec)

CAMERAS = OPERATOR_CAMERAS
from simulation.mujoco.dual_arm.action_contract import STATE_NAMES, ACTION_NAMES_V1 as ACTION_NAMES
from simulation.mujoco.dual_arm.action_contract import V2, ACTION_NAMES_V2


def dataset_features(width=640, height=480, schema_version=None, *, config=None):
    if config is not None:
        settings = config['recording']
        width, height = settings['width'], settings['height']
        schema_version = settings.get('schema_version')
    features = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (14,), "names": ACTION_NAMES_V2 if schema_version == V2 else ACTION_NAMES},
    }
    features.update({f"observation.images.{name}": {
        "dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]
    } for name in recording_camera_names(config)})
    return features


class MultiCameraRenderer:
    def __init__(self, model, width=640, height=480, *, camera_names=CAMERAS, config=None):
        self.model = model
        self.camera_names = recording_camera_names({'recording': {'cameras': camera_names}})
        self.training_camera = make_training_camera(config) if TRAINING_CAMERA_NAME in self.camera_names else None
        if self.training_camera is not None:
            specification = training_camera_spec(config)
            if (width, height) != (specification['width'], specification['height']):
                raise ValueError('The frozen oblique view requires 640x480 RGB')
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data, *, operator_view=None):
        images = {}
        # No mj_step or state mutation occurs between views: every RGB array
        # comes from this one restored snapshot, including the free camera.
        for name in self.camera_names:
            camera = self.training_camera if name == TRAINING_CAMERA_NAME else name
            self.renderer.update_scene(data, camera=camera)
            if operator_view is not None and name != TRAINING_CAMERA_NAME:
                apply_operator_view(self.renderer.scene, self.model, name, operator_view)
            images[name] = self.renderer.render().copy()
        return images

    def close(self):
        self.renderer.close()


def render_observations(controller, width=640, height=480, renderer=None):
    owned = renderer is None
    renderer = renderer or MultiCameraRenderer(controller.model, width, height)
    try:
        return renderer.render(controller.data)
    finally:
        if owned:
            renderer.close()


def apply_snapshot(model, data, snapshot):
    if "body_pos" in snapshot:
        model.body_pos[:] = snapshot["body_pos"]
    if "body_quat" in snapshot:
        model.body_quat[:] = snapshot["body_quat"]
    data.qpos[:] = snapshot["qpos"]
    data.qvel[:] = snapshot["qvel"]
    data.ctrl[:] = snapshot["ctrl"]
    data.time = float(snapshot["time"])
    mujoco.mj_forward(model, data)


def robot_geom_ids(model):
    """Panda body descendants only, excluding table, object and base pedestals."""
    roots = {model.body(name).id for name in ("left_link0", "right_link0")}
    return tuple(index for index, body in enumerate(model.geom_bodyid)
                 if model.body_rootid[body] in roots)


def sync_operator_view(viewer, model, camera, config):
    """Copy the operator appearance into a passive viewer, then restore model.

    Passive viewer.sync() copies model visual attributes into its private
    display state. The temporary override is limited to that synchronous copy;
    callers must run physics/snapshot capture outside this call. No material,
    collision, dynamics or saved model attributes are changed persistently.
    Do not hold viewer.lock() during sync(), which acquires its own lock.
    """
    alpha = float(config.get("operator_view", {}).get("top_robot_alpha", 1.0))
    geom_ids = np.asarray(robot_geom_ids(model), dtype=int)
    original = None
    try:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = model.camera(camera).id
            viewer.opt.sitegroup[:] = 0
            if camera == "top" and alpha < 1:
                original = model.geom_rgba[geom_ids].copy()
                effective = original.copy()
                materials = model.geom_matid[geom_ids]
                # Default geom RGBA inherits material RGBA. Preserve those
                # effective colors when setting an explicit alpha override.
                inherited = (materials >= 0) & np.all(original == [0.5, 0.5, 0.5, 1], axis=1)
                effective[inherited] = model.mat_rgba[materials[inherited]]
                effective[:, 3] = np.minimum(effective[:, 3], alpha)
                model.geom_rgba[geom_ids] = effective
        return viewer.sync()
    finally:
        if original is not None:
            with viewer.lock():
                model.geom_rgba[geom_ids] = original


def apply_operator_view(scene, model, camera, config):
    """Style a freshly updated display scene without changing the physics model.

    Only top-view robot geometry becomes translucent. The base pedestals,
    object and table retain their appearance; front/side robot geometry stays
    opaque. Known diagnostic sites are hidden in all operator views. Dataset
    renderers omit this helper entirely, including for historical configs.

    Call after every ``mjv_updateScene`` / Renderer.update_scene: those calls
    reconstruct the original visual geometry and prevent view-to-view leaks.
    An absent setting keeps the historical opaque robot appearance.
    """
    alpha = float(config.get("operator_view", {}).get("top_robot_alpha", 1.0))
    robot_geoms = set(robot_geom_ids(model))
    diagnostic_sites = {model.site(name).id for name in
                        ("left_ee", "right_ee", "left_grasp_site", "right_grasp_site")}
    for geom in scene.geoms[:scene.ngeom]:
        if geom.objtype == mujoco.mjtObj.mjOBJ_SITE and geom.objid in diagnostic_sites:
            geom.rgba[3] = 0
            geom.transparent = True
        elif camera == "top" and geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid in robot_geoms:
            # Never reveal geometry explicitly hidden by the scene/model.
            geom.rgba[3] = min(float(geom.rgba[3]), alpha)
            geom.transparent = geom.rgba[3] < 1


def _flush_snapshots(directory, snapshots, start):
    if start >= len(snapshots):
        return len(snapshots)
    fields = ("qpos", "qvel", "ctrl", "body_pos", "time", "state", "action")
    block = snapshots[start:]
    if all("body_quat" in frame for frame in block):
        fields += ("body_quat",)
    np.savez_compressed(directory / f"snapshots_{start:06d}.npz",
                        **{key: np.asarray([frame[key] for frame in block]) for key in fields})
    return len(snapshots)


class InspectionVideos:
    def __init__(self, directory, fps, width, height, camera_names=CAMERAS):
        import av
        directory.mkdir(parents=True, exist_ok=False)
        self.outputs = {}
        for name in camera_names:
            container = av.open(str(directory / f"{name}.mp4"), mode="w")
            stream = container.add_stream("libx264", rate=fps, options={"crf": "23", "preset": "veryfast"})
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            self.outputs[name] = container, stream

    def add(self, images):
        import av
        for name, image in images.items():
            container, stream = self.outputs[name]
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)

    def close(self):
        for container, stream in self.outputs.values():
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        self.outputs.clear()


def recording_worker(config, command_queue, reply_queue):
    from .task_controller import DualArmTaskController
    cache = Path(__file__).resolve().parents[3] / ".cache/huggingface"
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache / "datasets"))
    datasets = {}
    controller = None
    renderer = None
    frames = []
    pending = None
    persisted = 0
    try:
        controller = DualArmTaskController(config=config)
        settings = config.get("recording", {})
        fps, width, height = settings.get("fps", 30), settings.get("width", 640), settings.get("height", 480)
        camera_names = recording_camera_names(config)
        renderer = MultiCameraRenderer(controller.model, width, height, camera_names=camera_names, config=config)
        reply_queue.put({"type": "ready"})
        while True:
            command = command_queue.get()
            kind = command["type"]
            if kind == "start":
                frames, persisted = [], 0
                pending = Path(command["pending"])
                pending.mkdir(parents=True, exist_ok=True)
                reply_queue.put({"type": "started"})
            elif kind == "frame":
                frames.append(command["snapshot"])
                if len(frames) - persisted >= 30:
                    persisted = _flush_snapshots(pending, frames, persisted)
            elif kind == "flush":
                persisted = _flush_snapshots(pending, frames, persisted)
                reply_queue.put({"type": "flushed", "frames": len(frames)})
            elif kind == "save":
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
                from lerobot.configs.video import RGBEncoderConfig
                if not frames:
                    raise RuntimeError("No captured frames to save")
                persisted = _flush_snapshots(pending, frames, persisted)
                # This index travels with the physical snapshots when the
                # recorder moves the completed pending directory to research.
                # RGB generation never changes or advances the original clock.
                with (pending / 'camera_frame_index.jsonl').open('w', encoding='utf-8') as camera_index:
                    for index, snapshot in enumerate(frames):
                        camera_index.write(json.dumps({
                            'raw_frame_index': index, 'snapshot_index': index,
                            'snapshot_time_s': float(snapshot['time']),
                            'video_time_s': index / fps, 'cameras': list(camera_names),
                        }) + '\n')
                camera_manifest = {
                    'version': 'task3-shared-snapshot-rgb-v1',
                    'frames': len(frames), 'cameras': list(camera_names),
                    'width': width, 'height': height, 'fps': fps,
                    'recording_strategy': 'snapshot_replay_at_save',
                    'frame_index': 'camera_frame_index.jsonl',
                }
                if TRAINING_CAMERA_NAME in camera_names:
                    from .render_identity import render_identity
                    camera_manifest['training_camera'] = training_camera_metadata(config)
                    camera_manifest['render_identity'] = render_identity(config, model=controller.model)
                    if command['metadata'].get('render_identity') != camera_manifest['render_identity']:
                        raise ValueError('Recorder and save worker have different oblique render identities')
                (pending / 'camera_manifest.json').write_text(
                    json.dumps(camera_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
                outcome = command["outcome"]
                destination = Path(command["root"])
                if outcome not in datasets:
                    datasets[outcome] = LeRobotDataset.create(
                        repo_id=command["repo_id"], fps=fps, root=destination,
                        robot_type="dual_panda", features=dataset_features(config=config),
                        use_videos=True, video_backend="pyav", streaming_encoding=True,
                        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=23, preset="veryfast"),
                        encoder_threads=2,
                    )
                    metadata_path = destination / "meta/task3.json"
                    metadata_path.write_text(json.dumps(command["metadata"], ensure_ascii=False, indent=2), encoding="utf-8")
                elif datasets[outcome]._is_finalized:
                    datasets[outcome] = LeRobotDataset.resume(
                        repo_id=command["repo_id"], root=destination, video_backend="pyav",
                        streaming_encoding=True,
                        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=23, preset="veryfast"), encoder_threads=2,
                    )
                dataset = datasets[outcome]
                episode_index = dataset.meta.total_episodes
                inspection = None
                if settings.get("inspection_videos", True):
                    inspection = InspectionVideos(destination / f"episode_videos/episode_{episode_index:06d}", fps, width, height, camera_names)
                try:
                    for index, snapshot in enumerate(frames):
                        apply_snapshot(controller.model, controller.data, snapshot)
                        images = renderer.render(controller.data)
                        dataset.add_frame({
                            "observation.state": np.asarray(snapshot["state"], dtype=np.float32),
                            "action": np.asarray(snapshot["action"], dtype=np.float32),
                            "task": config.get("task", "bimanual_pick_carry_place_v1"),
                            **{f"observation.images.{name}": image for name, image in images.items()},
                        })
                        if inspection:
                            inspection.add(images)
                        if (index + 1) % 30 == 0:
                            reply_queue.put({"type": "progress", "frames": index + 1, "total": len(frames)})
                    dataset.save_episode()
                    # A successful save ACK must mean Parquet footers and all
                    # video/metadata writers are closed, even if the app later
                    # crashes. Resume append mode for the following episode.
                    dataset.finalize()
                finally:
                    if inspection:
                        inspection.close()
                reply_queue.put({"type": "saved", "episode_index": episode_index, "frames": len(frames), "root": str(destination)})
                frames, pending, persisted = [], None, 0
            elif kind == "discard":
                if pending:
                    _flush_snapshots(pending, frames, persisted)
                frames, pending, persisted = [], None, 0
                reply_queue.put({"type": "discarded"})
            elif kind == "close":
                if pending:
                    _flush_snapshots(pending, frames, persisted)
                for dataset in datasets.values():
                    dataset.finalize()
                reply_queue.put({"type": "closed"})
                break
    except Exception as error:
        reply_queue.put({"type": "fatal", "error": str(error), "traceback": traceback.format_exc()})
    finally:
        for dataset in datasets.values():
            try:
                dataset.finalize()
            except Exception:
                pass  # Original failure and the persisted snapshots remain available.
        if renderer:
            renderer.close()


def preview_worker(config, snapshots, outputs, ready):
    from .task_controller import DualArmTaskController
    renderer = None
    try:
        controller = DualArmTaskController(config=config)
        renderer = MultiCameraRenderer(controller.model, 640, 480, camera_names=OPERATOR_CAMERAS, config=config)
        ready.put({"type": "ready"})
        while True:
            snapshot = snapshots.get()
            if snapshot is None:
                break
            # Metadata belongs to previews only, never to recording snapshots.
            metadata = snapshot['_preview']
            apply_snapshot(controller.model, controller.data, snapshot)
            images = renderer.render(controller.data, operator_view=config)
            jpegs = {}
            for name, frame in images.items():
                buffer = io.BytesIO()
                Image.fromarray(frame).save(buffer, format="JPEG", quality=78)
                jpegs[name] = buffer.getvalue()
            try:
                outputs.put_nowait({"images": jpegs, "created_at": time.monotonic(), **metadata})
            except queue.Full:
                try:
                    outputs.get_nowait()
                except queue.Empty:
                    pass
                try:
                    outputs.put_nowait({"images": jpegs, "created_at": time.monotonic(), **metadata})
                except queue.Full:
                    pass
    except Exception as error:
        ready.put({"type": "fatal", "error": str(error), "traceback": traceback.format_exc()})
    finally:
        if renderer:
            renderer.close()
