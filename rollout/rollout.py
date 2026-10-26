"""Versioned, robot-only Paper A closed-loop evaluation in real MuJoCo physics.

Preview cases are engineering observations, never part of the fixed benchmark.
All model failures are retained. A batch is complete only with every planned
case, and infrastructure errors are reported separately from task failures.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import csv
import gzip
import hashlib
import html
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/evaluation_baseline.json"
VERSION = "papera-closed-loop-run-v2"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.writing")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def now():
    return datetime.now(timezone.utc).isoformat()


def git_commit():
    try:
        return subprocess.check_output([r"C:\Program Files\Git\cmd\git.exe", "-c",
            f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"], cwd=ROOT,
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def implementation_hashes():
    # Freeze the executor, protocol, observation adapter, physics and assets.
    paths = list((ROOT / "rollout").glob("*.py"))
    paths += [p for p in (ROOT / "tasks").rglob('*') if p.suffix in ('.py', '.json')]
    paths += list((ROOT / "simulation/mujoco/dual_arm").glob("*.py"))
    paths += [ROOT / "simulation/mujoco/assets/dual_arm/dual_arm_scene.xml"]
    paths += list((ROOT / "simulation/mujoco/assets/franka_emika_panda/assets").glob("*"))
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted(paths) if p.is_file()}


def validate_settings(settings):
    from rollout.action_chunks import ActionChunkScheduler, LEGACY
    from rollout.protocol import FULL_OPEN_SUCCESS_CRITERION, HALF_OPEN_SUCCESS_CRITERION
    if settings.get('media_mode', 'all_views') not in {'all_views', 'policy_only'}:
        raise ValueError('media_mode must be all_views or policy_only')
    if settings["version"] not in {"papera-closed-loop-evaluation-v1", "papera-closed-loop-evaluation-v2",
                                   "papera-single-view-half-open-evaluation-v1"}:
        raise ValueError("Unknown evaluation settings")
    open_fraction = settings.get("success_open_fraction", 1.0)
    criterion = settings.get("success_criterion", FULL_OPEN_SUCCESS_CRITERION)
    if (isinstance(open_fraction, bool) or not isinstance(open_fraction, (int, float))
            or not math.isfinite(open_fraction) or not 0 < open_fraction <= 1):
        raise ValueError("success_open_fraction must be in (0, 1]")
    expected_criterion = HALF_OPEN_SUCCESS_CRITERION if open_fraction < 1 else FULL_OPEN_SUCCESS_CRITERION
    if criterion != expected_criterion:
        raise ValueError("success_criterion does not match success_open_fraction")
    if settings["version"] == "papera-single-view-half-open-evaluation-v1" and open_fraction != .5:
        raise ValueError("The current single-view baseline protocol freezes half-open success")
    if settings["observation_hz"] != 30 or settings["control_hz"] != 60 or settings["video_fps"] != 30:
        raise ValueError("This executor requires 30 Hz observations and 60 Hz control")
    if settings["version"] == "papera-closed-loop-evaluation-v1" and settings["execution"] != LEGACY:
        raise ValueError("Legacy evaluation settings support only first-action execution")
    ActionChunkScheduler.from_settings(settings)
    if len(settings["test_seeds"]) != 50 or len(set(settings["test_seeds"])) != 50:
        raise ValueError("The first benchmark requires 50 unique seeds")
    if set(settings["test_seeds"]) & set(settings["engineering_seeds"]):
        raise ValueError("Engineering and test cases must be disjoint")
    if (isinstance(settings["max_seconds"], bool)
            or not isinstance(settings["max_seconds"], (int, float))
            or not math.isfinite(settings["max_seconds"]) or settings["max_seconds"] <= 0):
        raise ValueError("Invalid deadline")
    for name in ('material_intervention_translation_m', 'material_intervention_width_m', 'material_intervention_angle_rad'):
        if not math.isfinite(settings.get(name, .0001)) or settings.get(name, .0001) <= 0:
            raise ValueError(f'Invalid intervention tolerance: {name}')


def prepare_protocol(settings, output):
    from rollout.protocol import load_scene_provenance
    from tasks.registry import hook
    output = resolve(output)
    if output.exists():
        value = verify_protocol(output, settings)
        print(f"已核验并复用固定50次清单：{output}（未重新抽样）", flush=True)
        return value
    source = load_scene_provenance(resolve(settings["pack"]))
    config = source["config"]
    # The evaluation budget can end an episode before the frozen controller's
    # own timeout. Never mutate historical scene settings to shorten a run,
    # or advertise a budget beyond the controller's original upper limit.
    validate_settings(settings)
    scene_deadline = config["evaluation"]["max_episode_seconds"]
    if settings["max_seconds"] > scene_deadline:
        raise ValueError("Evaluation deadline exceeds the frozen task scene contract")
    if settings['lift_stable_seconds'] != config['cooperation']['grasp_stable_seconds']:
        raise ValueError('Lift confirmation differs from the frozen success monitor')
    check = hook(config, 'preflight')(config, settings["test_seeds"],
                            demonstration_seeds=source["demonstration_seeds"],
                            engineering_seeds=settings["engineering_seeds"])
    value = {"version": "papera-task-frozen-50-v1", "created_at": now(), "settings": settings,
             "source": source, "preflight": check, "seeds": settings["test_seeds"],
             "deadline": {"evaluation_seconds": settings["max_seconds"],
                          "frozen_controller_seconds": scene_deadline,
                          "policy": "evaluation_budget_within_frozen_controller_limit"},
             "code": implementation_hashes(), "git_commit": git_commit(),
             "protocol_scope": "single-checkpoint registered task; all geometric flags retained; no policy-based case selection"}
    value["signature"] = canonical(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, value)
    print(f"Frozen 50-case protocol: {output}", flush=True)
    print(json.dumps(check.get("summary", {}), ensure_ascii=False), flush=True)
    return value


def verify_protocol(path, settings):
    protocol = read_json(path)
    supplied = protocol.pop("signature")
    if canonical(protocol) != supplied:
        raise ValueError("Frozen protocol signature changed")
    protocol["signature"] = supplied
    if protocol["settings"] != settings or protocol["seeds"] != settings["test_seeds"]:
        raise ValueError("Settings/seed list differs from the frozen protocol")
    if protocol["code"] != implementation_hashes():
        raise ValueError("Evaluation or physics code/assets changed; create a new named protocol after engineering validation")
    return protocol


class PanelVideo:
    def __init__(self, path, fps, crf, *, cameras=("top", "front", "side")):
        import av
        self.av = av
        self.path, self.fps, self.frames = Path(path), int(fps), 0
        self.cameras = tuple(cameras)
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("libx264", rate=self.fps)
        self.stream.width, self.stream.height, self.stream.pix_fmt = 640*len(self.cameras), 480, "yuv420p"
        self.stream.options = {"crf": str(crf), "preset": "veryfast", "threads": "1"}

    def append(self, images):
        import numpy as np
        frame = self.av.VideoFrame.from_ndarray(np.concatenate([images[c] for c in self.cameras], axis=1), format="rgb24")
        frame.pts, frame.time_base = self.frames, Fraction(1, self.fps)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def decode_video(path, expected, *, width=1920):
    import av
    count = 0
    previous = -math.inf
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if frame.width != width or frame.height != 480 or frame.pts is None:
                raise ValueError("Malformed rollout video frame")
            current = float(frame.pts * frame.time_base)
            if current <= previous:
                raise ValueError("Non-increasing video presentation times")
            previous, count = current, count + 1
    if count != expected:
        raise ValueError(f"Video frame count mismatch: {count} != {expected}")
    return {"decoded_frames": count, "complete_decode": True, "width": width, "height": 480, "sha256": sha(path)}


_WORKER = {}


def measure_intervention(requested, issued, settings):
    """Count meaningful servo-goal changes, not Euler aliases or float noise."""
    import numpy as np
    residual = np.asarray(issued, dtype=float) - np.asarray(requested, dtype=float)
    angles = [3, 4, 5, 10, 11, 12]
    residual[angles] = (residual[angles] + np.pi) % (2*np.pi) - np.pi
    material = (np.max(np.abs(residual[[0,1,2,7,8,9]])) >= settings.get('material_intervention_translation_m', .0001)
        or np.max(np.abs(residual[[6,13]])) >= settings.get('material_intervention_width_m', .0001)
        or np.max(np.abs(residual[angles])) >= settings.get('material_intervention_angle_rad', .001))
    return bool(material), residual.tolist()


def render_camera_names(metadata, settings):
    cameras = tuple(metadata['cameras'])
    if settings.get('media_mode', 'all_views') == 'policy_only':
        if cameras != ('oblique',):
            raise ValueError('policy_only currently requires the single-oblique model contract')
        return cameras
    auxiliary = ('top', 'front', 'oblique') if 'oblique' in cameras else ('top', 'front', 'side')
    return tuple(dict.fromkeys((*auxiliary, *cameras)))


def initialize_worker(run_path, checkpoint, scene, settings, device, allow_overfit=False):
    import torch
    from rollout.policy import Policy
    from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
    from simulation.mujoco.dual_arm.render_workers import MultiCameraRenderer
    torch.set_num_threads(settings["cpu_threads_per_worker"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    policy = Policy(run_path, checkpoint=checkpoint, device=device, verify_cache=False,
                    allow_overfit=allow_overfit)
    if settings.get("chunk_size", policy.model.config.chunk_size) != policy.model.config.chunk_size:
        raise ValueError("Evaluation chunk_size differs from the frozen model")
    controller = DualArmTaskController(config=scene)
    camera_names = render_camera_names(policy.metadata, settings)
    if policy.metadata["cameras"] == ["oblique"]:
        from simulation.mujoco.dual_arm.training_camera import training_camera_metadata
        if any(camera != training_camera_metadata(scene)
               for camera in policy.metadata["training_camera_configs"].values()):
            raise ValueError("Live oblique camera differs from the verified training sources")
    renderer = MultiCameraRenderer(controller.model, width=640, height=480,
                                   camera_names=camera_names, config=scene)
    from OpenGL import GL
    gl_info = {name: (GL.glGetString(token) or b'unknown').decode(errors='replace')
               for name, token in [('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER), ('version', GL.GL_VERSION)]}
    runtime = {"python": sys.version, "executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in ('mujoco','torch','torchvision','av','numpy','pillow','glfw','PyOpenGL')},
        "gpu": torch.cuda.get_device_name() if device == 'cuda' else None,
        "cuda_runtime": torch.version.cuda, "device": device,
        "cpu_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "policy_batch_size": 1, "policy_metadata": policy.metadata,
        "opengl": gl_info, "render_cameras": list(camera_names),
        "media_mode": settings.get('media_mode', 'all_views')}
    print(f'Render cameras: {camera_names}; OpenGL: {gl_info["renderer"]}; inference: {runtime["gpu"]}', flush=True)
    _WORKER.update(policy=policy, controller=controller, renderer=renderer, settings=settings, runtime=runtime)


def restore_demonstration_start(controller, start):
    """Restore exactly one recorded free-grasp starting state for engineering.

No source future actions enter a learned rollout. Original snapshots omit
solver warm-start; this is a bounded reproduction check, not bitwise replay.
The current Task3 model has only fixed finger-joint equalities and no mocap.
"""
    import mujoco
    import numpy as np
    from simulation.mujoco.dual_arm.render_workers import apply_snapshot
    c, snapshot, row = controller, start["snapshot"], start["first_control"]
    if c.model.nmocap or not np.all(c.model.eq_type == mujoco.mjtEq.mjEQ_JOINT):
        raise ValueError("Demonstration initialization supports only fixed finger equalities and no mocap")
    contact = row["status"].get("physical_contact", {})
    if contact.get("ever_dual") or any(contact.get("stable_grasps", {}).values()):
        raise ValueError("Demonstration must start before a qualified grasp")
    for key, expected in (("qpos", c.data.qpos.shape), ("qvel", c.data.qvel.shape),
                          ("ctrl", c.data.ctrl.shape), ("body_pos", c.model.body_pos.shape),
                          ("body_quat", c.model.body_quat.shape)):
        value = np.asarray(snapshot[key])
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"Invalid demonstration initial snapshot: {key}")
    if not np.isfinite(snapshot["time"]) or abs(float(snapshot["time"])-row["sim_time"]) > 1e-8:
        raise ValueError("Demonstration initial snapshot/control times differ")
    apply_snapshot(c.model, c.data, snapshot)
    for arm in c.arms:
        saved = row["status"]["commanded_ee"][arm.side]
        arm.command_position = np.asarray(saved["position"], dtype=float).copy()
        arm.command_rotation = np.asarray(saved["rotation"], dtype=float).copy()
        grip = row["status"]["grippers"][arm.side]
        arm.gripper_requested_width_m = grip["requested_width_m"]
        arm.gripper_goal_width_m = grip["goal_width_m"]
        arm._write_gripper_width(grip["command_width_m"])
        arm._ik_tick_time = arm._ik_tick_origin = arm._gripper_tick = None
    c._previous_object_position = c.data.xpos[c.object_id].copy()
    c._previous_ee_positions = [a.get_actual_pose()[0].copy() for a in c.arms]
    remainder = (row["tick"]/60-float(snapshot["time"])) % c.model.opt.timestep
    c._physics_remainder = 0. if abs(remainder-c.model.opt.timestep) < 1e-8 else remainder
    c._task_started_at = float(c.data.time)
    return {
        "source_key": start["source_key"], "source_hashes": start["source_hashes"],
        "restored_once": True, "solver_warm_start_recorded": False,
        "initial_eq_active": c.data.eq_active.tolist(), "model_nmocap": int(c.model.nmocap),
        "initial_state_max_abs": float(np.max(np.abs(c.state_vector()-np.asarray(row["state"])))),
        "initial_time_s": float(c.data.time),
    }


def update_stage_progress(controller, evidence, stages, elapsed):
    """Diagnostic milestones only; never replace the physical success monitor."""
    flags = {
        "left_grasp": evidence.get("current_opposed_grasps", {}).get("left", False),
        "right_grasp": evidence.get("current_opposed_grasps", {}).get("right", False),
        "dual_grasp": evidence.get("ever_dual", False),
        "lift": evidence.get("lift_verified", False),
        "success": evidence.get("task_success", False),
    }
    if hasattr(controller, "arms"):
        import numpy as np
        c = controller
        actual = np.asarray([a.get_actual_pose()[0] for a in c.arms])
        position = c.data.xpos[c.object_id]
        rotation = c.data.xmat[c.object_id].reshape(3, 3)
        offsets = np.asarray(c.config["scene"]["grasp_offsets"])
        handles = position + offsets @ rotation.T
        flags["both_xy_aligned_30mm"] = bool(np.all(np.linalg.norm(actual[:, :2]-handles[:, :2], axis=1) <= .03))
        flags["both_near_grasp_height_30mm"] = bool(np.all(np.abs(actual[:, 2]-handles[:, 2]) <= .03))
        flags["any_finger_contact"] = any(any(c.contacts.get(side+"_fingers", [])) for side in ("left", "right"))
        flags["closed_below_45mm"] = all(arm.gripper_command_width_m <= .045 for arm in c.arms)
        flags["release_started"] = c.contact_control.release_count > 0
        flags["carry_xy_50mm"] = (flags["lift"] and
            np.linalg.norm(position[:2]-np.asarray(c.initial_state["object_position"][:2])) >= .05)
    for name, reached in flags.items():
        if reached and name not in stages:
            stages[name] = float(elapsed)


def initial_physics_state(controller):
    """Fields used to distinguish renderer roundoff from a physical reset change."""
    import numpy as np
    c = controller
    result = {"time": np.asarray(float(c.data.time))}
    for name in ("qpos", "qvel", "ctrl", "act", "eq_active", "mocap_pos", "mocap_quat",
                 "qacc_warmstart", "qfrc_applied", "xfrc_applied"):
        if hasattr(c.data, name):
            result[name] = np.asarray(getattr(c.data, name)).copy()
    for name in ("body_pos", "body_quat", "cam_pos", "cam_quat", "cam_fovy"):
        if hasattr(c.model, name):
            result["model_"+name] = np.asarray(getattr(c.model, name)).copy()
    return result


def compare_initial_observations(reference, current, *, maximum_channel_difference=2,
                                 maximum_changed_fraction=1e-4):
    """Strict state identity, bounded rasterization noise, never a model tolerance.

MuJoCo/OpenGL can vary a few 8-bit channels even without stepping physics.
Two levels in at most 0.01% of channels per camera are tolerated. Camera/state
changes and broader image differences remain failures. Raw observations stay
untouched, and both byte hashes plus per-camera differences are recorded.
"""
    import numpy as np
    cameras = tuple(name for name in ("top", "front", "side", "oblique") if name in reference)
    if cameras not in (("top", "front", "oblique"), ("oblique",),
                       ("top", "front", "side"), ("top", "front", "side", "oblique")):
        raise ValueError("Initial comparison requires recorded review views and optional oblique")
    if set(reference) != {"state", *cameras} or set(current) != set(reference):
        raise ValueError("Initial observation fields differ")
    report = {"version": "papera-initial-raster-tolerance-v1",
              "state_exact": np.array_equal(reference["state"], current["state"]),
              "maximum_channel_difference": maximum_channel_difference,
              "maximum_changed_fraction": maximum_changed_fraction, "cameras": {}}
    for name in cameras:
        a, b = np.asarray(reference[name]), np.asarray(current[name])
        if a.shape != (480, 640, 3) or b.shape != a.shape or a.dtype != np.uint8 or b.dtype != np.uint8:
            raise ValueError("Initial camera observation shape/type differs")
        delta = np.abs(a.astype(np.int16)-b.astype(np.int16))
        count, maximum = int(np.count_nonzero(delta)), int(delta.max())
        fraction = count/delta.size
        report["cameras"][name] = {
            "reference_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "current_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
            "max_channel_difference": maximum, "mean_absolute_difference": float(delta.mean()),
            "changed_channels": count, "changed_fraction": fraction,
            "within_tolerance": maximum <= maximum_channel_difference and fraction <= maximum_changed_fraction,
        }
    report["compatible"] = bool(report["state_exact"] and
                                 all(row["within_tolerance"] for row in report["cameras"].values()))
    return report


def run_case(job):
    import numpy as np
    import mujoco
    from rollout.executor import PolicyExecutor
    from rollout.policy import InvalidPolicyPrediction
    from rollout.protocol import SuccessMonitor
    from rollout.action_chunks import ActionChunkScheduler, InvalidActionChunk
    c, renderer, policy, settings = (_WORKER[k] for k in ("controller", "renderer", "policy", "settings"))
    seed, output, viewer_enabled = job["seed"], Path(job["output"]), job.get("viewer", False)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    c.reset(seed=seed)
    initial_state = c.initial_state.copy()
    executor = PolicyExecutor(c)
    if job.get("demonstration_start"):
        if not job.get("engineering_only"):
            raise ValueError("Demonstration initialization is not a benchmark case")
        restoration = restore_demonstration_start(c, job["demonstration_start"])
        initial_state.update(demonstration_initialization=restoration,
                             object_position=c.data.xpos[c.object_id].tolist())
    elif settings.get('initial_settle_seconds', 0.):
        c.set_episode_timeout_enabled(False)
        neutral = c.action_vector_m().copy()
        for _ in range(round(settings['initial_settle_seconds'] * settings['control_hz'])):
            result = executor.apply(neutral, dt=1/settings['control_hz'])
            c.step(1/settings['control_hz'])
            if not result['accepted'] or c.task_state.value == 'FAIL':
                raise RuntimeError('Initial neutral physics settling failed before policy execution')
        c.set_episode_timeout_enabled(True)
        initial_state['reset_object_position'] = initial_state['object_position']
        initial_state['object_position'] = c.data.xpos[c.object_id].tolist()
        initial_state['settled_at_sim_time_s'] = float(c.data.time)
    from tasks.registry import hook
    monitor = hook(c.config, "rollout_monitor")(c.config, initial_state, min_lift_m=settings["minimum_lift_m"],
                             minimum_open_fraction=settings.get("success_open_fraction", 1.0),
                             success_criterion=settings.get("success_criterion"))
    c.start_task_timer()
    simulation_started = float(c.data.time)
    viewer = None
    policy_cameras = getattr(policy, "metadata", {}).get("cameras", ["top", "front", "side"])
    render_camera_names({'cameras': policy_cameras}, settings)
    policy_only = settings.get('media_mode', 'all_views') == 'policy_only'
    primary_cameras = ('oblique',) if policy_only else (
        ('top', 'front', 'oblique') if 'oblique' in policy_cameras else ('top', 'front', 'side'))
    video_path = output / ('oblique.mp4' if policy_only else 'three_views.mp4')
    video = PanelVideo(video_path, settings["video_fps"], settings["video_crf"], cameras=primary_cameras)
    observation_video = (PanelVideo(output/"oblique.mp4", settings["video_fps"], settings["video_crf"],
                                   cameras=("oblique",)) if policy_cameras == ["oblique"] and not policy_only else None)
    timing = {'render': 0., 'video_encode': 0., 'control_physics_log': 0., 'video_decode': 0.}

    def capture_frame():
        start = time.perf_counter()
        images = renderer.render(c.data)
        timing['render'] += time.perf_counter()-start
        start = time.perf_counter()
        video.append(images)
        if observation_video is not None:
            observation_video.append(images)
        timing['video_encode'] += time.perf_counter()-start
        return images
    observations, predictions, states, video_times, control_times = [], [], [], [], []
    query_mask, query_origin, selected_offset, selected_actions = [], [], [], []
    chunk_size = getattr(getattr(getattr(policy, "model", None), "config", None), "chunk_size", settings.get("chunk_size", 30))
    scheduler = ActionChunkScheduler.from_settings(settings, chunk_size=chunk_size)
    stages = {}
    accepted, qpos, qvel, controls, forces, grip_forces, objects, rejects = [], [], [], [], [], [], [], []
    diagnostics_path = output / "control.jsonl.gz"
    inference_times, monitor_state = [], {}
    failure, error, control_steps = "", None, 0
    done_since = None
    rejected_count = intervention_count = any_projection_count = 0
    initial_state["actual_qpos"] = c.data.qpos.tolist()
    initial_state["model_fields"] = {"nq": c.model.nq, "nv": c.model.nv, "nu": c.model.nu,
        "joints": [{"name": mujoco.mj_id2name(c.model,mujoco.mjtObj.mjOBJ_JOINT,i),
                    "qpos_adr": int(c.model.jnt_qposadr[i]), "dof_adr": int(c.model.jnt_dofadr[i])}
                   for i in range(c.model.njnt)]}
    write_json(output / "initial_state.json", initial_state)
    initial_physics = initial_physics_state(c)
    np.savez_compressed(output / "initial_physics.npz", **initial_physics)
    if _WORKER.get('runtime'):
        write_json(output / 'runtime.json', _WORKER['runtime'])
    try:
        if viewer_enabled:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(c.model, c.data)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = c.model.camera("top").id
        with gzip.open(diagnostics_path, "wt", encoding="utf-8") as log:
            for frame_index in range(int(math.ceil(job["max_seconds"] * settings["observation_hz"]))):
                if viewer is not None and not viewer.is_running():
                    raise KeyboardInterrupt("Preview viewer closed by the operator")
                images = capture_frame()
                observation_time = float(c.data.time)
                video_times.append(observation_time)
                robot_state = np.asarray(c.state_vector(), dtype=np.float32)
                if frame_index == 0:
                    np.savez_compressed(output/'first_observation.npz', state=robot_state, **images)
                    if job.get("initial_reference_directory"):
                        if not job.get("engineering_only"):
                            raise ValueError("Initial comparison references are engineering-only")
                        reference = Path(job["initial_reference_directory"])
                        with np.load(reference/"first_observation.npz", allow_pickle=False) as archive:
                            previous = {key: archive[key] for key in archive.files}
                        comparison = compare_initial_observations(previous, dict(state=robot_state, **images))
                        with np.load(reference/"initial_physics.npz", allow_pickle=False) as archive:
                            comparison["physics_exact"] = (set(archive.files) == set(initial_physics)
                                and all(np.array_equal(archive[key], initial_physics[key]) for key in archive.files))
                        comparison["initial_metadata_exact"] = read_json(reference/"initial_state.json") == initial_state
                        comparison["reference_directory"] = str(reference.resolve())
                        comparison["compatible"] &= comparison["physics_exact"] and comparison["initial_metadata_exact"]
                        write_json(output/"initial_comparison.json", comparison)
                        if not comparison["compatible"]:
                            raise ValueError("Execution comparison initial physics/camera mismatch; see initial_comparison.json")
                query_elapsed = [0.]
                def predict_current():
                    infer_started = time.perf_counter()
                    chunk = policy.predict({name: images[name] for name in policy_cameras}, robot_state)
                    query_elapsed[0] = (time.perf_counter()-infer_started)*1000
                    return chunk
                try:
                    target, schedule = scheduler.next_action(frame_index, predict_current)
                except (InvalidPolicyPrediction, InvalidActionChunk) as exc:
                    failure = "invalid_policy_prediction"
                    write_json(output/'invalid_prediction.json', {'error': str(exc), 'time_s': observation_time})
                    break
                inference_times.append(query_elapsed[0])
                observations.append(observation_time)
                states.append(robot_state)
                # Preserve the per-observation array layout. Non-query rows
                # repeat the latest chunk and are explicitly identified.
                predictions.append(scheduler.latest_chunk)
                query_mask.append(schedule["queried"])
                query_origin.append(schedule["query_origin"])
                selected_offset.append(schedule["selected_offset"])
                selected_actions.append(target.copy())
                terminal = False
                control_started = time.perf_counter()
                for substep in range(2):
                    if done_since is None:
                        result = executor.apply(target, dt=1/settings["control_hz"])
                    else:
                        result = {"accepted": True, "reason": "post_done_settle_only", "model_action_applied": False}
                    c.step(dt=1/settings["control_hz"])
                    control_steps += 1
                    monitor_state = monitor.update(c)
                    update_stage_progress(c, monitor_state, stages, float(c.data.time)-simulation_started)
                    rejected_count += int(not result.get("accepted", False))
                    effective = np.asarray(c.action_vector_m(), dtype=float)
                    material, residual = measure_intervention(target, effective, settings)
                    intervention = done_since is None and (not result.get('accepted', False) or material)
                    any_projection = done_since is None and bool(result.get('intervened', False))
                    intervention_count += int(intervention)
                    any_projection_count += int(any_projection)
                    result.update(observation_index=frame_index, control_substep=substep,
                                  time_s=float(c.data.time), task_state=c.task_state.value,
                                  raw_model_target=target.tolist(), actual_accepted_target=effective.tolist(),
                                  action_schedule=schedule,
                                  intervention=intervention, any_projection=any_projection,
                                  issued_target_residual_wrapped=residual, success_evidence=monitor_state)
                    log.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    control_times.append(float(c.data.time))
                    accepted.append(effective)
                    qpos.append(c.data.qpos.copy()); qvel.append(c.data.qvel.copy()); controls.append(c.data.ctrl.copy())
                    objects.append(c.data.qpos[c.object_qpos_adr:c.object_qpos_adr+7].copy())
                    forces.append([*c.contacts["finger_normal_force_n"]["left"], *c.contacts["finger_normal_force_n"]["right"]])
                    grip_forces.append([*c.contacts["left_grip_forces_n"], *c.contacts["right_grip_forces_n"]])
                    rejects.append(not result.get("accepted", False))
                    if monitor_state.get("task_success"):
                        terminal = True
                    elif c.task_state.value == "FAIL":
                        failure, terminal = c.failure_reason or "physics_failure", True
                    elif c.task_state.value == "DONE":
                        # Measured-open settling can lag the legacy command-open
                        # predicate. Allow physics to settle existing goals only.
                        if monitor_state.get("terminal"):
                            failure, terminal = monitor_state.get("reason") or "controller_done_without_full_success_evidence", True
                        else:
                            if done_since is None:
                                done_since = float(c.data.time)
                            if float(c.data.time)-done_since >= settings["post_done_settle_seconds"]:
                                failure, terminal = "placement_physical_conditions_not_met", True
                    if terminal:
                        break
                timing['control_physics_log'] += time.perf_counter()-control_started
                if viewer is not None:
                    viewer.sync()
                    delay = (float(c.data.time) - simulation_started) - (time.perf_counter() - started)
                    if delay > 0:
                        time.sleep(min(delay, 1/settings["observation_hz"]))
                if terminal:
                    break
            if not monitor_state.get("task_success") and not failure:
                failure = "timeout"
            capture_frame()
            video_times.append(float(c.data.time))
    except KeyboardInterrupt:
        error = "operator_interrupted"
        raise
    except Exception:
        error = traceback.format_exc()
    finally:
        if viewer is not None:
            viewer.close()
        try:
            video.close()
            if observation_video is not None:
                observation_video.close()
        except Exception:
            error = error or traceback.format_exc()
        np.savez_compressed(output / "trajectory.npz",
            observation_time_s=np.asarray(observations), state=np.asarray(states), prediction_chunks=np.asarray(predictions),
            prediction_query_mask=np.asarray(query_mask, dtype=bool),
            prediction_query_origin=np.asarray(query_origin, dtype=np.int64),
            selected_chunk_offset=np.asarray(selected_offset, dtype=np.int64),
            selected_action=np.asarray(selected_actions),
            video_time_s=np.asarray(video_times), control_time_s=np.asarray(control_times), accepted_target=np.asarray(accepted),
            qpos=np.asarray(qpos), qvel=np.asarray(qvel), ctrl=np.asarray(controls), object_pose_wxyz=np.asarray(objects),
            finger_forces_n=np.asarray(forces), finger_opposed_grip_forces_n=np.asarray(grip_forces),
            rejected=np.asarray(rejects), inference_ms=np.asarray(inference_times),
            body_pos=c.model.body_pos.copy(), body_quat=c.model.body_quat.copy())
        if error:
            write_json(output / "error.json", {"seed": seed, "error": error, "infrastructure_error": True})
    if error:
        raise RuntimeError(f"Infrastructure error in seed {seed}; see {output / 'error.json'}")
    decode_started = time.perf_counter()
    media = decode_video(video_path, video.frames, width=640*len(primary_cameras))
    if observation_video is not None or policy_only:
        observation_media = media if policy_only else decode_video(output/"oblique.mp4", observation_video.frames, width=640)
        write_json(output/"observation_video.json", dict(observation_media, camera="oblique",
            time_index="video_index.json",
            note="Audit video of model RGB before input resize; encoding is not fed into the model"))
    timing['video_decode'] = time.perf_counter()-decode_started
    timing['inference'] = sum(inference_times)/1000
    # Frame zero is the initial observation; later frames refer to the last
    # completed control sample at/before their physics timestamp.
    video_control_index = np.searchsorted(np.asarray(control_times), np.asarray(video_times)+1e-9, side='right')-1
    write_json(output/'video_index.json', {'video_frames':video.frames, 'fps':settings['video_fps'],
        'physics_time_s':video_times, 'last_completed_control_index':video_control_index.tolist(),
        'camera_order':list(primary_cameras), 'initial_frame_control_index':-1})
    query_latencies = np.asarray(inference_times)[np.asarray(query_mask, dtype=bool)]
    row = {"seed": seed, "success": bool(monitor_state.get("task_success")),
           "failure_reason": failure, "sim_seconds": float(c.data.time)-simulation_started,
           "wall_seconds": time.perf_counter()-started, "observations": len(observations), "control_steps": control_steps,
           "rejected_actions": rejected_count, "intervention_steps": intervention_count,
           "any_projection_steps": any_projection_count,
           "intervention_fraction": intervention_count/max(1,control_steps),
           "model_queries": scheduler.query_count,
           "execution": scheduler.mode, "n_action_steps": scheduler.n_action_steps,
           "stage_first_seconds": stages,
           "mean_inference_ms": float(np.mean(query_latencies)) if len(query_latencies) else None,
           "p95_inference_ms": float(np.quantile(query_latencies,.95)) if len(query_latencies) else None,
           "final_object_position": c.data.xpos[c.object_id].tolist(), "target_position": c.target_position.tolist(),
           "final_xy_error_m": float(np.linalg.norm(c.data.xpos[c.object_id][:2]-c.target_position[:2])),
           "final_orientation_error_rad": monitor_state.get('target_error', {}).get('orientation_rad'),
           "final_z_error_m": monitor_state.get('target_error', {}).get('z_m'),
           "collision_count": int(c.metrics.collision_count), "contact_loss_count": int(c.metrics.contact_loss_count),
           "evidence": monitor_state, "media": media, "video": str(video_path.resolve()),
           "timing_seconds": timing, "render_cameras": list(render_camera_names({'cameras':policy_cameras}, settings)),
           "trajectory_sha256": sha(output/'trajectory.npz'), "completed_at": now(), "infrastructure_error": False}
    if observation_video is not None or policy_only:
        row.update(observation_video=str((output/'oblique.mp4').resolve()), observation_media=observation_media)
    write_json(output / "result.json", row)
    return row


def wilson(successes, count):
    if not count:
        return None
    z, p = 1.959963984540054, successes/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    radius = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [max(0.,center-radius), min(1.,center+radius)]


def summarize(output, metadata, rows, errors=None):
    output = Path(output)
    found = [row['seed'] for row in rows]
    if len(set(metadata['seeds'])) != len(metadata['seeds']) or len(set(found)) != len(found):
        raise ValueError('Duplicate planned/result seeds cannot count as independent trials')
    if not set(found).issubset(metadata['seeds']):
        raise ValueError('Result contains a seed outside the frozen roster')
    if any(row.get('infrastructure_error', False) for row in rows):
        raise ValueError('Infrastructure failures cannot be counted as model outcomes')
    rows = sorted(rows, key=lambda row: metadata["seeds"].index(row["seed"]))
    successes = sum(row["success"] for row in rows)
    planned, count = len(metadata["seeds"]), len(rows)
    complete = count == planned and not errors
    failures = {}
    for row in rows:
        if not row["success"]:
            failures[row["failure_reason"]] = failures.get(row["failure_reason"], 0) + 1
    summary = {"version": VERSION, "updated_at": now(), "mode": metadata["mode"], "complete": complete,
        "planned_episodes": planned, "completed_episodes": count, "successes": successes,
        "failures": count-successes, "task_success_rate": successes/planned if complete else None,
        "provisional_completed_case_rate": successes/count if count else None,
        "wilson95_given_checkpoint": wilson(successes,count) if complete else None,
        "interval_assumption": "independent initial-state draws for this fixed checkpoint, not training-seed uncertainty",
        "failure_reasons": failures, "infrastructure_errors": errors or [], "run": str(output.resolve()),
        "checkpoint": metadata["checkpoint"], "protocol_signature": metadata.get("protocol_signature"),
        "human_quality_review": "pending", "formal_paper_result": False}
    write_json(output / "summary.json", summary)
    fields = ("seed","success","failure_reason","sim_seconds","wall_seconds","observations","control_steps",
              "final_xy_error_m","final_orientation_error_rad","final_z_error_m","collision_count","contact_loss_count","rejected_actions","intervention_steps",
              "intervention_fraction","mean_inference_ms","p95_inference_ms","video")
    with (output/'episodes.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    title = "Paper A · 50次自主评估" if metadata["mode"] == "batch" else "Paper A · 工程演示（不计入50次）"
    camera_note = "单次工程演示不计入固定50次结果；相机沿用训练来源配置。"
    if metadata['mode'] == 'batch':
        camera_note = "相机视野检查详见本次protocol.json；静态视野检查不能证明动态无遮挡。"
        if (output / 'protocol.json').is_file():
            preflight = read_json(output / 'protocol.json').get('preflight', {}).get('summary', {})
            clipped = preflight.get('flag_counts', {}).get('all_object_waypoints_in_all_frusta')
            if clipped is not None:
                camera_note = (f"本次{preflight['case_count']}个初态中，{clipped}个的静态物体／抓柄检查点"
                               "超出至少一路相机视野；全部保留，不依据模型表现筛选。静态检查不代表动态无遮挡。")
    score = f"{successes}/{planned}（{100*successes/planned:.1f}%）" if complete else f"尚未完成：{count}/{planned}，当前成功{successes}"
    table = ''.join(f'<tr data-success="{int(r["success"])}"><td>{r["seed"]}</td><td>{"成功" if r["success"] else "失败"}</td>'
                    f'<td>{html.escape(r["failure_reason"])}</td><td>{r["sim_seconds"]:.2f}</td><td>{1000*r["final_xy_error_m"]:.1f}</td>'
                    f'<td>{r["intervention_fraction"]:.1%}</td><td><a href="episodes/seed_{r["seed"]}/{html.escape(Path(r["video"]).name, quote=True)}">播放视频</a></td></tr>' for r in rows)
    open_fraction = metadata.get('settings', {}).get('success_open_fraction', 1.)
    opening_text = (f"双爪命令与实测开度均达到各自最大值的{100*open_fraction:g}%"
                    f"（标准Panda最大开度80mm时不低于{80*open_fraction:g}mm）")
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{title}</title><style>
body{{font:16px/1.6 system-ui;margin:32px auto;max-width:1300px;padding:0 24px;color:#183047;background:#f7f9fc}}
table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}
.card{{background:white;padding:20px;margin:16px 0;border-radius:10px}}select{{font:inherit;padding:5px}}
</style><h1>{title}</h1><div class="card"><h2>{score}</h2><p>模型：{html.escape(str(metadata['run']))}<br>
checkpoint：{html.escape(metadata['checkpoint'])} · 30 Hz观察，60 Hz执行，{metadata['max_seconds']:g}秒仿真时间上限。</p>
<p>成功要求真实双爪抓持并抬高至少2厘米、到达目标、{html.escape(opening_text)}、脱离手指接触并支撑停稳。未经人工干预；单个训练种子的探索结果。</p>
<p>新单视角模式仅保存oblique；历史三视图依次为顶视、内侧近景、侧视。视频是实际模型执行；目标误差不是预测离线误差。
干预率统计拒绝指令，或限速／投影后任一位置、爪宽变化达到0.1毫米、角度达到0.001弧度的控制步；角度等价表示不重复计数。不是人工接管。失败、视频与轨迹全部保留。</p>
<p>{html.escape(camera_note)}</p></div>
<p>{'本批含加速前的已完成案例；继承范围和执行版本见continuation.json，原始结果未重算。' if metadata.get('continuation') else ''}</p>
<label>显示 <select onchange="document.querySelectorAll('tr[data-success]').forEach(r=>r.hidden=this.value!='all'&&r.dataset.success!=this.value)">
<option value="all">全部</option><option value="1">成功</option><option value="0">失败</option></select></label>
<table><thead><tr><th>初态种子</th><th>结果</th><th>失败原因</th><th>仿真秒</th><th>最终物体XY误差mm</th><th>控制干预率</th><th>视频</th></tr></thead><tbody>{table}</tbody></table>
<p>精确统计：summary.json；逐次结果：episodes.csv；原始预测、接受目标与物理状态：episodes/seed_*/trajectory.npz；控制原因：control.jsonl.gz。</p>
<p>基础设施错误数：{len(errors or [])}。未完成时不填充正式50次成功率。区间仅针对该固定checkpoint，不代表不同训练种子的波动。</p></html>'''
    (output/'report.html').write_text(report,encoding='utf-8')
    return summary


@contextmanager
def evaluation_lock():
    import msvcrt
    lock_path=ROOT/'.local/locks/rollout.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as error:raise RuntimeError('Another evaluation is running; do not start duplicate GPU jobs') from error
        try:yield
        finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)


def verify_saved_case(directory, seed):
    directory = Path(directory)
    row = read_json(directory/'result.json')
    if row['seed'] != seed or row.get('infrastructure_error'):
        raise ValueError('Saved case seed/error mismatch')
    name = Path(row['video']).name
    if name not in {'three_views.mp4', 'oblique.mp4'}:
        raise ValueError('Unexpected saved case video')
    if (sha(directory/'trajectory.npz') != row['trajectory_sha256']
            or sha(directory/name) != row['media']['sha256']):
        raise ValueError('Saved completed case artifacts changed')
    if row.get('observation_media') and sha(directory/'oblique.mp4') != row['observation_media']['sha256']:
        raise ValueError('Saved observation video changed')
    return row


def validate_continuation(old, current):
    """Explicit mixed-run provenance, never an exact-resume override.

    Only the media/orchestration module may change. Inference, scheduler,
    executor, scoring, all physics modules and assets must match byte-for-byte.
    The new protocol still independently verifies its complete current code.
    """
    for key in ('version', 'mode', 'run', 'checkpoint', 'scene', 'seeds',
                'max_seconds', 'device', 'workers'):
        if old[key] != current[key]:
            raise ValueError(f'Continuation mismatch: {key}')
    if current['mode'] != 'batch':
        raise ValueError('Continuation is only supported for batch')
    def relocated_policy(policy):
        # A Git worktree relocates verified source paths, not their bytes.
        # All artifact paths, cache counts and hashes still compare exactly.
        normalized = dict(policy)
        source_files = {'training/papera/'+name for name in ('act_model.py', 'data.py', 'pack_data.py', 'train.py')}
        source_files.add('dataset/task3/training_pack.py')
        hashes = {}
        for path, value in policy.get('checked_hashes', {}).items():
            suffix = '/'.join(Path(path).parts[-3:])
            key = 'source:'+suffix if suffix in source_files else path
            if key in hashes:
                raise ValueError('Duplicate relocated source identity')
            hashes[key] = value
        if 'checked_hashes' in policy:
            normalized['checked_hashes'] = hashes
        return normalized
    if relocated_policy(old['policy']) != relocated_policy(current['policy']):
        raise ValueError('Continuation checkpoint/input provenance changed')
    def scientific(settings):
        # Pack identity is checked through the full policy provenance above.
        return {k: v for k, v in settings.items()
                if k not in {'media_mode', 'preview_headless', 'pack', 'protocol_path', 'output_root', '_help'}}
    if scientific(old['settings']) != scientific(current['settings']):
        raise ValueError('Continuation scientific settings changed')
    if (old['settings'].get('media_mode', 'all_views') != 'all_views'
            or current['settings'].get('media_mode') != 'policy_only'):
        raise ValueError('Continuation only supports all_views to policy_only')
    changed = {key for key in set(old['code']) | set(current['code'])
               if old['code'].get(key) != current['code'].get(key)}
    if changed != {'rollout/rollout.py'}:
        raise ValueError(f'Continuation changed non-media code or missing rollout change: {sorted(changed)}')


def import_completed_cases(parent, output, metadata):
    """Copy and hash all completed evidence; do not edit or move the parent."""
    parent, output = Path(parent).resolve(), Path(output).resolve()
    old = read_json(parent/'run.json')
    validate_continuation(old, metadata)
    protocol = read_json(parent/'protocol.json')
    signature = protocol.pop('signature')
    if (canonical(protocol) != signature or signature != old['protocol_signature']
            or protocol['code'] != old['code'] or protocol['settings'] != old['settings']
            or protocol['seeds'] != old['seeds']):
        raise ValueError('Parent run/protocol integrity mismatch')
    imported = []
    for source in sorted((parent/'episodes').glob('seed_*/result.json')):
        seed = read_json(source)['seed']
        if seed not in metadata['seeds'] or source.parent.name != f'seed_{seed}':
            raise ValueError('Foreign completed case in continuation')
        row = verify_saved_case(source.parent, seed)
        destination = output/'episodes'/source.parent.name
        hashes = {p.relative_to(source.parent).as_posix(): sha(p)
                  for p in source.parent.rglob('*') if p.is_file()}
        shutil.copytree(source.parent, destination)
        if any(sha(destination/name) != value for name, value in hashes.items()):
            raise ValueError('Continuation copy hash mismatch')
        row['video'] = str(destination/Path(row['video']).name)
        if row.get('observation_video'):
            row['observation_video'] = str(destination/'oblique.mp4')
        row['inherited_from'] = str(source.parent)
        write_json(destination/'result.json', row)
        imported.append({'seed': seed, 'source': str(source.parent), 'source_hashes': hashes})
    evidence = {'parent': str(parent), 'parent_run_sha256': sha(parent/'run.json'),
                'parent_protocol_sha256': sha(parent/'protocol.json'),
                'old_code': old['code'], 'new_code': metadata['code'], 'imported': imported,
                'created_at': now(), 'kind': 'explicit_media_only_continuation',
                'note': 'Mixed execution versions; completed cases copied without rerunning or outcome selection. Not an exact resume or bitwise-equivalence claim.'}
    write_json(output/'continuation.json', evidence)
    return {'parent': str(parent), 'imported_seeds': [r['seed'] for r in imported],
            'manifest': 'continuation.json', 'manifest_sha256': sha(output/'continuation.json')}


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True)
    parser.add_argument('--pack',type=Path,help='prepare only: exact verified Pack for both experimental groups')
    parser.add_argument('--engineering',action='store_true',help='Allow completed smoke/overfit model for check or one short preview only')
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG, help='评估配置；默认使用 Task Manifest 的 evaluation_config，单 oblique 输入')
    parser.add_argument('--mode',choices=('check','prepare','preview','batch','summarize'),default='preview',
                        help='check只查来源；prepare固定50初态；preview单次仿真；batch跑50次；summarize重建汇总')
    parser.add_argument('--run',type=Path,help='本次训练目录，包含run.json和checkpoints；不是.pt文件，也不是评估输出目录')
    parser.add_argument('--checkpoint',choices=('best','last'),default='best', help='默认best按验证误差选模；batch只用best')
    parser.add_argument('--output',type=Path, help='新评估输出目录；summarize时指已有评估目录')
    parser.add_argument('--protocol',type=Path, help='固定50次清单；默认读取配置中的protocol_path')
    parser.add_argument('--workers',type=int,default=1, help='batch并行进程数1至4；不是任务次数，preview固定1')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda', help='模型推理设备；本机使用cuda')
    parser.add_argument('--headless',action='store_true',help='preview不弹MuJoCo窗口，仍保存视频；batch默认如此')
    parser.add_argument('--seed',type=int,help='仅preview：独立工程初态种子，不得与示范或固定50次重复')
    parser.add_argument('--max-seconds',type=float,help='仅preview：单次仿真时间上限；batch使用配置max_seconds并冻结到协议（当前60秒）')
    parser.add_argument('--resume',type=Path,help='已有评估输出目录；条件完全相同时续跑未完成任务，已完成任务不重跑')
    parser.add_argument('--continue-from', type=Path, help='显式新建加速续跑：仅允许all_views转policy_only、仅rollout媒体编排模块变化；保留父批次和混合版本来源')
    args=parser.parse_args(argv)
    from tasks.registry import get
    task=get(args.task)
    settings=read_json(resolve(task.manifest['evaluation_config'] if args.config == DEFAULT_CONFIG else args.config))
    settings.update(task_id=task.id, task_binding=task.binding(), scenario=task.manifest['scene_id'],
                    protocol_path=f'outputs/rollout/{task.id}/protocols/default/protocol.json',
                    output_root=f'outputs/rollout/{task.id}')
    if args.run:
        frozen_run=read_json(resolve(args.run)/'run.json')['provenance']
        if frozen_run.get('task_id')!=task.id or frozen_run.get('task_binding')!=task.binding():
            raise ValueError('Checkpoint/task mismatch')
        settings['pack']=frozen_run['pack']
    if args.pack: settings['pack']=str(args.pack.resolve())
    if args.mode=='prepare' and not args.pack: parser.error('prepare requires --pack')
    if args.mode=='prepare':
        from tasks.artifacts import check_artifact
        check_artifact(args.pack,task.id,False)
    if args.engineering and args.mode not in ('check','preview'):
        parser.error('engineering only supports check or short preview')
    if args.engineering and args.mode=='preview' and (args.max_seconds is None or args.max_seconds>10):
        parser.error('engineering preview requires --max-seconds <=10')
    validate_settings(settings)
    if args.continue_from and (args.resume or args.mode != 'batch'):
        parser.error('--continue-from requires a new batch, not --resume')
    if args.mode=='prepare':
        prepare_protocol(settings,args.protocol or settings['protocol_path']);return 0
    if args.mode=='summarize':
        if args.output is None:parser.error('summarize requires --output')
        output=resolve(args.output);metadata=read_json(output/'run.json')
        if metadata['settings'].get('task_id')!=task.id:raise ValueError('Cross-task evaluation summary rejected')
        rows=[read_json(p) for p in (output/'episodes').glob('seed_*/result.json')]
        errors = read_json(output/'status.json').get('errors', []) if (output/'status.json').exists() else []
        print(json.dumps(summarize(output,metadata,rows,errors),ensure_ascii=False,indent=2));return 0
    if not 1<=args.workers<=4:parser.error('Use 1 to 4 workers')
    if args.mode=='preview' and args.workers!=1:parser.error('Preview uses one worker')
    if args.mode=='batch' and (args.seed is not None or args.max_seconds is not None):
        parser.error('Batch seeds and timeout come from the frozen protocol')
    if args.mode=='batch' and args.checkpoint!='best':
        parser.error('This frozen comparison selects best by validation loss, not last or rollout performance')
    if args.mode=='check' and (args.output or args.resume or args.seed is not None or args.max_seconds is not None):
        parser.error('check只检查模型与来源，不接受仿真seed、时长或输出/续跑目录')
    with evaluation_lock():
        from rollout.policy import verify_run_provenance
        from rollout.protocol import load_scene_provenance
        selected_run = args.run or settings.get('baseline_run')
        if not selected_run:
            raise ValueError('缺少模型目录：请加 --run "outputs/training/<task_id>/<variant>/<run_id>"。'
                             '该目录须包含run.json、completion.json和checkpoints，不是单个.pt文件。')
        run_path=resolve(selected_run)
        try:
            protocol=verify_protocol(resolve(args.protocol or settings['protocol_path']),settings) if args.mode=='batch' else None
        except FileNotFoundError:
            if not resolve(args.protocol or settings['protocol_path']).is_file():
                raise ValueError('50次评估清单尚未建立。先在同一入口运行 --mode prepare；'
                                 '再运行 --mode batch --run "本次训练目录"。') from None
            raise
        print('Checking checkpoint, frozen training inputs and cache hashes ...',flush=True)
        frozen=read_json(run_path/'run.json')['provenance']
        if frozen.get('task_id')!=task.id or frozen.get('task_binding')!=task.binding():
            raise ValueError('Checkpoint/task mismatch')
        evaluation_output_root=f"outputs/rollout/{task.id}/{frozen['variant']}"
        if args.engineering:evaluation_output_root=f".local/checks/rollout/{task.id}/{frozen['variant']}"
        provenance=verify_run_provenance(run_path,args.checkpoint,verify_cache=True,allow_overfit=args.engineering)
        # A new single-view model binds its own explicit Pack. The old default
        # evaluation JSON describes a historical three-view Pack and is not a
        # valid source override for check/preview. Formal batches still require
        # the exact independently frozen protocol and its chosen Pack.
        if protocol is None and provenance.get('data_contract') == 'papera-safe-stitch-act-v1':
            settings = dict(settings, pack=provenance['pack'])
        source=protocol['source'] if protocol else load_scene_provenance(resolve(settings['pack']))
        if args.mode == 'preview':
            preview_seed = args.seed if args.seed is not None else settings['engineering_seeds'][0]
            if (preview_seed < 0 or preview_seed in settings['test_seeds']
                    or preview_seed in source.get('demonstration_seeds', [])):
                raise ValueError('preview seed必须非负，且不得与示范或固定50次种子重复；可使用910001等工程种子。')
        source_manifest=Path(source['pack_manifest_path']).resolve()
        if (Path(provenance['pack']).resolve() != source_manifest.parent
                or provenance['checked_hashes'].get(str(source_manifest)) != source['pack_manifest_sha256']):
            raise ValueError('Selected model pack differs from the frozen evaluation scene source')
        if args.mode == 'check':
            print(json.dumps({'checked': True, 'run': str(run_path),
                              'variant': provenance.get('variant'),
                              'checkpoint': args.checkpoint, 'checkpoint_step': provenance.get('checkpoint_step'),
                              'pack': str(source_manifest.parent),
                              'verified_cache_files': provenance.get('verified_cache_files'),
                              'simulations_started': 0, 'training_started': False}, ensure_ascii=False, indent=2))
            print('模型与场景来源核验通过；没有运行仿真或训练。下一步可运行 --mode preview。', flush=True)
            return 0
        seeds=protocol['seeds'] if protocol else [args.seed if args.seed is not None else settings['engineering_seeds'][0]]
        max_seconds=settings['max_seconds'] if args.max_seconds is None else args.max_seconds
        if not math.isfinite(max_seconds) or max_seconds<=0:parser.error('Invalid preview duration')
        metadata={'version':VERSION,'created_at':now(),'mode':args.mode,'run':str(run_path),'checkpoint':args.checkpoint,
            'policy':provenance,'settings':settings,'scene':source['config'],'seeds':seeds,'max_seconds':max_seconds,
            'code':implementation_hashes(),'git_commit':git_commit(),'protocol_signature':protocol['signature'] if protocol else None,
            'workers':args.workers,'device':args.device,'human_quality_review':'pending','training_seed_count':1}
        if args.resume:
            output=resolve(args.resume);old=read_json(output/'run.json')
            for key in ('version','mode','run','checkpoint','settings','scene','seeds','max_seconds','code','protocol_signature','device','workers'):
                if old[key]!=metadata[key]:raise ValueError(f'Resume mismatch: {key}')
            if old['policy']!=metadata['policy']:raise ValueError('Checkpoint/input provenance changed')
            if old.get('continuation') and sha(output/old['continuation']['manifest']) != old['continuation']['manifest_sha256']:
                raise ValueError('Continuation evidence changed')
            metadata=old
        else:
            if args.continue_from:
                validate_continuation(read_json(resolve(args.continue_from)/'run.json'), metadata)
            output=resolve(args.output or (resolve(evaluation_output_root)/f'{run_path.name.split("_seed")[0]}_{args.checkpoint}_{args.mode}_{timestamp()}'))
            output.mkdir(parents=True,exist_ok=False);(output/'episodes').mkdir()
            if args.continue_from:
                metadata['continuation'] = import_completed_cases(resolve(args.continue_from), output, metadata)
            write_json(output/'run.json',metadata)
            if protocol:write_json(output/'protocol.json',protocol)
        rows=[];jobs=[];errors=[]
        for seed in seeds:
            directory=output/'episodes'/f'seed_{seed}'
            if (directory/'result.json').exists():
                row=verify_saved_case(directory, seed)
                rows.append(row);continue
            if directory.exists():
                # Preserve interrupted evidence, with a checked destination under this run.
                archive=(output/'interrupted'/f'seed_{seed}_{timestamp()}').resolve()
                if output.resolve() not in archive.parents or output.resolve() not in directory.resolve().parents:
                    raise ValueError('Unsafe interrupted-case archive path')
                archive.parent.mkdir(exist_ok=True);directory.rename(archive)
            jobs.append({'seed':seed,'output':str(directory),'max_seconds':max_seconds,
                         'engineering_only':args.engineering, 'viewer':args.mode=='preview' and not args.headless and not settings.get('preview_headless', False)})
        summarize(output,metadata,rows)
        print(f'Output: {output}; cases: {len(jobs)} pending / {len(seeds)} planned',flush=True)
        if not jobs:
            summary = summarize(output, metadata, rows)
            write_json(output/'status.json', {'updated_at': now(), 'complete': summary['complete'],
                'completed_episodes': len(rows), 'planned_episodes': len(seeds), 'errors': []})
            print('全部任务已完成并核验，未重新初始化GPU或重复执行仿真。', flush=True)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            return 0 if summary['complete'] else 1
        init=(str(run_path),args.checkpoint,source['config'],settings,args.device,args.engineering)
        try:
            if args.workers==1:
                initialize_worker(*init)
                for job in jobs:
                    row=run_case(job);rows.append(row);summarize(output,metadata,rows)
                    print(f'[{len(rows)}/{len(seeds)}] seed={row["seed"]} success={row["success"]} reason={row["failure_reason"]} sim={row["sim_seconds"]:.2f}s',flush=True)
            else:
                pool = ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                                           initializer=initialize_worker,initargs=init)
                futures = {}
                try:
                    for job in jobs:
                        futures[pool.submit(run_case,job)] = job['seed']
                    for future in as_completed(futures):
                        try:
                            row=future.result();rows.append(row)
                            print(f'[{len(rows)}/{len(seeds)}] seed={row["seed"]} success={row["success"]} reason={row["failure_reason"]} sim={row["sim_seconds"]:.2f}s',flush=True)
                        except Exception:
                            errors.append({'seed':futures[future],'traceback':traceback.format_exc()})
                            print(f'Infrastructure error for seed {futures[future]}; preserved, no automatic retry',flush=True)
                        summarize(output,metadata,rows,errors)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    # Only processes created and owned by this pool are stopped.
                    # This CPython 3.12 environment lacks Executor.terminate_workers.
                    owned_processes = list((pool._processes or {}).values())
                    for process in owned_processes:
                        if process.is_alive():
                            process.terminate()
                    for process in owned_processes:
                        process.join(timeout=10)
                    pool.shutdown(wait=True,cancel_futures=True)
                    raise
                else:
                    pool.shutdown(wait=True)
        except KeyboardInterrupt:
            errors.append({'kind':'operator_interrupted', 'message':'Partial artifacts preserved; resume unchanged run.'})
            raise
        except Exception:
            errors.append({'kind':'evaluation_exception', 'traceback':traceback.format_exc()})
            raise
        finally:
            summary=summarize(output,metadata,rows,errors)
            write_json(output/'status.json',{'updated_at':now(),'complete':summary['complete'],
                'completed_episodes':len(rows),'planned_episodes':len(seeds),'errors':errors})
            if _WORKER.get('renderer'):_WORKER['renderer'].close();_WORKER.clear()
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
        return 0 if summary['complete'] else 1


def cli(argv=None):
    """Readable expected-error boundary; case-level runtime traces stay in outputs."""
    try:
        return main(argv)
    except (OSError, ValueError) as error:
        print(f'评估未启动或未完成：{error}', file=sys.stderr, flush=True)
        return 2


if __name__=='__main__':
    raise SystemExit(cli())
