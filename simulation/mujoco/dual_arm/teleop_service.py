"""Local Task3 teleoperation, research logging and outcome-separated recording."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from flask import Flask, Response, jsonify, request
from werkzeug.serving import WSGIRequestHandler, make_server

from .config import load_config
from .service_identity import backend_identity
from .task_controller import DualArmTaskController
from .recording_policy import physical_change
from .measurement_journal import MeasurementJournal
from .hand_measurement import inspect_measurement, HandInputGuard
from .task_scenes import scenario_config
from .command_protocol import CommandGate, GUARDED_COMMANDS, PROTOCOL
from .session_diagnostics import SessionDiagnostics
from .recording_timeline import capture_paper, paper_metadata, terminal_snapshot, compact_event
from .render_workers import CAMERAS, STATE_NAMES, ACTION_NAMES, recording_worker, preview_worker, sync_operator_view
from .training_camera import recording_camera_names, training_camera_metadata

ROOT = Path(__file__).resolve().parents[3]


def validate_recording_initial_state(status, paper_recording):
    """Recording is independent of current hand visibility, not grasp history."""
    contact = status.get('physical_contact') or {}
    level = paper_recording and contact.get('gripper_trigger_policy') == 'palm_fist_level_v2'
    if level:
        empty = (contact.get('state') == 'FREE' and contact.get('ever_dual') is False
            and all(len(status.get('contacts', {}).get(f'{s}_fingers', [])) == 2
                and all(v is False for v in status['contacts'][f'{s}_fingers']) for s in ('left', 'right')))
        if status.get('task_state') != 'PREGRASP' or not empty:
            raise ValueError('Start before grasping with no finger contact; reset the scene before recording')
    else:
        is_open = (all(status['grippers'][s]['command_width_m'] >= .079
            and status['grippers'][s]['actual_width_m'] >= .079 for s in ('left', 'right'))
            if paper_recording else status.get('gripper_latch') == 'OPEN')
        if status.get('task_state') != 'PREGRASP' or not is_open:
            raise ValueError('Start a complete demonstration in PREGRASP with open grippers; reset before recording')


class FixedPhaseSampler:
    """Choose one available control step per fixed simulation-time sample slot.

    Half a control step of tolerance selects the nearest tick, avoiding the
    20 Hz drift caused by restarting a 1/30 s wait after each rounded 2 ms
    physics snapshot. Ineligible slots are skipped, never backfilled.
    """
    VERSION = "fixed_phase_v1"

    def __init__(self, fps, control_hz=60):
        if not math.isfinite(fps) or not math.isfinite(control_hz) or not 0 < fps <= control_hz:
            raise ValueError("Sampling requires 0 < fps <= control_hz")
        self.fps = float(fps)
        self.tolerance = 0.5 / control_hz
        self.origin = self.last_time = None
        self.last_slot = -1

    def sample(self, simulation_time, eligible, force=False):
        moment = float(simulation_time)
        if not math.isfinite(moment) or (self.last_time is not None and moment < self.last_time - 1e-9):
            raise ValueError("Recording simulation time must be finite and monotonic")
        if self.origin is None:
            self.origin = moment
        self.last_time = moment
        slot = math.floor((moment - self.origin + self.tolerance) * self.fps + 1e-9)
        if not force and (not eligible or slot <= self.last_slot):
            return False
        self.last_slot = max(self.last_slot, slot)
        return True


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


class QuietRequests(WSGIRequestHandler):
    def log_request(self, code="-", size="-"):
        if self.path == "/control" and str(code).isdigit() and int(code) < 400:
            return  # Structured diagnostics retain the decisions and contents.
        if not self.path.startswith(("/health", "/top-preview", "/side-preview", "/front-preview")):
            super().log_request(code, size)


class DualArmRecorder:
    def __init__(self, args):
        self.args = args
        from tasks.registry import get
        from collection.config import recording_config
        self.task = get(args.task)
        self.config = recording_config(self.task)
        self.service_identity = backend_identity(self.config,
            input_kind=getattr(args, "input_kind", "unspecified"), seed=args.seed,
            headless=getattr(args, "headless", False), previews=not args.no_previews)
        self.controller = DualArmTaskController(config=self.config)
        self.controller.reset(seed=args.seed)
        self.scene_index = 0
        if self.controller.contact_control and self.controller.contact_control.rate_control:
            self.controller.contact_control.set_input_clock(time.monotonic, "server_monotonic")
        self.controller.set_episode_timeout_enabled(False)
        self.paper_recording = bool(self.controller.continuous_task and self.config['recording'].get('schema_version') == 'task3-papera-v2')
        self.config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = Path(args.dataset_root).resolve() if args.dataset_root else ROOT / "datasets" / self.task.id / self.session
        self.roots = {"success": base, "failure": base.with_name(base.name + "_failures")}
        if any(path.exists() for path in self.roots.values()):
            raise ValueError("Choose a new dataset root; existing datasets are never overwritten")
        self.ctx = mp.get_context("spawn")
        self.record_commands = self.ctx.Queue(maxsize=120)
        self.record_replies = self.ctx.Queue()
        self.worker = self.ctx.Process(target=recording_worker, args=(self.config, self.record_commands, self.record_replies), daemon=True)
        self.worker.start()
        self.wait_reply("ready")
        self.preview_in, self.preview_out, self.preview_ready = self.ctx.Queue(maxsize=1), self.ctx.Queue(maxsize=1), self.ctx.Queue()
        self.preview_process = None
        if not args.no_previews:
            self.preview_process = self.ctx.Process(target=preview_worker, args=(self.config, self.preview_in, self.preview_out, self.preview_ready), daemon=True)
            self.preview_process.start()
            result = self.preview_ready.get(timeout=90)
            if result["type"] != "ready":
                raise RuntimeError(result.get("traceback", str(result)))
        self.commands = queue.Queue(maxsize=64)
        self.motion_lock = threading.Lock()
        self.latest_motion = None
        self.latest_telemetry = None
        self.release_input_interruption = None
        self.telemetry = {}
        self.paper_events = queue.Queue(maxsize=4096)
        self.paper_event_dropped = 0
        self.latest_observation = None
        self.action_measurement_id = self.action_event_id = None
        self.recording_calibration = None
        self.last_command_result = None
        self.last_operation_result = None
        self.operation_results = []
        self.seen_events = set()
        self.last_input_at = time.monotonic()
        self.control_epoch = 0
        self.control_epoch_started_ms = time.time() * 1000
        self.command_gate = CommandGate(self.config["control"])
        self.hand_input_guard = HandInputGuard()
        self.frames = self.saved_success = self.saved_failure = self.attempt = 0
        self.recording = self.saving = self.stop_requested = False
        self.force_frame = False
        self.event_window_until = 0.0
        self.raw_file = self.pending = None
        self.preview_jpegs = {}
        self.preview_updated_at = float("-inf")
        self.preview_error = None
        self.preview_requested_at = float("-inf")
        self.tick = 0
        self.pending_events = []
        self.last_frame_at = float("-inf")
        self.last_training_snapshot = None
        self.sampling_clock = None
        self.gripper_audits = queue.Queue(maxsize=256)
        self.gripper_audit_lock = threading.Lock()
        self.gripper_audit_dropped = 0
        self.recording_operator_control = None
        self.status = {}
        self.server = None
        diagnostic_root = getattr(args, "diagnostics_root", None)
        if diagnostic_root is None:
            diagnostic_root = (base.parent / "diagnostics") if args.dataset_root else ROOT / "logs/task3/sessions"
        self.diagnostics = SessionDiagnostics(Path(diagnostic_root) / self.session, {
            "session": self.session, "config_hash": self.config_hash, "config": self.config,
            "task_id": self.config.get("task_binding", {}).get("task_id"),
            "session_seed": self.args.seed,
            "input_kind": getattr(args, "input_kind", "unspecified"),
            "contract": self.controller.action_contract(),
            "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
            "model_dimensions": {k: int(getattr(self.controller.model, k)) for k in ("nq", "nv", "nu")},
            "initial_state": self.controller.initial_state}, max_chunks=128 if self.paper_recording else 8)
        frontend = ROOT / 'collection/teleoperation/mediapipe'
        sources = [*(p for p in frontend.glob('*.js') if not p.name.endswith('.test.js')),
                   *Path(__file__).parent.glob('*.py'), frontend/'models/gesture_recognizer.task',
                   frontend/'node_modules/@mediapipe/tasks-vision/package.json', frontend/'node_modules/@mediapipe/tasks-vision/vision_bundle.js']
        self.measurement_journal = MeasurementJournal(self.diagnostics.root/'hand_measurements',self.session,
            {'config':self.config,'config_hash':self.config_hash,'contract':self.controller.action_contract(),
             'mediapipe_version':json.loads((frontend/'node_modules/@mediapipe/tasks-vision/package.json').read_text(encoding='utf-8'))['version'],
             'coordinates':'image_landmarks_21 first two values are normalized unmirrored image XY; multiply by per-measurement width/height for pixels; third value is not metric depth',
             'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()}})
        self._diagnostic_state_at = float("-inf")
        self.diagnostic("session_start")
        self.update_status()

    def diagnostic(self, kind, **fields):
        fields = compact_event(fields) if getattr(self, 'paper_recording', False) else fields
        logger = getattr(self, "diagnostics", None)
        if logger is not None:
            logger.emit(kind, session=self.session, control_epoch=self.control_epoch,
                        tick=self.tick, simulation_time=float(self.controller.data.time), **fields)
        if (getattr(self, 'paper_recording', False) and self.recording and not self.saving
                and not getattr(self, 'episode_end', None)):
            from .session_diagnostics import safe_json
            row = safe_json({'kind':kind, 'session':self.session, 'control_epoch':self.control_epoch,
                'server_monotonic_ns':time.monotonic_ns(), 'wall_time_ms':time.time()*1000,
                'tick':self.tick, 'sim_time':float(self.controller.data.time), **fields})
            try:
                self.paper_events.put_nowait(row)
            except queue.Full:
                self.paper_event_dropped += 1

    def diagnostic_state(self):
        now = time.monotonic()
        if getattr(self, "diagnostics", None) is None or now - self._diagnostic_state_at < .1:
            return
        self._diagnostic_state_at = now
        self.diagnostic("state", actual_state=self.controller.state_vector(),
            accepted_target_m=self.controller.action_vector_m(), qpos=self.controller.data.qpos,
            qvel=self.controller.data.qvel, ctrl=self.controller.data.ctrl,
            status=self.controller.get_status(), recording=self.recording,
            training_frames=self.frames, capture_status=self.capture_status,
            diagnostic_health=self.diagnostics.status())

    def wait_reply(self, expected, timeout=600):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                result = self.record_replies.get(timeout=1)
            except KeyboardInterrupt:
                if not getattr(self, "saving", False):
                    raise
                self.stop_requested = True
                print("[Task3] Finishing the current save before exiting; please wait.", flush=True)
                continue
            except queue.Empty:
                if not self.worker.is_alive():
                    raise RuntimeError("Task3 recording worker exited; pending snapshots retained")
                continue
            if result["type"] == "fatal":
                self.worker_failed = True
                raise RuntimeError(result["traceback"])
            if result["type"] == "progress":
                self.save_progress = result
                if hasattr(self, "status"):
                    self.update_status()
                continue
            if result["type"] != expected:
                raise RuntimeError(f"Unexpected recording reply: {result}")
            return result
        raise TimeoutError(f"Waiting for recording worker: {expected}")

    def update_status(self):
        self.status = jsonable({
            "ok": True, "service": "mujoco-task3-recorder", "task_mode": "dual_arm",
            **self.controller.get_status(), "recording": self.recording, "saving": self.saving,
            "episode_end": getattr(self, "episode_end", None) if self.recording else None,
            "frames": self.frames, "saved_success": self.saved_success, "saved_failure": self.saved_failure,
            "session": self.session, "config_hash": self.config_hash, "config": self.config,
            "last_command_result": self.last_command_result,
            "last_operation_result": getattr(self, "last_operation_result", None),
            "operation_results": getattr(self, "operation_results", []),
            "control_epoch": self.control_epoch,
            "session_seed": getattr(getattr(self, 'args', None), 'seed', None),
            "scene_index": getattr(self, 'scene_index', 0),
            "operator_id": getattr(getattr(self, 'args', None), 'operator_id', None),
            "input_kind": getattr(getattr(self, 'args', None), 'input_kind', 'unspecified'),
            "service_identity": getattr(self, 'service_identity', None),
            "recording_wall_seconds": (self.episode_end['wall_duration_s'] if getattr(self, 'episode_end', None)
                else time.monotonic() - self.started_at) if self.recording else 0,
            "save_progress": getattr(self, "save_progress", None),
            "preview_error": self.preview_error,
            "capture_status": self.capture_status,
            "command_protocol": PROTOCOL,
            "backend_input": self.hand_input_guard.status() if getattr(self, "hand_input_guard", None) else None,
            "hand_journal": self.measurement_journal.status() if getattr(self,"measurement_journal",None) else None,
            "research_log_dropped": getattr(self, 'paper_event_dropped', 0),
            "action_contract": self.controller.action_contract() if hasattr(self.controller, "action_contract") else None,
            "diagnostics": self.diagnostics.status() if getattr(self, "diagnostics", None) else None,
        })

    @property
    def capture_status(self):
        if not self.recording:
            return {"state": "not_recording", "training_eligible": False,
                    "last_frame_recorded": False, "frames": self.frames}
        if getattr(self, 'episode_end', None):
            return {"state": "awaiting_save", "training_eligible": False,
                    "last_frame_recorded": False, "frames": self.frames, "reason": "task_done"}
        return {**getattr(self, "_capture_status", {"state": "idle", "training_eligible": False,
                                                   "last_frame_recorded": False}), "frames": self.frames}

    def audit_gripper(self, payload, accepted, reason, stage):
        """Record request decisions separately from physical gripper events.

        HTTP threads enqueue diagnostics only. They never mutate controller
        state, the current ACK, or the input watchdog for rejected requests.
        """
        if payload.get("command") != "dual_gripper":
            return
        if getattr(self, "gripper_audit_lock", None) is None:
            self.gripper_audit_lock = threading.Lock()
        with self.gripper_audit_lock:
            if not getattr(self, "recording", False) or not getattr(self, "gripper_audit_open", True):
                return
            self._enqueue_gripper_audit(payload, accepted, reason, stage)

    def _enqueue_gripper_audit(self, payload, accepted, reason, stage):
        if getattr(self, "gripper_audits", None) is None:
            self.gripper_audits = queue.Queue(maxsize=256)
        row = {"eventId": payload.get("eventId"), "action": payload.get("action"),
               "request_session": payload.get("session"), "control_epoch": payload.get("control_epoch"),
               "sent_at_ms": payload.get("sentAt"), "received_at_ms": payload.get("_received_at_ms"),
               "observed_at_ms": time.time() * 1000,
               "stage": stage, "accepted": accepted, "reason": reason}
        try:
            self.gripper_audits.put_nowait(row)
        except queue.Full:
            self.gripper_audit_dropped = getattr(self, "gripper_audit_dropped", 0) + 1

    def drain_gripper_audits(self):
        audits = getattr(self, "gripper_audits", None)
        rows = []
        if audits is not None:
            while True:
                try:
                    rows.append(audits.get_nowait())
                except queue.Empty:
                    break
        return rows

    def finish_research_log(self):
        # Save can block on video rendering. Flush decisions made by HTTP
        # threads during that wait without adding frames after the saved video.
        self.drop_queued_controls()
        if getattr(self, "gripper_audit_lock", None) is None:
            self.gripper_audit_lock = threading.Lock()
        with self.gripper_audit_lock:
            self.gripper_audit_open = False
            self.capture(diagnostics_only=True)
            self.raw_file.close()
            self.raw_file = None

    def start_server(self):
        if getattr(self, "command_gate", None) is None:
            self.command_gate = CommandGate(self.config.get("control", {}))
        app = Flask(__name__)
        app.config["MAX_CONTENT_LENGTH"] = 128 * 1024

        @app.get("/health")
        def health():
            return jsonify({**self.status, "project_root": str(ROOT),
                'server_epoch_ms':time.time()*1000, 'server_monotonic_ns':time.monotonic_ns()})

        @app.post("/measurements")
        def measurements():
            # Independent threaded HTTP I/O; never update watchdog, observation,
            # hand guard, command queues or physics from this research channel.
            payload=request.get_json(silent=True)
            if not isinstance(payload,dict):return jsonify(ok=False,error="Expected a batch"),400
            try:
                result=self.measurement_journal.append(payload,{'session':self.session,
                    'control_epoch':self.control_epoch,'tick':self.tick,'sim_time':float(self.controller.data.time)})
                return jsonify(result)
            except ValueError as exc:return jsonify(ok=False,error=str(exc)),400
            except OSError as exc:return jsonify(ok=False,error=str(exc)),503

        @app.post("/control")
        def control():
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict) or not isinstance(payload.get("command"), str):
                return jsonify(ok=False, error="Expected a command object"), 400
            command = payload["command"]
            if command == "metric_action" and getattr(self.controller, "continuous_task", False):
                return jsonify(ok=False, reason="continuous_camera_input_required", error="Use measured continuous_input for the interactive Paper A task"), 409
            # Underscore fields are backend-owned; never accept an HTTP bypass.
            payload = {k: v for k, v in payload.items() if not k.startswith("_")}
            self.diagnostic("request_received", request=payload, received_at_ms=time.time()*1000)
            telemetry = payload.get("telemetry")
            recognition = telemetry.get("recognition_result") if isinstance(telemetry, dict) else None
            measurement = recognition.get("measurement") if isinstance(recognition, dict) else None
            if measurement is not None:
                problem = inspect_measurement(measurement, session=self.session, epoch=self.control_epoch)
                self.diagnostic("hand_measurement", measurement=measurement, schema_valid=problem is None,
                                reason=problem, eventId=payload.get("eventId"), received_at_ms=time.time()*1000)
            if command not in {"dual_motion", "dual_gripper", "metric_action", "continuous_input", "pause", "telemetry", "reset", "record_start", "record_save", "record_failure", "record_discard", "record_stop"}:
                self.diagnostic("request_rejected", request=payload, reason="unknown_command", stage="http")
                return jsonify(ok=False, error="Unknown Task3 command"), 400
            if command == "dual_gripper":
                payload["_received_at_ms"] = time.time() * 1000
            self.audit_gripper(payload, None, None, "received")
            if "session" in payload and payload["session"] != self.session:
                self.diagnostic("request_rejected", request=payload, reason="stale_session", stage="http")
                self.audit_gripper(payload, False, "stale_session", "http_rejected")
                return jsonify(ok=False, queued=False, reason="stale_session", session=self.session,
                               eventId=payload.get("eventId"),
                               error="Command belongs to another Task3 session; refresh backend status"), 409
            if self.saving and command not in {"pause", "telemetry"}:
                self.diagnostic("request_rejected", request=payload, reason="saving", stage="http")
                self.audit_gripper(payload, False, "saving", "http_rejected")
                return jsonify(ok=False, error="Saving the episode; wait for completion"), 409
            if command in GUARDED_COMMANDS:
                ticket, reason = self.command_gate.admit(payload, session=self.session, epoch=self.control_epoch,
                    now_ms=time.time()*1000, monotonic=time.monotonic(), epoch_started_ms=self.control_epoch_started_ms)
                if reason:
                    self.diagnostic("request_rejected", request=payload, reason=reason, stage="http")
                    self.audit_gripper(payload, False, reason, "http_rejected")
                    self.interrupt_release_input(payload, reason)
                    return jsonify(ok=False, queued=False, reason=reason, error=reason,
                                   eventId=payload.get("eventId")), 409
                payload["_admission"] = ticket
            if command == 'pause':
                # A newer camera result may replace the shared input slot;
                # it must not erase an already admitted pause boundary.
                self.interrupt_release_input(payload, payload.get('reason') or 'operator_pause')
            if measurement is not None and problem is None and not self.saving:
                self.latest_observation = {'measurement':copy.deepcopy(measurement), 'received_at_ms':time.time()*1000}
            try:
                if command in {"dual_motion", "metric_action", "continuous_input", "telemetry", "pause"}:
                    with self.motion_lock:
                        if command in {"dual_motion", "metric_action"}:
                            self.drop_input(self.latest_motion, "superseded_motion")
                            self.latest_motion = payload
                        else:
                            self.drop_input(self.latest_telemetry, "superseded_telemetry")
                            self.latest_telemetry = payload
                else:
                    self.commands.put_nowait(payload)
            except queue.Full:
                self.drop_input(payload, "command_queue_full")
                self.audit_gripper(payload, False, "command_queue_full", "http_rejected")
                return jsonify(ok=False, error="Command queue full"), 429
            self.diagnostic("request_queued", eventId=payload.get("eventId"), command=command)
            return jsonify(ok=True, queued=True, eventId=payload.get("eventId"))

        def preview(name):
            self.preview_requested_at = time.monotonic()
            # One immutable result reference keeps image, age and epoch together
            # even if the simulation publishes/reset its preview concurrently.
            frame = getattr(self, 'preview_frame', None)
            if not frame or frame['control_epoch'] != self.control_epoch:
                return Response(status=204)
            age = time.monotonic() - frame['captured_at']
            jpeg = frame['images'].get(name) if age <= 1 else None
            return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store",
                "X-Preview-Age-Ms": str(max(0., age*1000)),
                "X-Preview-Epoch": str(frame['control_epoch'])}) if jpeg else Response(status=204)

        for name in CAMERAS:
            app.add_url_rule(f"/{name}-preview", f"{name}_preview", lambda name=name: preview(name))
        self.server = make_server("127.0.0.1", self.args.port, app, threaded=True, request_handler=QuietRequests)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def metadata(self, outcome):
        transforms = {}
        for name, key in (("left_link0", "T_WL"), ("right_link0", "T_WR")):
            index = mujoco.mj_name2id(self.controller.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if index >= 0:
                matrix = np.eye(4)
                matrix[:3, :3] = self.controller.data.xmat[index].reshape(3, 3)
                matrix[:3, 3] = self.controller.data.xpos[index]
                transforms[key] = matrix.tolist()
        task_transform = np.eye(4)
        task_transform[:3, :3] = self.controller.left.task_rotation
        task_transform[:3, 3] = self.controller.left.task_origin
        transforms["T_WT"] = task_transform.tolist()
        metadata = {
            "task_id": self.task.id, "task_binding": self.config['task_binding'],
            "scene_id": self.task.manifest['scene_id'], "session_id": self.session,
            "task_version": self.task.manifest['version'],
            "sampling_policy": {"version": FixedPhaseSampler.VERSION, "clock": "simulation",
                                "fps": self.config["recording"]["fps"], "control_hz": 60,
                                "nearest_control_tick": True, "backfill_idle": False},
            "schema_version": "task3-v1", "task": self.config.get("task", "bimanual_pick_carry_place_v1"),
            "config": self.config, "config_hash": self.config_hash, "session": self.session,
            "operator_id": self.args.operator_id, "seed": self.controller.seed, "outcome": outcome,
            "session_seed": self.args.seed, "scene_index": getattr(self, 'scene_index', 0),
            "scene_seed_policy": "session_seed_plus_scene_index_mod_2^31",
            "coordinate_frame": "task_frame", "field_order": "left_then_right",
            "state_names": STATE_NAMES, "action_names": ACTION_NAMES,
            "action_units": ["m", "m", "m", "rad", "rad", "rad", "0..255"] * 2,
            "transforms": transforms, "cameras": copy.deepcopy(self.config["scene"]["cameras"]),
            "recording_strategy": "snapshot_replay_at_save", "confirmation_gate": True,
            "operator_control": copy.deepcopy(self.recording_operator_control),
        }
        names = recording_camera_names(self.config)
        metadata['recording_cameras'] = list(names)
        metadata['camera_frame_alignment'] = {
            'clock': 'snapshot.time', 'index': 'shared_raw_frame_index',
            'same_snapshot_for_all_cameras': True, 'rendering': 'offline_at_save',
        }
        if 'oblique' in names:
            from .render_identity import render_identity
            metadata['cameras']['oblique'] = training_camera_metadata(self.config)
            metadata['training_camera'] = copy.deepcopy(metadata['cameras']['oblique'])
            metadata['render_identity'] = render_identity(self.config, model=self.controller.model)
        contact = self.controller.contact_control
        if contact and contact.rate_control:
            metadata["release_timing"] = {"clock": contact.release_clock,
                "confirm_seconds": contact.limits["release_confirm_seconds"],
                "max_input_gap_seconds": contact.limits["release_input_max_gap_seconds"],
                "physical_support_clock": "simulation"}
        return paper_metadata(self, metadata) if getattr(self, 'paper_recording', False) else metadata

    def advance_scene(self):
        """Advance every reset, including practice resets, with replayable seeds.

        The session seed remains part of process identity. Actual scene seeds
        and ordinals are independent of the recording attempt counter.
        """
        next_index = getattr(self, 'scene_index', 0) + 1
        seed = (int(self.args.seed) + next_index) % (2**31)
        self.controller.reset(seed=seed)
        self.scene_index = next_index
        self.controller.set_episode_timeout_enabled(False)

    def start_recording(self):
        self.paper_recording = getattr(self, 'paper_recording', False)
        if not hasattr(self, 'paper_events'):
            self.paper_events = queue.Queue(maxsize=4096)
        if getattr(self.controller, "continuous_gripper", False) and not getattr(self, 'paper_recording', False):
            raise ValueError("Continuous gripper validation cannot create formal demonstrations; wait for M03/M09")
        if self.recording:
            raise ValueError("An episode is already recording")
        sampling_clock = FixedPhaseSampler(self.config["recording"]["fps"])
        if self.controller.get_status().get("task_state") in {"DONE", "FAIL"}:
            self.advance_scene()
            self.new_control_epoch()
        status = self.controller.get_status()
        validate_recording_initial_state(status, self.paper_recording)
        contact = status.get('physical_contact') or {}
        if (self.paper_recording and contact.get('control_protocol') in {'gesture-grasp-release-v2', 'gesture-rate-gripper-v3'}
                and (any(contact.get('release_executed', {}).values())
                     or any(contact.get('release_opening', {}).values()))):
            # This episode cannot inherit an authorization whose causal event
            # exists only in the preceding practice log. Preserve that log and
            # require an explicit scene reset instead of fabricating a source.
            raise ValueError('recording_requires_clean_control_cycle')
        if self.paper_recording and getattr(self.args,'input_kind','human_teleop') == 'human_teleop':
            rate = contact.get('control_protocol') == 'gesture-rate-gripper-v3'
            aperture = self.telemetry.get('aperture_calibration')
            ready = (isinstance(aperture, dict) and
                (aperture == {'phase':'NOT_REQUIRED', 'reason':'categorical_rate_control'} if rate else aperture.get('phase') == 'READY'))
            if not self.telemetry.get('calibrated') or not ready:
                raise ValueError('Bind both hands before recording' if rate else
                    'Bind both hands and complete aperture calibration before recording')
        self.recording_calibration = copy.deepcopy(self.telemetry.get('aperture_calibration'))
        operator_control = self.telemetry.get("operator_control")
        if not isinstance(operator_control, dict):
            operator_control = {key: self.config["control"][key] for key in (
                "motion_mapping", "motion_speed_scale", "rate_motion_range",
                "rate_dead_zone", "wrist_sensitivity", "wrist_dead_zone",
            ) if key in self.config["control"]}
        self.recording_operator_control = copy.deepcopy(operator_control)
        self.pending = ROOT / ".local/task3-pending" / self.session / f"attempt_{self.attempt:06d}"
        self.pending.mkdir(parents=True, exist_ok=False)
        (self.pending / "task3.json").write_text(json.dumps(self.metadata("pending"), ensure_ascii=False, indent=2), encoding="utf-8")
        self.raw_file = (self.pending / "research.jsonl").open("w", encoding="utf-8")
        self.record_commands.put({"type": "start", "pending": str(self.pending)})
        self.wait_reply("started")
        self.drain_gripper_audits()
        self.gripper_audit_dropped = 0
        self.gripper_audit_open = True
        while not self.paper_events.empty():
            self.paper_events.get_nowait()
        self.paper_event_dropped = 0
        self.action_measurement_id = self.action_event_id = None
        if self.paper_recording and getattr(self.controller.contact_control, 'edge_control', False):
            # Practice inputs are absent from this episode's causal ledger.
            # Preserve the physical pose/width, but revoke the old edge lease
            # before the first snapshot; the first fresh shape is baseline
            # only. Never invent a recorded action for a practice mode ID.
            self.controller.pause('record_start_baseline')
        self.recording, self.frames, self.force_frame = True, 0, True
        self.episode_end = self.episode_terminal_snapshot = None
        self.episode_success = None
        self.started_at = time.monotonic()
        self.hand_recording_start_ms = time.time()*1000
        self.hand_recording_epoch = self.control_epoch
        self.controller.set_episode_timeout_enabled(True)
        self.controller.start_task_timer()
        self.initial_snapshot = self.controller.snapshot()
        self.initial_snapshot['scene_index'] = getattr(self, 'scene_index', 0)
        self.initial_snapshot["operator_control"] = copy.deepcopy(self.recording_operator_control)
        self.last_frame_at = float("-inf")
        self.last_training_snapshot = None
        self.sampling_clock = sampling_clock
        self.attempt += 1
        self.pending_events = [{"type": "record_start", "operator_control": copy.deepcopy(self.recording_operator_control)}]
        if self.paper_recording and self.latest_observation:
            self.diagnostic('hand_measurement', measurement=self.latest_observation['measurement'],
                            schema_valid=True, before_recording=True, received_at_ms=self.latest_observation['received_at_ms'])
        self.capture(force=True)

    def save_recording(self, outcome, hand_barrier=None):
        if not self.recording or not self.frames:
            raise ValueError("Start recording before saving an episode")
        status = self.controller.get_status()
        tail = status.get('success_tail')
        if tail and status.get('simulation_time', 0.) < tail['end_time'] - 1e-9:
            raise ValueError('success_tail_recording: wait for the post-success recording to finish')
        if outcome == "success" and status.get("task_state") != "DONE":
            raise ValueError("Success save requires the automatic DONE condition; use failure save for incomplete tasks")
        episode_end = getattr(self, 'episode_end', None)
        hand_recording_end_ms = episode_end['wall_time_ms'] if episode_end else time.time()*1000
        strict_hands = (isinstance(hand_barrier, dict) and hand_barrier.get('version') == 'hand-save-boundary-v1')
        if getattr(self.args, 'input_kind', None) == 'human_teleop' and not strict_hands:
            raise ValueError('hand_save_boundary_required: refresh the collection page')
        staged_hands = None
        hand_summary = None
        if strict_hands:
            try:
                acknowledged_end = self.measurement_journal.validate_boundary(hand_barrier)
                if episode_end and acknowledged_end < hand_recording_end_ms:
                    raise ValueError('hand_save_boundary_precedes_done: retry with the completed episode boundary')
                hand_recording_end_ms = min(hand_recording_end_ms, acknowledged_end)
                if hand_recording_end_ms < self.hand_recording_start_ms:
                    raise ValueError('hand_save_boundary_before_recording')
                # Export before committing a robot episode: failure leaves this
                # attempt open for an identical retry, without duplicate episodes.
                staged_hands = self.pending / 'sealed_hands.jsonl'
                hand_summary = self.measurement_journal.export(staged_hands,
                    self.hand_recording_epoch, self.hand_recording_start_ms, hand_recording_end_ms, hand_barrier)
                hand_summary.update(research_warning=None, requires_review=False)
            except OSError as error:
                raise ValueError(f'hand_save_not_ready: {error}; retry this episode') from error
        duration = episode_end['wall_duration_s'] if episode_end else time.monotonic() - self.started_at
        self.pending_events.append({"type": "save", "outcome": outcome})
        self.capture(force=True)
        self.raw_file.flush()
        if self.paper_recording:
            terminal_snapshot(self, outcome)
        self.saving = True
        self.update_status()
        try:
            self.record_commands.put({"type": "save", "outcome": outcome, "root": str(self.roots[outcome]),
                                      "repo_id": "local/task3_dual_arm" + ("_failures" if outcome == "failure" else ""),
                                      "metadata": self.metadata(outcome)})
            result = self.wait_reply("saved")
            self.finish_research_log()
            research = self.roots[outcome] / "research"
            research.mkdir(exist_ok=True)
            hand_target = research/f"episode_{result['episode_index']:06d}_hands.jsonl"
            if staged_hands is not None:
                shutil.copyfile(staged_hands, hand_target)
                shutil.copyfile(staged_hands.with_suffix('.manifest.json'), hand_target.with_suffix('.manifest.json'))
            else:
                hand_summary = self.measurement_journal.export_for_operation(hand_target,
                    self.hand_recording_epoch,self.hand_recording_start_ms,hand_recording_end_ms,hand_barrier) if getattr(self,'measurement_journal',None) else None
            shutil.move(str(self.pending / "research.jsonl"), str(research / f"episode_{result['episode_index']:06d}.jsonl"))
            initial = self.initial_snapshot
            entry = {
                "episode_index": result["episode_index"], "outcome": outcome, "frames": result["frames"],
                "episode_end": copy.deepcopy(episode_end),
                "hand_measurements": {"file":f"episode_{result['episode_index']:06d}_hands.jsonl", "measurements":hand_summary["measurements"], "sha256":hand_summary["sha256"], "research_warning":hand_summary.get("research_warning"), "requires_review":hand_summary.get("requires_review", False), "source_journal":hand_summary.get("path")} if hand_summary else None,
                "wall_duration_s": duration, "training_duration_s": result["frames"] / self.config.get("recording", {}).get("fps", 30),
                "seed": initial["initial_state"]["seed"], "config_hash": self.config_hash,
                "session_seed": self.args.seed, "scene_index": initial.get('scene_index', 0),
                "initial_state": initial['initial_state'],
                "metrics": jsonable(status.get("metrics", {})),
                "failure_reason": status.get("failure_reason") if outcome == "failure" else None,
                "initial_qpos": jsonable(initial["qpos"]), "initial_body_pos": jsonable(initial["body_pos"]),
                "operator_id": self.args.operator_id, "session": self.session,
                "operator_control": initial.get("operator_control"),
                "recording_cameras": list(recording_camera_names(self.config)),
                "camera_frame_alignment": "shared_snapshot_and_raw_frame_index",
            }
            if self.paper_recording:
                entry.update(input_kind=getattr(self.args,'input_kind','unspecified'), quality_review='pending',
                             schema_version='task3-papera-v2', raw_duration_s=float(
                                 (episode_end['simulation_time'] if episode_end else self.controller.data.time)-self.initial_snapshot['time']))
            if outcome == "failure" and not entry["failure_reason"]:
                entry["failure_reason"] = "operator_abort"
            with (research / "episodes.jsonl").open("a", encoding="utf-8") as manifest:
                manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
            actual = self.pending.resolve()
            if not actual.is_relative_to((ROOT / ".local/task3-pending").resolve()):
                raise RuntimeError("Unexpected pending directory")
            # Retain compact physics snapshots for recovery/re-rendering,
            # rather than deleting the only source after a save ACK.
            shutil.move(str(actual), str(research / f"episode_{result['episode_index']:06d}_snapshots"))
            self.recording = False
            self.saved_success += outcome == "success"
            self.saved_failure += outcome == "failure"
            self.advance_scene()
            self.controller.set_episode_timeout_enabled(False)
            self.new_control_epoch()
            self.telemetry = {}
            return_warning = hand_summary.get("research_warning") if hand_summary else None
            print(f"[Task3] Saved {outcome} episode {result['episode_index']}: {result['frames']} frames at {result['root']}", flush=True)
            return return_warning
        finally:
            self.saving = False
            self.update_status()

    def discard_recording(self, reason="operator_abort", hand_barrier=None):
        if not self.recording:
            return
        if self.paper_recording:
            terminal_snapshot(self, 'discarded')
        end = getattr(self, 'episode_end', None)
        hand_recording_end_ms=end['wall_time_ms'] if end else time.time()*1000
        self.saving = True
        try:
            if self.worker.is_alive() and not getattr(self, "worker_failed", False):
                self.record_commands.put({"type": "discard"})
                self.wait_reply("discarded")
            if self.raw_file:
                self.finish_research_log()
            destination = self.roots["success"].with_name(self.roots["success"].name + "_rejected") / self.session / self.pending.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            (self.pending / "rejection.json").write_text(json.dumps({"reason": reason, "config_hash": self.config_hash,
                "config": self.config, "session": self.session, "seed": self.initial_snapshot["initial_state"]["seed"]}), encoding="utf-8")
            hand_summary = None
            if getattr(self,'measurement_journal',None):
                hand_summary = self.measurement_journal.export_for_operation(self.pending/'hand_measurements.jsonl',
                    self.hand_recording_epoch,self.hand_recording_start_ms,hand_recording_end_ms,hand_barrier)
            if hand_summary and hand_summary.get("research_warning"):
                rejection = self.pending / "rejection.json"
                info = json.loads(rejection.read_text(encoding="utf-8"))
                info["hand_measurements"] = hand_summary
                rejection.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
            shutil.move(str(self.pending), str(destination))
            self.recording = False
            self.advance_scene()
            self.controller.set_episode_timeout_enabled(False)
            self.new_control_epoch()
            self.telemetry = {}
            return hand_summary.get("research_warning") if hand_summary else None
        finally:
            self.saving = False

    def handle(self, payload):
        operation_session, operation_epoch = getattr(self, 'session', None), getattr(self, 'control_epoch', None)
        self._execution_result = None
        self._hand_guard_check = None
        try:
            return self._handle(payload)
        finally:
            self.retain_operation_result(payload, operation_session, operation_epoch)
            if (payload.get('command') in {'continuous_input','metric_action','pause'}
                    and self._execution_result and not self._execution_result.get('accepted')
                    and self._execution_result.get('gripper_rejection_handled') != 'trusted_motion_hold_v2'
                    and payload.get('session', getattr(self,'session',None)) == getattr(self,'session',None)
                    and payload.get('control_epoch',self.control_epoch) == self.control_epoch):
                contact = getattr(self.controller,'contact_control',None)
                if contact:
                    contact.cancel_release(self._execution_result.get('reason') or 'release_input_rejected')
            if payload.get('command') == 'continuous_input' and self._execution_result and self._execution_result.get('accepted'):
                self.action_measurement_id = payload.get('measurement_id')
                self.action_event_id = payload.get('eventId')
            self.diagnostic("request_executed", request=payload, result=self._execution_result,
                hand_guard_check=self._hand_guard_check,
                actual_state_phase="before_next_physics_step",
                actual_state=self.controller.state_vector() if hasattr(self.controller, "state_vector") else None,
                accepted_target_m=self.controller.action_vector_m() if hasattr(self.controller, "action_vector_m") else None)

    def retain_operation_result(self, payload, session, epoch):
        """Keep discrete completion ACKs independent of the 20 Hz hand stream."""
        if (session is None or epoch is None or payload.get('command') not in {'record_start', 'record_save', 'record_failure',
                'record_discard', 'record_stop', 'reset'} or not self._execution_result
                or payload.get('session', session) != session
                or payload.get('control_epoch', epoch) != epoch):
            return
        event_id = payload.get('eventId')
        if not isinstance(event_id, str) or not event_id:
            return
        history = getattr(self, 'operation_results', [])
        if any(row['eventId'] == event_id for row in history):
            return  # A retransmitted/rejected duplicate cannot rewrite a completion.
        result = self._execution_result
        self.last_operation_result = {
            'command': payload['command'], 'eventId': event_id,
            'accepted': result.get('accepted') is True, 'reason': result.get('reason'),
            'research_warning': result.get('research_warning'),
            'session': session, 'control_epoch': epoch,
            'completed_control_epoch': self.control_epoch,
            'frames': getattr(self, 'frames', 0), 'recording': getattr(self, 'recording', False),
            'saved_success': getattr(self, 'saved_success', 0),
            'saved_failure': getattr(self, 'saved_failure', 0),
            'completed_at_ms': time.time()*1000}
        self.operation_results = [*history, self.last_operation_result][-16:]
        self.update_status()

    def reject_continuous_input(self, payload, reason):
        """Keep a rejected camera command visible and cancel its recovery state."""
        self.controller.pause(reason)
        self._execution_result = {"accepted": False, "reason": reason, "eventId": payload.get("eventId")}
        self.last_command_result = {"command": payload["command"], **self._execution_result}
        self.update_status()

    def _handle(self, payload):
        command = payload["command"]
        if "_admission" in payload:
            reason = self.command_gate.consume(payload["_admission"], session=self.session, epoch=self.control_epoch,
                                               monotonic=time.monotonic(), now_ms=time.time()*1000)
            if reason:
                self._execution_result = {"accepted": False, "reason": reason}
                self.audit_gripper(payload, False, reason, "execution_rejected")
                return
        # Session IDs fence backend restarts, where the control epoch starts at
        # zero again. Reject old packets without replacing the current ACK or
        # refreshing the input watchdog. Missing IDs support existing clients.
        if "session" in payload and payload["session"] != self.session:
            self._execution_result = {"accepted": False, "reason": "stale_session"}
            self.audit_gripper(payload, False, "stale_session", "execution_rejected")
            return
        # Reject stale tracking information before it can resume a paused arm
        # or overwrite hand identity after a save/reset.
        if command in {"dual_motion", "dual_gripper", "metric_action", "continuous_input", "telemetry", "pause"}:
            try:
                sent_at = float(payload["sentAt"])
                age = time.time() * 1000 - sent_at
                control = self.config["control"]
                fresh = math.isfinite(sent_at) and -control.get("max_future_command_ms", 1000) <= age <= control.get("max_command_age_ms", 150)
                fresh = fresh and sent_at >= self.control_epoch_started_ms
                fresh = fresh and payload.get("control_epoch", self.control_epoch) == self.control_epoch
            except (ValueError, KeyError, TypeError):
                fresh = False
            if not fresh:
                self._execution_result = {"accepted": False, "reason": "stale_command"}
                if (command == "continuous_input" and self.config.get("contact_control", {}).get("control_protocol") == "gesture-rate-gripper-v3"
                        and payload.get("control_epoch", self.control_epoch) == self.control_epoch):
                    self.hand_input_guard.invalidate()
                    self.reject_continuous_input(payload, "stale_command")
                if command == "dual_gripper":
                    self.last_command_result = {"command": command, "eventId": payload.get("eventId"), "accepted": False, "reason": "stale_command"}
                    self.audit_gripper(payload, False, "stale_command", "execution_rejected")
                    self.update_status()
                return
        if command == "continuous_input":
            telemetry = payload.get('telemetry')
            if not isinstance(telemetry, dict):
                self.hand_input_guard.invalidate()
                self.reject_continuous_input(payload, 'hand_schema_required')
                return
            if self.recording and self.paper_recording and telemetry.get('aperture_calibration') != self.recording_calibration:
                self.hand_input_guard.invalidate()
                self.reject_continuous_input(payload, 'calibration_changed_during_recording')
                return
            # Preserve the clock value actually used by the freshness guard.
            # IK/protection work may finish substantially later than this check.
            self._hand_guard_check = {'epoch_ms':time.time()*1000, 'monotonic_ns':time.monotonic_ns()}
            reason = self.hand_input_guard.check(payload, session=self.session, epoch=self.control_epoch,
                now_ms=self._hand_guard_check['epoch_ms'],
                release_threshold=self.config.get('contact_control',{}).get('release_intent_aperture_normalized',.85),
                control_protocol=self.config.get('contact_control',{}).get('control_protocol'),
                gesture_parameters=self.config.get('gesture_control'))
            if reason:
                self.reject_continuous_input(payload, reason)
                return
        elif command == "pause" and getattr(self, "hand_input_guard", None):
            self.hand_input_guard.invalidate()
        self.last_input_at = time.monotonic()
        if isinstance(payload.get("telemetry"), dict):
            telemetry = payload["telemetry"]
            hands = telemetry.get("hands")
            if hands is not None and (not isinstance(hands, dict) or any(
                not isinstance(hands.get(side, {}), dict) for side in ("left", "right")
            )):
                if command == "continuous_input":
                    self.hand_input_guard.invalidate()
                    self.reject_continuous_input(payload, "invalid_tracking_metadata")
                    return
                self.audit_gripper(payload, False, "invalid_tracking_metadata", "execution_rejected")
                self._execution_result = {"accepted": False, "reason": "invalid_tracking_metadata"}
                return
            self.telemetry = telemetry
            self.controller.update_telemetry(self.telemetry)
        result = {"command": command, "eventId": payload.get("eventId"), "accepted": True}
        try:
            if command == "record_start":
                self.start_recording()
            elif command in {"record_save", "record_failure"}:
                result["research_warning"] = self.save_recording("success" if command == "record_save" else "failure",payload.get("hand_journal_barrier"))
            elif command == "record_discard":
                result["research_warning"] = self.discard_recording(hand_barrier=payload.get("hand_journal_barrier"))
            elif command == "record_stop":
                if self.recording:
                    raise ValueError("Save or explicitly discard the current episode before exiting")
                self.stop_requested = True
            elif command == "reset":
                if self.recording:
                    raise ValueError("Save or discard the current episode before resetting")
                self.advance_scene()
                self.controller.set_episode_timeout_enabled(False)
                self.new_control_epoch()
                self.telemetry = {}
            elif command != "telemetry":
                event_id = payload.get("eventId")
                if event_id and event_id in self.seen_events:
                    result.update(accepted=False, reason="duplicate_event")
                else:
                    answer = self.controller.apply_command(payload, now_ms=time.time() * 1000)
                    if isinstance(answer, dict):
                        result.update(answer)
                    elif answer is False:
                        result.update(accepted=False, reason="command_rejected")
                    if command == "dual_gripper" and result.get("accepted"):
                        self.force_frame = True
                        self.event_window_until = time.monotonic() + 0.5
                        self.pending_events.append({"type": "gripper", "action": payload["action"], "eventId": event_id})
                        if event_id:
                            self.seen_events.add(event_id)
            if command not in {"telemetry", "dual_motion", "pause"}:
                self.last_command_result = result
        except (ValueError, KeyError, TypeError) as error:
            result = {**result, "accepted": False, "reason": str(error)}
            if command not in {"telemetry", "dual_motion", "pause"}:
                self.last_command_result = {**result, "accepted": False, "reason": str(error)}
        if command == "dual_gripper":
            self.audit_gripper(payload, self.last_command_result["accepted"], self.last_command_result.get("reason"), "executed")
        self._execution_result = result
        self.update_status()

    def drop_input(self, payload, reason):
        if payload:
            if getattr(self, "command_gate", None):
                self.command_gate.discard(payload.get("_admission"))
            self.diagnostic("request_dropped", request=payload, reason=reason)

    def new_control_epoch(self):
        self.control_epoch += 1
        self.preview_jpegs = {}
        self.preview_frame = None
        self.preview_updated_at = float("-inf")
        self.control_epoch_started_ms = time.time() * 1000
        self.latest_observation = None
        self.action_measurement_id = self.action_event_id = None
        with self.motion_lock:
            self.drop_input(self.latest_motion, "scene_reset")
            self.drop_input(self.latest_telemetry, "scene_reset")
            self.latest_motion = self.latest_telemetry = None
        self.drop_queued_controls()
        self.diagnostic("scene_reset")

    def drop_queued_controls(self):
        # Old movement/gripper messages cannot cross the scene reset. Record
        # operations retain their order and receive an explicit backend ACK.
        if getattr(self, "commands", None) is None:
            return
        retained = []
        while True:
            try:
                payload = self.commands.get_nowait()
            except queue.Empty:
                break
            if payload["command"].startswith("record_") or payload["command"] == "reset":
                retained.append(payload)
            else:
                self.drop_input(payload, "epoch_reset_dropped")
                self.audit_gripper(payload, False, "epoch_reset_dropped", "execution_rejected")
        for payload in retained:
            self.commands.put_nowait(payload)

    def interrupt_release_input(self, payload, reason):
        """HTTP may flag cancellation; only the simulation thread changes grip state."""
        if (payload.get('command') in {'continuous_input','metric_action','pause'}
                and payload.get('session') == self.session and payload.get('control_epoch') == self.control_epoch):
            with self.motion_lock:
                self.release_input_interruption = (self.control_epoch, reason)

    def process_commands(self):
        with self.motion_lock:
            motion, telemetry = self.latest_motion, self.latest_telemetry
            self.latest_motion = self.latest_telemetry = None
            interruption = getattr(self,'release_input_interruption',None)
            self.release_input_interruption = None
        if interruption and interruption[0] == self.control_epoch:
            contact = getattr(self.controller,'contact_control',None)
            if contact:
                contact.cancel_release(interruption[1])
                self.diagnostic('release_confirmation_cancelled',reason=interruption[1],stage='http_rejection')
        if telemetry:
            self.handle(telemetry)
            try:
                if motion and float(motion.get("sentAt", 0)) < float(telemetry.get("sentAt", 0)):
                    self.drop_input(motion, "newer_telemetry")
                    motion = None
            except (TypeError, ValueError):
                self.drop_input(motion, "invalid_tracking_timestamp")
                motion = None
        had_event = False
        while True:
            try:
                payload = self.commands.get_nowait()
            except queue.Empty:
                break
            self.handle(payload)
            had_event = True
        if motion and not had_event:
            self.handle(motion)
        elif motion:
            self.drop_input(motion, "discrete_event_priority")
        # Local watchdogs are not captured input. Advancing the remote timestamp
        # here would falsely reject a fresh frame still in transport.
        if time.monotonic() - self.last_input_at > 0.35:
            self.controller.pause("command_timeout")

    def capture(self, force=False, diagnostics_only=False):
        if not self.recording or getattr(self, 'episode_end', None):
            return
        if getattr(self, 'paper_recording', False):
            status = self.controller.get_status()
            done = not diagnostics_only and status.get('task_state') == 'DONE'
            end_ms, end_ns = time.time()*1000, time.monotonic_ns()
            if done:
                if not getattr(self, 'episode_success', None):
                    self.episode_success = {'simulation_time': float(self.controller.snapshot()['time']),
                        'tick': self.tick, 'wall_time_ms': end_ms, 'server_monotonic_ns': end_ns}
                    self.diagnostic('episode_success', boundary=self.episode_success)
                tail = status.get('success_tail')
                end_time = tail['end_time'] if tail else self.episode_success['simulation_time']
                done = float(self.controller.snapshot()['time']) >= end_time - 1e-9
            if done:
                # Keep recording=True until save/discard: it denotes an unsaved
                # episode. Physics may keep settling, but its episode is sealed.
                self.episode_terminal_snapshot = copy.deepcopy(self.controller.snapshot())
                boundary = {'version':'episode-done-boundary-v2', 'reason':'task_done',
                    'success': copy.deepcopy(self.episode_success),
                    'post_success_seconds': float(status.get('success_tail', {}).get('seconds', 0.)) if status.get('success_tail') else 0.,
                    'session':self.session, 'control_epoch':self.control_epoch, 'tick':self.tick,
                    'simulation_time':float(self.episode_terminal_snapshot['time']),
                    'wall_time_ms':end_ms, 'server_monotonic_ns':end_ns,
                    'wall_duration_s':max(0., end_ns/1e9-self.started_at)}
                self.diagnostic('episode_capture_end', boundary=boundary)
                self.episode_end = boundary
            return capture_paper(self, self.sampling_clock, jsonable, diagnostics_only=diagnostics_only)
        now = time.monotonic()
        status = self.controller.get_status()
        snapshot = self.controller.snapshot()
        mode = self.telemetry.get("control_mode", status.get("control_mode", "IDLE"))
        # UI can remain in confirmation until its next health poll. Once the
        # atomic event happened, capture actual grasp/release physics instead.
        if mode in {"GRASP_CONFIRM", "RELEASE_CONFIRM"} and status.get("task_state") in {"GRASP_VERIFY", "RELEASE_VERIFY", "DUAL_GRASPED"} and now < self.event_window_until:
            mode = status.get("task_state")
        confirm = mode in {"GRASP_CONFIRM", "RELEASE_CONFIRM"}
        previous = getattr(self, "last_training_snapshot", None)
        physical = physical_change({**snapshot, "qvel": snapshot.get("qvel", self.controller.data.qvel)},
                                   previous, status.get("task_state"))
        backend_paused = status.get("paused", False) or status.get("control_mode") == "PAUSE"
        wanted = (force or self.force_frame or
                  physical or
                  (not confirm and not backend_paused and (now < self.event_window_until or
                                    self.telemetry.get("training_recordable", True))))
        fps = self.config.get("recording", {}).get("fps", 30)
        # Sampling follows physics time, not computer load/wall-clock jitter.
        simulation_time = float(snapshot.get("time", self.controller.data.time))
        if getattr(self, "sampling_clock", None) is None:
            self.sampling_clock = FixedPhaseSampler(fps)
        recordable = not diagnostics_only and self.sampling_clock.sample(simulation_time, wanted, force or self.force_frame)
        # Report the actual training gate, not the browser's requested mode.
        # FPS throttling changes last_frame_recorded, not the visible activity.
        capture_reason = ("diagnostics_only" if diagnostics_only else
                          "boundary_event" if force or self.force_frame else
                          "physical_change" if physical else
                          "operator_control" if wanted else
                          "stationary_pause" if backend_paused or mode == "PAUSE" else
                          "gesture_confirmation" if confirm else "idle")
        self._capture_status = {"state": capture_reason, "training_eligible": bool(wanted and not diagnostics_only),
                                "last_frame_recorded": bool(recordable)}
        frame_index = None
        if recordable:
            self.record_commands.put({"type": "frame", "snapshot": snapshot}, timeout=2)
            frame_index = self.frames
            self.frames += 1
            self.last_frame_at = simulation_time
            self.last_training_snapshot = {key: np.asarray(snapshot[key]).copy()
                                           for key in ("state", "action", "qpos") if key in snapshot}
        events, self.pending_events = self.pending_events, []
        metrics = {**status.get("metrics", {}), **{key: status[key] for key in (
            "object_position", "relative_pose_error_m", "relative_orientation_error_rad", "input_disagreement_m", "contacts"
        ) if key in status}}
        row = {
            "tick": self.tick, "sim_time": float(self.controller.data.time), "wall_time_ms": time.time() * 1000,
            "control_mode": mode, "task_state": status.get("task_state"), "gripper_latch": status.get("gripper_latch"),
            "state": snapshot["state"], "action": snapshot["action"],
            "actual_ee": status.get("actual_ee", {}), "commanded_ee": status.get("commanded_ee", {}),
            "metrics": metrics, "telemetry": self.telemetry,
            "training_recordable": bool(recordable), "events": events, "frame_index": frame_index,
            "capture_reason": capture_reason,
            "sampling_policy": FixedPhaseSampler.VERSION,
            "backend": {key: status.get(key) for key in ("control_mode", "paused", "pause_reason", "release_ready")},
            "last_command_result": getattr(self, "last_command_result", None),
            "gripper_requests": self.drain_gripper_audits(),
            "gripper_request_log_dropped": getattr(self, "gripper_audit_dropped", 0),
        }
        self.raw_file.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
        self.raw_file.flush()
        self.force_frame = False

    def update_previews(self, now):
        if not self.preview_process:
            return
        if not self.preview_process.is_alive():
            self.preview_jpegs = {}
            self.preview_error = "Preview renderer exited; restart Task3 to restore live views"
            return
        if now - self.preview_requested_at < 1 and now - getattr(self, "last_preview_at", 0) >= 1/12:
            snapshot = self.controller.snapshot()
            snapshot['_preview'] = {'captured_at': time.monotonic(),
                                    'control_epoch': self.control_epoch, 'session': self.session}
            # Replace only an unrendered preview; research/recording queues are untouched.
            try:
                self.preview_in.get_nowait()
            except queue.Empty:
                pass
            try:
                self.preview_in.put_nowait(snapshot)
                self.last_preview_at = now
            except queue.Full:
                pass
        while True:
            try:
                result = self.preview_out.get_nowait()
                if result.get('control_epoch') != self.control_epoch or result.get('session') != self.session:
                    continue
                self.preview_frame = result
                self.preview_jpegs = result["images"]
                self.preview_updated_at = result["captured_at"]
                self.preview_error = None
            except queue.Empty:
                break
        if now - self.preview_updated_at > 1:
            self.preview_jpegs = {}
            if now - self.preview_requested_at < 1:
                self.preview_error = "Waiting for fresh camera previews"

    def run(self):
        self.start_server()
        print(f"Recorder ready: http://127.0.0.1:{self.args.port}/health; use the collection launcher's web URL", flush=True)
        viewer = None
        if not self.args.headless:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(self.controller.model, self.controller.data)
        viewer_camera = "front"
        started = time.monotonic()
        try:
            while not self.stop_requested and (viewer is None or viewer.is_running()):
                tick_started = time.monotonic()
                self.process_commands()
                self.controller.step(1 / 60)
                self.tick += 1
                self.capture()
                self.controller.consume_gripper_event()
                self.update_previews(tick_started)
                self.update_status()
                self.diagnostic_state()
                if viewer:
                    mode = self.controller.get_status().get("control_mode")
                    if mode == "XY":
                        viewer_camera = "top"
                    elif mode == "Z":
                        viewer_camera = "front"
                    sync_operator_view(viewer, self.controller.model, viewer_camera, self.config)
                if self.args.max_seconds and tick_started - started >= self.args.max_seconds:
                    break
                time.sleep(max(0, 1 / 60 - (time.monotonic() - tick_started)))
        finally:
            if viewer:
                viewer.close()

    def close(self):
        if self.server:
            self.server.shutdown()
        try:
            if self.recording:
                self.discard_recording("shutdown_without_save")
            if self.worker.is_alive() and not getattr(self, "worker_failed", False):
                self.record_commands.put({"type": "close"})
                self.wait_reply("closed")
        finally:
            if getattr(self,"measurement_journal",None):
                try:
                    self.measurement_journal.close()
                except OSError as exc:
                    self.diagnostic("hand_journal_close_failed",error=str(exc))
            self.diagnostic("session_end", recording=self.recording)
            if getattr(self, "diagnostics", None):
                self.diagnostics.close()
            if self.raw_file and not self.raw_file.closed:
                self.raw_file.close()
            self.worker.join(timeout=20)
            if self.worker.is_alive():
                self.worker.terminate()
                self.worker.join(timeout=5)
            if self.preview_process and self.preview_process.is_alive():
                try:
                    self.preview_in.put(None, timeout=2)
                except queue.Full:
                    pass
                self.preview_process.join(timeout=10)
                if self.preview_process.is_alive():
                    self.preview_process.terminate()
                    self.preview_process.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/simulation_base.json")
    parser.add_argument("--port", type=int, default=5013)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--task", required=True)
    parser.add_argument("--operator-id", default="anonymous")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--diagnostics-root", type=Path, help="Separate diagnostic session parent directory")
    parser.add_argument("--input-kind", choices=["human_teleop", "scripted_test", "synthetic_test", "policy_rollout"], default="human_teleop")
    parser.add_argument("--max-seconds", type=float, default=0)
    args = parser.parse_args()
    recorder = DualArmRecorder(args)
    try:
        recorder.run()
    except KeyboardInterrupt:
        pass
    finally:
        recorder.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()
