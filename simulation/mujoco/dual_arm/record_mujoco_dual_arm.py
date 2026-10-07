"""Local Task3 teleoperation, research logging and outcome-separated recording."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
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
from .dual_arm_task_controller import DualArmTaskController
from .recording_policy import physical_change
from .dual_arm_render_workers import CAMERAS, STATE_NAMES, ACTION_NAMES, recording_worker, preview_worker, sync_operator_view

ROOT = Path(__file__).resolve().parents[3]


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
        if not self.path.startswith(("/health", "/top-preview", "/side-preview", "/front-preview")):
            super().log_request(code, size)


class DualArmRecorder:
    def __init__(self, args):
        self.args = args
        self.config = load_config(args.config)
        self.controller = DualArmTaskController(config=self.config)
        self.controller.reset(seed=args.seed)
        self.controller.set_episode_timeout_enabled(False)
        self.config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = Path(args.dataset_root).resolve() if args.dataset_root else ROOT / "datasets" / f"task3_dual_arm_{self.session}"
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
        self.telemetry = {}
        self.last_command_result = None
        self.seen_events = set()
        self.last_input_at = time.monotonic()
        self.control_epoch = 0
        self.control_epoch_started_ms = time.time() * 1000
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
        self.update_status()

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
            "frames": self.frames, "saved_success": self.saved_success, "saved_failure": self.saved_failure,
            "session": self.session, "config_hash": self.config_hash, "config": self.config,
            "last_command_result": self.last_command_result,
            "control_epoch": self.control_epoch,
            "recording_wall_seconds": time.monotonic() - self.started_at if self.recording else 0,
            "save_progress": getattr(self, "save_progress", None),
            "preview_error": self.preview_error,
            "capture_status": self.capture_status,
        })

    @property
    def capture_status(self):
        if not self.recording:
            return {"state": "not_recording", "training_eligible": False,
                    "last_frame_recorded": False, "frames": self.frames}
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
        app = Flask(__name__)
        app.config["MAX_CONTENT_LENGTH"] = 128 * 1024

        @app.get("/health")
        def health():
            return jsonify(self.status)

        @app.post("/control")
        def control():
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict) or not isinstance(payload.get("command"), str):
                return jsonify(ok=False, error="Expected a command object"), 400
            command = payload["command"]
            if command not in {"dual_motion", "dual_gripper", "pause", "telemetry", "reset", "record_start", "record_save", "record_failure", "record_discard", "record_stop"}:
                return jsonify(ok=False, error="Unknown Task3 command"), 400
            if command == "dual_gripper":
                payload["_received_at_ms"] = time.time() * 1000
            self.audit_gripper(payload, None, None, "received")
            if "session" in payload and payload["session"] != self.session:
                self.audit_gripper(payload, False, "stale_session", "http_rejected")
                return jsonify(ok=False, queued=False, reason="stale_session", session=self.session,
                               eventId=payload.get("eventId"),
                               error="Command belongs to another Task3 session; refresh backend status"), 409
            if self.saving and command not in {"pause", "telemetry"}:
                self.audit_gripper(payload, False, "saving", "http_rejected")
                return jsonify(ok=False, error="Saving the episode; wait for completion"), 409
            try:
                if command in {"dual_motion", "telemetry", "pause"}:
                    with self.motion_lock:
                        if command == "dual_motion":
                            self.latest_motion = payload
                        else:
                            self.latest_telemetry = payload
                else:
                    self.commands.put_nowait(payload)
            except queue.Full:
                self.audit_gripper(payload, False, "command_queue_full", "http_rejected")
                return jsonify(ok=False, error="Command queue full"), 429
            return jsonify(ok=True, queued=True, eventId=payload.get("eventId"))

        def preview(name):
            self.preview_requested_at = time.monotonic()
            jpeg = self.preview_jpegs.get(name) if self.preview_requested_at - self.preview_updated_at <= 1 else None
            return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"}) if jpeg else Response(status=204)

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
        return {
            "sampling_policy": {"version": FixedPhaseSampler.VERSION, "clock": "simulation",
                                "fps": self.config["recording"]["fps"], "control_hz": 60,
                                "nearest_control_tick": True, "backfill_idle": False},
            "schema_version": "task3-v1", "task": self.config.get("task", "bimanual_pick_carry_place_v1"),
            "config": self.config, "config_hash": self.config_hash, "session": self.session,
            "operator_id": self.args.operator_id, "seed": self.args.seed, "outcome": outcome,
            "coordinate_frame": "task_frame", "field_order": "left_then_right",
            "state_names": STATE_NAMES, "action_names": ACTION_NAMES,
            "action_units": ["m", "m", "m", "rad", "rad", "rad", "0..255"] * 2,
            "transforms": transforms, "cameras": self.config["scene"]["cameras"],
            "recording_strategy": "snapshot_replay_at_save", "confirmation_gate": True,
            "operator_control": copy.deepcopy(self.recording_operator_control),
        }

    def start_recording(self):
        if self.recording:
            raise ValueError("An episode is already recording")
        sampling_clock = FixedPhaseSampler(self.config["recording"]["fps"])
        if self.controller.get_status().get("task_state") in {"DONE", "FAIL"}:
            self.controller.reset(seed=self.args.seed + self.attempt)
            self.new_control_epoch()
        status = self.controller.get_status()
        if status.get("task_state") != "PREGRASP" or status.get("gripper_latch") != "OPEN":
            raise ValueError("Start a complete demonstration in PREGRASP with open grippers; reset before recording")
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
        self.recording, self.frames, self.force_frame = True, 0, True
        self.started_at = time.monotonic()
        self.controller.set_episode_timeout_enabled(True)
        self.controller.start_task_timer()
        self.initial_snapshot = self.controller.snapshot()
        self.initial_snapshot["operator_control"] = copy.deepcopy(self.recording_operator_control)
        self.last_frame_at = float("-inf")
        self.last_training_snapshot = None
        self.sampling_clock = sampling_clock
        self.attempt += 1
        self.pending_events = [{"type": "record_start", "operator_control": copy.deepcopy(self.recording_operator_control)}]
        self.capture(force=True)

    def save_recording(self, outcome):
        if not self.recording or not self.frames:
            raise ValueError("Start recording before saving an episode")
        status = self.controller.get_status()
        if outcome == "success" and status.get("task_state") != "DONE":
            raise ValueError("Success save requires the automatic DONE condition; use failure save for incomplete tasks")
        duration = time.monotonic() - self.started_at
        self.pending_events.append({"type": "save", "outcome": outcome})
        self.capture(force=True)
        self.raw_file.flush()
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
            shutil.move(str(self.pending / "research.jsonl"), str(research / f"episode_{result['episode_index']:06d}.jsonl"))
            initial = self.initial_snapshot
            entry = {
                "episode_index": result["episode_index"], "outcome": outcome, "frames": result["frames"],
                "wall_duration_s": duration, "training_duration_s": result["frames"] / self.config.get("recording", {}).get("fps", 30),
                "seed": initial["initial_state"]["seed"], "config_hash": self.config_hash,
                "metrics": jsonable(status.get("metrics", {})),
                "failure_reason": status.get("failure_reason") if outcome == "failure" else None,
                "initial_qpos": jsonable(initial["qpos"]), "initial_body_pos": jsonable(initial["body_pos"]),
                "operator_id": self.args.operator_id, "session": self.session,
                "operator_control": initial.get("operator_control"),
            }
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
            self.controller.reset(seed=self.args.seed + self.attempt)
            self.controller.set_episode_timeout_enabled(False)
            self.new_control_epoch()
            self.telemetry = {}
            print(f"[Task3] Saved {outcome} episode {result['episode_index']}: {result['frames']} frames at {result['root']}", flush=True)
        finally:
            self.saving = False
            self.update_status()

    def discard_recording(self, reason="operator_abort"):
        if not self.recording:
            return
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
            shutil.move(str(self.pending), str(destination))
            self.recording = False
            self.controller.reset(seed=self.args.seed + self.attempt)
            self.controller.set_episode_timeout_enabled(False)
            self.new_control_epoch()
            self.telemetry = {}
        finally:
            self.saving = False

    def handle(self, payload):
        command = payload["command"]
        # Session IDs fence backend restarts, where the control epoch starts at
        # zero again. Reject old packets without replacing the current ACK or
        # refreshing the input watchdog. Missing IDs support existing clients.
        if "session" in payload and payload["session"] != self.session:
            self.audit_gripper(payload, False, "stale_session", "execution_rejected")
            return
        # Reject stale tracking information before it can resume a paused arm
        # or overwrite hand identity after a save/reset.
        if command in {"dual_motion", "dual_gripper", "telemetry", "pause"}:
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
                if command == "dual_gripper":
                    self.last_command_result = {"command": command, "eventId": payload.get("eventId"), "accepted": False, "reason": "stale_command"}
                    self.audit_gripper(payload, False, "stale_command", "execution_rejected")
                    self.update_status()
                return
        self.last_input_at = time.monotonic()
        if isinstance(payload.get("telemetry"), dict):
            telemetry = payload["telemetry"]
            hands = telemetry.get("hands")
            if hands is not None and (not isinstance(hands, dict) or any(
                not isinstance(hands.get(side, {}), dict) for side in ("left", "right")
            )):
                self.audit_gripper(payload, False, "invalid_tracking_metadata", "execution_rejected")
                return
            self.telemetry = telemetry
            self.controller.update_telemetry(self.telemetry)
        result = {"command": command, "eventId": payload.get("eventId"), "accepted": True}
        try:
            if command == "record_start":
                self.start_recording()
            elif command in {"record_save", "record_failure"}:
                self.save_recording("success" if command == "record_save" else "failure")
            elif command == "record_discard":
                self.discard_recording()
            elif command == "record_stop":
                if self.recording:
                    raise ValueError("Save or explicitly discard the current episode before exiting")
                self.stop_requested = True
            elif command == "reset":
                if self.recording:
                    raise ValueError("Save or discard the current episode before resetting")
                self.controller.reset(seed=self.args.seed + self.attempt)
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
            if command not in {"telemetry", "dual_motion", "pause"}:
                self.last_command_result = {**result, "accepted": False, "reason": str(error)}
        if command == "dual_gripper":
            self.audit_gripper(payload, self.last_command_result["accepted"], self.last_command_result.get("reason"), "executed")
        self.update_status()

    def new_control_epoch(self):
        self.control_epoch += 1
        self.control_epoch_started_ms = time.time() * 1000
        with self.motion_lock:
            self.latest_motion = self.latest_telemetry = None
        self.drop_queued_controls()

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
                self.audit_gripper(payload, False, "epoch_reset_dropped", "execution_rejected")
        for payload in retained:
            self.commands.put_nowait(payload)

    def process_commands(self):
        with self.motion_lock:
            motion, telemetry = self.latest_motion, self.latest_telemetry
            self.latest_motion = self.latest_telemetry = None
        if telemetry:
            self.handle(telemetry)
            try:
                if motion and float(motion.get("sentAt", 0)) < float(telemetry.get("sentAt", 0)):
                    motion = None
            except (TypeError, ValueError):
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
        # Recenter after an absent browser; retain the gripper latch.
        if time.monotonic() - self.last_input_at > 0.35:
            self.controller.apply_command({"command": "pause", "reason": "command_timeout", "sentAt": time.time() * 1000}, now_ms=time.time() * 1000)

    def capture(self, force=False, diagnostics_only=False):
        if not self.recording:
            return
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
        if now - self.preview_requested_at < 1 and now - getattr(self, "last_preview_at", 0) >= 0.125:
            try:
                self.preview_in.put_nowait(self.controller.snapshot())
                self.last_preview_at = now
            except queue.Full:
                pass
        while True:
            try:
                result = self.preview_out.get_nowait()
                self.preview_jpegs = result["images"]
                self.preview_updated_at = result["created_at"]
                self.preview_error = None
            except queue.Empty:
                break
        if now - self.preview_updated_at > 1:
            self.preview_jpegs = {}
            if now - self.preview_requested_at < 1:
                self.preview_error = "Waiting for fresh camera previews"

    def run(self):
        self.start_server()
        print(f"Task3 ready: http://127.0.0.1:8000/dual_arm.html (recorder {self.args.port})", flush=True)
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
    parser.add_argument("--config", type=Path, default=ROOT / "config/task3_dual_arm.json")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--operator-id", default="anonymous")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-previews", action="store_true")
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
