"""Test the gate against timed research rows without opening any renderer."""
import io
import json
import queue
import threading
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np
import mujoco

from simulation.mujoco.dual_arm.record_mujoco_dual_arm import DualArmRecorder, FixedPhaseSampler
from simulation.mujoco.dual_arm.dual_arm_task_controller import DualArmTaskController


class FakeController:
    data = SimpleNamespace(qvel=np.zeros(38), time=1.0)

    def get_status(self):
        return {"task_state": "PREGRASP", "control_mode": "IDLE", "gripper_latch": "OPEN", "metrics": {}}

    def snapshot(self):
        return {"state": np.zeros(16), "action": np.zeros(14)}


class RecordingGateTests(unittest.TestCase):
    def recorder(self):
        result = DualArmRecorder.__new__(DualArmRecorder)
        result.controller = FakeController()
        result.recording = True
        result.raw_file = io.StringIO()
        result.record_commands = queue.Queue()
        result.telemetry = {"control_mode": "GRASP_CONFIRM", "training_recordable": False}
        result.force_frame = False
        result.event_window_until = 0
        result.config = {"recording": {"fps": 30}}
        result.pending_events = []
        result.tick = result.frames = 0
        result.last_frame_at = float("-inf")
        return result

    def test_confirmation_is_logged_but_does_not_add_training_frames(self):
        recorder = self.recorder()
        for _ in range(40):
            recorder.capture()
        self.assertEqual(recorder.frames, 0)
        self.assertTrue(recorder.record_commands.empty())
        self.assertEqual(len(recorder.raw_file.getvalue().splitlines()), 40)

    def test_atomic_gripper_event_survives_confirmation_gate(self):
        recorder = self.recorder()
        recorder.force_frame = True
        recorder.pending_events = [{"type": "gripper", "action": "close"}]
        recorder.capture()
        recorder.capture()
        rows = [json.loads(line) for line in recorder.raw_file.getvalue().splitlines()]
        self.assertEqual(recorder.frames, 1)
        self.assertEqual(rows[0]["frame_index"], 0)
        self.assertEqual(rows[0]["events"][0]["action"], "close")
        self.assertIsNone(rows[1]["frame_index"])

    def test_physical_transition_is_recorded_even_without_hand_motion(self):
        recorder = self.recorder()
        recorder.telemetry = {"control_mode": "IDLE", "training_recordable": False}
        recorder.controller = FakeController()
        recorder.controller.data = SimpleNamespace(qvel=np.ones(38), time=1.0)
        recorder.capture()
        self.assertEqual(recorder.frames, 1)

    def test_frame_cadence_follows_simulation_time(self):
        recorder = self.recorder()
        recorder.telemetry = {"training_recordable": True}
        recorder.controller = FakeController()
        recorder.controller.data = SimpleNamespace(qvel=np.zeros(38), time=1.0)
        recorder.capture()
        recorder.controller.data.time += 1 / 60
        recorder.capture()
        self.assertEqual(recorder.frames, 1)
        recorder.controller.data.time += 1 / 60
        recorder.capture()
        self.assertEqual(recorder.frames, 2)

    def test_ten_seconds_real_physics_preserves_300_sampling_intervals(self):
        recorder = self.recorder()
        recorder.controller = DualArmTaskController()
        recorder.controller.control_mode = "XY"
        recorder.controller.paused = False
        recorder.telemetry = {"control_mode": "XY", "training_recordable": True}
        # The gate remains eligible so this isolates cadence from gesture and
        # stationarity decisions. The clock still comes from real MuJoCo steps.
        status = recorder.controller.get_status
        recorder.controller.get_status = lambda: {**status(), "paused": False, "control_mode": "XY"}
        recorder.capture()
        for _ in range(600):
            recorder.controller.step(1 / 60)
            recorder.capture()
        samples = []
        while not recorder.record_commands.empty():
            samples.append(recorder.record_commands.get_nowait()["snapshot"])
        intervals = np.diff([sample["time"] for sample in samples])
        self.assertEqual(len(intervals), 300)
        self.assertAlmostEqual(sum(intervals), 10.0, places=7)
        self.assertTrue(np.all((intervals >= 0.032 - 1e-9) & (intervals <= 0.034 + 1e-9)))

    def test_fixed_phase_skips_long_idle_and_keeps_off_grid_boundary(self):
        sampler = FixedPhaseSampler(30)
        kept = [0.0]
        self.assertTrue(sampler.sample(0, True))
        for tick in range(1, 600):
            self.assertFalse(sampler.sample(tick / 60, False))
        self.assertTrue(sampler.sample(10, True))
        self.assertFalse(sampler.sample(10, True))
        self.assertTrue(sampler.sample(10 + 1 / 60, False, force=True))
        self.assertFalse(sampler.sample(10 + 1 / 60, False))
        for tick in range(602, 1201):
            if sampler.sample(tick / 60, True):
                kept.append(tick / 60)
        self.assertEqual(len(kept) - 1, 300)

    def test_raw_diagnostics_record_backend_reason_and_event_decision_without_frames(self):
        recorder = self.recorder()
        recorder.controller.get_status = lambda: {
            "task_state": "PREGRASP", "paused": True, "control_mode": "PAUSE",
            "pause_reason": "command_timeout", "release_ready": False,
        }
        payload = {"command": "dual_gripper", "eventId": "blocked-open", "action": "open",
                   "sentAt": 123, "_received_at_ms": 145}
        recorder.audit_gripper(payload, False, "release_not_ready", "executed")
        recorder.capture()
        row = json.loads(recorder.raw_file.getvalue())
        self.assertEqual(row["backend"]["pause_reason"], "command_timeout")
        self.assertEqual(row["gripper_requests"][0]["eventId"], "blocked-open")
        self.assertFalse(row["gripper_requests"][0]["accepted"])
        self.assertEqual(row["gripper_requests"][0]["received_at_ms"], 145)
        self.assertEqual(row["events"], [])
        self.assertFalse(row["training_recordable"])

    def test_save_tail_logs_http_rejection_without_appending_video_frames(self):
        recorder = self.recorder()
        recorder.capture(force=True)
        recorder.audit_gripper({"command": "dual_gripper", "eventId": "during-save"},
                               False, "saving", "http_rejected")
        recorder.commands = queue.Queue()
        recorder.commands.put({"command": "dual_gripper", "eventId": "queued-before-save"})
        recorder.commands.put({"command": "record_start"})
        output = recorder.raw_file
        with patch.object(output, "close"):
            recorder.finish_research_log()
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(recorder.frames, 1)
        self.assertEqual(rows[-1]["capture_reason"], "diagnostics_only")
        self.assertIsNone(rows[-1]["frame_index"])
        self.assertEqual(rows[-1]["gripper_requests"][0]["reason"], "saving")
        self.assertEqual(rows[-1]["gripper_requests"][1]["reason"], "epoch_reset_dropped")
        self.assertEqual(rows[-1]["gripper_requests"][1]["eventId"], "queued-before-save")
        self.assertEqual(recorder.commands.get_nowait(), {"command": "record_start"})
        self.assertTrue(recorder.commands.empty())
        recorder.audit_gripper({"command": "dual_gripper"}, False, "saving", "http_rejected")
        self.assertEqual(recorder.drain_gripper_audits(), [])

    def test_stationary_hand_loss_pause_is_raw_only(self):
        recorder = self.recorder()
        recorder.telemetry = {"control_mode": "PAUSE", "training_recordable": False}
        for _ in range(40):
            recorder.capture()
        self.assertEqual(recorder.frames, 0)
        self.assertEqual(len(recorder.raw_file.getvalue().splitlines()), 40)

    def test_long_hand_loss_reports_pause_without_duplicate_training_and_preserves_physics(self):
        recorder = self.recorder()
        recorder.telemetry = {"control_mode": "PAUSE", "pause_reason": "hand_lost", "training_recordable": False}
        recorder.controller.data = SimpleNamespace(qvel=np.zeros(38), time=0.0)
        recorder.controller.get_status = lambda: {"task_state": "PREGRASP", "paused": True, "control_mode": "PAUSE"}
        sample = {"state": np.zeros(16), "action": np.zeros(14), "qpos": np.zeros(39)}
        recorder.controller.snapshot = lambda: {**sample, "time": recorder.controller.data.time}
        recorder.capture(force=True)
        for _ in range(600):
            recorder.controller.data.time += 1 / 60
            recorder.capture()
        self.assertEqual(recorder.frames, 1)
        self.assertEqual(recorder.capture_status["state"], "stationary_pause")
        self.assertFalse(recorder.capture_status["training_eligible"])
        self.assertEqual(len(recorder.raw_file.getvalue().splitlines()), 601)
        # A change during loss remains observable even after a long idle period.
        sample["qpos"][-1] = 0.002
        recorder.controller.data.time += 1 / 60
        recorder.capture()
        self.assertEqual(recorder.frames, 2)
        self.assertEqual(recorder.capture_status["state"], "physical_change")
        self.assertTrue(recorder.capture_status["last_frame_recorded"])
        row = json.loads(recorder.raw_file.getvalue().splitlines()[-1])
        self.assertEqual(row["capture_reason"], "physical_change")
        recorder.recording = False
        self.assertEqual(recorder.capture_status["state"], "not_recording")
        self.assertFalse(recorder.capture_status["training_eligible"])

    def test_backend_pause_overrides_stale_motion_but_preserves_physics(self):
        recorder = self.recorder()
        recorder.telemetry = {"control_mode": "XY", "training_recordable": True}
        recorder.controller.get_status = lambda: {
            "task_state": "PREGRASP", "control_mode": "PAUSE", "paused": True, "metrics": {},
        }
        recorder.controller.data = SimpleNamespace(qvel=np.zeros(38), time=1.0)
        sample = {"state": np.zeros(16), "action": np.zeros(14)}
        recorder.controller.snapshot = lambda: sample
        recorder.capture()
        self.assertEqual(recorder.frames, 0)
        recorder.controller.data.qvel[:] = 0.1
        recorder.capture()
        self.assertEqual(recorder.frames, 1)
        recorder.controller.data.qvel[:] = 0
        recorder.controller.data.time += 1 / 30
        sample["state"][0] += 0.0002
        recorder.capture()
        self.assertEqual(recorder.frames, 2)
        rows = [json.loads(line) for line in recorder.raw_file.getvalue().splitlines()]
        self.assertEqual([row["frame_index"] for row in rows], [None, 0, 1])

    def test_confirmation_cannot_hide_dynamic_frames_or_duplicate_force_frame(self):
        recorder = self.recorder()
        recorder.controller.data = SimpleNamespace(qvel=np.ones(38), time=1.0)
        recorder.force_frame = True
        recorder.capture()
        recorder.capture()
        self.assertEqual(recorder.frames, 1)
        self.assertFalse(recorder.force_frame)
        recorder.controller.data.time += 1 / 30
        recorder.capture()
        self.assertEqual(recorder.frames, 2)
        rows = [json.loads(line) for line in recorder.raw_file.getvalue().splitlines()]
        self.assertEqual([row["frame_index"] for row in rows], [0, None, 1])

    def test_physical_verification_preserves_stationary_settling_time(self):
        for task_state in ("GRASP_VERIFY", "RELEASE_VERIFY"):
            with self.subTest(task_state=task_state):
                recorder = self.recorder()
                recorder.controller.get_status = lambda: {
                    "task_state": task_state, "control_mode": "PAUSE", "metrics": {},
                }
                recorder.controller.data = SimpleNamespace(qvel=np.zeros(38), time=1.0)
                recorder.capture()
                self.assertEqual(recorder.frames, 1)

    def test_slow_object_or_target_changes_survive_stationary_pause(self):
        for changed_key in ("qpos", "action"):
            with self.subTest(changed_key=changed_key):
                recorder = self.recorder()
                recorder.telemetry = {"control_mode": "PAUSE", "training_recordable": False}
                recorder.controller.data = SimpleNamespace(qvel=np.zeros(38), time=1.0)
                sample = {"state": np.zeros(16), "action": np.zeros(14), "qpos": np.zeros(39)}
                recorder.controller.snapshot = lambda: sample
                recorder.capture(force=True)
                recorder.controller.data.time += 1 / 30
                sample[changed_key][-1] = 0.00005
                recorder.capture()
                self.assertEqual(recorder.frames, 1)
                recorder.controller.data.time += 1 / 30
                sample[changed_key][-1] = 0.0002
                recorder.capture()
                self.assertEqual(recorder.frames, 2)

    def test_real_falling_object_is_recorded_during_confirmation_and_pause(self):
        recorder = self.recorder()
        controller = DualArmTaskController()
        controller.pause("gripper_confirmation")
        controller.data.qpos[controller.object_qpos_adr + 2] += 0.04
        mujoco.mj_forward(controller.model, controller.data)
        recorder.controller = controller
        recorder.capture(force=True)
        for step in range(14):
            if step == 3:
                recorder.telemetry = {"control_mode": "PAUSE", "training_recordable": False}
            controller.step(1 / 60)
            recorder.capture()
        samples = []
        while not recorder.record_commands.empty():
            samples.append(recorder.record_commands.get_nowait()["snapshot"])
        self.assertGreaterEqual(len(samples), 4)
        self.assertLess(samples[-1]["qpos"][controller.object_qpos_adr + 2],
                        samples[0]["qpos"][controller.object_qpos_adr + 2])
        self.assertEqual(controller.gripper_latch, "OPEN")
        intervals = np.diff([sample["time"] for sample in samples])
        self.assertTrue(np.all(intervals >= 1 / 30 - controller.model.opt.timestep - 1e-9))
        self.assertAlmostEqual(samples[-1]["qpos"][controller.object_qpos_adr + 2],
                               controller.config["scene"]["table_top_z"] +
                               controller.config["scene"]["object_half_size"][2], delta=0.002)
        rows = [json.loads(line) for line in recorder.raw_file.getvalue().splitlines()]
        self.assertTrue(any(row["training_recordable"] and row["control_mode"] == "GRASP_CONFIRM" for row in rows))
        self.assertTrue(any(row["training_recordable"] and row["control_mode"] == "PAUSE" for row in rows))
        for sample in samples:
            self.assertEqual(sample["state"].shape, (16,))
            self.assertEqual(sample["action"].shape, (14,))

    def test_recording_freezes_operator_controls_and_save_returns_to_practice(self):
        test_root = Path(__file__).resolve().parents[2] / ".local/task3-check"
        test_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="operator-controls-", dir=test_root) as directory:
            root = Path(directory)
            recorder = self.recorder()
            recorder.recording = False
            recorder.controller = DualArmTaskController()
            recorder.controller.set_episode_timeout_enabled(False)
            recorder.config = recorder.controller.config
            recorder.config_hash = recorder.controller.config_hash
            recorder.args = SimpleNamespace(seed=1000, operator_id="test")
            recorder.session = "test-session"
            recorder.roots = {"success": root / "success", "failure": root / "failure"}
            recorder.roots["failure"].mkdir()
            recorder.attempt = recorder.saved_success = recorder.saved_failure = 0
            recorder.wait_reply = Mock(side_effect=lambda kind: {} if kind == "started" else {
                "episode_index": 0, "frames": recorder.frames, "root": str(recorder.roots["failure"]),
            })
            recorder.update_status = Mock()
            recorder.new_control_epoch = Mock()
            controls = {"motion_mapping": "rate", "motion_speed_scale": 0.4,
                        "wrist_sensitivity": 0.04, "wrist_dead_zone": 0.003}
            recorder.telemetry = {"operator_control": controls, "training_recordable": False}
            with patch("simulation.mujoco.dual_arm.record_mujoco_dual_arm.ROOT", root):
                recorder.start_recording()
            self.assertTrue(recorder.controller.episode_timeout_enabled)
            controls["motion_speed_scale"] = 0.9
            self.assertEqual(recorder.initial_snapshot["operator_control"]["motion_speed_scale"], 0.4)
            pending_meta = json.loads((recorder.pending / "task3.json").read_text(encoding="utf-8"))
            self.assertEqual(pending_meta["operator_control"]["motion_speed_scale"], 0.4)
            raw = (recorder.pending / "research.jsonl").read_text(encoding="utf-8")
            self.assertEqual(json.loads(raw.splitlines()[0])["events"][0]["operator_control"]["motion_speed_scale"], 0.4)
            with patch("simulation.mujoco.dual_arm.record_mujoco_dual_arm.ROOT", root):
                recorder.save_recording("failure")
            manifest = json.loads((recorder.roots["failure"] / "research/episodes.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(manifest["operator_control"]["motion_speed_scale"], 0.4)
            self.assertFalse(recorder.controller.episode_timeout_enabled)
            self.assertFalse(recorder.recording)


class ControlEpochTests(unittest.TestCase):
    def recorder(self):
        recorder = DualArmRecorder.__new__(DualArmRecorder)
        recorder.controller = Mock()
        recorder.config = {"control": {"max_command_age_ms": 150}}
        recorder.control_epoch = 2
        recorder.control_epoch_started_ms = time.time() * 1000 - 50
        recorder.telemetry = {"control_mode": "PAUSE"}
        recorder.update_status = Mock()
        recorder.last_input_at = time.monotonic()
        recorder.commands = queue.Queue()
        recorder.motion_lock = threading.Lock()
        recorder.latest_motion = recorder.latest_telemetry = None
        return recorder

    def test_stale_telemetry_does_not_resume_or_change_identity(self):
        recorder = self.recorder()
        recorder.handle({"command": "telemetry", "sentAt": time.time() * 1000 - 300,
                         "telemetry": {"control_mode": "XY"}})
        recorder.controller.update_telemetry.assert_not_called()
        self.assertEqual(recorder.telemetry["control_mode"], "PAUSE")

    def test_previous_epoch_gripper_is_rejected(self):
        recorder = self.recorder()
        recorder.handle({"command": "dual_gripper", "action": "close", "eventId": "old",
                         "sentAt": time.time() * 1000, "control_epoch": 1})
        self.assertFalse(recorder.last_command_result["accepted"])
        recorder.controller.apply_command.assert_not_called()

    def test_reset_drops_queued_controls_but_preserves_record_operations(self):
        recorder = self.recorder()
        recorder.latest_motion = {"command": "dual_motion"}
        for command in ("pause", "dual_gripper", "record_start"):
            recorder.commands.put({"command": command})
        recorder.new_control_epoch()
        self.assertEqual(recorder.control_epoch, 3)
        self.assertIsNone(recorder.latest_motion)
        self.assertEqual(recorder.commands.get_nowait()["command"], "record_start")
        self.assertTrue(recorder.commands.empty())

    def test_malformed_tracking_timestamps_cannot_crash_command_loop(self):
        recorder = self.recorder()
        recorder.latest_motion = {"command": "dual_motion", "sentAt": None}
        recorder.latest_telemetry = {"command": "telemetry", "sentAt": [1]}
        recorder.process_commands()
        recorder.controller.update_telemetry.assert_not_called()


class ControlSessionTests(unittest.TestCase):
    commands = ("dual_motion", "dual_gripper", "pause", "telemetry", "reset",
                "record_start", "record_save", "record_failure", "record_discard", "record_stop")

    def recorder(self):
        recorder = ControlEpochTests().recorder()
        recorder.session = "new-backend-session"
        recorder.control_epoch = 0
        recorder.saving = False
        recorder.args = SimpleNamespace(port=0)
        recorder.seen_events = set()
        recorder.last_command_result = {"eventId": "current-event", "accepted": True}
        recorder.start_recording = Mock()
        recorder.save_recording = Mock()
        recorder.discard_recording = Mock()
        return recorder

    def client(self, recorder):
        module = "simulation.mujoco.dual_arm.record_mujoco_dual_arm"
        with patch(f"{module}.make_server") as server, patch(f"{module}.threading.Thread"):
            recorder.start_server()
        return server.call_args.args[2].test_client()

    def test_old_session_requests_never_enter_motion_or_operation_queues(self):
        recorder = self.recorder()
        client = self.client(recorder)
        current_ack = recorder.last_command_result.copy()
        for command in self.commands:
            with self.subTest(command=command):
                response = client.post("/control", json={
                    "command": command, "session": "old-backend-session", "control_epoch": 0,
                    "sentAt": time.time() * 1000, "eventId": "old-event",
                })
                self.assertEqual(response.status_code, 409)
                self.assertEqual(response.json["reason"], "stale_session")
                self.assertEqual(response.json["session"], recorder.session)
                self.assertFalse(response.json["queued"])
                self.assertTrue(recorder.commands.empty())
                self.assertIsNone(recorder.latest_motion)
                self.assertIsNone(recorder.latest_telemetry)
                self.assertEqual(recorder.last_command_result, current_ack)

    def test_current_and_legacy_requests_keep_existing_queue_behavior(self):
        recorder = self.recorder()
        client = self.client(recorder)
        for session in (recorder.session, None):
            for command in ("dual_motion", "telemetry", "pause", "dual_gripper", "record_start"):
                with self.subTest(session=session, command=command):
                    payload = {"command": command, "eventId": "current-event"}
                    if session is not None:
                        payload["session"] = session
                    response = client.post("/control", json=payload)
                    self.assertEqual(response.status_code, 200)
                    self.assertTrue(response.json["queued"])
                    if command == "dual_motion":
                        self.assertEqual(recorder.latest_motion, payload)
                    elif command in {"telemetry", "pause"}:
                        self.assertEqual(recorder.latest_telemetry, payload)
                    else:
                        queued = recorder.commands.get_nowait()
                        if command == "dual_gripper":
                            self.assertIsInstance(queued.pop("_received_at_ms"), float)
                        self.assertEqual(queued, payload)
        recorder.start_recording.assert_not_called()

    def test_old_session_execution_cannot_change_watchdog_telemetry_or_current_ack(self):
        recorder = self.recorder()
        last_input_at = recorder.last_input_at
        current_ack = recorder.last_command_result.copy()
        for command in self.commands:
            with self.subTest(command=command):
                recorder.handle({
                    "command": command, "session": "old-backend-session", "control_epoch": 0,
                    "sentAt": time.time() * 1000, "eventId": "old-event",
                    "telemetry": {"control_mode": "XY"},
                })
                self.assertEqual(recorder.last_input_at, last_input_at)
                self.assertEqual(recorder.telemetry, {"control_mode": "PAUSE"})
                self.assertEqual(recorder.last_command_result, current_ack)
        recorder.controller.apply_command.assert_not_called()
        recorder.controller.update_telemetry.assert_not_called()
        recorder.controller.reset.assert_not_called()
        recorder.start_recording.assert_not_called()
        recorder.save_recording.assert_not_called()
        recorder.discard_recording.assert_not_called()
        recorder.update_status.assert_not_called()

    def test_current_and_legacy_fresh_motion_still_execute(self):
        recorder = self.recorder()
        recorder.controller.apply_command.return_value = {"accepted": True}
        for session in (recorder.session, None):
            payload = {"command": "dual_motion", "sentAt": time.time() * 1000, "control_epoch": 0}
            if session is not None:
                payload["session"] = session
            recorder.handle(payload)
        self.assertEqual(recorder.controller.apply_command.call_count, 2)

    def test_gripper_http_rejections_are_audited_without_changing_ack_or_watchdog(self):
        for reason in ("stale_session", "saving", "command_queue_full"):
            with self.subTest(reason=reason):
                recorder = self.recorder()
                recorder.recording = True
                client = self.client(recorder)
                ack, last_input = recorder.last_command_result.copy(), recorder.last_input_at
                payload = {"command": "dual_gripper", "action": "close", "eventId": reason,
                           "session": recorder.session, "sentAt": time.time() * 1000}
                if reason == "stale_session":
                    payload["session"] = "old"
                elif reason == "saving":
                    recorder.saving = True
                else:
                    recorder.commands = queue.Queue(maxsize=1)
                    recorder.commands.put({"command": "reset"})
                response = client.post("/control", json=payload)
                self.assertEqual(response.status_code, 429 if reason == "command_queue_full" else 409)
                audits = recorder.drain_gripper_audits()
                self.assertEqual([entry["stage"] for entry in audits], ["received", "http_rejected"])
                self.assertEqual(audits[-1]["reason"], reason)
                self.assertFalse(audits[-1]["accepted"])
                self.assertEqual(audits[-1]["eventId"], reason)
                self.assertEqual(recorder.last_command_result, ack)
                self.assertEqual(recorder.last_input_at, last_input)
                recorder.controller.apply_command.assert_not_called()

    def test_gripper_execution_audit_links_acceptance_and_duplicate_to_physical_event(self):
        recorder = self.recorder()
        recorder.recording = True
        recorder.pending_events = []
        recorder.controller.apply_command.return_value = {"accepted": True}
        payload = {"command": "dual_gripper", "action": "close", "eventId": "close-1",
                   "sentAt": time.time() * 1000, "control_epoch": 0}
        recorder.handle(payload)
        recorder.handle(payload)
        audits = recorder.drain_gripper_audits()
        self.assertEqual([entry["accepted"] for entry in audits], [True, False])
        self.assertEqual(audits[-1]["reason"], "duplicate_event")
        self.assertEqual([entry["eventId"] for entry in audits], ["close-1", "close-1"])
        self.assertEqual(recorder.pending_events, [{"type": "gripper", "action": "close", "eventId": "close-1"}])


if __name__ == "__main__":
    unittest.main()
