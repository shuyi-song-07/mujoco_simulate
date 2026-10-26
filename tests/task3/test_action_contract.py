"""M01 contracts, shared execution, HTTP admission and diagnostic persistence."""
import json
import queue
import threading
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, PropertyMock

import numpy as np

from simulation.mujoco.dual_arm.action_contract import V1, V2, describe, legacy_to_metric, metric_to_legacy
from simulation.mujoco.dual_arm.action_contract import require_legacy_checkpoint
from simulation.mujoco.dual_arm.command_protocol import CommandGate, PROTOCOL
from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
from simulation.mujoco.dual_arm.panda_arm_controller import PandaArmController, rpy_to_matrix, matrix_to_rpy
from simulation.mujoco.dual_arm.session_diagnostics import SessionDiagnostics
from simulation.mujoco.dual_arm.teleop_service import DualArmRecorder


class ActionContractTests(unittest.TestCase):
    def setUp(self):
        self.c = DualArmTaskController()

    def test_field_order_and_model_total_width(self):
        spec = describe(self.c.gripper_width_limits)
        self.assertEqual(len(spec["state_names"]), 16)
        self.assertEqual(spec["action_names"][6], "left_gripper_width_m")
        self.assertEqual(spec["action_names"][13], "right_gripper_width_m")
        np.testing.assert_allclose(self.c.gripper_width_limits, [.08, .08])
        np.testing.assert_allclose(self.c.state_vector()[[7, 15]], [.08, .08], atol=1e-7)
        self.assertFalse(spec["capabilities"]["continuous_gripper"])

    def test_checkpoint_units_and_versions_rejected_before_loading_weights(self):
        require_legacy_checkpoint({"task3": {"schema_version": V1}})
        for metadata in ({"task3": {"schema_version": V2}}, {},
                         {"task3": {"schema_version": V1, "action_units": ["m"]*14}}):
            with self.assertRaises(ValueError):
                require_legacy_checkpoint(metadata)

    def test_explicit_legacy_round_trip_preserves_independent_width_values(self):
        original = self.c.action_vector().astype(float)
        original[[6, 13]] = [63.75, 191.25]
        metric = legacy_to_metric(original, self.c.gripper_width_limits, version=V1)
        np.testing.assert_allclose(metric[[6, 13]], [.02, .06])
        np.testing.assert_allclose(metric_to_legacy(metric, self.c.gripper_width_limits, version=V2), original)
        with self.assertRaises(ValueError):
            legacy_to_metric(original, self.c.gripper_width_limits, version=V2)

    def test_translated_rotated_task_frame_and_rpy_convention(self):
        cfg = json.loads(json.dumps(self.c.config))
        cfg["scene"]["task_frame_position"] = [.1, -.2, .3]
        cfg["scene"]["task_frame_quaternion_wxyz"] = [2**-.5, 0, 0, 2**-.5]
        arm = PandaArmController(self.c.model, self.c.data, "left", cfg)
        np.testing.assert_allclose(arm.task_to_world([.1, 0, 0]), [.1, -.1, .3], atol=1e-10)
        np.testing.assert_allclose(arm.world_to_task(arm.task_to_world([.2, .3, .4])), [.2, .3, .4])
        rpy = [.2, -.3, .4]
        np.testing.assert_allclose(matrix_to_rpy(rpy_to_matrix(rpy)), rpy)

    def test_legacy_script_and_metric_commands_share_execution(self):
        other = DualArmTaskController()
        target = self.c.action_vector_m()
        target[0] += .002
        target[7] += .002
        result = self.c.apply_action(target, schema_version=V2, components=("motion",), mode="XY")
        legacy = other.apply_command({"command": "dual_motion", "sentAt": 0, "mode": "xy",
                                      "left": {"dx": .5}, "right": {"dx": .5}}, now_ms=0)
        self.assertTrue(result["accepted"] and legacy["accepted"])
        np.testing.assert_allclose(self.c.data.ctrl, other.data.ctrl, atol=1e-8)

    def test_policy_v1_uses_same_metric_executor(self):
        other = DualArmTaskController()
        legacy = self.c.action_vector()
        legacy[0] += .002
        legacy[7] += .002
        a = self.c.apply_policy_action(legacy)
        b = other.apply_action(legacy_to_metric(legacy, other.gripper_width_limits, version=V1), schema_version=V2)
        self.assertTrue(a["accepted"] and b["accepted"])
        np.testing.assert_allclose(self.c.data.ctrl, other.data.ctrl, atol=1e-8)

    def test_rejections_are_atomic_and_do_not_invent_continuous_control(self):
        good = self.c.action_vector_m()
        for value, version in [(np.full(14, np.nan), V2), (np.full(14, np.inf), V2),
                               (good[:13], V2), (good, V1)]:
            before = self.c.data.ctrl.copy()
            self.assertFalse(self.c.apply_action(value, schema_version=version)["accepted"])
            np.testing.assert_array_equal(self.c.data.ctrl, before)
        for widths, reason in [([.02, .02], "continuous_gripper_requires_M02"),
                               ([0, .08], "asymmetric_gripper_action"), ([255, 255], "gripper_width_out_of_range")]:
            value = good.copy(); value[[6, 13]] = widths; value[0] += .01
            before = self.c.data.ctrl.copy()
            result = self.c.apply_action(value, schema_version=V2)
            self.assertEqual(result["reason"], reason)
            np.testing.assert_array_equal(self.c.data.ctrl, before)

    def test_limited_target_and_actual_state_are_separate(self):
        target = self.c.action_vector_m(); previous = target.copy()
        target[0] += .1
        state = self.c.state_vector().copy()
        result = self.c.apply_action(target, schema_version=V2, components=("motion",))
        self.assertTrue(result["accepted"])
        self.assertLessEqual(result["accepted_target"][0]-previous[0], .004001)
        self.assertEqual(result["requested_target"][0], target[0])
        self.assertIn(0, result["intervened_fields"])
        np.testing.assert_array_equal(result["actual_state"], state)

    def test_release_rejection_cannot_move_one_arm_first(self):
        closed = self.c.action_vector_m(); closed[[6, 13]] = 0
        self.c.apply_action(closed, schema_version=V2, components=("gripper",))
        target = self.c.action_vector_m(); target[[6, 13]] = .08; target[0] += .02
        before = self.c.data.ctrl.copy()
        with patch.object(DualArmTaskController, "release_ready", new_callable=PropertyMock, return_value=False):
            self.assertFalse(self.c.apply_action(target, schema_version=V2)["accepted"])
        np.testing.assert_array_equal(self.c.data.ctrl, before)

    def test_local_timeout_does_not_make_fresh_recovery_input_out_of_order(self):
        recorder = DualArmRecorder.__new__(DualArmRecorder)
        recorder.controller = self.c
        recorder.motion_lock = threading.Lock()
        recorder.latest_motion = recorder.latest_telemetry = None
        recorder.commands = queue.Queue()
        recorder.last_input_at = 0
        # A frame captured at 990 ms is still in transport when the local
        # watchdog pauses at 1000 ms; it arrives at 1010 ms (20 ms old).
        with patch("simulation.mujoco.dual_arm.teleop_service.time.time", return_value=1), \
                patch("simulation.mujoco.dual_arm.teleop_service.time.monotonic", return_value=1):
            recorder.process_commands()
        self.assertEqual(self.c.pause_reason, "command_timeout")
        packet = {"command": "dual_motion", "mode": "xy", "sentAt": 990,
                  "left": {"dx": .2}, "right": {"dx": .2}}
        result = self.c.apply_command(packet, now_ms=1010)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(self.c.apply_command({**packet, "sentAt": 980}, now_ms=1020)["reason"], "out_of_order_command")


class CommandProtocolTests(unittest.TestCase):
    def setUp(self):
        self.gate = CommandGate({"max_command_age_ms": 150, "max_future_command_ms": 1000})
        self.packet = {"protocol": PROTOCOL, "session": "session", "control_epoch": 2, "client_id": "page",
                       "sequence": 1, "eventId": "event-1", "sentAt": 990, "command": "dual_motion"}
        self.context = dict(session="session", epoch=2, now_ms=1000, monotonic=10., epoch_started_ms=500)

    def admit(self, **changes):
        return self.gate.admit({**self.packet, **changes}, **self.context)

    def test_session_epoch_age_numeric_and_required_context(self):
        for changes, reason in [({"protocol": None}, "command_protocol_required"),
                                ({"session": "old"}, "stale_session"), ({"control_epoch": 1}, "stale_epoch"),
                                ({"sentAt": 800}, "stale_command"), ({"sentAt": 2100}, "stale_command"),
                                ({"left": {"dx": float("nan")}}, "nonfinite_payload"),
                                ({"sequence": True}, "invalid_sequence_context"), ({"eventId": None}, "event_id_required")]:
            with self.subTest(changes=changes):
                self.assertEqual(self.admit(**changes)[1], reason)
        self.assertEqual(self.gate.sequences, {})

    def test_sequence_and_event_replay_rejected(self):
        self.assertIsNone(self.admit()[1])
        self.assertEqual(self.admit()[1], "duplicate_or_out_of_order_sequence")
        self.assertEqual(self.admit(sequence=0, eventId="other")[1], "duplicate_or_out_of_order_sequence")
        self.assertEqual(self.admit(sequence=2)[1], "duplicate_event")

    def test_full_input_history_reserves_recovery_and_reset_clears_ledger(self):
        ticket, _ = self.admit()
        self.gate.discard(ticket)
        self.gate.events = {f"old-{i}" for i in range(65536)}
        self.assertEqual(self.admit(sequence=2)[1], "command_history_full_reset_scene")
        ticket, reason = self.admit(sequence=3, command="reset", eventId="reset")
        self.assertIsNone(reason)
        self.assertIsNone(self.gate.consume(ticket, session="session", epoch=2, monotonic=10.01, now_ms=1010))
        self.context["epoch"] = 3
        self.assertIsNone(self.admit(control_epoch=3)[1])
        self.assertEqual(len(self.gate.events), 1)

    def test_execution_queue_age_and_scene_are_rechecked(self):
        ticket, _ = self.admit()
        self.assertEqual(self.gate.consume(ticket, session="session", epoch=2, monotonic=10.2, now_ms=1200), "expired_before_execution")
        ticket, _ = self.admit(sequence=2, eventId="event-2")
        self.assertEqual(self.gate.consume(ticket, session="session", epoch=3, monotonic=10.02, now_ms=1020), "stale_execution_context")

    def test_ticket_is_owned_and_consumed_exactly_once(self):
        self.assertEqual(self.gate.consume({"trusted": True}, session="session", epoch=2, monotonic=10, now_ms=1000), "missing_or_consumed_admission")
        ticket, _ = self.admit()
        context = dict(session="session", epoch=2, monotonic=10.01, now_ms=1010)
        self.assertIsNone(self.gate.consume(ticket, **context))
        self.assertEqual(self.gate.consume(ticket, **context), "missing_or_consumed_admission")


class DiagnosticTests(unittest.TestCase):
    def test_practice_trace_preserves_request_copy_and_strict_json(self):
        with tempfile.TemporaryDirectory() as temp:
            logger = SessionDiagnostics(Path(temp)/"session", {"source": "synthetic_test"})
            request = {"eventId": "1", "dx": float("nan"), "_admission": object()}
            logger.emit("request_rejected", request=request, reason="nonfinite_payload")
            request["eventId"] = "changed"
            logger.emit("state", actual_state=np.arange(16), accepted_target_m=np.zeros(14))
            logger.close()
            rows = [json.loads(s) for p in logger.root.glob("events-*.jsonl") for s in p.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["request"]["eventId"], "1")
            self.assertEqual(rows[0]["request"]["dx"], "nan")
            self.assertNotIn("_admission", rows[0]["request"])
            self.assertEqual(len(rows[1]["actual_state"]), 16)
            self.assertEqual(logger.status()["dropped"], 0)
            self.assertFalse((Path(temp)/"datasets").exists())

    def test_capacity_stops_logging_without_deleting_existing_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            logger = SessionDiagnostics(Path(temp)/"session", {}, chunk_bytes=250, max_chunks=2)
            for i in range(20):
                logger.emit("input", text="a"*100, sequence=i)
            logger.close()
            self.assertEqual(len(list(logger.root.glob("events-*.jsonl"))), 2)
            self.assertEqual(logger.error, "diagnostic_capacity_reached")
            self.assertGreater(logger.dropped, 0)


if __name__ == "__main__":
    unittest.main()
