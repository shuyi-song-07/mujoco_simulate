"""HTTP admission is separate from the trusted in-process action interface."""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading

PROTOCOL = "task3-command-v2"
INPUT_COMMANDS = {"dual_motion", "dual_gripper", "metric_action", "continuous_input", "telemetry", "pause"}
GUARDED_COMMANDS = INPUT_COMMANDS | {"reset", "record_start", "record_save", "record_failure", "record_discard"}


def finite_json(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_json(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_json(v) for v in value)
    return True


@dataclass(frozen=True)
class Admission:
    session: str
    epoch: int
    received_monotonic: float
    sent_at_ms: float


class CommandGate:
    def __init__(self, control):
        self.maximum_age_ms = float(control.get("max_command_age_ms", 150))
        self.future_ms = float(control.get("max_future_command_ms", 1000))
        self.lock = threading.Lock()
        self.context = None
        self.sequences = {}
        self.events = set()
        self.pending = {}

    def admit(self, payload, *, session, epoch, now_ms, monotonic, epoch_started_ms):
        """Return (server-owned ticket, reason). Do not refresh rejected inputs."""
        with self.lock:
            context = (session, epoch)
            if self.context != context:
                self.context = context
                self.sequences.clear()
                self.events.clear()
                self.pending.clear()
            if payload.get("protocol") != PROTOCOL:
                return None, "command_protocol_required"
            if payload.get("session") != session:
                return None, "stale_session"
            if type(payload.get("control_epoch")) is not int or payload["control_epoch"] != epoch:
                return None, "stale_epoch"
            client, sequence, event = payload.get("client_id"), payload.get("sequence"), payload.get("eventId")
            if not isinstance(client, str) or not 1 <= len(client) <= 128 or type(sequence) is not int or sequence < 0:
                return None, "invalid_sequence_context"
            if not isinstance(event, str) or not 1 <= len(event) <= 160:
                return None, "event_id_required"
            if not finite_json(payload):
                return None, "nonfinite_payload"
            sent = payload.get("sentAt")
            if type(sent) not in {float, int} or not -self.future_ms <= now_ms-sent <= self.maximum_age_ms or sent < epoch_started_ms:
                return None, "stale_command"
            if sequence <= self.sequences.get(client, -1):
                return None, "duplicate_or_out_of_order_sequence"
            if event in self.events:
                return None, "duplicate_event"
            recovery = payload.get("command") in {"reset", "record_save", "record_failure", "record_discard"}
            # Reserve bounded space to save/discard and reset after a long scene;
            # the normal-input limit must not block its own recovery operation.
            if client not in self.sequences and len(self.sequences) >= (36 if recovery else 32):
                return None, "command_client_limit_reset_scene"
            if len(self.events) >= (65792 if recovery else 65536):
                return None, "command_history_full_reset_scene"
            # Queue replacements retire their tickets explicitly. This cap also
            # bounds admission if an internal consumer fails to drain requests.
            if len(self.pending) >= 256:
                return None, "command_admission_full"
            ticket = Admission(session, epoch, monotonic, float(sent))
            self.pending[id(ticket)] = ticket
            self.sequences[client] = sequence
            self.events.add(event)
            return ticket, None

    def discard(self, ticket):
        with self.lock:
            if isinstance(ticket, Admission):
                self.pending.pop(id(ticket), None)

    def consume(self, ticket, *, session, epoch, monotonic, now_ms):
        with self.lock:
            if not isinstance(ticket, Admission) or self.pending.pop(id(ticket), None) is not ticket:
                return "missing_or_consumed_admission"
        if (ticket.session, ticket.epoch) != (session, epoch):
            return "stale_execution_context"
        queue_age = (monotonic-ticket.received_monotonic)*1000
        if queue_age < 0 or queue_age > self.maximum_age_ms or not -self.future_ms <= now_ms-ticket.sent_at_ms <= self.maximum_age_ms:
            return "expired_before_execution"
        return None
