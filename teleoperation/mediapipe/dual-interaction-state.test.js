import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_DUAL_CONFIG, DualInteractionState, DualHandTracker, WristFilter, detectionsFromMediaPipe, resolveOperation } from "./dual-interaction-state.js";

function pair(gesture = "None", overrides = {}) {
  return ["left", "right"].map((side) => ({ side, handedness_score: 0.99, gesture, confidence: 0.97, wrist: [side === "left" ? 0.2 : 0.8, 0.5], ...overrides[side] }));
}
function calibrationPair(overrides = {}) {
  return pair("Open_Palm", { left: { confidence: 0.65, ...overrides.left }, right: { confidence: 0.65, ...overrides.right } });
}
function session(latch = "OPEN", config = {}) {
  // Keep the original mapping as an explicit regression baseline. Rate/default
  // mapping behaviour is exercised independently below.
  const engine = new DualInteractionState({ wrist_filter: "ema", ema_alpha: 1, motion_mapping: "incremental", motion_speed_scale: 1, wrist_sensitivity: 0.04, ...config });
  engine.syncHealth({ gripper_latch: latch, task_state: "PREGRASP" }, 0);
  let now = -40;
  return {
    engine, results: [], time: () => now,
    tick(detections = pair(), dt = 40) { now += dt; const result = engine.update(detections, now); this.results.push(result); return result; },
    bind() { let result; for (let i = 0; i <= 50; i++) result = this.tick(pair("Open_Palm")); return result; },
  };
}
function setup(latch = "OPEN", config = {}) {
  const s = session(latch, config); s.bind();
  assert.equal(s.engine.tracker.calibrated, true);
  assert.equal(s.tick(pair("ILoveYou")).reason, "binding_gesture_released");
  return s;
}
function enter(s, gesture, detections = pair(gesture)) {
  let result; for (let i = 0; i < 6; i++) result = s.tick(detections); return result;
}
function recover(s, detections = pair("ILoveYou")) {
  let result; for (let i = 0; i < 6; i++) {
    result = s.tick(detections);
    assert.ok(!["dual_motion", "dual_gripper"].includes(result.command?.command));
  }
  assert.equal(result.reason, "tracking_recovered");
  return result;
}
function noActuation(result) { assert.ok(!["dual_motion", "dual_gripper"].includes(result.command?.command)); }

test("MediaPipe identity follows handedness even when detection array order changes", () => {
  const result = { landmarks: [[{ x: 0.8, y: 0.5 }], [{ x: 0.2, y: 0.5 }]], handedness: [[{ categoryName: "Right", score: 0.99 }], [{ categoryName: "Left", score: 0.98 }]], gestures: [[{ categoryName: "Victory", score: 0.9 }], [{ categoryName: "ILoveYou", score: 0.9 }]] };
  assert.equal(detectionsFromMediaPipe(result)[0].side, "right");
  assert.equal(detectionsFromMediaPipe(result, true)[0].side, "left");
  const tracker = new DualHandTracker();
  tracker.update(pair(), 0); tracker.update(pair().reverse(), 40);
  assert.deepEqual(tracker.hands.left.wrist_raw, [0.2, 0.5]);
  assert.deepEqual(tracker.hands.right.wrist_raw, [0.8, 0.5]);
});

test("two seconds of palms bind automatically in the same frame with no second confirmation API or stage", () => {
  const s = session();
  for (let i = 0; i < 50; i++) { noActuation(s.tick(pair("Open_Palm"))); assert.equal(s.engine.tracker.calibrated, false); }
  const bound = s.tick(pair("Open_Palm")), status = s.engine.tracker.gestureCalibrationStatus();
  assert.equal(s.engine.tracker.calibrated, true); assert.equal(bound.reason, "binding_complete");
  assert.equal(bound.command.command, "pause"); assert.equal(bound.telemetry.training_recordable, false);
  assert.equal(status.phase, "CONFIRMED"); assert.equal(status.source, "dual_open_palm");
  assert.equal(status.method, "dual_open_palm_stable"); assert.equal(status.evidence_ms, 2000);
  assert.equal(status.palm_binding_guard, true);
  assert.equal("confirm_progress" in status, false); assert.equal("ready_for_confirmation" in status, false);
  assert.equal("calibration_confirm_ms" in DEFAULT_DUAL_CONFIG, false);
  assert.equal(typeof s.engine.tracker.confirmCalibration, "undefined");
  assert.equal(typeof s.engine.tracker.isCalibrationReady, "undefined");
});

test("binding while CLOSED and continuing to hold palms never opens either gripper", () => {
  const s = session("CLOSED"), bound = s.bind(), boundAt = s.engine.tracker.calibrationConfirmedAt;
  assert.equal(bound.reason, "binding_complete");
  for (let i = 0; i < 100; i++) {
    const result = s.tick(pair("Open_Palm"));
    assert.equal(result.reason, "binding_complete"); noActuation(result);
    assert.equal(s.engine.evidence.open.left, 0); assert.equal(s.engine.evidence.open.right, 0);
  }
  assert.equal(s.engine.gripperLatch, "CLOSED"); assert.equal(s.engine.pendingGripper, null);
  assert.equal(s.engine.tracker.calibrationConfirmedAt, boundAt);
});

test("palm guard requires the same known confident non-palm gesture and pauses the first release frame", () => {
  const s = session(); s.bind();
  for (const detections of [pair("None"), pair("ILoveYou", { right: { confidence: 0.7 } }), pair("ILoveYou", { right: { gesture: "Victory" } }), pair("Closed_Fist", { left: { confidence: 0.7 } })]) {
    for (let i = 0; i < 20; i++) { noActuation(s.tick(detections)); assert.equal(s.engine.tracker.palmBindingGuard, true); }
  }
  const released = s.tick(pair("ILoveYou"));
  assert.equal(released.reason, "binding_gesture_released"); assert.equal(released.command.command, "pause");
  assert.equal(s.engine.tracker.palmBindingGuard, false);
  assert.equal(s.engine.tracker.hands.left.anchor, null);
  assert.equal(enter(s, "ILoveYou").command, null);
});

test("binding advanced through the app's tracker-only pending branch still keeps palm guard on the next normal frame", () => {
  const s = session("CLOSED");
  for (let i = 0; i <= 50; i++) {
    s.engine.tracker.update(pair("Open_Palm"), i * 40);
    noActuation(s.engine.pause("operation_pending"));
  }
  assert.equal(s.engine.tracker.calibrated, true);
  const result = s.engine.update(pair("Open_Palm"), 2040);
  assert.equal(result.reason, "binding_complete"); noActuation(result);
  assert.equal(s.engine.gripperLatch, "CLOSED");
});

test("XY input is independent, normalized and preserves the backend gripper latch", () => {
  const s = setup(); assert.equal(enter(s, "ILoveYou").command, null);
  const result = s.tick(pair("ILoveYou", { left: { wrist: [0.25, 0.45] } }));
  assert.equal(result.command.command, "dual_motion"); assert.equal(result.command.mode, "xy");
  assert.ok(result.command.left.dx < 0 && result.command.left.dy < 0);
  assert.ok(Math.hypot(result.command.left.dx, result.command.left.dy) <= 1 + 1e-12);
  assert.equal(Math.hypot(result.command.right.dx, result.command.right.dy), 0);
  assert.equal(s.engine.gripperLatch, "OPEN"); assert.equal(result.telemetry.training_recordable, true);
});

test("XY and Z alternate with a fresh anchor after each mode confirmation", () => {
  const s = setup(); enter(s, "ILoveYou");
  const shifted = pair("Victory", { left: { wrist: [0.2, 0.65] }, right: { wrist: [0.8, 0.65] } });
  let result = s.tick(shifted); assert.equal(result.command.command, "pause");
  for (let i = 0; i < 5; i++) result = s.tick(shifted);
  assert.equal(result.mode, "Z"); assert.equal(result.command, null);
  result = s.tick(pair("Victory", { left: { wrist: [0.2, 0.61] }, right: { wrist: [0.8, 0.65] } }));
  assert.equal(result.command.mode, "z"); assert.ok(result.command.left.dz > 0);
  assert.equal(enter(s, "ILoveYou").command, null);
});

test("gripper evidence tolerates one flicker and sends one event until authoritative acknowledgement", () => {
  const s = setup(), commands = [];
  for (let i = 0; i < 10; i++) commands.push(s.tick(pair("Closed_Fist")).command);
  commands.push(s.tick(pair("Closed_Fist", { left: { gesture: "None" } })).command);
  assert.ok(s.engine.evidence.close.left > 0);
  for (let i = 0; i < 25; i++) commands.push(s.tick(pair("Closed_Fist")).command);
  const events = commands.filter((command) => command?.command === "dual_gripper");
  assert.equal(events.length, 1); assert.equal(events[0].action, "close");
  assert.equal(s.engine.gripperLatch, "OPEN");
  s.engine.syncHealth({ gripper_latch: "CLOSED", task_state: "GRASP_VERIFY" }, s.time());
  assert.equal(s.engine.pendingGripper, null);
  for (let i = 0; i < 25; i++) assert.notEqual(s.tick(pair("Closed_Fist")).command?.command, "dual_gripper");
});

test("one-sided grasp or low-confidence fists cannot trigger either gripper", () => {
  const s = setup();
  for (let i = 0; i < 50; i++) noActuation(s.tick(pair("Closed_Fist", { right: { gesture: "ILoveYou" } })));
  for (let i = 0; i < 50; i++) noActuation(s.tick(pair("Closed_Fist", { left: { confidence: 0.7 }, right: { confidence: 0.7 } })));
});

test("rejected grasp cannot automatically retry before leaving the triggering gesture", () => {
  const s = setup(); let event;
  for (let i = 0; i < 20; i++) { const result = s.tick(pair("Closed_Fist")); if (result.command?.command === "dual_gripper") event = result.command; }
  s.engine.syncHealth({ gripper_latch: "OPEN", last_command_result: { eventId: event.eventId, accepted: false, reason: "grasp_not_ready" } }, s.time());
  for (let i = 0; i < 25; i++) noActuation(s.tick(pair("Closed_Fist")));
  s.tick(pair("ILoveYou"));
  const events = []; for (let i = 0; i < 20; i++) events.push(s.tick(pair("Closed_Fist")).command);
  assert.equal(events.filter((command) => command?.command === "dual_gripper").length, 1);
});

test("unassigned gestures, disagreement and camera time gaps pause without immediate grasp", () => {
  const s = setup(); enter(s, "ILoveYou");
  assert.equal(s.tick(pair("ILoveYou", { right: { gesture: "Pointing_Up" } })).reason, "gesture_disagreement");
  enter(s, "ILoveYou");
  assert.equal(s.tick(pair("ILoveYou", { right: { gesture: "Victory" } })).reason, "gesture_disagreement");
  noActuation(s.tick(pair("Closed_Fist"), 2000));
  assert.equal(s.engine.evidence.close.left, 0); assert.equal(s.engine.tracker.calibrated, true);
});

test("release requires its own evidence and motion keeps CLOSED", () => {
  const s = setup("CLOSED"); s.engine.syncHealth({ gripper_latch: "CLOSED", task_state: "DUAL_GRASPED" }, s.time());
  enter(s, "Victory"); s.tick(pair("Victory", { left: { wrist: [0.2, 0.45] } }));
  assert.equal(s.engine.gripperLatch, "CLOSED");
  const events = []; for (let i = 0; i < 20; i++) events.push(s.tick(pair("Open_Palm")).command);
  assert.equal(events.filter((command) => command?.command === "dual_gripper" && command.action === "open").length, 1);
});

test("a failed closed grasp permits confirmed palms to open once while preserving the failure", () => {
  const s = setup("CLOSED");
  s.engine.syncHealth({ gripper_latch: "CLOSED", task_state: "FAIL", failure_reason: "grasp_failure" }, s.time());
  for (let i = 0; i < 18; i++) noActuation(s.tick(pair("Open_Palm")));
  const event = s.tick(pair("Open_Palm")).command;
  assert.equal(event.command, "dual_gripper"); assert.equal(event.action, "open");
  assert.equal(event.gesture_confirm_ms, 700); assert.equal(s.engine.health.task_state, "FAIL");
  for (let i = 0; i < 25; i++) noActuation(s.tick(pair("Open_Palm")));
  s.engine.syncHealth({ gripper_latch: "OPEN", task_state: "FAIL", failure_reason: "grasp_failure" }, s.time());
  assert.equal(s.engine.pendingGripper, null);
  for (const gesture of ["Open_Palm", "Closed_Fist", "ILoveYou", "Victory"]) {
    for (let i = 0; i < 25; i++) {
      const result = s.tick(pair(gesture));
      noActuation(result); assert.equal(result.reason, "grasp_failure");
    }
  }
});

test("FAIL release still requires both qualified hands and never admits motion or closing", () => {
  const s = setup("CLOSED");
  s.engine.syncHealth({ gripper_latch: "CLOSED", task_state: "FAIL", failure_reason: "timeout" }, s.time());
  for (const detections of [pair("Closed_Fist"), pair("ILoveYou"), pair("Victory"), pair("Open_Palm", { right: { gesture: "Closed_Fist" } }), pair("Open_Palm", { right: { confidence: 0.54 } })]) {
    for (let i = 0; i < 25; i++) noActuation(s.tick(detections));
  }
  assert.equal(s.engine.evidence.open.right, 0);
  for (let i = 0; i < 18; i++) noActuation(s.tick(pair("Open_Palm")));
  assert.equal(s.tick(pair("Open_Palm")).command?.action, "open");
});

test("binding palms in FAIL cannot release a closed grasp until the operator leaves the binding gesture", () => {
  const s = session("CLOSED");
  s.engine.syncHealth({ gripper_latch: "CLOSED", task_state: "FAIL", failure_reason: "grasp_failure" }, s.time());
  s.bind();
  for (let i = 0; i < 40; i++) {
    const result = s.tick(pair("Open_Palm"));
    noActuation(result); assert.equal(result.reason, "binding_complete");
  }
  assert.equal(s.tick(pair("ILoveYou")).reason, "binding_gesture_released");
  const events = []; for (let i = 0; i < 25; i++) events.push(s.tick(pair("Open_Palm")).command);
  assert.equal(events.filter((command) => command?.command === "dual_gripper" && command.action === "open").length, 1);
});

test("temporary release speed rejection permits only a freshly reconfirmed retry with a new event id", () => {
  const s = setup("CLOSED");
  s.engine.syncHealth({ gripper_latch: "CLOSED", task_state: "DUAL_GRASPED" }, s.time());
  let event;
  for (let i = 0; i < 20; i++) {
    const result = s.tick(pair("Open_Palm")); if (result.command?.command === "dual_gripper") event = result.command;
  }
  for (let rejection = 0; rejection < 2; rejection++) {
    const previousEvent = event;
    s.engine.syncHealth({ gripper_latch: "CLOSED", task_state: "DUAL_GRASPED", last_command_result: { eventId: event.eventId, accepted: false, reason: "release_rejected_stop_before_opening" } }, s.time());
    assert.equal(s.engine.pendingGripper, null); assert.equal(s.engine.gripperBlocked, null);
    assert.equal(s.engine.evidence.open.left, 0); assert.equal(s.engine.evidence.open.right, 0);
    for (let i = 0; i < 18; i++) {
      const result = s.tick(pair("Open_Palm"));
      noActuation(result); assert.equal(result.reason, "release_waiting_for_stop");
      assert.equal(result.telemetry.gripper_retry_reason, "release_rejected_stop_before_opening");
    }
    event = s.tick(pair("Open_Palm")).command;
    assert.equal(event.command, "dual_gripper"); assert.equal(event.action, "open");
    assert.notEqual(event.eventId, previousEvent.eventId);
    assert.equal(s.engine.gripperLatch, "CLOSED");
  }
  s.engine.syncHealth({ gripper_latch: "OPEN", task_state: "RELEASE_VERIFY", last_command_result: { eventId: event.eventId, accepted: true } }, s.time());
  assert.equal(s.engine.pendingGripper, null); assert.equal(s.engine.gripperRetryReason, null);
  for (let i = 0; i < 25; i++) noActuation(s.tick(pair("Open_Palm")));
});

test("release retry evidence cannot survive a long sample gap or bypass saving", () => {
  for (const taskState of ["DUAL_GRASPED", "FAIL"]) {
    const s = setup("CLOSED"); let event;
    s.engine.syncHealth({ gripper_latch: "CLOSED", task_state: taskState }, s.time());
    for (let i = 0; i < 20; i++) {
      const result = s.tick(pair("Open_Palm")); if (result.command?.command === "dual_gripper") event = result.command;
    }
    s.engine.syncHealth({ gripper_latch: "CLOSED", task_state: taskState, last_command_result: { eventId: event.eventId, accepted: false, reason: "release_rejected_stop_before_opening" } }, s.time());
    for (let i = 0; i < 15; i++) noActuation(s.tick(pair("Open_Palm")));
    noActuation(s.tick(pair("Open_Palm"), 260));
    assert.equal(s.engine.evidence.open.left, 0); assert.equal(s.engine.evidence.open.right, 0);
    s.engine.syncHealth({ gripper_latch: "CLOSED", task_state: taskState, saving: true }, s.time());
    for (let i = 0; i < 30; i++) {
      const result = s.tick(pair("Open_Palm"));
      noActuation(result); assert.equal(result.reason, "saving");
    }
    assert.equal(s.engine.evidence.open.left, 0); assert.equal(s.engine.evidence.open.right, 0);
  }
});

test("other release rejections and acknowledgement timeout still require leaving the gesture", () => {
  for (const reason of ["stale_command", "episode_ended_reset_required", "ack_timeout"]) {
    const s = setup("CLOSED"); let event;
    for (let i = 0; i < 20; i++) {
      const result = s.tick(pair("Open_Palm")); if (result.command?.command === "dual_gripper") event = result.command;
    }
    if (reason === "ack_timeout") s.engine.syncHealth({ gripper_latch: "CLOSED" }, s.time() + 3001);
    else s.engine.syncHealth({ gripper_latch: "CLOSED", last_command_result: { eventId: event.eventId, accepted: false, reason } }, s.time());
    assert.equal(s.engine.gripperRetryReason, null); assert.equal(s.engine.gripperBlocked, "open");
    for (let i = 0; i < 30; i++) {
      const result = s.tick(pair("Open_Palm"));
      noActuation(result); assert.equal(result.reason, "gripper_rejected_change_gesture");
    }
  }
});

test("bilateral 70 percent palms release after 700ms in normal and failed closed states", () => {
  for (const task_state of ["DUAL_GRASPED", "FAIL"]) {
    const s = setup("CLOSED");
    s.engine.syncHealth({ task_state, gripper_latch: "CLOSED", failure_reason: "grasp_failure" }, s.time());
    const palms = pair("Open_Palm", { left: { confidence: 0.7 }, right: { confidence: 0.7 } });
    for (let i = 0; i < 18; i++) noActuation(s.tick(palms));
    const event = s.tick(palms).command;
    assert.equal(event.command, "dual_gripper"); assert.equal(event.action, "open");
    assert.deepEqual(event.telemetry.gripper_thresholds, { open: 0.55, close: 0.8, confirm_ms: 700 });
    assert.equal(s.engine.health.task_state, task_state);
    for (let i = 0; i < 30; i++) noActuation(s.tick(palms));
  }
});

test("each releasing hand needs at least 55 percent and each closing hand at least 80 percent", () => {
  const open = setup("CLOSED");
  const asymmetric = pair("Open_Palm", { left: { confidence: 0.7 }, right: { confidence: 0.54 } });
  for (let i = 0; i < 40; i++) noActuation(open.tick(asymmetric));
  assert.equal(open.engine.evidence.open.right, 0);
  const atThreshold = pair("Open_Palm", { left: { confidence: 0.55 }, right: { confidence: 0.55 } });
  for (let i = 0; i < 18; i++) noActuation(open.tick(atThreshold));
  assert.equal(open.tick(atThreshold).command?.action, "open");

  const close = setup("OPEN");
  for (const score of [0.7, 0.79]) {
    for (let i = 0; i < 40; i++) noActuation(close.tick(pair("Closed_Fist", { left: { confidence: score }, right: { confidence: score } })));
  }
  const fists = pair("Closed_Fist", { left: { confidence: 0.8 }, right: { confidence: 0.8 } });
  for (let i = 0; i < 18; i++) noActuation(close.tick(fists));
  assert.equal(close.tick(fists).command?.action, "close");
});

test("55 percent palm release cannot proceed with one hand or uncertain handedness", () => {
  const s = setup("CLOSED");
  const palms = pair("Open_Palm", { left: { confidence: 0.55 }, right: { confidence: 0.55 } });
  for (let i = 0; i < 25; i++) noActuation(s.tick(palms.slice(0, 1)));
  for (let i = 0; i < 25; i++) noActuation(s.tick(palms.map((hand) => ({ ...hand, handedness_score: 0.79 }))));
  assert.equal(s.engine.gripperLatch, "CLOSED"); assert.equal(s.engine.tracker.calibrated, true);
  assert.equal(s.engine.evidence.open.left, 0); assert.equal(s.engine.evidence.open.right, 0);
  recover(s, palms);
  for (let i = 0; i < 18; i++) noActuation(s.tick(palms));
  assert.equal(s.tick(palms).command?.action, "open");
});

test("one high-scoring fist cannot compensate for the other hand below 80 percent", () => {
  const s = setup();
  const asymmetric = pair("Closed_Fist", { left: { confidence: 0.99 }, right: { confidence: 0.79 } });
  for (let i = 0; i < 35; i++) noActuation(s.tick(asymmetric));
  assert.equal(s.engine.evidence.close.left, 700); assert.equal(s.engine.evidence.close.right, 0);
  const qualified = pair("Closed_Fist", { left: { confidence: 0.99 }, right: { confidence: 0.8 } });
  for (let i = 0; i < 18; i++) noActuation(s.tick(qualified));
  assert.equal(s.tick(qualified).command?.action, "close");
});

test("frozen legacy configs retain their shared threshold through construction and reconfiguration", () => {
  const legacy = { ...DEFAULT_DUAL_CONFIG };
  delete legacy.gripper_open_confidence_min; delete legacy.gripper_close_confidence_min;
  const s = setup("CLOSED", legacy);
  assert.deepEqual(s.engine.telemetry().gripper_thresholds, { open: 0.85, close: 0.85, confirm_ms: 700 });
  const palms = pair("Open_Palm", { left: { confidence: 0.7 }, right: { confidence: 0.7 } });
  for (let i = 0; i < 40; i++) noActuation(s.tick(palms));
  s.engine.reconfigure({ ...DEFAULT_DUAL_CONFIG }); recover(s, palms);
  const events = []; for (let i = 0; i < 25; i++) events.push(s.tick(palms).command);
  assert.equal(events.filter((command) => command?.action === "open").length, 1);
  s.engine.reconfigure(legacy);
  assert.deepEqual(s.engine.telemetry().gripper_thresholds, { open: 0.85, close: 0.85, confirm_ms: 700 });
});

test("an explicitly supplied action threshold wins while a missing action uses the legacy value", () => {
  const engine = new DualInteractionState({ gripper_confidence_min: 0.8, gripper_close_confidence_min: 0.9 });
  assert.deepEqual(engine.telemetry().gripper_thresholds, { open: 0.8, close: 0.9, confirm_ms: 700 });
  engine.reconfigure({ gripper_confidence_min: 0.85, gripper_open_confidence_min: 0.65 });
  assert.equal(engine.gripperConfidence("open"), 0.65); assert.equal(engine.gripperConfidence("close"), 0.85);
});

test("lower palm release threshold cannot bypass binding protection with either unassigned pointing or thumb gestures", () => {
  const s = session("CLOSED"); s.bind();
  const palms = pair("Open_Palm", { left: { confidence: 0.7 }, right: { confidence: 0.7 } });
  for (const detections of [palms, pair("Pointing_Up"), pair("Thumb_Up"), palms]) {
    for (let i = 0; i < 30; i++) {
      const result = s.tick(detections);
      noActuation(result); assert.equal(result.reason, "binding_complete");
    }
  }
  assert.equal(s.tick(pair("ILoveYou")).reason, "binding_gesture_released");
  for (let i = 0; i < 18; i++) noActuation(s.tick(palms));
  assert.equal(s.tick(palms).command?.action, "open");
});

test("70 percent release still rejects uncertain identity and clears evidence over a sample gap", () => {
  const s = setup("CLOSED");
  const palms = pair("Open_Palm", { left: { confidence: 0.7 }, right: { confidence: 0.7 } });
  for (let i = 0; i < 14; i++) noActuation(s.tick(palms));
  noActuation(s.tick(palms, 260));
  assert.equal(s.engine.evidence.open.left, 0);
  const uncertain = palms.map((hand) => ({ ...hand, handedness_score: 0.79 }));
  for (let i = 0; i < 30; i++) noActuation(s.tick(uncertain));
  assert.equal(s.engine.evidence.open.left, 0); assert.equal(s.engine.evidence.open.right, 0);
  recover(s, palms);
  for (let i = 0; i < 18; i++) noActuation(s.tick(palms));
  assert.equal(s.tick(palms).command?.action, "open");
});

test("pointing and thumb gestures follow the same unassigned rules as None without dedicated pause behavior", () => {
  for (const other of [null, "ILoveYou", "Open_Palm"]) {
    const results = [];
    for (const unassigned of ["None", "Pointing_Up", "Thumb_Up"]) {
      const s = setup("CLOSED"); enter(s, "ILoveYou");
      const detections = pair(unassigned, other ? { left: { gesture: other } } : {});
      const result = s.tick(detections); noActuation(result);
      assert.equal(s.engine.tracker.hands.left.anchor, null);
      results.push({ mode: result.mode, reason: result.reason, command: result.command?.command });
    }
    assert.deepEqual(results[0], results[1]); assert.deepEqual(results[0], results[2]);
  }
});

test("motion diagnostics separate low confidence, unassigned gestures and disagreement", () => {
  const s = setup(); enter(s, "ILoveYou");
  const low = s.tick(pair("ILoveYou", { right: { confidence: 0.74 } }));
  assert.equal(low.mode, "PAUSE"); assert.equal(low.reason, "motion_low_confidence");
  assert.equal(low.telemetry.pause_reason, "motion_low_confidence"); noActuation(low);
  assert.equal(s.tick(pair("None")).reason, "unassigned_gesture");
  assert.equal(s.tick(pair("ILoveYou", { right: { gesture: "Victory" } })).reason, "gesture_disagreement");
  assert.equal(s.tick(pair("Thumb_Up")).reason, "unassigned_gesture");
  const ready = enter(s, "Victory");
  assert.equal(ready.mode, "Z"); assert.equal(ready.reason, null); assert.equal(ready.command, null);
});

test("lowering hands retains binding and returns through recovery plus a fresh motion anchor", () => {
  const s = setup(); enter(s, "ILoveYou"); const boundAt = s.engine.tracker.calibrationConfirmedAt;
  const lost = s.tick([]); assert.equal(lost.reason, "hand_lost"); assert.equal(lost.command.command, "pause");
  s.tick([], 1000); assert.equal(s.engine.tracker.calibrated, true);
  const relocated = pair("ILoveYou", { left: { wrist: [0.35, 0.65] }, right: { wrist: [0.65, 0.65] } }).reverse();
  recover(s, relocated); assert.equal(s.engine.tracker.hands.left.anchor, null);
  assert.equal(enter(s, "ILoveYou", relocated).command, null);
  const moved = s.tick(pair("ILoveYou", { left: { wrist: [0.36, 0.64] }, right: { wrist: [0.65, 0.65] } }).reverse());
  assert.equal(moved.command.command, "dual_motion"); assert.equal(Math.hypot(moved.command.right.dx, moved.command.right.dy), 0);
  assert.equal(s.engine.tracker.calibrationConfirmedAt, boundAt);
});

test("unreliable single detections never update trusted wrists or cancel an existing binding", () => {
  for (const latch of ["OPEN", "CLOSED"]) {
    const s = setup(latch), boundAt = s.engine.tracker.calibrationConfirmedAt;
    const previous = structuredClone(s.engine.tracker.hands);
    for (const override of [{ handedness_score: 0.6 }, { handedness_score: NaN }, { side: undefined }, { wrist: [NaN, 0.5] }, { wrist: [] }]) {
      for (let i = 0; i < 8; i++) {
        const result = s.tick([{ ...pair(latch === "OPEN" ? "Closed_Fist" : "Open_Palm")[0], ...override }]);
        assert.equal(result.reason, "hand_lost"); noActuation(result);
        assert.equal(s.engine.tracker.calibrated, true);
        for (const side of ["left", "right"]) assert.deepEqual(s.engine.tracker.hands[side].wrist_raw, previous[side].wrist_raw);
      }
    }
    assert.equal(s.engine.gripperLatch, latch); assert.equal(s.engine.tracker.calibrationConfirmedAt, boundAt);
  }
});

test("two-hand low scores, duplicate labels and invalid wrists pause but retain binding source and time", () => {
  const duplicate = pair("Closed_Fist"); duplicate[1].side = "left";
  for (const detections of [pair("Closed_Fist", { right: { handedness_score: 0.79 } }), duplicate, pair("Open_Palm", { left: { wrist: [] } })]) {
    const s = setup(), boundAt = s.engine.tracker.calibrationConfirmedAt;
    for (let i = 0; i < 40; i++) { assert.equal(s.tick(detections).reason, "ambiguous_handedness"); assert.equal(s.engine.tracker.calibrated, true); }
    assert.equal(s.engine.tracker.calibrationSource, "dual_open_palm"); assert.equal(s.engine.tracker.calibrationConfirmedAt, boundAt);
    recover(s); assert.equal(s.engine.pendingGripper, null);
  }
});

test("persistent label reversal cannot age out of conflict or bypass it through a long hand loss", () => {
  const s = setup(), previous = structuredClone(s.engine.tracker.hands);
  const reversedLabels = pair("Closed_Fist", { left: { wrist: [0.8, 0.5] }, right: { wrist: [0.2, 0.5] } }).reverse();
  for (let i = 0; i < 100; i++) {
    const result = s.tick(reversedLabels); assert.equal(result.reason, "identity_conflict"); noActuation(result);
    assert.equal(s.engine.tracker.calibrated, true); assert.equal(s.engine.tracker.recoveryCandidate, null);
  }
  s.tick([], 3000);
  assert.equal(s.tick(reversedLabels).reason, "identity_conflict");
  for (const side of ["left", "right"]) assert.deepEqual(s.engine.tracker.hands[side].wrist_raw, previous[side].wrist_raw);
  assert.equal(s.engine.pendingGripper, null);
});

test("identity conflict recovers only after the original consistent labelled pair is stable again", () => {
  const s = setup();
  s.tick(pair("ILoveYou", { left: { wrist: [0.8, 0.5] }, right: { wrist: [0.2, 0.5] } }));
  recover(s, pair("ILoveYou").reverse());
  assert.equal(s.engine.tracker.identityConflictReference, null);
  assert.equal(enter(s, "ILoveYou").command, null);
  const moved = s.tick(pair("ILoveYou", { right: { wrist: [0.81, 0.49] } }).reverse());
  assert.equal(moved.command.command, "dual_motion"); assert.equal(Math.hypot(moved.command.left.dx, moved.command.left.dy), 0);
});

test("a conflicting pair cannot recover near the centre until labels agree and hands are separated", () => {
  const s = setup();
  s.tick(pair("Closed_Fist", { left: { wrist: [0.8, 0.5] }, right: { wrist: [0.2, 0.5] } }));
  for (const wrists of [[[0.52, 0.5], [0.48, 0.5]], [[0.48, 0.5], [0.52, 0.5]]]) {
    for (let i = 0; i < 30; i++) {
      const result = s.tick(pair("Closed_Fist", { left: { wrist: wrists[0] }, right: { wrist: wrists[1] } }));
      assert.equal(result.reason, "identity_conflict"); noActuation(result);
      assert.equal(s.engine.tracker.recoveryCandidate, null); assert.equal(s.engine.tracker.calibrated, true);
    }
  }
  recover(s, pair("ILoveYou", { left: { wrist: [0.3, 0.5] }, right: { wrist: [0.7, 0.5] } }).reverse());
  assert.equal(s.engine.pendingGripper, null);
});

test("a reasonable common translation after long hand loss can recover without silently exchanging arms", () => {
  const s = setup(); s.tick([], 1500);
  const shifted = pair("ILoveYou", { left: { wrist: [0.3, 0.85] }, right: { wrist: [0.9, 0.85] } }).reverse();
  recover(s, shifted); assert.equal(enter(s, "ILoveYou", shifted).command, null);
  const moved = s.tick(pair("ILoveYou", { left: { wrist: [0.31, 0.84] }, right: { wrist: [0.9, 0.85] } }).reverse());
  assert.equal(moved.command.command, "dual_motion"); assert.equal(Math.hypot(moved.command.right.dx, moved.command.right.dy), 0);
});

test("recovery requires both enough samples and 200ms of stable wrists", () => {
  const s = setup(); s.tick([]);
  for (let i = 0; i < 30; i++) {
    const delta = (i % 6) * 0.02;
    const result = s.tick(pair("Closed_Fist", { left: { wrist: [0.2 + delta, 0.5] }, right: { wrist: [0.8 + delta, 0.5] } }));
    assert.equal(result.reason, "tracking_recovery"); noActuation(result);
  }
  recover(s); assert.equal(s.engine.pendingGripper, null);
});

test("duplicate and backwards camera timestamps pause without removing binding or supplying recovery evidence", () => {
  const s = setup();
  for (let i = 0; i < 100; i++) {
    const result = s.engine.update(pair("Closed_Fist"), s.time());
    assert.equal(result.reason, "tracking_stalled"); noActuation(result);
  }
  assert.equal(s.engine.tracker.calibrated, true);
  const backwards = s.engine.update(pair("Closed_Fist"), s.time() - 80);
  assert.equal(backwards.reason, "clock_reversal"); noActuation(backwards);
  recover(s); assert.equal(s.engine.tracker.calibrated, true);
});

test("explicit unbind clears pending control and binding while keeping the observed CLOSED latch", () => {
  const s = setup();
  for (let i = 0; i < 20; i++) s.tick(pair("Closed_Fist"));
  assert.ok(s.engine.pendingGripper);
  s.engine.syncHealth({ gripper_latch: "CLOSED", task_state: "PREGRASP" }, s.time());
  const released = s.engine.unbind();
  assert.equal(released.command.command, "pause"); assert.equal(released.reason, "binding_released");
  assert.equal(s.engine.tracker.calibrated, false); assert.equal(s.engine.tracker.calibrationSource, null);
  assert.equal(s.engine.pendingGripper, null); assert.equal(s.engine.gripperLatch, "CLOSED");
  assert.equal(s.bind().reason, "binding_complete");
  for (let i = 0; i < 30; i++) { noActuation(s.tick(pair("Open_Palm"))); assert.equal(s.engine.gripperLatch, "CLOSED"); }
});

test("reconfigure preserves source, time and palm guard while requiring fresh tracking", () => {
  for (const releaseGuard of [false, true]) {
    const s = session("CLOSED"); s.bind();
    if (releaseGuard) s.tick(pair("ILoveYou"));
    const tracker = s.engine.tracker, boundAt = tracker.calibrationConfirmedAt, guard = tracker.palmBindingGuard;
    const result = s.engine.reconfigure({ wrist_filter: "ema", ema_alpha: 0.5, calibration_ms: 3000 });
    assert.equal(result.reason, "config_updated"); assert.equal(tracker.calibrated, true);
    assert.equal(tracker.calibrationSource, "dual_open_palm"); assert.equal(tracker.calibrationConfirmedAt, boundAt);
    assert.equal(tracker.palmBindingGuard, guard); assert.equal(tracker.recoveryPending, true);
    assert.equal(tracker.filters.left.config.ema_alpha, 0.5);
    recover(s, pair("Open_Palm"));
    if (!releaseGuard) assert.equal(s.tick(pair("Open_Palm")).reason, "binding_complete");
  }
});

test("scene or backend-session reset preserves binding and clears old motion and gripper input", () => {
  const s = setup(); enter(s, "ILoveYou");
  for (let i = 0; i < 20; i++) s.tick(pair("Closed_Fist"));
  assert.ok(s.engine.pendingGripper);
  const boundAt = s.engine.tracker.calibrationConfirmedAt, result = s.engine.resetScene();
  assert.equal(result.command.command, "pause"); assert.equal(result.reason, "scene_reset");
  assert.equal(s.engine.pendingGripper, null); assert.equal(s.engine.evidence.close.left, 0);
  assert.equal(s.engine.tracker.calibrated, true); assert.equal(s.engine.tracker.calibrationConfirmedAt, boundAt);
  recover(s); assert.equal(enter(s, "ILoveYou").command, null);
});

test("configuration changes before binding clear old calibration evidence", () => {
  const s = session(); for (let i = 0; i < 40; i++) s.tick(calibrationPair());
  assert.ok(s.engine.tracker.palmsEvidenceMs > 1000);
  s.engine.reconfigure({ calibration_ms: 3000 });
  assert.equal(s.engine.tracker.palmsEvidenceMs, 0); assert.equal(s.engine.tracker.calibrated, false);
  s.bind(); assert.equal(s.engine.tracker.calibrated, false);
});

test("0.65-confidence palms bind despite isolated classifier flickers without any actuation", () => {
  const s = session("CLOSED"); let bound;
  for (let i = 0; i < 160 && !s.engine.tracker.calibrated; i++) bound = s.tick(calibrationPair(i % 4 === 3 ? { right: { gesture: "None" } } : {}));
  assert.equal(s.engine.tracker.calibrated, true); assert.equal(bound.reason, "binding_complete");
  assert.equal(s.engine.gripperLatch, "CLOSED"); s.results.forEach(noActuation);
});

test("persistent palm scores below 0.60 cannot bind and explain the side and required score", () => {
  const s = session(); for (let i = 0; i < 150; i++) s.tick(calibrationPair({ right: { confidence: 0.59 } }));
  const status = s.engine.tracker.gestureCalibrationStatus();
  assert.equal(status.evidence_ms, 0); assert.equal(status.phase, "PALMS");
  assert.deepEqual(status.blocker, { code: "gesture_confidence_low", side: "right", actual: 0.59, required: 0.60 });
  assert.equal(status.detection_count, 2); assert.equal(status.detections[1].confidence, 0.59);
  assert.ok(status.reset_count > 0); assert.equal(s.engine.tracker.calibrated, false);
});

test("short flickers add no missing time and expired flickers reset even on the recovery frame", () => {
  const s = session(); for (let i = 0; i < 11; i++) s.tick(calibrationPair());
  assert.equal(s.engine.tracker.palmsEvidenceMs, 400);
  s.tick(calibrationPair({ right: { gesture: "None" } }));
  s.tick(calibrationPair(), 120); assert.equal(s.engine.tracker.palmsEvidenceMs, 400);
  s.tick(calibrationPair()); assert.equal(s.engine.tracker.palmsEvidenceMs, 440);
  s.tick(pair("Victory")); s.tick(calibrationPair(), 161);
  assert.equal(s.engine.tracker.palmsEvidenceMs, 0);
  assert.equal(s.engine.tracker.gestureCalibrationStatus().last_reset_reason, "gesture_mismatch");
});

test("trusted 300ms samples bind automatically without lowering motion, grasp or identity thresholds", () => {
  const s = session();
  for (let i = 0; i < 7; i++) { noActuation(s.tick(calibrationPair(), 300)); assert.equal(s.engine.tracker.calibrated, false); }
  assert.equal(s.tick(calibrationPair(), 300).reason, "binding_complete");
  assert.equal(s.engine.tracker.calibrated, true);
  for (let i = 0; i < 30; i++) noActuation(s.tick(pair("ILoveYou", { left: { confidence: 0.65 }, right: { confidence: 0.65 } })));
  for (let i = 0; i < 30; i++) noActuation(s.tick(pair("Closed_Fist", { left: { confidence: 0.65 }, right: { confidence: 0.65 } })));
  assert.equal(s.engine.tracker.palmBindingGuard, true);
  s.tick(pair("ILoveYou", { right: { handedness_score: 0.79 } }));
  assert.equal(s.engine.tracker.calibrated, true); assert.equal(s.engine.tracker.calibrationBlocker.code, "handedness_confidence_low");
});

test("before binding hard loss, identity faults, wrist motion and long gaps reset calibration evidence", () => {
  const duplicate = calibrationPair(); duplicate[1].side = "left";
  for (const [detections, code, dt] of [
    [[], "hand_lost", 40],
    [calibrationPair({ left: { side: "unknown" } }), "unknown_handedness", 40],
    [duplicate, "duplicate_handedness", 40],
    [calibrationPair({ left: { handedness_score: 0.79 } }), "handedness_confidence_low", 40],
    [calibrationPair({ left: { wrist: [] } }), "invalid_wrist", 40],
    [calibrationPair({ left: { wrist: [0.7, 0.5] }, right: { wrist: [0.3, 0.5] } }), "identity_conflict", 40],
    [calibrationPair({ left: { wrist: [0.32, 0.5] } }), "wrist_motion", 40],
    [calibrationPair(), "sample_gap", 1000],
    [calibrationPair(), "clock_reversal", -80],
  ]) {
    const s = session(); for (let i = 0; i < 15; i++) s.tick(calibrationPair());
    noActuation(s.tick(detections, dt));
    const status = s.engine.tracker.gestureCalibrationStatus();
    assert.equal(status.blocker.code, code); assert.equal(status.last_reset_reason, code);
    assert.equal(status.evidence_ms, 0); assert.equal(s.engine.tracker.calibrated, false);
  }
});

test("frozen camera timestamps, UI polling and thumbs alone cannot complete binding", () => {
  const s = session();
  for (let i = 0; i < 100; i++) s.tick(pair("Thumb_Up"));
  assert.equal(s.engine.tracker.calibrated, false);
  for (let i = 0; i < 20; i++) s.tick(calibrationPair());
  const progress = s.engine.tracker.palmsEvidenceMs;
  for (let i = 0; i < 200; i++) noActuation(s.engine.update(calibrationPair(), s.time()));
  assert.equal(s.engine.tracker.gestureCalibrationStatus(s.time() + 10000).evidence_ms, progress);
  assert.equal(s.engine.tracker.advanceGestureCalibration(s.time() + 10000), false);
  assert.equal(s.engine.tracker.calibrated, false);
});

test("One Euro smooths wrist jitter and EMA is still a reproducible baseline", () => {
  const filter = new WristFilter(); filter.update([0.5, 0.5], 0);
  const point = filter.update([0.52, 0.48], 40); assert.ok(point[0] > 0.5 && point[0] < 0.52);
  const ema = new WristFilter({ wrist_filter: "ema", ema_alpha: 0.5 });
  ema.update([0.5, 0.5], 0); assert.deepEqual(ema.update([0.6, 0.4], 40), [0.55, 0.45]);
});

test("save stays pending during encoding and only its own backend ACK resolves it", () => {
  const pending = { command: "record_save", eventId: "save-1", startedAt: 1000 };
  assert.equal(resolveOperation(pending, { saving: true }, 120000).pending, pending);
  assert.equal(resolveOperation(pending, { last_command_result: { eventId: "other", accepted: true } }, 5000).pending, pending);
  assert.equal(resolveOperation(pending, { last_command_result: { eventId: "save-1", accepted: true } }, 130000).outcome, "accepted");
  assert.equal(resolveOperation(pending, { last_command_result: { eventId: "save-1", accepted: false, reason: "not_done" } }, 2000).reason, "not_done");
  assert.equal(resolveOperation(pending, {}, 17000).outcome, "timeout");
});

test("queued exit may close health before ACK without pretending that shutdown was confirmed", () => {
  const pending = { command: "record_stop", eventId: "stop-1", startedAt: 0, queued: true };
  assert.equal(resolveOperation(pending, {}, 300, false).outcome, "connection_closed");
  assert.equal(resolveOperation({ ...pending, queued: false }, {}, 300, false).outcome, null);
  assert.equal(resolveOperation({ ...pending, command: "record_save" }, {}, 300, false).outcome, null);
});

function rateSetup(speed = 0.7) {
  return setup("OPEN", { motion_mapping: "rate", motion_speed_scale: speed, wrist_sensitivity: 0.012 });
}

test("default incremental settings and the actual operator parameters are explicit in telemetry", () => {
  const engine = new DualInteractionState();
  assert.equal(engine.config.gripper_sample_gap_ms, 250);
  assert.equal(engine.config.motion_confidence_min, 0.75); assert.equal(engine.config.gripper_confidence_min, 0.85);
  assert.deepEqual(engine.telemetry().operator_control, {
    motion_mapping: "incremental", motion_speed_scale: 0.7, rate_motion_range: 0.08,
    rate_dead_zone: 0.01, wrist_sensitivity: 0.012, wrist_dead_zone: 0.003,
  });
});

test("default mapping settles while holding and returning from an unassigned gesture never jumps back", () => {
  const s = setup("OPEN", { ...DEFAULT_DUAL_CONFIG });
  enter(s, "ILoveYou");
  const offset = pair("ILoveYou", { left: { wrist: [0.245, 0.47] } });
  let moves = 0;
  for (let i = 0; i < 25; i++) if (s.tick(offset).command?.command === "dual_motion") moves += 1;
  assert.ok(moves > 0 && moves < 20, "the filter may settle briefly, but a held position must not become continuous velocity");
  for (let i = 0; i < 25; i++) assert.equal(s.tick(offset).command, null);
  assert.equal(s.tick(pair("Thumb_Up")).reason, "unassigned_gesture");
  const comfortable = pair("Thumb_Up", { left: { wrist: [0.16, 0.54] } });
  for (let i = 0; i < 20; i++) noActuation(s.tick(comfortable));
  const resumed = comfortable.map((hand) => ({ ...hand, gesture: "ILoveYou" }));
  assert.equal(enter(s, "ILoveYou", resumed).command, null);
  for (let i = 0; i < 10; i++) assert.equal(s.tick(resumed).command, null);
});

test("rate holds a fixed neutral and keeps moving from a small steady offset on fresh frames", () => {
  const s = rateSetup(); enter(s, "ILoveYou");
  const neutral = [...s.engine.tracker.hands.left.anchor];
  const offset = pair("ILoveYou", { left: { wrist: [0.245, 0.5] } });
  for (let i = 0; i < 10; i++) {
    const result = s.tick(offset);
    assert.equal(result.command.command, "dual_motion"); assert.equal(result.command.mode, "xy");
    assert.ok(Math.abs(result.command.left.dy + 0.35) < 1e-9); assert.equal(result.command.left.dx, 0);
    assert.equal(Math.hypot(result.command.right.dx, result.command.right.dy), 0);
    assert.deepEqual(s.engine.tracker.hands.left.anchor, neutral);
  }
  const centered = s.tick(pair("ILoveYou", { left: { wrist: [0.208, 0.508] } }));
  assert.equal(centered.command, null); assert.equal(centered.telemetry.training_recordable, false);
  assert.deepEqual(s.engine.tracker.hands.left.anchor, neutral);
});

test("rate speed choices preserve the per-command norm cap and mirror directions", () => {
  for (const speed of [0.3, 0.7, 1]) {
    const s = rateSetup(speed); enter(s, "ILoveYou");
    const result = s.tick(pair("ILoveYou", { left: { wrist: [0.3, 0.4] } }));
    const delta = result.command.left;
    assert.ok(delta.dx < 0 && delta.dy < 0);
    assert.ok(Math.abs(Math.hypot(delta.dx, delta.dy) - speed) < 1e-9);
    assert.equal(result.telemetry.operator_control.motion_speed_scale, speed);
  }
});

test("rate does not repeat motion above 25Hz or from a duplicated camera timestamp", () => {
  const s = rateSetup(); enter(s, "ILoveYou");
  const offset = pair("ILoveYou", { left: { wrist: [0.245, 0.5] } });
  let commands = 0;
  for (let i = 0; i < 10; i++) if (s.tick(offset, 20).command?.command === "dual_motion") commands += 1;
  assert.equal(commands, 5);
  const repeated = s.engine.update(offset, s.time());
  assert.equal(repeated.reason, "tracking_stalled"); noActuation(repeated);
  assert.equal(s.engine.tracker.calibrated, true);
});

test("rate unassigned gesture, mode change and scene reset establish a new neutral before accepting motion", () => {
  const s = rateSetup(); enter(s, "ILoveYou");
  const offset = pair("ILoveYou", { left: { wrist: [0.245, 0.5] } });
  assert.equal(s.tick(offset).command.command, "dual_motion");
  assert.equal(s.tick(pair("Thumb_Up")).reason, "unassigned_gesture");
  assert.equal(enter(s, "ILoveYou", offset).command, null);
  assert.equal(s.tick(offset).command, null);
  const newZ = pair("Victory", { left: { wrist: [0.245, 0.56] } });
  assert.equal(enter(s, "Victory", newZ).command, null); assert.equal(s.tick(newZ).command, null);
  s.engine.resetScene(); recover(s, newZ);
  assert.equal(enter(s, "Victory", newZ).command, null); assert.equal(s.tick(newZ).command, null);
});

test("changing mappings and recovering lost hands cannot carry an old rate offset into the new neutral", () => {
  const s = rateSetup(); enter(s, "ILoveYou");
  const offset = pair("ILoveYou", { left: { wrist: [0.245, 0.5] } });
  s.tick(offset); s.tick([]); s.tick([], 1000);
  recover(s, offset); assert.equal(enter(s, "ILoveYou", offset).command, null);
  assert.equal(s.tick(offset).command, null);
  s.engine.reconfigure({ ...s.engine.config, motion_mapping: "incremental" });
  recover(s, offset); assert.equal(enter(s, "ILoveYou", offset).command, null);
  assert.equal(s.tick(offset).command, null);
  assert.equal(s.engine.telemetry().operator_control.motion_mapping, "incremental");
});

test("incremental motion preserves a slow axis while the other axis crosses its dead zone every frame", () => {
  const s = setup(); enter(s, "ILoveYou");
  let dx = 0, dy = 0;
  for (let i = 1; i <= 50; i++) {
    const result = s.tick(pair("ILoveYou", { left: { wrist: [0.2 + i * 0.004, 0.5 + i * 0.001] } }));
    if (result.command?.command === "dual_motion") { dx += result.command.left.dx; dy += result.command.left.dy; }
  }
  assert.ok(dx > 1); assert.ok(Math.abs(dy + 5) < 1e-9);
});

test("incremental Z updates only its vertical anchor and retains the inactive horizontal coordinate", () => {
  const s = setup(); enter(s, "Victory");
  for (let i = 1; i <= 10; i++) {
    const result = s.tick(pair("Victory", { left: { wrist: [0.2 + i * 0.001, 0.5 - i * 0.004] } }));
    assert.equal(result.command.command, "dual_motion"); assert.ok(result.command.left.dz > 0);
    assert.equal(s.engine.tracker.hands.left.anchor[0], 0.2);
  }
});

test("fresh 0.90-confidence bilateral fists can confirm at 160ms and 200ms sample cadence exactly once", () => {
  for (const interval of [160, 200]) {
    const s = setup(); let firstSeen = null, eventTime = null, count = 0;
    for (let i = 0; i < 15; i++) {
      const result = s.tick(pair("Closed_Fist", { left: { confidence: 0.90 }, right: { confidence: 0.90 } }), interval);
      firstSeen ??= s.time();
      if (result.command?.command === "dual_gripper") { count += 1; eventTime = s.time(); }
    }
    assert.equal(count, 1); assert.ok(eventTime - firstSeen >= 700);
    assert.equal(s.engine.gripperLatch, "OPEN");
  }
});

test("gripper sample gaps above 250ms drop accumulated evidence rather than counting an unseen interval", () => {
  const s = setup(); for (let i = 0; i < 10; i++) s.tick(pair("Closed_Fist"));
  assert.ok(s.engine.evidence.close.left > 0);
  noActuation(s.tick(pair("Closed_Fist"), 300));
  assert.equal(s.engine.evidence.close.left, 0); assert.equal(s.engine.evidence.close.right, 0);
  for (let i = 0; i < 17; i++) noActuation(s.tick(pair("Closed_Fist")));
  assert.equal(s.tick(pair("Closed_Fist")).command.command, "dual_gripper");
});

test("FAIL/DONE and a below-threshold second hand still block grasp despite the first hand displaying 90 percent", () => {
  for (const state of ["FAIL", "DONE"]) {
    const s = setup(); s.engine.syncHealth({ task_state: state, failure_reason: "timeout", gripper_latch: "OPEN" }, s.time());
    for (let i = 0; i < 50; i++) {
      const result = s.tick(pair("Closed_Fist", { left: { confidence: 0.9 }, right: { confidence: 0.9 } }));
      noActuation(result); assert.equal(s.engine.evidence.close.left, 0);
    }
  }
  const s = setup();
  for (let i = 0; i < 30; i++) noActuation(s.tick(pair("Closed_Fist", { left: { confidence: 0.90 }, right: { confidence: 0.79 } })));
  assert.equal(s.engine.evidence.close.left, 700); assert.equal(s.engine.evidence.close.right, 0);
  assert.equal(s.engine.pendingGripper, null);
});
