/** Task 3 control logic, independent of the camera, DOM, and single-arm baseline. */
export const DEFAULT_DUAL_CONFIG = Object.freeze({
  motion_mode_confirm_ms: 200,
  motion_confidence_min: 0.75,
  gripper_confirm_ms: 700,
  gripper_confidence_min: 0.85,
  gripper_open_confidence_min: 0.55,
  gripper_close_confidence_min: 0.80,
  gripper_sample_gap_ms: 250,
  evidence_decay: 1.5,
  hand_lost_grace_ms: 250,
  control_interval_ms: 40,
  calibration_ms: 2000,
  calibration_gesture_confidence_min: 0.60,
  calibration_flicker_grace_ms: 160,
  calibration_sample_gap_ms: 500,
  calibration_max_result_age_ms: 500,
  calibration_wrist_tolerance: 0.08,
  recovery_frames: 5,
  identity_recovery_ms: 200,
  handedness_confidence_min: 0.8,
  max_wrist_jump: 0.25,
  identity_swap_margin: 0.08,
  wrist_filter: "one_euro",
  motion_mapping: "incremental",
  motion_speed_scale: 0.7,
  rate_motion_range: 0.08,
  rate_dead_zone: 0.01,
  wrist_sensitivity: 0.012,
  wrist_dead_zone: 0.003,
  ema_alpha: 0.35,
  one_euro_min_cutoff: 1.7,
  one_euro_beta: 0.3,
  one_euro_d_cutoff: 1,
  gripper_ack_timeout_ms: 3000,
});

const sides = ["left", "right"];
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const validWrist = (wrist) => Array.isArray(wrist) && wrist.length === 2 && wrist.every(Number.isFinite);

function interactionConfig(config) {
  const merged = { ...DEFAULT_DUAL_CONFIG, ...config };
  // Frozen older configs used one threshold for both actions. Do not inject
  // the new, lower open threshold when replaying their explicit legacy value.
  for (const key of ["gripper_open_confidence_min", "gripper_close_confidence_min"]) {
    if (config[key] === undefined && config.gripper_confidence_min !== undefined) merged[key] = config.gripper_confidence_min;
  }
  return merged;
}

/** Resolve a discrete operation from authoritative backend ACKs, not HTTP 200. */
export function resolveOperation(pending, health, now, connected = true) {
  if (!pending) return { pending: null, outcome: null };
  const result = health?.last_command_result;
  if (result?.eventId === pending.eventId) return { pending: null, outcome: result.accepted === false ? "rejected" : "accepted", command: pending.command, reason: result.reason };
  if (!connected && pending.command === "record_stop" && pending.queued) return { pending: null, outcome: "connection_closed", command: pending.command };
  if (connected && !health?.saving && now - pending.startedAt > 15000) return { pending: null, outcome: "timeout", command: pending.command };
  return { pending, outcome: null };
}

function alpha(cutoff, dt) {
  return 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
}

export class WristFilter {
  constructor(config = {}) {
    this.config = { ...DEFAULT_DUAL_CONFIG, ...config };
    this.reset();
  }
  reset() {
    this.raw = this.filtered = this.time = null;
    this.derivative = [0, 0];
  }
  update(point, now) {
    if (!this.filtered || now <= this.time || now - this.time > 300) {
      this.raw = [...point];
      this.filtered = [...point];
      this.derivative = [0, 0];
    } else if (this.config.wrist_filter === "ema") {
      const a = clamp(this.config.ema_alpha, 0.001, 1);
      this.filtered = point.map((x, i) => a * x + (1 - a) * this.filtered[i]);
    } else {
      const dt = Math.max(0.001, (now - this.time) / 1000);
      const aD = alpha(this.config.one_euro_d_cutoff, dt);
      this.filtered = point.map((x, i) => {
        this.derivative[i] = aD * (x - this.raw[i]) / dt + (1 - aD) * this.derivative[i];
        const a = alpha(this.config.one_euro_min_cutoff + this.config.one_euro_beta * Math.abs(this.derivative[i]), dt);
        return a * x + (1 - a) * this.filtered[i];
      });
    }
    this.raw = [...point];
    this.time = now;
    return [...this.filtered];
  }
}

/** Convert each detection using its handedness label, never its array position. */
export function detectionsFromMediaPipe(result, swapLabels = false) {
  return (result.landmarks ?? []).map((landmarks, i) => {
    const handedness = result.handedness?.[i]?.[0];
    const gesture = result.gestures?.[i]?.[0];
    let side = handedness?.categoryName?.toLowerCase();
    if (swapLabels && (side === "left" || side === "right")) side = side === "left" ? "right" : "left";
    return {
      side,
      handedness_score: handedness?.score ?? 0,
      gesture: gesture?.categoryName ?? "None",
      confidence: gesture?.score ?? 0,
      wrist: [landmarks[0]?.x, landmarks[0]?.y],
    };
  });
}

export class DualHandTracker {
  constructor(config = {}) {
    this.config = { ...DEFAULT_DUAL_CONFIG, ...config };
    this.hands = Object.fromEntries(sides.map((side) => [side, {
      side, visible: false, gesture: "None", confidence: 0, handedness_score: 0,
      wrist_raw: null, wrist_filtered: null, anchor: null,
      last_seen_ms: -Infinity, stable_gesture_ms: 0, gesture_since_ms: null,
    }]));
    this.filters = Object.fromEntries(sides.map((side) => [side, new WristFilter(this.config)]));
    this.calibrated = false;
    this.validFrames = 0;
    this.reason = "calibration_required";
    this.calibrationSource = this.calibrationConfirmedAt = null;
    this.calibrationResetCount = 0;
    this.calibrationLastResetReason = null;
    this.calibrationDetectionCount = 0;
    this.calibrationDetections = [];
    this.calibrationFrameGapMs = this.lastObservedFrameAt = null;
    this.palmBindingGuard = false;
    this.recoveryPending = this.controlResumeRequired = false;
    this.recoveryCandidate = this.identityConflictReference = null;
    this.resetGestureCalibration("waiting_for_hands", false);
  }
  resetGestureCalibration(reason = "show_palms", recordReset = true) {
    this.calibrationPhase = "PALMS";
    this.palmsEvidenceMs = this.palmsMatchedFrames = 0;
    this.lastMatchingFrameAt = this.calibrationMismatchStartedAt = null;
    this.calibrationWristAnchor = this.lastCalibrationFrameAt = null;
    this.gestureCalibrationReason = reason;
    this.calibrationEligible = false;
    this.calibrationBlocker = reason === "waiting_for_hands" ? { code: reason, side: "both", actual: 0, required: 2 } : null;
    if (recordReset) {
      this.calibrationLastResetReason = reason;
      this.calibrationResetCount += 1;
    }
  }
  resetBinding(reason) {
    this.calibrated = false;
    this.validFrames = 0;
    this.calibrationSource = this.calibrationConfirmedAt = null;
    this.palmBindingGuard = false;
    this.recoveryPending = this.controlResumeRequired = false;
    this.recoveryCandidate = this.identityConflictReference = null;
    this.lastObservedFrameAt = null;
    this.resetGestureCalibration(reason);
    for (const side of sides) {
      Object.assign(this.hands[side], { visible: false, gesture: "None", confidence: 0, handedness_score: 0, wrist_raw: null, wrist_filtered: null, last_seen_ms: -Infinity, anchor: null, gesture_since_ms: null, stable_gesture_ms: 0 });
      this.filters[side].reset();
    }
  }
  requireRecovery(reason) {
    this.validFrames = 0;
    this.recoveryCandidate = null;
    this.controlResumeRequired = this.recoveryPending = this.calibrated;
    if (!this.calibrated) this.resetGestureCalibration(reason);
  }
  gestureCalibrationStatus() {
    return {
      method: "dual_open_palm_stable", source: this.calibrationSource,
      phase: this.calibrated ? "CONFIRMED" : "PALMS",
      palm_progress: this.calibrated ? 1 : clamp(this.palmsEvidenceMs / this.config.calibration_ms, 0, 1),
      reason: this.calibrated ? this.reason ?? "binding_complete" : this.gestureCalibrationReason,
      confirmed_at_monotonic_ms: this.calibrationConfirmedAt,
      eligible: this.calibrationEligible,
      blocker: this.calibrationBlocker ? { ...this.calibrationBlocker } : null,
      evidence_ms: this.calibrated ? this.config.calibration_ms : this.palmsEvidenceMs,
      required_ms: this.config.calibration_ms,
      frame_gap_ms: this.calibrationFrameGapMs,
      detection_count: this.calibrationDetectionCount,
      detections: this.calibrationDetections.map((detection) => ({ ...detection })),
      last_reset_reason: this.calibrationLastResetReason, reset_count: this.calibrationResetCount,
      palm_binding_guard: this.palmBindingGuard,
      tracking_recovery: {
        pending: this.recoveryPending, valid_samples: this.recoveryCandidate?.samples ?? 0,
        stable_ms: this.recoveryCandidate ? Math.max(0, this.recoveryCandidate.lastAt - this.recoveryCandidate.startedAt) : 0,
        required_samples: Math.max(5, this.config.recovery_frames), required_ms: this.config.identity_recovery_ms,
      },
    };
  }
  advanceGestureCalibration(now) {
    if (this.calibrated) { this.calibrationBlocker = null; this.calibrationEligible = true; return false; }
    if (!sides.every((side) => this.hands[side].visible && this.hands[side].last_seen_ms === now)) return false;
    if (now === this.lastCalibrationFrameAt) return false;
    let timingBlocker = null;
    if (this.lastCalibrationFrameAt !== null && (now < this.lastCalibrationFrameAt || now - this.lastCalibrationFrameAt > this.config.calibration_sample_gap_ms)) {
      timingBlocker = { code: now < this.lastCalibrationFrameAt ? "clock_reversal" : "sample_gap", side: "both", actual: now - this.lastCalibrationFrameAt, required: this.config.calibration_sample_gap_ms };
      this.resetGestureCalibration(timingBlocker.code);
    }
    this.lastCalibrationFrameAt = now;
    if (this.calibrationWristAnchor) {
      const movedSide = sides.find((side) => distance(this.hands[side].wrist_raw, this.calibrationWristAnchor[side]) > this.config.calibration_wrist_tolerance);
      if (movedSide) {
        const blocker = { code: "wrist_motion", side: movedSide, actual: distance(this.hands[movedSide].wrist_raw, this.calibrationWristAnchor[movedSide]), required: this.config.calibration_wrist_tolerance };
        this.resetGestureCalibration("wrist_motion");
        this.lastCalibrationFrameAt = now; this.calibrationBlocker = blocker;
        return false;
      }
    }
    if (this.calibrationMismatchStartedAt !== null && now - this.calibrationMismatchStartedAt > this.config.calibration_flicker_grace_ms) {
      this.resetGestureCalibration(this.calibrationBlocker?.code ?? "gesture_mismatch");
      this.lastCalibrationFrameAt = now;
    }
    const mismatchedSide = sides.find((side) => this.hands[side].gesture !== "Open_Palm" || this.hands[side].confidence < this.config.calibration_gesture_confidence_min || !Number.isFinite(this.hands[side].confidence));
    if (mismatchedSide) {
      const hand = this.hands[mismatchedSide];
      this.calibrationEligible = false;
      this.lastMatchingFrameAt = null;
      this.calibrationMismatchStartedAt ??= now;
      this.calibrationBlocker = hand.gesture !== "Open_Palm" ? { code: "gesture_mismatch", side: mismatchedSide, actual: hand.gesture, required: "Open_Palm" } : { code: "gesture_confidence_low", side: mismatchedSide, actual: Number.isFinite(hand.confidence) ? hand.confidence : null, required: this.config.calibration_gesture_confidence_min };
      this.gestureCalibrationReason = "show_palms";
      return false;
    }
    const dt = this.lastMatchingFrameAt === null ? 0 : now - this.lastMatchingFrameAt;
    this.lastMatchingFrameAt = now;
    this.calibrationMismatchStartedAt = null;
    this.calibrationEligible = true; this.calibrationBlocker = timingBlocker;
    this.calibrationWristAnchor ??= Object.fromEntries(sides.map((side) => [side, [...this.hands[side].wrist_raw]]));
    this.palmsEvidenceMs = clamp(this.palmsEvidenceMs + dt, 0, this.config.calibration_ms);
    this.palmsMatchedFrames += 1;
    this.gestureCalibrationReason = "hold_palms";
    if (this.palmsEvidenceMs < this.config.calibration_ms || this.palmsMatchedFrames < this.config.recovery_frames) return false;
    this.calibrated = true;
    this.calibrationSource = "dual_open_palm";
    this.calibrationConfirmedAt = now;
    this.palmBindingGuard = true;
    this.gestureCalibrationReason = "binding_complete";
    for (const side of sides) this.hands[side].anchor = null;
    return true;
  }
  identityIssue(assigned, now) {
    const reference = this.identityConflictReference ?? Object.fromEntries(sides.map((side) => [side, this.hands[side].wrist_raw]));
    if (!reference.left || !reference.right) return null;
    const previousPair = reference.right.map((x, i) => x - reference.left[i]);
    const currentPair = assigned.right.wrist.map((x, i) => x - assigned.left.wrist[i]);
    const referenceSeparation = Math.hypot(...previousPair), separation = Math.hypot(...currentPair);
    const pairDirection = previousPair[0] * currentPair[0] + previousPair[1] * currentPair[1];
    // Close to the centre, both nearest-reference distances can fall inside
    // the swap margin. Reversed pair direction still cannot resolve a conflict.
    if (referenceSeparation > 0.001 && pairDirection < 0) return { code: "identity_conflict", side: "both", actual: pairDirection, required: 0 };
    const minimumSeparation = Math.min(0.15, referenceSeparation * 0.5);
    if (this.identityConflictReference && separation < minimumSeparation) return { code: "identity_conflict", side: "both", actual: separation, required: minimumSeparation };
    const shift = Object.fromEntries(sides.map((side) => [side, assigned[side].wrist.map((x, i) => x - reference[side][i])]));
    // A common translation retains the labelled pair's geometry. A label swap
    // does not, and cannot be cured merely by letting the old sample expire.
    const commonTranslation = distance(shift.left, shift.right) <= this.config.calibration_wrist_tolerance;
    for (const side of sides) {
      const ownDistance = distance(assigned[side].wrist, reference[side]);
      const otherDistance = distance(assigned[side].wrist, reference[side === "left" ? "right" : "left"]);
      const swapped = !commonTranslation && ownDistance > 0.15 && ownDistance > otherDistance + this.config.identity_swap_margin;
      const recentJump = !this.identityConflictReference && now - this.hands[side].last_seen_ms <= this.config.hand_lost_grace_ms && ownDistance > this.config.max_wrist_jump;
      if (swapped || recentJump) return { code: "identity_conflict", side, actual: ownDistance, required: swapped ? otherDistance + this.config.identity_swap_margin : this.config.max_wrist_jump };
    }
    return null;
  }
  acceptPair(assigned, now, resetFilters = false) {
    for (const side of sides) {
      const detection = assigned[side], hand = this.hands[side];
      if (resetFilters || now - hand.last_seen_ms > this.config.hand_lost_grace_ms) this.filters[side].reset();
      if (hand.gesture !== detection.gesture || hand.gesture_since_ms === null || resetFilters) hand.gesture_since_ms = now;
      Object.assign(hand, {
        visible: true, gesture: detection.gesture, confidence: detection.confidence,
        handedness_score: detection.handedness_score, wrist_raw: [...detection.wrist],
        wrist_filtered: this.filters[side].update(detection.wrist, now), last_seen_ms: now,
        stable_gesture_ms: now - hand.gesture_since_ms,
      });
    }
  }
  advanceRecovery(assigned, now) {
    const candidate = this.recoveryCandidate;
    const changed = !candidate || now <= candidate.lastAt || now - candidate.lastAt > this.config.calibration_sample_gap_ms ||
      sides.some((side) => distance(assigned[side].wrist, candidate.wrists[side]) > this.config.calibration_wrist_tolerance);
    if (changed) {
      this.recoveryCandidate = { startedAt: now, lastAt: now, samples: 1, wrists: Object.fromEntries(sides.map((side) => [side, [...assigned[side].wrist]])) };
    } else { candidate.lastAt = now; candidate.samples += 1; }
    const current = this.recoveryCandidate;
    const ready = current.samples >= Math.max(5, this.config.recovery_frames) && now - current.startedAt >= this.config.identity_recovery_ms;
    if (!ready) return false;
    this.acceptPair(assigned, now, true);
    this.recoveryPending = false;
    this.identityConflictReference = null;
    this.controlResumeRequired = true;
    return true;
  }
  update(detections, now) {
    this.calibrationDetectionCount = detections.length;
    this.calibrationDetections = detections.map((detection) => ({ reported_side: detection.side ?? null, handedness_score: Number.isFinite(detection.handedness_score) ? detection.handedness_score : null, gesture: detection.gesture ?? "None", confidence: Number.isFinite(detection.confidence) ? detection.confidence : null, wrist_valid: validWrist(detection.wrist) }));
    this.calibrationFrameGapMs = this.lastObservedFrameAt === null ? null : now - this.lastObservedFrameAt;
    this.lastObservedFrameAt = now;
    for (const side of sides) this.hands[side].visible = false;
    const assigned = {};
    let issue = detections.length === 2 ? null : "hand_lost";
    let blocker = issue ? { code: "hand_lost", side: "both", actual: detections.length, required: 2 } : null;
    for (const detection of detections) {
      let invalid = null;
      if (!sides.includes(detection.side)) invalid = { code: "unknown_handedness", side: "unknown", actual: detection.side ?? null, required: "Left/Right" };
      else if (assigned[detection.side]) invalid = { code: "duplicate_handedness", side: detection.side, actual: detection.side, required: "one_left_one_right" };
      else if (!Number.isFinite(detection.handedness_score) || detection.handedness_score < this.config.handedness_confidence_min) invalid = { code: "handedness_confidence_low", side: detection.side, actual: Number.isFinite(detection.handedness_score) ? detection.handedness_score : null, required: this.config.handedness_confidence_min };
      else if (!validWrist(detection.wrist)) invalid = { code: "invalid_wrist", side: detection.side, actual: false, required: true };
      if (invalid) {
        if (!(this.calibrated && detections.length < 2)) { issue = "ambiguous_handedness"; if (!blocker || blocker.code === "hand_lost") blocker = invalid; }
      } else assigned[detection.side] = detection;
    }
    if (!assigned.left || !assigned.right) issue ??= "hand_lost";
    if (!issue && this.calibrated && this.calibrationFrameGapMs !== null && this.calibrationFrameGapMs <= 0) {
      issue = this.calibrationFrameGapMs < 0 ? "clock_reversal" : "tracking_stalled";
      blocker = { code: issue, side: "both", actual: this.calibrationFrameGapMs, required: "fresh_increasing_timestamp" };
    }
    if (!issue) {
      const conflict = this.identityIssue(assigned, now);
      if (conflict) { issue = "identity_conflict"; blocker = conflict; }
    }
    if (issue) {
      if (this.calibrated && issue === "identity_conflict" && !this.identityConflictReference) this.identityConflictReference = Object.fromEntries(sides.map((side) => [side, this.hands[side].wrist_raw ? [...this.hands[side].wrist_raw] : null]));
      this.requireRecovery(blocker?.code ?? issue);
      this.calibrationEligible = false; this.calibrationBlocker = blocker; this.reason = issue;
      return { valid: false, reason: issue, hands: this.hands };
    }
    if (this.calibrated && this.calibrationFrameGapMs > this.config.calibration_sample_gap_ms) this.requireRecovery("sample_gap");
    if (this.calibrated && this.recoveryPending) {
      if (!this.advanceRecovery(assigned, now)) {
        this.reason = "tracking_recovery";
        this.calibrationEligible = false;
        this.calibrationBlocker = { code: "tracking_recovery", side: "both", actual: this.recoveryCandidate.samples, required: Math.max(5, this.config.recovery_frames) };
        return { valid: false, reason: this.reason, hands: this.hands };
      }
    } else this.acceptPair(assigned, now);
    this.validFrames += 1;
    const calibrationConfirmed = this.advanceGestureCalibration(now);
    this.reason = !this.calibrated && this.validFrames < this.config.recovery_frames ? "tracking_recovery" : this.calibrated ? null : "calibration_required";
    return { valid: !this.reason, reason: this.reason, hands: this.hands, calibrationConfirmed };
  }
}

export class DualInteractionState {
  constructor(config = {}) {
    this.config = interactionConfig(config);
    this.tracker = new DualHandTracker(this.config);
    this.mode = "PAUSE";
    this.reason = "calibration_required";
    this.candidate = null;
    this.candidateSince = null;
    this.evidence = { close: { left: 0, right: 0 }, open: { left: 0, right: 0 } };
    this.previousGripperMatches = { close: { left: false, right: false }, open: { left: false, right: false } };
    this.gripperLatch = null;
    this.pendingGripper = null;
    this.gripperBlocked = null;
    this.gripperRetryReason = null;
    this.lastUpdate = null;
    this.lastMotion = -Infinity;
    this.eventCounter = 0;
    this.health = {};
  }
  gripperConfidence(action) {
    return this.config[action === "open" ? "gripper_open_confidence_min" : "gripper_close_confidence_min"] ?? this.config.gripper_confidence_min;
  }
  syncHealth(health, now) {
    this.health = health;
    if (["OPEN", "CLOSED"].includes(health.gripper_latch)) this.gripperLatch = health.gripper_latch;
    if (this.gripperLatch === "OPEN") this.gripperRetryReason = null;
    if (!this.pendingGripper) return;
    const result = health.last_command_result;
    if (result?.eventId === this.pendingGripper.eventId && result.accepted === false) {
      // A moving object is a temporary release condition, not a latched input
      // error. Holding fresh, qualified palms may reconfirm after another
      // full evidence interval; the backend still checks speed on every try.
      const waitForStop = this.pendingGripper.action === "open" && result.reason === "release_rejected_stop_before_opening";
      this.gripperBlocked = waitForStop ? null : this.pendingGripper.action;
      this.gripperRetryReason = waitForStop ? result.reason : null;
      this.reason = result.reason || "gripper_rejected";
      this.pendingGripper = null;
      this.clearEvidence();
    } else if ((this.pendingGripper.action === "close" && this.gripperLatch === "CLOSED") || (this.pendingGripper.action === "open" && this.gripperLatch === "OPEN")) {
      this.pendingGripper = null;
      this.gripperRetryReason = null;
      this.clearEvidence();
    } else if (now - this.pendingGripper.createdAt > this.config.gripper_ack_timeout_ms) {
      this.gripperBlocked = this.pendingGripper.action;
      this.gripperRetryReason = null;
      this.reason = "gripper_ack_timeout";
      this.pendingGripper = null;
      this.clearEvidence();
    }
  }
  clearEvidence() {
    for (const action of ["close", "open"]) for (const side of sides) {
      this.evidence[action][side] = 0;
      this.previousGripperMatches[action][side] = false;
    }
  }
  clearAnchors() {
    for (const side of sides) this.tracker.hands[side].anchor = null;
    this.candidate = this.candidateSince = null;
  }
  unbind(reason = "binding_released") {
    this.tracker.resetBinding(reason);
    this.clearAnchors();
    this.clearEvidence();
    this.pendingGripper = this.gripperBlocked = null;
    this.gripperRetryReason = null;
    this.lastUpdate = null;
    this.lastMotion = -Infinity;
    this.mode = "PAUSE";
    this.reason = reason;
    return this.output({ command: "pause", reason });
  }
  resetInput(reason) {
    this.clearAnchors();
    this.clearEvidence();
    this.pendingGripper = this.gripperBlocked = null;
    this.gripperRetryReason = null;
    this.lastUpdate = null;
    this.lastMotion = -Infinity;
    this.tracker.requireRecovery(reason);
    this.mode = "PAUSE";
    this.reason = reason;
    return this.output({ command: "pause", reason });
  }
  reconfigure(config) {
    this.config = interactionConfig(config);
    this.tracker.config = { ...this.config };
    for (const side of sides) {
      this.tracker.filters[side].config = { ...this.config };
      this.tracker.filters[side].reset();
    }
    return this.resetInput("config_updated");
  }
  resetScene() {
    // Scene/session changes reset robot input while the operator stays bound.
    return this.resetInput("scene_reset");
  }
  telemetry(recordable = false, delta = null) {
    return {
      control_mode: this.mode, training_recordable: recordable,
      pause_reason: this.reason, calibrated: this.tracker.calibrated,
      calibration: this.tracker.gestureCalibrationStatus(this.lastUpdate ?? this.tracker.lastCalibrationFrameAt ?? 0),
      gripper_latch_observed: this.gripperLatch,
      gripper_pending: this.pendingGripper?.action ?? null,
      gripper_retry_reason: this.gripperRetryReason,
      gripper_thresholds: { open: this.gripperConfidence("open"), close: this.gripperConfidence("close"), confirm_ms: this.config.gripper_confirm_ms },
      hands: Object.fromEntries(sides.map((side) => [side, { ...this.tracker.hands[side], last_seen_ms: Number.isFinite(this.tracker.hands[side].last_seen_ms) ? this.tracker.hands[side].last_seen_ms : null }])),
      evidence_ms: structuredClone(this.evidence),
      user_delta: delta, filter: this.config.wrist_filter,
      operator_control: Object.fromEntries(["motion_mapping", "motion_speed_scale", "rate_motion_range", "rate_dead_zone", "wrist_sensitivity", "wrist_dead_zone"].map((key) => [key, this.config[key]])),
    };
  }
  pause(reason, clearEvidence = true) {
    const changed = this.mode !== "PAUSE" || this.reason !== reason;
    this.mode = "PAUSE";
    this.reason = reason;
    this.clearAnchors();
    if (clearEvidence) this.clearEvidence();
    return this.output(changed ? { command: "pause", reason } : null);
  }
  output(command = null, recordable = false, delta = null) {
    const telemetry = this.telemetry(recordable, delta);
    return { command: command ? { ...command, telemetry } : null, telemetry, mode: this.mode, reason: this.reason };
  }
  update(detections, now) {
    const elapsed = this.lastUpdate === null ? 0 : now - this.lastUpdate;
    // Sample spacing is separate from result age. The app rejects results
    // older than 150ms; fresh results can legitimately arrive every 160–250ms.
    // Only adjacent qualified samples count, and a larger gap drops evidence.
    if (elapsed > this.config.gripper_sample_gap_ms) this.clearEvidence();
    const dt = elapsed > 0 && elapsed <= this.config.gripper_sample_gap_ms ? elapsed : 0;
    this.lastUpdate = now;
    const tracked = this.tracker.update(detections, now);
    if (!tracked.valid) return this.pause(tracked.reason);
    if (tracked.calibrationConfirmed) {
      this.pendingGripper = this.gripperBlocked = null;
      this.gripperRetryReason = null;
      return this.pause("binding_complete");
    }
    if (this.tracker.controlResumeRequired) {
      this.tracker.controlResumeRequired = false;
      return this.pause("tracking_recovered");
    }
    const hands = this.tracker.hands;
    if (this.tracker.palmBindingGuard) {
      const gesture = hands.left.gesture;
      const threshold = { ILoveYou: this.config.motion_confidence_min, Victory: this.config.motion_confidence_min, Closed_Fist: this.gripperConfidence("close") }[gesture];
      const released = threshold !== undefined && sides.every((side) => hands[side].gesture === gesture && hands[side].confidence >= threshold);
      if (!released) return this.pause("binding_complete");
      this.tracker.palmBindingGuard = false;
      return this.pause("binding_gesture_released");
    }
    // A failed grasp can leave both grippers closed. Match the backend's
    // recovery rule: permit only a deliberate, confirmed release in FAIL;
    // motion/closing remain frozen and releasing never clears the failure.
    const failedRelease = this.health.task_state === "FAIL" && this.gripperLatch === "CLOSED" && sides.every((side) => hands[side].gesture === "Open_Palm");
    if (this.health.task_state === "DONE" || (this.health.task_state === "FAIL" && !failedRelease)) return this.pause(this.health.task_state === "FAIL" ? this.health.failure_reason || "task_failed" : "task_done");
    if (this.health.saving) return this.pause("saving");
    if (!this.gripperLatch) return this.pause("backend_state_unknown");
    if (this.pendingGripper) {
      this.mode = this.pendingGripper.action === "close" ? "GRASP_CONFIRM" : "RELEASE_CONFIRM";
      this.reason = "gripper_pending";
      this.clearAnchors();
      return this.output();
    }
    if (this.gripperBlocked) {
      const gesture = this.gripperBlocked === "close" ? "Closed_Fist" : "Open_Palm";
      if (sides.some((side) => hands[side].gesture !== gesture)) this.gripperBlocked = null;
      else return this.pause("gripper_rejected_change_gesture");
    }
    if (sides.some((side) => hands[side].gesture !== "Open_Palm")) this.gripperRetryReason = null;

    for (const [action, gesture] of [["close", "Closed_Fist"], ["open", "Open_Palm"]]) {
      for (const side of sides) {
        const matches = hands[side].gesture === gesture && hands[side].confidence >= this.gripperConfidence(action);
        const evidenceDelta = matches ? (this.previousGripperMatches[action][side] ? dt : 0) : -this.config.evidence_decay * dt;
        this.evidence[action][side] = clamp(this.evidence[action][side] + evidenceDelta, 0, this.config.gripper_confirm_ms);
        this.previousGripperMatches[action][side] = matches;
      }
    }
    const desiredAction = this.gripperLatch === "OPEN" ? "close" : "open";
    const desiredGesture = desiredAction === "close" ? "Closed_Fist" : "Open_Palm";
    const anyGripper = sides.some((side) => ["Closed_Fist", "Open_Palm"].includes(hands[side].gesture));
    if (anyGripper) {
      const wasMoving = this.mode === "XY" || this.mode === "Z";
      this.clearAnchors();
      this.mode = sides.some((side) => hands[side].gesture === desiredGesture) ? (desiredAction === "close" ? "GRASP_CONFIRM" : "RELEASE_CONFIRM") : "IDLE";
      this.reason = this.mode === "IDLE" ? "gripper_already_latched" : "gripper_confirmation";
      if (desiredAction === "open" && this.gripperRetryReason && this.mode === "RELEASE_CONFIRM") this.reason = "release_waiting_for_stop";
      const confirmed = sides.every((side) => hands[side].gesture === desiredGesture && hands[side].confidence >= this.gripperConfidence(desiredAction) && this.evidence[desiredAction][side] >= this.config.gripper_confirm_ms);
      if (confirmed) {
        const eventId = `gripper-${Date.now()}-${++this.eventCounter}`;
        this.pendingGripper = { action: desiredAction, eventId, createdAt: now };
        this.clearEvidence();
        return this.output({ command: "dual_gripper", action: desiredAction, eventId, gesture_confirm_ms: this.config.gripper_confirm_ms }, true);
      }
      return this.output(wasMoving ? { command: "pause", reason: "gripper_confirmation" } : null);
    }

    const candidate = sides.every((side) => hands[side].gesture === "ILoveYou" && hands[side].confidence >= this.config.motion_confidence_min) ? "XY" : sides.every((side) => hands[side].gesture === "Victory" && hands[side].confidence >= this.config.motion_confidence_min) ? "Z" : null;
    if (!candidate) {
      const sameGesture = hands.left.gesture === hands.right.gesture;
      const reason = !sameGesture ? "gesture_disagreement" : ["ILoveYou", "Victory"].includes(hands.left.gesture) ? "motion_low_confidence" : "unassigned_gesture";
      return this.pause(reason, false);
    }
    if (this.candidate !== candidate) {
      const wasMoving = this.mode === "XY" || this.mode === "Z";
      this.clearAnchors();
      this.candidate = candidate;
      this.candidateSince = now;
      this.mode = "IDLE";
      this.reason = "motion_confirmation";
      return this.output(wasMoving ? { command: "pause", reason: "motion_confirmation" } : null);
    }
    if (now - this.candidateSince < this.config.motion_mode_confirm_ms) return this.output();
    this.mode = candidate;
    this.reason = null;
    if (sides.some((side) => !hands[side].anchor)) {
      for (const side of sides) hands[side].anchor = [...hands[side].wrist_filtered];
      this.lastMotion = now;
      return this.output();
    }
    if (now - this.lastMotion < this.config.control_interval_ms) return this.output();
    const deltas = {};
    let nonzero = false;
    for (const side of sides) {
      const hand = hands[side];
      const du = hand.wrist_filtered[0] - hand.anchor[0];
      const dv = hand.wrist_filtered[1] - hand.anchor[1];
      const isRate = this.config.motion_mapping === "rate";
      const zeroed = (x) => {
        if (!isRate) return Math.abs(x) <= this.config.wrist_dead_zone ? 0 : x / this.config.wrist_sensitivity;
        const deadZone = this.config.rate_dead_zone;
        const span = Math.max(0.001, this.config.rate_motion_range - deadZone);
        return Math.abs(x) <= deadZone ? 0 : Math.sign(x) * clamp((Math.abs(x) - deadZone) / span, 0, 1);
      };
      // Match the fixed top camera: screen up is task -X, screen right +Y.
      // Hand webcam is mirrored (raw du is opposite display horizontal delta).
      let delta = candidate === "XY" ? { dx: zeroed(dv), dy: -zeroed(du), dz: 0 } : { dx: 0, dy: 0, dz: -zeroed(dv) };
      const norm = Math.hypot(delta.dx, delta.dy, delta.dz);
      if (norm > 1) delta = Object.fromEntries(Object.entries(delta).map(([key, value]) => [key, value / norm]));
      const speed = clamp(this.config.motion_speed_scale, 0, 1);
      delta = Object.fromEntries(Object.entries(delta).map(([key, value]) => [key, value * speed]));
      deltas[side] = delta;
      if (Math.hypot(delta.dx, delta.dy, delta.dz) > 0) {
        nonzero = true;
        // Rate control keeps a fixed neutral until mode/pause/recovery reset.
        // Incremental control consumes only axes that actually crossed the
        // dead zone; another axis's slow displacement remains accumulated.
        if (!isRate) {
          if (candidate === "XY" && Math.abs(du) > this.config.wrist_dead_zone) hand.anchor[0] = hand.wrist_filtered[0];
          if (Math.abs(dv) > this.config.wrist_dead_zone) hand.anchor[1] = hand.wrist_filtered[1];
        }
      }
    }
    this.lastMotion = now;
    return this.output(nonzero ? { command: "dual_motion", mode: candidate.toLowerCase(), ...deltas } : null, nonzero, deltas);
  }
}
