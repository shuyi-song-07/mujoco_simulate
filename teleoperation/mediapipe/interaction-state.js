export function mapPlanarMotion({
  gesture,
  screenDx,
  screenDy,
  sensitivity,
  carrying,
  reverseVictory = false,
  allowEmptyDiagonal = false,
}) {
  const clamp = (value) => Math.min(1, Math.max(-1, value));
  // 摄像头画面是镜像的：画面中的右/上分别对应原始坐标的 -X/-Y。
  const visualRight = clamp(-screenDx / sensitivity);
  const visualForward = clamp(-screenDy / sensitivity);

  if (gesture === "ILoveYou" && (carrying || allowEmptyDiagonal)) {
    return { dx: visualForward, dy: -visualRight };
  }
  if (gesture === "Victory") {
    return { dx: 0, dy: reverseVictory ? -visualRight : visualRight };
  }
  if (gesture === "Thumb_Up") {
    return { dx: visualForward, dy: 0 };
  }
  return { dx: 0, dy: 0 };
}

export function shouldStartPlanarLock({ gesture, gestureChanged, carrying }) {
  const planarGestures = new Set(["Victory", "Thumb_Up", "ILoveYou"]);
  return planarGestures.has(gesture) && gestureChanged && carrying;
}

export function nextPlanarAnchor(anchor, point, moved, accumulate) {
  return !anchor || moved || !accumulate ? { x: point.x, y: point.y } : anchor;
}

export function planarGestureAfterDiagonal(gesture, now, lastDiagonalAt, holdMs = 180) {
  if (["Open_Palm", "Closed_Fist", "Victory", "Thumb_Up", "Pointing_Up"].includes(gesture)) return gesture;
  return now - lastDiagonalAt <= holdMs ? "ILoveYou" : gesture;
}

export function mapILoveYouLateralStep(screenDx, deadZone) {
  if (!Number.isFinite(screenDx) || Math.abs(screenDx) <= deadZone) return 0;
  // ILoveYou maps camera-horizontal motion to the robot's Y axis.
  return Math.sign(screenDx);
}

export function mapVictoryLateralStep(screenDx, deadZone) {
  if (!Number.isFinite(screenDx) || Math.abs(screenDx) <= deadZone) return 0;
  return -Math.sign(screenDx);
}

// MediaPipe coordinates are unmirrored while the operator sees a mirrored
// webcam. Positive angles therefore mean counterclockwise on that display.
export function pointingUpAngle(landmarks) {
  const base = landmarks?.[0];
  const tip = landmarks?.[8];
  if (![base?.x, base?.y, tip?.x, tip?.y].every(Number.isFinite)) return null;
  const dx = tip.x - base.x;
  const dy = tip.y - base.y;
  if (Math.hypot(dx, dy) < 0.05) return null;
  return Math.atan2(dx, -dy);
}

export function indexFingerPointing(landmarks) {
  const wrist = landmarks?.[0];
  if (!Number.isFinite(wrist?.x) || !Number.isFinite(wrist?.y)) return false;
  const distance = (index) => {
    const point = landmarks[index];
    return Number.isFinite(point?.x) && Number.isFinite(point?.y)
      ? Math.hypot(point.x - wrist.x, point.y - wrist.y)
      : null;
  };
  const indexTip = distance(8);
  const indexPip = distance(6);
  if (indexTip === null || indexPip === null || indexTip < 0.06 || indexTip < indexPip * 1.18) return false;
  return [[12, 10], [16, 14], [20, 18]].every(([tip, pip]) => {
    const tipDistance = distance(tip);
    const pipDistance = distance(pip);
    return tipDistance !== null && pipDistance !== null && tipDistance <= pipDistance * 1.1;
  });
}

export function pointingControlCandidate(gesture, landmarks, { active = false } = {}) {
  // A shape heuristic may keep an existing Pointing_Up adjustment alive, but
  // must not start one from None during transport or another control gesture.
  if (gesture === "Pointing_Up") return true;
  if (!active || gesture !== "None") return false;
  return indexFingerPointing(landmarks);
}

export function pointingYawFromWrist(rawAngle, clockwiseGain = 1.2) {
  if (!Number.isFinite(rawAngle)) return null;
  const adjusted = rawAngle < 0 ? rawAngle * clockwiseGain : rawAngle;
  return Math.max(-Math.PI / 2, Math.min(Math.PI / 2, adjusted));
}

export function openPalmAllowedAfterDiagonal({ now, lastDiagonalAt }) {
  return now - lastDiagonalAt >= 50;
}

// Partial hands at the image boundary must never authorize releasing a load.
export function handClearForRelease(landmarks, score, margin = 0.02) {
  return Number.isFinite(score) && score >= 0.5 && landmarks?.length === 21 &&
    landmarks.every(point => Number.isFinite(point?.x) && Number.isFinite(point?.y) &&
      point.x >= margin && point.x <= 1 - margin &&
      point.y >= margin && point.y <= 1 - margin);
}

export function openPalmReadyToRelease({ now, openPalmSince, confirmMs = 50 }) {
  return openPalmSince !== null && now - openPalmSince >= confirmMs;
}

export function shouldCloseGripper({
  taskMode,
  gesture,
  gripperClosed,
  fistConfirmed,
}) {
  if (gesture !== "Closed_Fist" || gripperClosed === true) return false;
  if (taskMode !== "pick_place") return true;
  // A cached alignment poll must not suppress an explicit closing gesture.
  return fistConfirmed;
}

export class SingleHandRearmGate {
  constructor({ waitMs = 500, lossGraceMs = 250 } = {}) {
    Object.assign(this, { waitMs, lossGraceMs });
    this.reset();
  }

  reset() {
    this.armed = false;
    this.candidate = null;
  }

  observe(result, now) {
    const top = result.gestures?.[0]?.[0];
    const wrist = result.landmarks?.[0]?.[0];
    if (!Number.isFinite(wrist?.x) || !Number.isFinite(wrist?.y)) {
      if (!this.armed && this.candidate && now - this.candidate.lastSeenAt <= this.lossGraceMs) {
        return { state: "waiting", remainingMs: Math.max(0, this.waitMs - (now - this.candidate.since)) };
      }
      this.reset();
      return { state: "lost" };
    }
    if (this.armed) return { state: "armed" };
    if (!this.candidate || now < this.candidate.since) {
      this.candidate = { since: now, lastSeenAt: now, gestureSince: now, gesture: top?.categoryName ?? "None" };
    }
    this.candidate.lastSeenAt = now;
    const gesture = top?.categoryName ?? "None";
    if (gesture !== this.candidate.gesture) this.candidate.gestureSince = now;
    this.candidate.gesture = gesture;
    const remainingMs = Math.max(0, this.waitMs - (now - this.candidate.since));
    return remainingMs === 0 ? { state: "ready" } : { state: "waiting", remainingMs };
  }

  arm() {
    if (this.candidate) this.armed = true;
  }
}
