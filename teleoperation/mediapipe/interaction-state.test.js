import test from "node:test";
import assert from "node:assert/strict";

import { mapPlanarMotion, mapILoveYouLateralStep, mapVictoryLateralStep, pointingUpAngle, pointingControlCandidate, pointingYawFromWrist, shouldStartPlanarLock, nextPlanarAnchor, planarGestureAfterDiagonal, SingleHandRearmGate, openPalmAllowedAfterDiagonal, openPalmReadyToRelease, handClearForRelease, shouldCloseGripper } from "./interaction-state.js";

test("one-frame Open_Palm flicker cannot release a closed gripper", () => {
  assert.equal(openPalmReadyToRelease({ now: 1033, openPalmSince: 1000 }), false);
  assert.equal(openPalmReadyToRelease({ now: 1050, openPalmSince: 1000 }), true);
  assert.equal(openPalmReadyToRelease({ now: 1100, openPalmSince: null }), false);
});

test("Victory and Thumb_Up take control immediately after ILoveYou", () => {
  assert.equal(planarGestureAfterDiagonal("Victory", 1080, 1000), "Victory");
  assert.equal(planarGestureAfterDiagonal("Thumb_Up", 1080, 1000), "Thumb_Up");
  assert.equal(planarGestureAfterDiagonal("Open_Palm", 1080, 1000), "Open_Palm");
});

test("confirmed fist requests grasp while moving and ignores stale readiness polls", () => {
  const state = {
    taskMode: "pick_place",
    gesture: "Closed_Fist",
    gripperClosed: false,
    holdingCube: false,
    graspReady: true,
  };
  assert.equal(shouldCloseGripper({ ...state, fistConfirmed: false }), false);
  assert.equal(shouldCloseGripper({ ...state, graspReady: false, fistConfirmed: true }), true);
  assert.equal(shouldCloseGripper({ ...state, fistConfirmed: true }), true);
  assert.equal(shouldCloseGripper({ ...state, gripperClosed: true, fistConfirmed: true }), false);
});

test("Pointing_Up follows the wrist-to-fingertip angle through both horizontal directions", () => {
  const hand = (dx, dy = -0.2) => { const points = Array.from({length: 21}, () => ({x: 0.5, y: 0.5})); points[5] = {x: 0.6, y: 0.4}; points[8] = {x: 0.5 + dx, y: 0.5 + dy}; return points; };
  assert.equal(pointingUpAngle(hand(0)), 0);
  assert.ok(pointingUpAngle(hand(0.1)) > 0);
  assert.ok(pointingUpAngle(hand(-0.1)) < 0);
  assert.ok(Math.abs(pointingUpAngle(hand(-0.2, 0)) + Math.PI / 2) < 1e-9);
  assert.ok(Math.abs(pointingUpAngle(hand(0.2, 0)) - Math.PI / 2) < 1e-9);
  assert.equal(pointingUpAngle([]), null);
});

test("single extended index finger retains Pointing_Up control when the classifier flickers", () => {
  const hand = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5 }));
  hand[6] = { x: 0.42, y: 0.46 };
  hand[8] = { x: 0.27, y: 0.50 };
  for (const [pip, tip] of [[10, 12], [14, 16], [18, 20]]) {
    hand[pip] = { x: 0.5, y: 0.42 };
    hand[tip] = { x: 0.5, y: 0.43 };
  }
  assert.equal(pointingControlCandidate("Pointing_Up", hand), true);
  assert.equal(pointingControlCandidate("None", hand), false);
  assert.equal(pointingControlCandidate("Thumb_Up", hand), false);
  assert.equal(pointingControlCandidate("Open_Palm", hand), false);
  assert.equal(pointingControlCandidate("None", hand, { active: true }), true);
  assert.equal(pointingControlCandidate("Thumb_Up", hand, { active: true }), false);
  assert.equal(pointingControlCandidate("Open_Palm", hand, { active: true }), false);
  assert.equal(pointingControlCandidate("Victory", hand, { active: true }), false);
  assert.equal(pointingControlCandidate("ILoveYou", hand, { active: true }), false);
  assert.ok(pointingUpAngle(hand) < -1.5);
  const distantHand = hand.map(({ x, y }) => ({ x: 0.5 + (x - 0.5) * 0.35, y: 0.5 + (y - 0.5) * 0.35 }));
  assert.equal(pointingControlCandidate("None", distantHand, { active: true }), true);
  hand[8] = { x: 0.48, y: 0.47 };
  assert.equal(pointingControlCandidate("Closed_Fist", hand), false);
});

test("vertical wrist-to-fingertip is absolute zero, including first clockwise detection", () => {
  assert.equal(pointingYawFromWrist(0), 0);
  assert.ok(Math.abs(pointingYawFromWrist(0.3) - 0.3) < 1e-9);
  assert.ok(Math.abs(pointingYawFromWrist(-0.3) + 0.36) < 1e-9);
  assert.equal(pointingYawFromWrist(-2), -Math.PI / 2);
  assert.equal(pointingYawFromWrist(null), null);
});

test("Open_Palm becomes available directly after ILoveYou without a third gesture", () => {
  const allowed = (now) => openPalmAllowedAfterDiagonal({ now, lastDiagonalAt: 1000 });
  assert.equal(allowed(1000), false);
  assert.equal(allowed(1049), false);
  assert.equal(allowed(1050), true);
  assert.equal(allowed(1800), true);
});

test("ILoveYou horizontal motion uses a full lateral step after the vertical threshold", () => {
  assert.equal(mapILoveYouLateralStep(-0.007, 0.008), 0);
  assert.equal(mapILoveYouLateralStep(-0.008, 0.008), 0);
  assert.equal(mapILoveYouLateralStep(-0.009, 0.008), -1);
  assert.equal(mapILoveYouLateralStep(0.009, 0.008), 1);
});

test("Victory retains its full lateral step and mirrored direction", () => {
  assert.equal(mapVictoryLateralStep(-0.007, 0.008), 0);
  assert.equal(mapVictoryLateralStep(-0.009, 0.008), 1);
  assert.equal(mapVictoryLateralStep(0.009, 0.008), -1);
});

test("re-entering hand uses 500 ms of wall time without restarting on movement or gesture changes", () => {
  const gate = new SingleHandRearmGate();
  const frame = (gesture = "Victory", x = 0.5, score = 0.9) => ({
    gestures: [[{ categoryName: gesture, score }]], landmarks: [[{ x, y: 0.5 }]],
  });
  assert.equal(gate.observe(frame(), 0).state, "waiting");
  assert.deepEqual(gate.observe(frame("Closed_Fist", 0.8, 0.4), 50), { state: "waiting", remainingMs: 450 });
  assert.deepEqual(gate.observe({ landmarks: [], gestures: [] }, 75), { state: "waiting", remainingMs: 425 });
  assert.deepEqual(gate.observe(frame("Open_Palm", 0.2), 150), { state: "waiting", remainingMs: 350 });
  assert.deepEqual(gate.observe(frame("Victory", 0.9), 499), { state: "waiting", remainingMs: 1 });
  assert.equal(gate.observe(frame("Victory", 0.9), 500).state, "ready");
  gate.arm();
  assert.equal(gate.observe(frame(), 520).state, "armed");
  assert.equal(gate.observe({ landmarks: [], gestures: [] }, 540).state, "lost");
  assert.deepEqual(gate.observe(frame(), 560), { state: "waiting", remainingMs: 500 });
  assert.equal(gate.observe({ landmarks: [], gestures: [] }, 820).state, "lost");
});

test("stable re-entry holds robot until the wrist and gesture settle", () => {
  const gate = new SingleHandRearmGate({ waitMs: 350, stableMs: 250, stableRadius: 0.012 });
  const frame = (x, gesture = "Victory") => ({
    gestures: [[{ categoryName: gesture, score: 0.9 }]],
    landmarks: [[{ x, y: 0.5 }]],
  });
  assert.equal(gate.observe(frame(0.5), 0).state, "waiting");
  assert.equal(gate.observe(frame(0.55), 200).state, "waiting");
  assert.equal(gate.observe(frame(0.55), 350).state, "waiting");
  assert.equal(gate.observe(frame(0.55), 450).state, "ready");
  gate.arm();
  assert.equal(gate.observe({ gestures: [], landmarks: [] }, 500).state, "lost");
  assert.equal(gate.observe(frame(0.2), 600).state, "waiting");
  assert.equal(gate.observe(frame(0.2), 950).state, "ready");
});

test("mirrored upper-right ILoveYou motion points toward +X/-Y", () => {
  assert.deepEqual(
    mapPlanarMotion({
      gesture: "ILoveYou",
      screenDx: -0.02,
      screenDy: -0.02,
      sensitivity: 0.04,
      carrying: true,
    }),
    { dx: 0.5, dy: -0.5 },
  );
});

test("Victory and Thumb_Up remain single-axis controls", () => {
  assert.deepEqual(
    mapPlanarMotion({
      gesture: "Victory",
      screenDx: -0.02,
      screenDy: -0.02,
      sensitivity: 0.04,
      carrying: true,
    }),
    { dx: 0, dy: 0.5 },
  );
  assert.deepEqual(
    mapPlanarMotion({
      gesture: "Thumb_Up",
      screenDx: -0.02,
      screenDy: -0.02,
      sensitivity: 0.04,
      carrying: true,
    }),
    { dx: 0.5, dy: 0 },
  );
});

test("MuJoCo reverses Victory only", () => {
  const input = { screenDx: -0.02, screenDy: -0.02, sensitivity: 0.04, carrying: true, reverseVictory: true };
  assert.deepEqual(mapPlanarMotion({ ...input, gesture: "Victory" }), { dx: 0, dy: -0.5 });
  assert.deepEqual(mapPlanarMotion({ ...input, gesture: "Thumb_Up" }), { dx: 0.5, dy: 0 });
  assert.deepEqual(mapPlanarMotion({ ...input, gesture: "ILoveYou" }), { dx: 0.5, dy: -0.5 });
});

test("MuJoCo ILoveYou moves diagonally with an empty gripper", () => {
  const input = { screenDx: -0.02, screenDy: -0.02, sensitivity: 0.04, carrying: false };
  assert.deepEqual(mapPlanarMotion({ ...input, gesture: "ILoveYou" }), { dx: 0, dy: 0 });
  assert.deepEqual(mapPlanarMotion({ ...input, gesture: "ILoveYou", allowEmptyDiagonal: true }), { dx: 0.5, dy: -0.5 });
});

test("repeated slow ILoveYou movement accumulates until it sends a step", () => {
  let anchor = { x: 0.5, y: 0.5 };
  let moves = 0;
  for (let frame = 1; frame <= 12; frame += 1) {
    const point = { x: 0.5 - frame * 0.001, y: 0.5 - frame * 0.001 };
    const motion = mapPlanarMotion({
      gesture: "ILoveYou", screenDx: point.x - anchor.x,
      screenDy: point.y - anchor.y, sensitivity: 0.03,
      carrying: false, allowEmptyDiagonal: true,
    });
    const moved = Math.hypot(motion.dx, motion.dy) > 0.09;
    if (moved) moves += 1;
    anchor = nextPlanarAnchor(anchor, point, moved, true);
  }
  assert.ok(moves >= 3);
});

test("every newly entered planar gesture locks Z while carrying", () => {
  for (const gesture of ["Victory", "Thumb_Up", "ILoveYou"]) {
    assert.equal(
      shouldStartPlanarLock({ gesture, gestureChanged: true, carrying: true }),
      true,
    );
  }
  for (const gesture of ["Open_Palm", "Closed_Fist", "Pointing_Up"]) {
    assert.equal(
      shouldStartPlanarLock({ gesture, gestureChanged: true, carrying: true }),
      false,
    );
  }
});


test("a clipped, missing or uncertain hand cannot authorize releasing a load", () => {
  const hand = Array.from({length: 21}, () => ({x: 0.5, y: 0.5}));
  assert.equal(handClearForRelease(hand, 0.95), true);
  assert.equal(handClearForRelease([], 0.95), false);
  assert.equal(handClearForRelease(hand, 0.499), false);
  assert.equal(handClearForRelease(hand, 0.5), true);
  assert.equal(handClearForRelease(hand, 0.6), true);
  for (const point of [{x: 0.99, y: 0.5}, {x: 0.5, y: -0.1}, {x: NaN, y: 0.5}]) {
    assert.equal(handClearForRelease(hand.map((p, i) => i === 8 ? point : p), 0.95), false);
  }
});
