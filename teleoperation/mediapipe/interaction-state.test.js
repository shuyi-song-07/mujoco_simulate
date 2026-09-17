import test from "node:test";
import assert from "node:assert/strict";

import { mapPlanarMotion, shouldStartPlanarLock } from "./interaction-state.js";

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
