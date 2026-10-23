const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const app = fs.readFileSync(path.join(root, 'teleoperation/mediapipe/app.js'), 'utf8');
const interaction = fs.readFileSync(path.join(root, 'teleoperation/mediapipe/interaction-state.js'), 'utf8');

test('entry dropout invalidates resume token and recovers within the 500ms budget', async () => {
  const commands = [];
  const ctx = vm.createContext({
    robotConnected: true, robotBackend: 'MuJoCo', lastGestureReportAt: 0,
    sceneResetInFlight: false, resumePending: false, resumeConfirmed: false,
    pendingResumeToken: null, controlToken: null, controlGeneration: 0,
    backendContext: 'session:1', armMotionLocked: false,
    gripperOrientationFinalized: true, pointingFinalizationPending: false,
    gripperClosed: false, lastPointingAngle: null, GRIPPER_OPEN_RETRY_MS: 150,
    GRIPPER_CLOSE_RETRY_MS: 300, robotFeedback: {},
    crypto: { randomUUID: () => `token-${commands.length}` },
    sendRobotCommand: () => Promise.resolve(true),
    queueSafetyCommand: (command, extra) => { commands.push({ command, extra }); return Promise.resolve(true); },
  });
  const gate = interaction.slice(interaction.indexOf('export class SingleHandRearmGate')).replace('export class', 'class');
  const disarm = app.slice(app.indexOf('function disarmHand('), app.indexOf('\nfunction isPushingTask'));
  const control = app.slice(app.indexOf('function processRobotControl('), app.indexOf('  const topGesture =', app.indexOf('function processRobotControl(')));
  vm.runInContext(gate + '\nconst rearmGate = new SingleHandRearmGate();\n' + disarm + '\n' + control + '\n processedAt = timestamp; }', ctx);
  const frame = { landmarks: [[{ x: 0.5, y: 0.5 }]], gestures: [[{ categoryName: 'Victory', score: 0.9 }]] };
  ctx.processRobotControl(frame, 0);
  await Promise.resolve();
  assert.equal(ctx.resumeConfirmed, true);
  ctx.processRobotControl({ landmarks: [], gestures: [] }, 100);
  assert.equal(ctx.resumeConfirmed, false);
  assert.equal(ctx.controlToken, null);
  ctx.processRobotControl(frame, 150);
  await Promise.resolve();
  assert.equal(commands.filter(c => c.command === 'resume').length, 2);
  ctx.processRobotControl(frame, 350);
  assert.equal(ctx.processedAt, 350);
  assert.ok(ctx.controlToken);
  ctx.processRobotControl({ landmarks: [], gestures: [] }, 520);
  assert.equal(ctx.controlToken, null);
  assert.equal(commands.at(-1).command, 'pause');
});

test('repeated reentry with the same gesture sends motion within 500ms', async () => {
  const commands = [];
  const ctx = vm.createContext({
    robotConnected: true, robotBackend: 'MuJoCo', robotTaskMode: 'pick_place',
    lastGestureReportAt: 0, sceneResetInFlight: false,
    resumePending: false, resumeConfirmed: false, pendingResumeToken: null,
    controlToken: null, controlGeneration: 0, backendContext: 'session:1',
    armMotionLocked: false, gripperOrientationFinalized: true,
    pointingFinalizationPending: false, gripperClosed: false, lastPointingAngle: null,
    GRIPPER_OPEN_RETRY_MS: 150, GRIPPER_CLOSE_RETRY_MS: 300,
    GRIPPER_FIST_CONFIRM_MS: 50, POINTING_EXIT_HOLD_MS: 200,
    DIAGONAL_GESTURE_HOLD_MS: 180, CONTROL_INTERVAL_MS: 40,
    lastControlSentAt: 0, lastDiagonalGestureAt: -Infinity,
    robotFeedback: {}, motionSpeed: () => ({}),
    verticalDeadZone: () => 0.015, planarSensitivity: () => 0.03,
    planarDeadZone: () => 0.09, isPushingTask: () => false,
    crypto: { randomUUID: () => `token-${commands.length}` },
  });
  ctx.sendRobotCommand = (command, extra) => { commands.push({command, extra}); return Promise.resolve(true); };
  ctx.queueSafetyCommand = ctx.sendRobotCommand;
  ctx.sendLatestMotionCommand = ctx.sendRobotCommand;
  const disarm = app.slice(app.indexOf('function disarmHand('), app.indexOf('\nfunction isPushingTask'));
  const control = app.slice(app.indexOf('function processRobotControl('), app.indexOf('\nfunction processPushingControl('));
  vm.runInContext(interaction.replaceAll('export ', '') + '\nconst rearmGate = new SingleHandRearmGate();\n' + disarm + '\n' + control, ctx);
  const frame = (x, gesture) => ({ landmarks: [[{ x, y: x }]], gestures: [[{ categoryName: gesture, score: 0.95 }]] });
  for (const gesture of ['ILoveYou', 'Victory', 'Thumb_Up', 'Open_Palm', 'Closed_Fist']) {
    for (let cycle = 0; cycle < 5; cycle++) {
      const now = 1000 + commands.length * 1000;
      ctx.processRobotControl({landmarks: [], gestures: []}, now);
      ctx.processRobotControl(frame(0.5, gesture), now + 20);
      await Promise.resolve();
      ctx.processRobotControl(frame(0.5, gesture), now + 370);
      const before = commands.length;
      ctx.processRobotControl(frame(0.55, gesture), now + 410);
      assert.ok(commands.slice(before).some(c => ['move_xy', 'down'].includes(c.command)), `${gesture}: no movement after reentry`);
      assert.ok(ctx.controlToken);
    }
  }
});
