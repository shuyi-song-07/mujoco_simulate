import test from "node:test";
import assert from "node:assert/strict";
import {ContinuousInteractionState} from "./continuous-interaction-state.js";
import {detectionsFromMediaPipe} from "./dual-interaction-state.js";
import {gestureResult,gestureMeasure} from "../../../tests/task3/gesture_hand_fixture.mjs";
const protocol="gesture-rate-gripper-v3",sides=["left","right"];
function setup({bind=true,modeConfirmMs=200}={}) {
  const e=new ContinuousInteractionState({wrist_filter:"ema",ema_alpha:1});let now=-40;
  e.syncHealth({task_state:"PREGRASP",gripper_latch:"CONTINUOUS",grippers:{left:{command_width_m:.08},right:{command_width_m:.08}},
    physical_contact:{control_protocol:protocol,hold_width_m:{left:null,right:null}},
    action_contract:{control_protocol:protocol,gesture_parameters:{mode_confirm_ms:modeConfirmMs,open_score_min:.5,close_score_min:.8,motion_score_min:.75}}},0);
  const s={e,tick(shapes=["ILY","ILY"],opts={}) {
    now+=opts.stepMs??40;const m=gestureMeasure(shapes,now,opts),result=gestureResult(shapes,opts);
    if(opts.invalid)m.hands[0].valid=false;
    if(opts.stale)m.accepted_recognition=false;
    if(opts.single){m.hands.pop();result.landmarks.pop();result.handedness.pop();result.gestures.pop();}
    if(opts.lowIdentity){m.hands[0].identity_score=.5;m.hands[0].valid=false;result.handedness[0][0].score=.5;}
    e.setMeasurement(m);return e.update(detectionsFromMediaPipe(result),now);
  },settle(shapes=["ILY","ILY"],opts={},n=10){let out;for(let i=0;i<n;i++)out=this.tick(shapes,opts);return out;}};
  if(bind)s.settle(["PALM","PALM"],{},54);
  return s;
}
const noMotion=out=>sides.forEach(s=>assert.equal(Math.hypot(...out.command.delta_task_m[s]),0));
test("left fist uses its lower close threshold while the right retains the shared threshold",()=>{
  const s=setup({modeConfirmMs:0});s.settle(["FIST","FIST"]);
  s.e.gestureParameters={...s.e.gestureParameters,left_close_score_min:.5,close_score_min:.6};
  const out=s.settle(["FIST","FIST"],{scores:[.55,.55]});
  assert.deepEqual(out.command.interaction_modes,["Z_CLOSE","HOLD"]);
  assert.deepEqual(s.tick(["FIST","FIST"],{scores:[.49,.6]}).command.interaction_modes,["HOLD","Z_CLOSE"]);
});
test("open palm at 50 percent qualifies and anything below it holds the gripper",()=>{
  const s=setup({modeConfirmMs:0});s.settle();
  const boundary=s.settle(["PALM","PALM"],{scores:[.5,.5]});
  assert.deepEqual(boundary.command.interaction_modes,["Z_OPEN","Z_OPEN"]);
  const below=s.tick(["PALM","PALM"],{scores:[.499,.5]});
  assert.deepEqual(below.command.interaction_modes,["HOLD","Z_OPEN"]);
});
test("100 ms current-shape confirmation holds old ramps then requests independent endpoints at the exact boundary",()=>{
  const s=setup({modeConfirmMs:100});s.settle();
  s.e.health.grippers.left.command_width_m=.025;s.e.health.grippers.right.command_width_m=.043;
  const transition=s.tick(["PALM","FIST"],{stepMs:10});
  assert.equal(transition.reason,"gesture_mode_confirmation");
  assert.deepEqual(transition.command.interaction_modes,["HOLD","HOLD"]);
  assert.deepEqual(transition.command.widths_m,[.025,.043]);noMotion(transition);
  for(let i=0;i<9;i++)assert.deepEqual(s.tick(["PALM","FIST"],{stepMs:10}).command.interaction_modes,["HOLD","HOLD"]);
  const confirmed=s.tick(["PALM","FIST"],{stepMs:10});
  assert.deepEqual(confirmed.command.interaction_modes,["Z_OPEN","Z_CLOSE"]);
  assert.deepEqual(confirmed.command.release_intent,[true,false]);assert.deepEqual(confirmed.command.widths_m,[.08,0]);noMotion(confirmed);
  const neutral=s.tick(["UNKNOWN","UNKNOWN"]);assert.deepEqual(neutral.command.widths_m,[.025,.043]);noMotion(neutral);
  const resumed=s.settle(["PALM","FIST"]);
  assert.deepEqual(resumed.command.interaction_modes,["Z_OPEN","Z_CLOSE"],"the same shape resumes without requiring an opposite-shape edge");
});
test("an explicit zero confirmation window enables the new mode immediately and still establishes a fresh wrist baseline",()=>{
  const s=setup({modeConfirmMs:0});s.settle();
  const first=s.tick(["PALM","FIST"],{dy:-.04});
  assert.deepEqual(first.command.interaction_modes,["Z_OPEN","Z_CLOSE"]);assert.deepEqual(first.command.widths_m,[.08,0]);noMotion(first);
  const second=s.tick(["PALM","FIST"],{dy:-.05});assert.ok(second.command.delta_task_m.left[2]>0);
});
test("v3 binds without aperture calibration and protects the binding palms",()=>{
  const s=setup();assert.equal(s.e.tracker.calibrated,true);
  const held=s.tick(["PALM","PALM"]);assert.equal(held.reason,"binding_complete");
  assert.deepEqual(held.command.interaction_modes,["HOLD","HOLD"]);assert.deepEqual(held.command.release_intent,[false,false]);
  assert.deepEqual(held.telemetry.aperture_calibration,{phase:"NOT_REQUIRED",reason:"categorical_rate_control"});
  assert.equal(held.telemetry.pinch_mapping,null);assert.equal(held.telemetry.aperture_takeover,false);
  assert.equal(s.e.apertureCalibration.phase,"CLOSE","unused legacy calibration is not falsely marked READY");
  const active=s.settle(["FIST","FIST"]);assert.deepEqual(active.command.interaction_modes,["Z_CLOSE","Z_CLOSE"]);
});
test("one binding palm cannot accidentally release while the other side exits binding",()=>{
  const s=setup();const held=s.settle(["PALM","FIST"]);
  assert.equal(held.reason,"binding_complete");assert.deepEqual(held.command.interaction_modes,["HOLD","HOLD"]);
  s.settle(["ILY","ILY"]);assert.deepEqual(s.settle(["PALM","FIST"]).command.interaction_modes,["Z_OPEN","Z_CLOSE"]);
});
test("open and close request independent endpoints while both allow wrist Z movement",()=>{
  const s=setup();s.settle();s.settle(["PALM","FIST"]);
  const out=s.tick(["PALM","FIST"],{dy:[-.01,.01]});
  assert.equal(out.command.control_protocol,protocol);assert.deepEqual(out.command.widths_m,[.08,0]);
  assert.deepEqual(out.command.release_intent,[true,false]);assert.equal(out.command.mode,"Z");
  assert.ok(out.command.delta_task_m.left[2]>0);assert.ok(out.command.delta_task_m.right[2]<0);
  assert.ok(Math.hypot(...out.command.delta_task_m.left)<=.004+1e-12);
});
test("ILY holds currently issued width instead of continuing unfinished opening or closing",()=>{
  const s=setup();s.settle(["FIST","FIST"]);
  s.e.health.grippers.left.command_width_m=.025;s.e.health.grippers.right.command_width_m=.043;
  const transition=s.tick();assert.deepEqual(transition.command.interaction_modes,["HOLD","HOLD"]);
  assert.deepEqual(transition.command.widths_m,[.025,.043]);noMotion(transition);
  s.settle();const out=s.tick(["ILY","ILY"],{dy:-.01});
  assert.deepEqual(out.command.widths_m,[.025,.043]);assert.deepEqual(out.command.interaction_modes,["XY_HOLD","XY_HOLD"]);
  assert.ok(out.command.delta_task_m.left[0]<0);assert.equal(out.command.delta_task_m.left[2],0);
});
test("category confidence controls mode without additional finger-angle or stop/takeover gates",()=>{
  const s=setup();s.settle();
  for(let i=0;i<8;i++)s.tick(["UNKNOWN","UNKNOWN"],{gestures:["Open_Palm","Closed_Fist"],scores:[.55,.8],dy:-i*.004});
  const out=s.tick(["UNKNOWN","UNKNOWN"],{gestures:["Open_Palm","Closed_Fist"],scores:[.55,.8],dy:-.04});
  assert.deepEqual(out.command.interaction_modes,["Z_OPEN","Z_CLOSE"]);assert.ok(out.command.delta_task_m.left[2]>0);
});
test("low category confidence stops that side immediately and reconfirmation starts a fresh baseline",()=>{
  const s=setup();s.settle(["FIST","FIST"]);
  const stopped=s.tick(["FIST","FIST"],{scores:[.79,.8],dy:-.01});
  assert.deepEqual(stopped.command.interaction_modes,["HOLD","Z_CLOSE"]);assert.equal(stopped.command.widths_m[0],.08);
  assert.equal(Math.hypot(...stopped.command.delta_task_m.left),0);
  for(let i=0;i<5;i++)assert.equal(s.tick(["FIST","FIST"],{dy:-.02}).command.interaction_modes[0],"HOLD");
  const baseline=s.tick(["FIST","FIST"],{dy:-.02});assert.equal(baseline.command.interaction_modes[0],"Z_CLOSE");
  assert.equal(Math.hypot(...baseline.command.delta_task_m.left),0);
});
test("unknown categories lock the jaw and never use measured finger distance",()=>{
  const s=setup();s.settle(["FIST","FIST"]);s.e.health.grippers.left.command_width_m=.031;
  const out=s.settle(["PINCH","PINCH"],{ratios:[.15,1.2]});
  assert.deepEqual(out.command.interaction_modes,["HOLD","HOLD"]);assert.deepEqual(out.command.widths_m,[.031,.08]);noMotion(out);
});
test("mixed XY and Z hold both sides and reset position anchors",()=>{
  const s=setup();s.settle();const out=s.settle(["ILY","FIST"],{dy:-.02});
  assert.equal(out.reason,"mixed_motion_planes");assert.deepEqual(out.command.interaction_modes,["HOLD","HOLD"]);noMotion(out);
  let resumed;for(let i=0;i<6;i++)resumed=s.tick(["ILY","ILY"],{dy:-.04});
  assert.deepEqual(resumed.command.interaction_modes,["XY_HOLD","XY_HOLD"]);noMotion(resumed);
});
for(const fault of ["invalid","single","stale","lowIdentity"])test(`v3 ${fault} input pauses both sides and preserves binding`,()=>{
  const s=setup();s.settle(["FIST","FIST"]);const out=s.tick(["FIST","FIST"],{[fault]:true});
  assert.equal(out.mode,"PAUSE");assert.equal(out.command.command,"pause");assert.equal(s.e.tracker.calibrated,true);
  for(let i=0;i<5;i++){
    const recovery=s.tick(["FIST","FIST"]);assert.equal(recovery.command.command,"continuous_input");
    assert.deepEqual(recovery.command.interaction_modes,["HOLD","HOLD"]);noMotion(recovery);
  }
  const active=s.settle(["FIST","FIST"]);assert.deepEqual(active.command.interaction_modes,["Z_CLOSE","Z_CLOSE"]);
});
test("manual pause remains effective without requiring or resetting an aperture calibration",()=>{
  const s=setup();s.settle(["FIST","FIST"]);s.e.pause("user_pause");
  for(let i=0;i<8;i++)assert.equal(s.tick(["PALM","PALM"]).reason,"user_pause");
  s.e.resume();const out=s.settle(["FIST","FIST"]);assert.ok(["HOLD","Z_CLOSE"].includes(out.command.interaction_modes[0]));
  assert.equal(out.telemetry.aperture_calibration.phase,"NOT_REQUIRED");
});
