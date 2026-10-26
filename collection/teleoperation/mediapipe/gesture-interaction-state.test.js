import test from "node:test";
import assert from "node:assert/strict";
import {ContinuousInteractionState,anchoredWidth,GESTURE_CONTROL_PROTOCOL} from "./continuous-interaction-state.js";
import {DEFAULT_GESTURE_PARAMETERS} from "./hand-measurement.js";
import {detectionsFromMediaPipe} from "./dual-interaction-state.js";
import {gestureResult,gestureMeasure} from "../../../tests/task3/gesture_hand_fixture.mjs";
const sides=["left","right"];
function setup() {
  const e=new ContinuousInteractionState({wrist_filter:"ema",ema_alpha:1});let now=-40;
  e.syncHealth({task_state:"PREGRASP",gripper_latch:"CONTINUOUS",grippers:{left:{command_width_m:.08},right:{command_width_m:.08}},
    physical_contact:{control_protocol:GESTURE_CONTROL_PROTOCOL,hold_width_m:{left:null,right:null},operation_hold_width_m:{left:null,right:null},release_opening:{left:false,right:false},release_cycle_state:{left:"IDLE",right:"IDLE"},release_reentry_pending:{left:false,right:false}},
    action_contract:{control_protocol:GESTURE_CONTROL_PROTOCOL,gesture_parameters:{...DEFAULT_GESTURE_PARAMETERS}}},0);
  const s={e,time:()=>now,tick(shapes=["PINCH","PINCH"],opts={}) {
    now+=opts.stepMs??40;const m=gestureMeasure(shapes,now,opts),result=gestureResult(shapes,opts);
    if(opts.invalid)m.hands[0].valid=false;
    e.setMeasurement(m);const out=e.update(detectionsFromMediaPipe(result),now);
    if(out.command?.command==="continuous_input")sides.forEach((side,i)=>{
      if(out.command.interaction_modes[i]==="PINCH")e.health.grippers[side].command_width_m=out.command.widths_m[i];
      if(out.command.interaction_modes[i]==="Z_HOLD")e.health.physical_contact.operation_hold_width_m[side]=e.health.grippers[side].command_width_m;
    });
    return out;
  },settle(shapes=["PINCH","PINCH"],opts={},n=25){let out;for(let i=0;i<n;i++)out=this.tick(shapes,opts);return out;}};
  for(let i=0;i<53;i++)assert.notEqual(s.tick(["PALM","PALM"]).command?.command,"continuous_input");
  for(let i=0;i<40;i++)s.tick(["PINCH","PINCH"],{ratios:[.15,.15]});
  for(let i=0;i<41;i++)s.tick(["PINCH","PINCH"],{ratios:[1.2,1.2]});
  assert.equal(e.apertureCalibration.phase,"READY");return s;
}
const noMotion=out=>sides.forEach(s=>assert.equal(Math.hypot(...out.command.delta_task_m[s]),0));
test("anchored piecewise mapping holds the matching baseline and preserves both endpoints",()=>{
  assert.equal(anchoredWidth(.4,.4,.03),.03);assert.equal(anchoredWidth(0,.4,.03),0);assert.equal(anchoredWidth(1,.4,.03),.08);
  assert.ok(anchoredWidth(.2,.4,.03)<.03);assert.ok(anchoredWidth(.8,.4,.03)>.03);
  assert.equal(anchoredWidth(0,0,0),0);assert.equal(anchoredWidth(1,0,0),.08);
  assert.equal(anchoredWidth(1,1,.08),.08);assert.equal(anchoredWidth(0,1,.08),0);
  assert.equal(anchoredWidth(.5,0,.03),.055);assert.equal(anchoredWidth(.5,1,.03),.015);
  assert.equal(anchoredWidth(1,1,.079803),.079803);assert.equal(anchoredWidth(0,1,.079803),0);
  assert.equal(anchoredWidth(0,0,.0002),.0002);assert.equal(anchoredWidth(1,0,.0002),.08);
  assert.ok(Number.isFinite(anchoredWidth(.00005,.0001,.01)));
});
test("ILoveYou moves XY with the existing jaw command independently of finger aperture",()=>{
  const s=setup();s.e.health.grippers.left.command_width_m=.03;s.e.health.grippers.right.command_width_m=.05;
  s.settle(["ILY","ILY"],{ratios:[.2,1.2]});
  const out=s.tick(["ILY","ILY"],{ratios:[1.2,.2],dy:-.01});
  assert.equal(out.command.control_protocol,GESTURE_CONTROL_PROTOCOL);assert.equal(out.command.mode,"XY");
  assert.deepEqual(out.command.interaction_modes,["XY_HOLD","XY_HOLD"]);assert.deepEqual(out.command.widths_m,[.03,.05]);
  assert.deepEqual(out.command.release_intent,[false,false]);assert.ok(out.command.delta_task_m.left[0]<0);
});
test("fist is a permanent operation lock at the current command, never forced closing or physical grasp",()=>{
  const s=setup();s.e.health.grippers.left.command_width_m=.025;s.e.health.grippers.right.command_width_m=.045;
  const out=s.settle(["FIST","FIST"]);assert.deepEqual(out.command.interaction_modes,["Z_HOLD","Z_HOLD"]);
  assert.deepEqual(out.command.widths_m,[.025,.045]);assert.deepEqual(s.e.health.physical_contact.hold_width_m,{left:null,right:null});
  const held=s.settle(["PINCH","PINCH"],{ratios:[.15,.15]});assert.deepEqual(held.command.interaction_modes,["HOLD","HOLD"]);
  assert.deepEqual(held.command.widths_m,[.025,.045]);assert.deepEqual(held.command.release_intent,[false,false]);
});
test("None with idle-finger geometry controls independent pinch only after stationary takeover",()=>{
  const s=setup();s.e.health.grippers.left.command_width_m=.04;s.e.health.grippers.right.command_width_m=.04;
  s.settle(["PINCH_CLOSED","PINCH_CLOSED"],{ratios:[.675,.675]});
  const out=s.tick(["PINCH_CLOSED","PINCH_CLOSED"],{ratios:[.78,.675]});
  assert.deepEqual(out.command.interaction_modes,["PINCH","PINCH"]);assert.ok(out.command.widths_m[0]>.04);
  assert.ok(Math.abs(out.command.widths_m[1]-.04)<1e-9);noMotion(out);
  assert.equal(out.telemetry.pinch_mapping.version,"anchored-piecewise-v2");assert.equal(out.telemetry.pinch_mapping.left.active,true);
});
test("nearly open jaw accepts a saturated open hand without jumping and can then close independently",()=>{
  const s=setup(),held=.07980318905676966;
  s.e.health.grippers.left.command_width_m=held;s.e.resetGestureModes();
  for(let i=0;i<25;i++) {
    const out=s.tick(["PINCH","PINCH"],{ratios:[1.2,1.2]});
    assert.equal(out.command.widths_m[0],held);assert.equal(out.command.widths_m[1],.08);noMotion(out);
  }
  assert.equal(s.e.interaction.left.takeover,false);
  const closed=s.tick(["PINCH","PINCH"],{ratios:[.675,1.2]});
  assert.deepEqual(closed.command.interaction_modes,["PINCH","PINCH"]);
  assert.ok(closed.command.widths_m[0]<held);assert.equal(closed.command.widths_m[1],.08);noMotion(closed);
});
test("nearly closed jaw accepts a saturated closed hand without jumping and can then open independently",()=>{
  const s=setup(),held=.0002;
  s.e.health.grippers.left.command_width_m=held;s.e.resetGestureModes();
  for(let i=0;i<25;i++) {
    const out=s.tick(["PINCH","PINCH"],{ratios:[.15,1.2]});
    assert.equal(out.command.widths_m[0],held);assert.equal(out.command.widths_m[1],.08);noMotion(out);
  }
  assert.equal(s.e.interaction.left.takeover,false);
  const opened=s.tick(["PINCH","PINCH"],{ratios:[.675,1.2]});
  assert.deepEqual(opened.command.interaction_modes,["PINCH","PINCH"]);
  assert.ok(opened.command.widths_m[0]>held);assert.equal(opened.command.widths_m[1],.08);noMotion(opened);
});
test("endpoint takeover still rejects more than 12 mm mismatch without blocking the matching other hand",()=>{
  for(const [ratio,width] of [[1.2,.067],[.15,.013]]) {
    const s=setup();s.e.health.grippers.left.command_width_m=width;s.e.resetGestureModes();
    const out=s.settle(["PINCH","PINCH"],{ratios:[ratio,1.2]});
    assert.equal(out.reason,"aperture_takeover");assert.deepEqual(out.command.interaction_modes,["HOLD","PINCH"]);
    assert.deepEqual(out.command.widths_m,[width,.08]);assert.equal(s.e.interaction.left.takeover,true);noMotion(out);
  }
});
test("approach holds width and each stop requires takeover without replaying aperture drift or wrist displacement",()=>{
  const s=setup();s.e.health.grippers.left.command_width_m=.04;s.e.health.grippers.right.command_width_m=.04;
  s.settle(["PINCH","PINCH"],{ratios:[.675,.675]});
  for(let i=1;i<=12;i++) {
    const out=s.tick(["PINCH","PINCH"],{ratios:[1.2,1.2],dy:-i*.01});
    assert.deepEqual(out.command.widths_m,[.04,.04]);assert.equal(out.command.delta_task_m.left[0],0);
    assert.ok(out.command.interaction_modes.every(v=>["HOLD","Z_APPROACH"].includes(v)));
  }
  const wait=s.settle(["PINCH","PINCH"],{ratios:[1.2,1.2],dy:-.12});
  assert.equal(wait.reason,"aperture_takeover");assert.deepEqual(wait.command.widths_m,[.04,.04]);noMotion(wait);
  const baseline=s.tick(["PINCH","PINCH"],{ratios:[.675,.675],dy:-.12});
  assert.equal(baseline.reason,"pinch_baseline");assert.deepEqual(baseline.command.widths_m,[.04,.04]);noMotion(baseline);
  const resumed=s.tick(["PINCH","PINCH"],{ratios:[.675,.675],dy:-.12});
  assert.deepEqual(resumed.command.interaction_modes,["PINCH","PINCH"]);assert.deepEqual(resumed.command.widths_m,[.04,.04]);noMotion(resumed);
});
test("mode confirmation uses valid HOLD heartbeats instead of restarting recovery on every frame",()=>{
  const s=setup();s.settle(["ILY","ILY"]);
  for(let i=0;i<4;i++) {
    const out=s.tick(["FIST","FIST"]);assert.equal(out.command.command,"continuous_input");
    assert.deepEqual(out.command.interaction_modes,["HOLD","HOLD"]);noMotion(out);
  }
  assert.deepEqual(s.settle(["FIST","FIST"]).command.interaction_modes,["Z_HOLD","Z_HOLD"]);
});
test("mixed Z allows fist holding on one side and pinch on the other, while XY/Z mixes hold the pair",()=>{
  const s=setup();const z=s.settle(["FIST","PINCH"]);
  assert.deepEqual(z.command.interaction_modes,["Z_HOLD","PINCH"]);assert.equal(z.command.mode,"Z");
  const mixed=s.settle(["ILY","PINCH"]);assert.equal(mixed.reason,"mixed_motion_planes");
  assert.deepEqual(mixed.command.interaction_modes,["HOLD","HOLD"]);noMotion(mixed);
});
test("locked release is complete-palm evidence even at 20 percent aperture and never None or index-flexed geometry",()=>{
  const s=setup();s.e.health.physical_contact.operation_hold_width_m={left:.03,right:.04};
  s.e.health.grippers.left.command_width_m=.03;s.e.health.grippers.right.command_width_m=.04;
  const none=s.settle(["PINCH","PINCH"],{ratios:[.2,.2]});assert.deepEqual(none.command.release_intent,[false,false]);
  const wrong=s.settle(["PINCH_CLOSED","PINCH_CLOSED"],{gestures:["Open_Palm","Open_Palm"],ratios:[.2,.2]});assert.deepEqual(wrong.command.release_intent,[false,false]);
  const release=s.settle(["PALM","PALM"],{ratios:[.2,.2],scores:[.6,.6]});
  assert.deepEqual(release.command.interaction_modes,["RELEASE","RELEASE"]);assert.deepEqual(release.command.release_intent,[true,true]);
  assert.deepEqual(release.command.widths_m,[.03,.04]);noMotion(release);
});
test("authorized opening tolerates None without a persistent aperture but explicit modes or motion cancel immediately",()=>{
  for(const cancel of ["FIST","ILY","MOVE"]) {
    const s=setup();s.e.health.physical_contact.release_opening={left:true,right:true};
    const opening=s.tick(["PINCH_CLOSED","PINCH_CLOSED"],{ratios:[.2,.2]});
    assert.deepEqual(opening.command.interaction_modes,["OPENING","OPENING"]);assert.deepEqual(opening.command.release_intent,[false,false]);noMotion(opening);
    const stopped=cancel==="MOVE"?s.tick(["PINCH_CLOSED","PINCH_CLOSED"],{ratios:[.2,.2],dy:.01}):s.tick([cancel,cancel]);
    assert.deepEqual(stopped.command.interaction_modes,["HOLD","HOLD"]);assert.equal(stopped.reason,"opening_cancelled");noMotion(stopped);
    assert.deepEqual(s.tick(["PINCH_CLOSED","PINCH_CLOSED"],{ratios:[.2,.2]}).command.interaction_modes,["HOLD","HOLD"],"stale opening health cannot resume an interrupted authorization");
  }
});
test("release completion keeps jaws open until an explicit cycle exit and a new pinch takeover",()=>{
  const s=setup();s.e.health.physical_contact.release_opening={left:true,right:true};s.tick(["PINCH","PINCH"]);
  s.e.health.physical_contact.release_opening={left:false,right:false};
  s.e.health.physical_contact.release_cycle_state={left:"COMPLETE",right:"COMPLETE"};
  s.e.health.physical_contact.release_reentry_pending={left:true,right:true};
  const held=s.settle(["PALM","PALM"]);assert.deepEqual(held.command.interaction_modes,["HOLD","HOLD"]);assert.deepEqual(held.command.widths_m,[.08,.08]);
  s.tick(["ILY","ILY"]);s.e.health.physical_contact.release_reentry_pending={left:false,right:false};
  s.settle(["ILY","ILY"]);const retaken=s.settle(["PINCH","PINCH"]);
  assert.deepEqual(retaken.command.interaction_modes,["PINCH","PINCH"]);assert.deepEqual(retaken.command.widths_m,[.08,.08]);
});
test("ABORTED retains locked width but permits a fresh continuous palm confirmation without resuming the old opening",()=>{
  const s=setup(),contact=s.e.health.physical_contact;
  s.e.health.grippers.left.command_width_m=.03;s.e.health.grippers.right.command_width_m=.04;
  contact.release_opening={left:true,right:true};
  s.tick(["PINCH_CLOSED","PINCH_CLOSED"],{ratios:[.2,.2]});
  assert.equal(s.tick(["ILY","ILY"]).reason,"opening_cancelled");
  contact.release_opening={left:false,right:false};contact.release_cycle_state={left:"ABORTED",right:"ABORTED"};
  contact.release_reentry_pending={left:true,right:true};contact.operation_hold_width_m={left:.03,right:.04};
  const none=s.settle(["PINCH_CLOSED","PINCH_CLOSED"],{ratios:[.2,.2]});
  assert.deepEqual(none.command.interaction_modes,["HOLD","HOLD"]);assert.deepEqual(none.command.widths_m,[.03,.04]);
  for(let i=0;i<3;i++)assert.deepEqual(s.tick(["PALM","PALM"],{ratios:[.2,.2]}).command.interaction_modes,["HOLD","HOLD"]);
  const move=s.settle(["ILY","ILY"]);assert.deepEqual(move.command.interaction_modes,["XY_HOLD","XY_HOLD"]);
  assert.deepEqual(move.command.release_intent,[false,false]);
  const fresh=s.settle(["PALM","PALM"],{ratios:[.2,.2],scores:[.6,.6]});
  assert.deepEqual(fresh.command.interaction_modes,["RELEASE","RELEASE"]);assert.deepEqual(fresh.command.release_intent,[true,true]);
  assert.deepEqual(fresh.command.widths_m,[.03,.04]);noMotion(fresh);
  assert.equal(s.e.interaction.left.release_cycle_guard,false);
  for(let i=0;i<20;i++)assert.deepEqual(s.tick(["PALM","PALM"],{ratios:[.2,.2]}).command.interaction_modes,["RELEASE","RELEASE"],"only the backend can authorize a new OPENING after its fresh 700ms window");
});
test("invalid tracking and manual pause abort permissions; reliable recovery emits HOLD before new takeover",()=>{
  const s=setup();s.settle();const bad=s.tick(["PINCH","PINCH"],{invalid:true});assert.equal(bad.command.command,"pause");
  for(let i=0;i<5;i++)assert.deepEqual(s.tick().command.interaction_modes,["HOLD","HOLD"]);
  assert.equal(s.tick().reason,"tracking_recovered");
  s.e.pause("user_pause");assert.notEqual(s.tick().command?.command,"continuous_input");
  s.e.resume();assert.deepEqual(s.tick().command.interaction_modes,["HOLD","HOLD"]);
});
test("small cumulative wrist motion cannot count as a stable pinch interval",()=>{
  const s=setup();let modes=[];
  for(let i=0;i<24;i++)modes=s.tick(["PINCH","PINCH"],{dy:i*.001}).command.interaction_modes;
  assert.ok(modes.every(v=>v!=="PINCH"));assert.equal(s.e.interaction.left.takeover,true);
});

test("bounded sub-five-pixel wrist jitter permits independent pinch without moving the arm",()=>{
  const s=setup();
  // A fixed-reference oscillation: bounded measurement/pose jitter, not a
  // progressively moving wrist. Full-open takeover precedes finger closure.
  for(let i=0;i<20;i++)s.tick(["PINCH","PINCH"],{dy:i%2?.003:-.003});
  for(let i=0;i<20;i++) {
    const out=s.tick(["PINCH","PINCH"],{dy:i%2?.003:-.003,ratios:[1.2-i*.025,1.2]});
    assert.deepEqual(out.command.interaction_modes,["PINCH","PINCH"]);
    noMotion(out);
  }
  assert.ok(s.e.health.grippers.left.command_width_m<.06);
  assert.equal(s.e.health.grippers.right.command_width_m,.08);
  const moved=s.tick(["PINCH","PINCH"],{dy:.03,ratios:[.7,1.2]});
  assert.ok(moved.command.interaction_modes.every(v=>v!=="PINCH"));
  assert.equal(s.e.interaction.left.takeover,true);
});
