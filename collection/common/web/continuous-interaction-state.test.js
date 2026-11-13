import test from "node:test";
import assert from "node:assert/strict";
import {ContinuousInteractionState} from "./continuous-interaction-state.js";
import {detectionsFromMediaPipe} from "./dual-interaction-state.js";
import {resultPair,measure} from "../../../tests/task3/hand_fixture.mjs";

function setup({manualPause=false,stopAtPhase="READY"}={}) {
  const e=new ContinuousInteractionState({wrist_filter:"ema",ema_alpha:1});let now=-40;
  e.syncHealth({task_state:"PREGRASP",gripper_latch:"CONTINUOUS",grippers:{left:{command_width_m:.08},right:{command_width_m:.08}}},0);
  if(manualPause)e.pause("user_pause");
  const s={e,tick(r=[1.2,1.3],opts={}) {
    now+=opts.stepMs??40;const m=measure(r,now,opts);
    if(opts.invalidEdgeSide)Object.assign(m.hands.find(h=>h.bound_side===opts.invalidEdgeSide),{valid:false,invalid_reason:"hand_near_frame_edge",near_edge:true});
    e.setMeasurement(m);return e.update(detectionsFromMediaPipe(resultPair(r,opts)),now);
  },time:()=>now};
  for(let i=0;i<53;i++) {const out=s.tick();assert.notEqual(out.command?.command,"continuous_input");}
  assert.equal(e.tracker.calibrated,true);
  if(stopAtPhase==="CLOSE")return s;
  for(let i=0;i<40;i++) {const out=s.tick([.15,.2]);assert.notEqual(out.command?.command,"continuous_input");}
  if(stopAtPhase==="OPEN") {assert.equal(e.apertureCalibration.phase,"OPEN");return s;}
  for(let i=0;i<41;i++)s.tick([1.2,1.3]);
  assert.equal(e.apertureCalibration.phase,"READY");
  for(let i=0;i<15;i++)s.tick();
  return s;
}
test("binding and calibration never actuate; movement and independent aperture coexist",()=>{
  const s=setup();const out=s.tick([.5,1.3],{dy:-.01});
  assert.equal(out.command.command,"continuous_input");
  assert.ok(out.command.delta_task_m.left[0]<0);
  assert.ok(out.command.widths_m[0]<out.command.widths_m[1]);
});

test("release intention uses each trusted calibrated opening independently, not smoothed actuator width",()=>{
  const s=setup();
  for(let i=0;i<15;i++)s.tick([.15,.2]);
  const out=s.tick([1.2,.2]);
  assert.equal(out.command.command,"continuous_input");
  assert.deepEqual(out.command.release_intent,[true,false]);
  assert.ok(out.command.widths_m[0]<.068,"intent is separate from the delayed aperture target");
  const closed=s.tick([.15,.2]);assert.deepEqual(closed.command.release_intent,[false,false]);
  const edge=s.tick([1.2,1.3],{invalidEdgeSide:"left"});
  assert.equal(edge.command.command,"pause");assert.equal(edge.command.release_intent,undefined);
});

test("held grasp recovery rebuilds wrist baseline without forcing hand opening to match locked jaw",()=>{
  const s=setup();s.e.pause("user_pause");
  s.e.syncHealth({...s.e.health,physical_contact:{control_protocol:"grasp-hold-release-v1",hold_width_m:{left:.02,right:.03}},
    grippers:{left:{command_width_m:.02},right:{command_width_m:.03}}},s.time());
  s.e.resume();let first=null;
  for(let i=0;i<30;i++) {const out=s.tick([1.2,1.3],{dy:.03});if(out.command?.command==="continuous_input"&&!first)first=out;}
  assert.ok(first,"stable hands may resume held grasp without aperture matching");
  for(const side of ["left","right"])assert.ok(Math.hypot(...first.command.delta_task_m[side])<1e-9);
  assert.deepEqual(first.command.release_intent,[true,true],"backend still decides arming, support and confirmation");
});

test("partial grasp recovery still requires aperture matching for the free side",()=>{
  const s=setup();s.e.pause("user_pause");
  s.e.syncHealth({...s.e.health,physical_contact:{control_protocol:"grasp-hold-release-v1",hold_width_m:{left:.02,right:null}},
    grippers:{left:{command_width_m:.02},right:{command_width_m:.03}}},s.time());
  s.e.resume();
  for(let i=0;i<25;i++)assert.notEqual(s.tick().command?.command,"continuous_input");
  assert.equal(s.e.reason,"aperture_takeover");
  const ratios=[1.2,.2+(1.3-.2)*.03/.08];
  assert.equal(s.tick(ratios).reason,"continuous_baseline");
  assert.equal(s.tick(ratios).command.command,"continuous_input");
});
test("mode transition preserves width then rebuilds wrist baseline",()=>{
  const s=setup();
  const out=s.tick([.4,.4],{roll:Math.PI/2});assert.equal(out.command.command,"pause");
  for(let i=0;i<10;i++) {const r=s.tick([.4,.4],{roll:Math.PI/2});assert.notEqual(r.command?.command,"continuous_input");}
  for(let i=0;i<10;i++)s.tick([1.2,1.3],{roll:Math.PI/2});
  const moved=s.tick([1.2,1.3],{roll:Math.PI/2,dy:-.01});
  assert.equal(moved.command.mode,"Z");assert.ok(moved.command.delta_task_m.left[2]>0);
  assert.equal(moved.command.delta_task_m.left[0],0);
});
test("clutch permits reposition within frame and cannot replay accumulated displacement",()=>{
  const s=setup();for(let i=0;i<10;i++)assert.notEqual(s.tick([1.2,1.3],{roll:.75,dy:-.04}).command?.command,"continuous_input");
  let first=null;
  for(let i=0;i<20;i++){const out=s.tick([1.2,1.3],{dy:-.04});if(out.command?.command==="continuous_input"&&!first)first=out;}
  assert.ok(first);assert.ok(Math.hypot(...first.command.delta_task_m.left)<1e-7);
});
test("loss/recovery and manual pause hold independent widths without auto opening",()=>{
  const s=setup();s.e.setMeasurement(null);const out=s.e.update([],s.time()+20);assert.equal(out.command.command,"pause");
  s.e.syncHealth({task_state:"DUAL_GRASPED",gripper_latch:"CONTINUOUS",grippers:{left:{command_width_m:.005},right:{command_width_m:.01}}},s.time());
  for(let i=0;i<20;i++)assert.notEqual(s.tick().command?.command,"continuous_input");
  assert.equal(s.e.reason,"aperture_takeover");
  s.e.pause("user_pause");for(let i=0;i<20;i++)assert.notEqual(s.tick([.15,.2]).command?.command,"continuous_input");
  s.e.resume();for(let i=0;i<20;i++)s.tick([.15,.2]);
  assert.equal(s.e.mode,"XY");
});
test("unbinding clears calibration and failed episode never emits motion",()=>{
  const s=setup();s.e.syncHealth({...s.e.health,task_state:"FAIL",failure_reason:"contact_loss"},s.time());
  assert.notEqual(s.tick([.15,.2]).command?.command,"continuous_input");
  s.e.unbind();assert.equal(s.e.apertureCalibration.phase,"CLOSE");assert.equal(s.e.tracker.calibrated,false);
});

test("manual pause before binding allows aperture calibration but requires explicit recovery and takeover",()=>{
  const s=setup({manualPause:true});
  assert.equal(s.e.manualPaused,true);assert.equal(s.e.reason,"user_pause");
  for(let i=0;i<12;i++)assert.notEqual(s.tick([.5,.7],{dy:.03}).command?.command,"continuous_input");
  const held=[.025,.06];
  s.e.syncHealth({...s.e.health,grippers:{left:{command_width_m:held[0]},right:{command_width_m:held[1]}}},s.time());
  s.e.resume();
  for(let i=0;i<5;i++) {
    const out=s.tick([1.2,1.3],{dy:.03});
    assert.notEqual(out.command?.command,"continuous_input");assert.equal(s.e.tracker.recoveryPending,true);
  }
  for(let i=0;i<15;i++)assert.notEqual(s.tick([1.2,1.3],{dy:.03}).command?.command,"continuous_input");
  assert.equal(s.e.reason,"aperture_takeover");
  const matching=[.15+(1.2-.15)*held[0]/.08,.2+(1.3-.2)*held[1]/.08];
  const baseline=s.tick(matching,{dy:.03});
  assert.equal(baseline.reason,"continuous_baseline");assert.notEqual(baseline.command?.command,"continuous_input");
  const resumed=s.tick(matching,{dy:.03});assert.equal(resumed.command.command,"continuous_input");
  for(const side of ["left","right"])assert.ok(Math.hypot(...resumed.command.delta_task_m[side])<1e-9);
  held.forEach((v,i)=>assert.ok(Math.abs(resumed.command.widths_m[i]-v)<1e-9));
});

for(const edgeSide of ["left","right"])test(`${edgeSide} edge measurements stop control and reset recovery before matching widths at a new wrist position`,()=>{
  const s=setup();const held=[.03,.05];
  s.e.syncHealth({...s.e.health,grippers:{left:{command_width_m:held[0]},right:{command_width_m:held[1]}}},s.time());
  const edge=()=>s.tick([.25,.25],{dy:.03,invalidEdgeSide:edgeSide});
  const stopped=edge();assert.equal(stopped.command.command,"pause");assert.equal(stopped.reason,"hand_near_frame_edge");
  assert.equal(s.e.tracker.recoveryPending,true);assert.equal(s.e.takeover,true);
  for(let i=0;i<4;i++)assert.notEqual(s.tick([1.2,1.3],{dy:.03}).command?.command,"continuous_input");
  assert.equal(s.e.tracker.recoveryCandidate.samples,4);
  assert.equal(edge().reason,"hand_near_frame_edge");assert.equal(s.e.tracker.recoveryCandidate,null);
  for(let i=0;i<5;i++) {
    assert.notEqual(s.tick([1.2,1.3],{dy:.03}).command?.command,"continuous_input");
    assert.equal(s.e.tracker.recoveryPending,true);
  }
  assert.equal(s.tick([1.2,1.3],{dy:.03}).reason,"tracking_recovered");
  for(let i=0;i<8;i++)assert.notEqual(s.tick([1.2,1.3],{dy:.03}).command?.command,"continuous_input");
  assert.equal(s.e.reason,"aperture_takeover");
  const matching=[.15+(1.2-.15)*held[0]/.08,.2+(1.3-.2)*held[1]/.08];
  const baseline=s.tick(matching,{dy:.03});assert.equal(baseline.reason,"continuous_baseline");
  assert.notEqual(baseline.command?.command,"continuous_input");
  const resumed=s.tick(matching,{dy:.03});assert.equal(resumed.command.command,"continuous_input");
  for(const side of ["left","right"])assert.ok(Math.hypot(...resumed.command.delta_task_m[side])<1e-9);
  held.forEach((v,i)=>assert.ok(Math.abs(resumed.command.widths_m[i]-v)<1e-9));
});

for(const phase of ["CLOSE","OPEN"])test(`${phase} calibration cannot count a short edge interruption toward its stable sample window`,()=>{
  const s=setup({stopAtPhase:phase}),ratios=phase==="CLOSE"?[.15,.2]:[1.2,1.3];
  for(let i=0;i<30;i++)s.tick(ratios);
  assert.equal(s.e.apertureCalibration.phase,phase);assert.ok(s.e.apertureCalibration.progress>.7);
  const endpoints=structuredClone(s.e.apertureCalibration.endpoints);
  // Real handMeasurement rejects this wrist at y=.95. A fast camera can return
  // and finish tracking recovery before calibration's 250 ms gap limit expires.
  const stopped=s.tick(ratios,{dy:.3,stepMs:10});assert.equal(stopped.reason,"hand_near_frame_edge");
  for(let i=0;i<120;i++) {
    const out=s.tick(ratios,{stepMs:10});
    assert.notEqual(out.command?.command,"continuous_input");
    assert.equal(s.e.apertureCalibration.phase,phase,"a new stable window must last 1500 ms after trusted recovery");
  }
  assert.deepEqual(s.e.apertureCalibration.endpoints,endpoints,"completed endpoints survive an interrupted current phase");
  for(let i=0;i<100&&s.e.apertureCalibration.phase===phase;i++)s.tick(ratios,{stepMs:10});
  assert.equal(s.e.apertureCalibration.phase,phase==="CLOSE"?"OPEN":"READY");
});
