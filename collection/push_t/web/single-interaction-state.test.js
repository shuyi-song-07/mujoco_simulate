import test from 'node:test';
import assert from 'node:assert/strict';
import {SingleInteractionState,recordingButtons,continuousPacket,singleHandTelemetry,operationExplicitlyRejected} from './single-interaction-state.js';
const hand=(gesture='Open_Palm',label='right',u=.5,v=.5)=>({raw_label:label,bound_side:label,valid:true,identity_score:.99,gesture,gesture_score:.99,image_landmarks_21:Array.from({length:21},()=>[u,v,0]),invalid_reason:null});
const measurement=(t,h=[hand()])=>({measurement_id:`sample-${t}`,capture_epoch_ms:t,accepted_recognition:true,hands:h});
function tick(e,t,gesture='Open_Palm',label='right',u=.5,v=.5){return e.update(measurement(t,[hand(gesture,label,u,v)]),t+20);}
function bind(e){for(let t=1000;t<=3000;t+=50)tick(e,t);assert.equal(e.boundLabel,'right');}
function activate(e,gesture='ILoveYou') {let next;for(let t=3050;t<=3650;t+=50)next=tick(e,t,gesture);return next;}

test('binds one actual right hand to left robot channel after two seconds',()=>{
  const e=new SingleInteractionState();for(let t=1000;t<3000;t+=50)tick(e,t);assert.equal(e.boundLabel,null);
  const n=tick(e,3000);assert.equal(e.boundLabel,'right');assert.equal(n.measurement.hands.length,1);
  assert.equal(n.hand.raw_label,'right');assert.equal(n.hand.bound_side,'left');assert.equal(n.canSendControl,false);
});
test('two palms are ambiguous and do not silently choose a hand',()=>{
  const e=new SingleInteractionState();for(let t=1000;t<=4000;t+=50)e.update(measurement(t,[hand('Open_Palm','left'),hand()]),t+10);
  assert.equal(e.boundLabel,null);
});
test('scene change preserves an unfinished binding guard and resets motion history',()=>{
  const e=new SingleInteractionState();bind(e);e.resetForScene();
  assert.equal(e.boundLabel,'right');assert.equal(e.palmBindingGuard,true);
  assert.equal(e.lastCapture,null);assert.equal(e.anchor,null);assert.equal(e.manualPaused,false);
  const held=tick(e,4000,'Open_Palm');
  assert.equal(held.canSendControl,false);assert.deepEqual(held.delta,[0,0,0]);
});
test('other hand cannot take over after binding; raw invalid evidence survives',()=>{
  const e=new SingleInteractionState();bind(e);activate(e);
  const wrong=tick(e,3700,'ILoveYou','left');assert.equal(wrong.canSendControl,false);assert.equal(e.boundLabel,'right');
  const invalid=hand('ILoveYou');invalid.valid=false;invalid.identity_score=.6;invalid.invalid_reason='identity_low';
  const n=e.update(measurement(3750,[invalid]),3770);assert.equal(n.canSendControl,false);assert.equal(n.measurement.hands[0].identity_score,.6);
});
test('binding palm cannot move; mode confirmation establishes a zero-displacement baseline',()=>{
  const e=new SingleInteractionState();bind(e);assert.equal(tick(e,3050).canSendControl,false);
  const n=activate(e);assert.equal(n.mode,'XY_HOLD');assert.deepEqual(n.delta,[0,0,0]);
  const moved=tick(e,3700,'ILoveYou','right',.52,.52);assert.ok(Math.hypot(...moved.delta)>0);assert.ok(Math.hypot(...moved.delta)<=.004+1e-12);assert.equal(moved.delta[2],0);
});
test('mirrored wrist movement follows the single-arm top view in all four directions',()=>{
  const directions=[
    ['right',.48,.5,1,-1],['left',.52,.5,1,1],
    ['up',.5,.48,0,1],['down',.5,.52,0,-1],
  ];
  for(const label of ['left','right'])for(const [direction,u,v,axis,sign] of directions) {
    const e=new SingleInteractionState();
    for(let t=1000;t<=3000;t+=50)tick(e,t,'Open_Palm',label);
    for(let t=3050;t<=3650;t+=50)tick(e,t,'ILoveYou',label);
    const moved=tick(e,3700,'ILoveYou',label,u,v);
    assert.ok(moved.delta[axis]*sign>0,`${label} hand moving ${direction}`);
    assert.equal(moved.delta[1-axis],0);assert.equal(moved.delta[2],0);
    assert.ok(Math.hypot(...moved.delta)<=.004+1e-12);
    assert.deepEqual(continuousPacket(moved,{}).delta_task_m.left,moved.delta);
    assert.deepEqual(tick(e,3750,'ILoveYou',label,u,v).delta,[0,0,0]);
  }
});
test('unknown gesture allows recenter without a jump; missing hand keeps identity',()=>{
  const e=new SingleInteractionState();bind(e);activate(e);tick(e,3700,'ILoveYou','right',.52,.52);
  const hold=tick(e,3750,'None','right',.3,.3);assert.equal(hold.mode,'HOLD');assert.deepEqual(hold.delta,[0,0,0]);
  let n;for(let t=3800;t<=4050;t+=50)n=tick(e,t,'ILoveYou','right',.5,.5);assert.deepEqual(n.delta,[0,0,0]);
  n=e.update(measurement(4100,[]),4120);assert.equal(n.canSendControl,false);assert.equal(e.boundLabel,'right');
  n=tick(e,4150,'ILoveYou');assert.equal(n.mode,'HOLD');
});
test('stale/repeated samples and manual pause cannot move the arm',()=>{
  const e=new SingleInteractionState();bind(e);activate(e);
  const stale=e.update(measurement(3700,[hand('ILoveYou')]),4000);assert.equal(stale.canSendControl,false);
  const repeated=tick(e,3700,'ILoveYou');assert.equal(repeated.canSendControl,false);
  e.setPaused(true);let n;for(let t=4050;t<4500;t+=50)n=tick(e,t,'Closed_Fist','right',.5,.4);assert.equal(n.canSendControl,false);assert.deepEqual(n.delta,[0,0,0]);
});
test('palm and fist give height-only motion and never an open gripper target',()=>{
  for(const gesture of ['Closed_Fist','Open_Palm']) {
    const e=new SingleInteractionState();bind(e);activate(e);let n;
    for(let t=3700;t<=4000;t+=50)n=tick(e,t,gesture);n=tick(e,4050,gesture,'right',.55,.48);
    assert.deepEqual(n.delta.slice(0,2),[0,0]);assert.ok(n.delta[2]>0);
    const p=continuousPacket(n,{});assert.equal(p.interaction_modes.length,1);assert.deepEqual(p.widths_m,[0]);
    assert.deepEqual(Object.keys(p.delta_task_m),['left']);assert.equal(p.release_intent[0],gesture==='Open_Palm');
  }
});
test('DONE while recording enables manual save; never changes recording to practice',()=>{
  const state={task_state:'DONE',recording:true,saving:false};
  const enabled=recordingButtons(state,true);assert.equal(enabled.record_save,true);assert.equal(enabled.record_failure,true);assert.equal(enabled.record_discard,true);assert.equal(enabled.record_start,false);
  assert.equal(state.recording,true);assert.equal(recordingButtons({...state,recording:false},true).record_save,false);
  assert.equal(recordingButtons(state,true,true).record_save,false);assert.equal(recordingButtons(state,false).record_save,false);
});
test('actual UI packet exposes visible for the backend and never fabricates another hand',()=>{
  const e=new SingleInteractionState();bind(e);const n=activate(e);
  const p=continuousPacket(n,{hands:singleHandTelemetry(n)});
  assert.deepEqual(Object.keys(p.telemetry.hands),['left']);
  assert.equal(p.telemetry.hands.left.visible,true);assert.equal(p.telemetry.hands.left.valid,true);
  assert.equal(p.telemetry.hands.left.gesture,'ILoveYou');
  const lost=e.update(measurement(3700,[]),3720);
  assert.equal(singleHandTelemetry(lost).left.visible,false);
  assert.equal(singleHandTelemetry(null).left.visible,false);
});
test('explicit HTTP rejection unlocks retry while network/accepted-queue uncertainty does not',()=>{
  assert.equal(operationExplicitlyRejected({status:409,response:{queued:false,reason:'stale_command'}}),true);
  assert.equal(operationExplicitlyRejected({status:400,response:{error:'bad_request'}}),true);
  assert.equal(operationExplicitlyRejected({status:409,response:{queued:true}}),false);
  assert.equal(operationExplicitlyRejected({status:500,response:{}}),false);
  assert.equal(operationExplicitlyRejected(Error('Timeout')),false);
});
