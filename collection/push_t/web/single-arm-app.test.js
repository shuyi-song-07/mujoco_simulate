import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createContext,runInContext} from 'node:vm';
import {SingleInteractionState,recordingButtons} from './single-interaction-state.js';

const source=readFileSync(new URL('./single-arm-app.js',import.meta.url),'utf8');
function tick(engine,t,u=.5) {
  return engine.update({measurement_id:`sample-${t}`,capture_epoch_ms:t,accepted_recognition:true,
    hands:[{raw_label:'right',valid:true,identity_score:.99,gesture:t<=3000?'Open_Palm':'ILoveYou',
      gesture_score:.99,image_landmarks_21:Array.from({length:21},()=>[u,.5,0])}]},t+10);
}
function setup() {
  const engine=new SingleInteractionState(),sent=[];
  for(let t=1000;t<=3650;t+=50)tick(engine,t);
  tick(engine,3700,.52);
  const state=createContext({engine,recordingButtons,connected:true,operation:null,operationError:'',
    runtime:{task_state:'PREGRASP',recording:false,saving:false,session:'test',control_epoch:0},
    latestInput:{command:'continuous_input'},lastNext:{},lastDiagnostic:{},lastSaved:null,
    eventQueue:[],views:[],previews:{},workerBusy:false,workerReady:true,shutdown:false,
    stream:{},video:{readyState:2,currentTime:0},lastVideoTime:0,lastResultAt:0,workerSentAt:0,
    performance:{now:()=>2000},Date,crypto:{randomUUID:()=>`operation-${sent.length}`},
    render:()=>{},telemetry:()=>({}),queueEvent:packet=>sent.push(packet),requestAnimationFrame:()=>{},
    journal:{freeze:()=>({}),flushBoundary:async()=>({})},
  });
  for(const [start,end] of [
    ['function finishOperation(', 'async function pump('],
    ["function pause(reason=", 'async function pollHealth('],
    ['async function cameraFrame(', 'async function startCamera('],
    ['async function operate(', "document.querySelectorAll('[data-command]').forEach(button=>"],
  ])runInContext(source.slice(source.indexOf(start),source.indexOf(end)),state);
  return {engine,state,sent,run:code=>runInContext(code,state)};
}
function confirm(s,command,epochChanged=true) {
  const pending=s.state.operation;
  if(epochChanged) {s.state.runtime.control_epoch++;s.run("clearContext('scene_reset')");}
  s.state.runtime.last_operation_result={eventId:pending.eventId,command,accepted:true};
  s.run('resolveOperation()');
}
function assertFreshRecovery(engine) {
  const first=tick(engine,4000,.7);
  assert.deepEqual(first.delta,[0,0,0]);
  let next;for(let t=4050;t<=4650;t+=50)next=tick(engine,t,.7);
  assert.equal(next.mode,'XY_HOLD');assert.deepEqual(next.delta,[0,0,0]);
  assert.ok(Math.hypot(...tick(engine,4700,.72).delta)>0);
}

test('save, discard and next scene automatically recover with the same bound hand',async()=>{
  for(const command of ['record_save','record_failure','record_discard','reset']) {
    const s=setup();
    s.state.runtime.recording=command!=='reset';
    if(command==='record_save')s.state.runtime.task_state='DONE';
    await s.run(`operate('${command}')`);
    assert.equal(s.sent.at(-1).command,command);
    assert.equal(s.engine.manualPaused,false);
    confirm(s,command);
    assert.equal(s.state.operation,null);assert.equal(s.engine.boundLabel,'right');
    assert.equal(s.engine.manualPaused,false);assert.equal(s.state.latestInput,null);
    assertFreshRecovery(s.engine);
  }
});
test('explicit manual pause remains paused across operations and scene changes',async()=>{
  const s=setup();s.run('pause()');await s.run("operate('reset')");confirm(s,'reset');
  assert.equal(s.engine.manualPaused,true);assert.equal(tick(s.engine,4000,.7).canSendControl,false);
  s.engine.setPaused(false);assertFreshRecovery(s.engine);
});
test('camera stall automatically recovers and intentional operation waits do not count as stalls',async()=>{
  const s=setup();s.state.operation={command:'record_save',uncertain:true};
  await s.run('cameraFrame()');assert.equal(s.sent.length,0);
  s.state.operation=null;s.state.runtime.saving=true;
  await s.run('cameraFrame()');assert.equal(s.sent.length,0);
  s.state.runtime.saving=false;
  await s.run('cameraFrame()');await s.run('cameraFrame()');
  assert.equal(s.sent.length,1);assert.equal(s.sent[0].reason,'camera_stalled');
  assert.equal(s.engine.manualPaused,false);assertFreshRecovery(s.engine);
});
test('a failed journal flush releases the temporary operation gate and requires fresh recovery',async()=>{
  const s=setup();s.state.journal.flushBoundary=async()=>{throw Error('journal unavailable');};
  await s.run("operate('reset')");
  assert.equal(s.sent.length,0);assert.equal(s.state.operation,null);
  assert.equal(s.engine.manualPaused,false);assertFreshRecovery(s.engine);
});
test('uncertain operations remain gated until an acknowledgement arrives',()=>{
  const s=setup();s.state.operation={command:'reset',eventId:'pending',epoch:0,uncertain:true};
  s.run('resolveOperation()');assert.equal(s.state.operation.eventId,'pending');
  confirm(s,'reset');assert.equal(s.state.operation,null);
});
test('backend restart requires binding again without creating a manual pause',()=>{
  const s=setup();s.run("clearContext('backend_session_changed')");
  assert.equal(s.engine.boundLabel,null);assert.equal(s.engine.manualPaused,false);
  assert.equal(tick(s.engine,4000,.7).canSendControl,false);
});
