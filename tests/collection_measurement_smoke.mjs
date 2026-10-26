// Scripted landmarks through the real frontend engine + HTTP recorder. No camera claim.
import {writeFileSync} from 'node:fs';
import {setTimeout as sleep} from 'node:timers/promises';
import {ContinuousInteractionState} from '../collection/teleoperation/mediapipe/continuous-interaction-state.js';
import {detectionsFromMediaPipe} from '../collection/teleoperation/mediapipe/dual-interaction-state.js';
import {handMeasurement} from '../collection/teleoperation/mediapipe/hand-measurement.js';
import {MeasurementJournal,controlObservation} from '../collection/teleoperation/mediapipe/measurement-journal.js';
import {gestureResult} from './task3/gesture_hand_fixture.mjs';
const [url,reportPath]=process.argv.slice(2);
const health=async()=>{const r=await fetch(url+'/health');return r.json();};
let runtime;
for(let i=0;i<150;i++){try{runtime=await health();break;}catch{await sleep(100);}}
if(!runtime)throw Error('backend unavailable');
const engine=new ContinuousInteractionState(runtime.config.control);engine.syncHealth(runtime,performance.now());
let seq=0,frame=0;const client='scripted-hand-'+Date.now(),stages=[];
const context={input_kind:'scripted_test',camera:{width:640,height:480},display_mirrored:true,inference_mirrored:false,
  filter_and_mapping:engine.config,gesture_parameters:engine.gestureParameters,control_protocol:engine.gestureProtocol()};
const journal=new MeasurementJournal({clientId:client,send:async b=>{
  const r=await fetch(url+'/measurements',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const ack=await r.json();if(!r.ok)throw Error(JSON.stringify(ack));return ack;
}});
async function post(command){
  const packet={...command,protocol:'task3-command-v2',client_id:client,sequence:++seq,eventId:`${client}:${seq}`,
    sentAt:command.command==='continuous_input'?command.telemetry.recognition_result.measurement.capture_epoch_ms:Date.now(),session:runtime.session,control_epoch:runtime.control_epoch};
  journal.add('dispatch',{event_id:packet.eventId,session:packet.session,control_epoch:packet.control_epoch,
    measurement_id:packet.measurement_id??packet.telemetry?.recognition_result?.measurement?.measurement_id??null,
    command:packet.command,delta_task_m:packet.delta_task_m??null},context);
  if(command.command==='record_stop')await journal.flush();
  const r=await fetch(url+'/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(packet)});
  const result=await r.json();if(!r.ok)throw Error(JSON.stringify(result));return {...result,eventId:packet.eventId};
}
async function stage(name,shapes,n,options={}){
  const summary={name,measurements:0,modes:[],reasons:[],nonzero_deltas:0};
  for(let i=0;i<n;i++){
    const began=performance.now();runtime=await health();engine.syncHealth(runtime,began);
    const now=performance.now(),epoch=Date.now();
    const result=gestureResult(shapes,{...options,dy:options.move?-.0015*i:0});
    if(options.absent){result.landmarks=[];result.worldLandmarks=[];result.handedness=[];result.gestures=[];}
    const measurement=handMeasurement(result,{id:`${client}:m${++frame}`,session:runtime.session,epoch:runtime.control_epoch,
      capturedAt:now,capturedEpochMs:epoch,recognitionCompletedEpochMs:epoch,receivedAt:now,
      inferenceMs:0,width:640,height:480});
    const before=Object.fromEntries(['left','right'].map(s=>[s,engine.interaction[s].motionAnchor]));
    engine.setMeasurement(measurement);const next=engine.update(detectionsFromMediaPipe(result),now);
    next.telemetry.recognition_result={measurement,accepted:true};
    journal.add('measurement',{measurement,control:controlObservation(engine,next,measurement,before),stage:name},context);
    await post(next.command??{command:'telemetry',telemetry:next.telemetry});
    if(i%4===0)void journal.flush().catch(()=>{});
    summary.measurements++;summary.modes.push(next.command?.interaction_modes??['PAUSE','PAUSE']);
    summary.reasons.push(next.reason);summary.nonzero_deltas+=Object.values(next.command?.delta_task_m??{}).some(d=>Math.hypot(...d)>0);
    await sleep(Math.max(0,50-(performance.now()-began)));
  }
  stages.push(summary);
}
await stage('enter_and_bind',['PALM','PALM'],60);
if(!engine.tracker.calibrated)throw Error('binding failed');
await post({command:'record_start',telemetry:engine.telemetry(false)});await sleep(150);
await stage('xy_control',['ILY','ILY'],18,{move:true});
await stage('unknown_recentre',['UNKNOWN','UNKNOWN'],12,{move:true});
await stage('resume_xy',['ILY','ILY'],18,{move:true});
await stage('switch_fist_z',['FIST','FIST'],16,{move:true});
await stage('left_unknown_right_valid',['UNKNOWN','FIST'],12);
await stage('leave_frame',['PALM','PALM'],8,{absent:true});
await stage('return_recover',['PALM','PALM'],20);
await post(engine.pause('user_pause').command??{command:'pause',reason:'user_pause'});
async function saveSealed(expectedCount) {
  const frozen=journal.freeze(Date.now());
  // A newer observation must remain in the session and never enter this file.
  const last=engine.measurement;
  journal.add('measurement',{measurement:{...last,measurement_id:`${client}:m${++frame}`,
    capture_epoch_ms:frozen.end_capture_epoch_ms+1},stage:'after_save_boundary'},context);
  const barrier=await journal.flushBoundary(frozen);
  let retries=0;
  for(let attempt=0;attempt<2;attempt++) {
    const request=await post({command:'record_failure',telemetry:engine.telemetry(false),hand_journal_barrier:barrier});
    await journal.flush();
    for(let i=0;i<600;i++) {
      await sleep(100);runtime=await health();
      if(runtime.last_operation_result?.eventId===request.eventId)break;
    }
    if(runtime.last_operation_result?.eventId!==request.eventId)throw Error('save ACK timeout');
    if(runtime.last_operation_result.accepted)break;
    if(!runtime.recording||runtime.saved_failure!==expectedCount-1)throw Error('failed hand export committed/reset an episode');
    retries++;
  }
  if(runtime.recording||runtime.saved_failure!==expectedCount)throw Error('sealed save did not complete');
  stages.push({name:'sealed_save',boundary:barrier,retries});
}
await saveSealed(1);
const firstEpoch=runtime.control_epoch,firstOperation=runtime.last_operation_result;
if(!firstOperation?.accepted||firstEpoch<1)throw Error('save did not reset scene/acknowledge');
engine.resetScene();engine.syncHealth(runtime,performance.now());
await post({command:'record_start',telemetry:engine.telemetry(false)});
await sleep(200);runtime=await health();
if(!runtime.recording)throw Error('next episode did not start after save/reset');
await stage('next_episode_after_save',['PALM','PALM'],12);
await post(engine.pause('user_pause').command??{command:'pause',reason:'user_pause'});
await saveSealed(2);
if(runtime.recording||runtime.saved_failure!==2||runtime.control_epoch!==firstEpoch+1)throw Error('second save/reset failed');
stages.push({name:'save_reset_record_save',first_epoch:firstEpoch,second_epoch:runtime.control_epoch,
  first_research_warning:firstOperation.research_warning??null});
await post({command:'record_stop'});await journal.flush();
writeFileSync(reportPath,JSON.stringify({input_kind:'scripted_test',camera_used:false,session:runtime.session,
  saved_failure:runtime.saved_failure,journal:journal.status(),stages},null,2));
