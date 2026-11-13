// Synthetic camera input through the actual browser protocol/journal helpers.
// No webcam is accessed and these records never claim to be human data.
import {handMeasurement} from '../../collection/common/web/hand-measurement.js';
import {SingleInteractionState,continuousPacket,singleTelemetry,createSingleJournal,logSingleControlResult}
  from '../../collection/push_t/web/single-interaction-state.js';
import {gestureResult} from './gesture_hand_fixture.mjs';

const engine=new SingleInteractionState(),batches=[],packets=[],measurements=[],clientId='single-ui-contract';
let sequence=0,frame=0,now=100000;
const context={input_kind:'scripted_test',synthetic:true,arm_order:['left']};
const journal=createSingleJournal({clientId,send:async batch=>{
  batches.push(structuredClone(batch));
  return {ok:true,last_sequence:batch.rows.at(-1).sequence};
}});
Date.now=()=>now;
function emit(shape='ILY',dx=0){
  const result=shape===null?{landmarks:[]} : gestureResult([shape],{dx});
  const measurement=handMeasurement(result,{id:`ui-contract:${frame}`,session:'ui-contract',epoch:2,
    capturedAt:frame*50,capturedEpochMs:now,inferenceMs:10,receivedAt:frame*50+10,width:640,height:480});
  const next=engine.update(measurement,now+10,.7);
  const telemetry=singleTelemetry(engine,next,{measurement},.7);
  journal.add('measurement',{measurement},context);measurements.push(structuredClone(measurement));
  const input=continuousPacket(next,telemetry);
  if(input){
    const packet={...input,protocol:'task3-command-v2',client_id:clientId,sequence:++sequence,
      eventId:`ui-contract-event:${frame}`,session:'ui-contract',control_epoch:2};
    packets.push({frame,packet:structuredClone(packet)});
    journal.add('dispatch',{event_id:packet.eventId,measurement_id:packet.measurement_id,command:packet.command,packet},context);
    logSingleControlResult(journal,{event_id:packet.eventId,response:{ok:true,queued:true}},context);
  }
  frame++;now+=50;
}
for(let i=0;i<42;i++)emit('PALM');
for(let i=0;i<20;i++)emit('ILY',i*.001);
engine.resetMotion();
const start=await journal.flushBoundary(journal.freeze(now));
const startFrame=frame;
for(let i=0;i<20;i++)emit('ILY',i*.001);
for(let i=0;i<5;i++)emit('UNKNOWN');
for(let i=0;i<15;i++)emit('ILY',i*.001);
for(let i=0;i<20;i++)emit('PALM');
for(let i=0;i<5;i++)emit(null);
for(let i=0;i<20;i++)emit('ILY',i*.001);
const endFrame=frame;
const end=journal.freeze(now-1);
emit('ILY'); // A result completed after the fixed save prefix.
const saved=await journal.flushBoundary(end);
await journal.flush();
process.stdout.write(JSON.stringify({batches,packets,measurements,start,saved,startFrame,endFrame,
  expectedRecordingIds:measurements.slice(startFrame,endFrame).map(m=>m.measurement_id)}));
