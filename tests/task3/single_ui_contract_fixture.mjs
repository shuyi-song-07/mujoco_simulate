// Synthetic camera input through the actual browser protocol/journal helpers.
// No webcam is accessed and these records never claim to be human data.
import {handMeasurement} from '../../collection/common/web/hand-measurement.js';
import {SingleInteractionState,continuousPacket,singleTelemetry,createSingleJournal,logSingleControlResult}
  from '../../collection/push_t/web/single-interaction-state.js';
import {gestureResult} from './gesture_hand_fixture.mjs';
import {readFileSync} from 'node:fs';
import {createContext,runInContext} from 'node:vm';

const engine=new SingleInteractionState(),batches=[],packets=[],measurements=[],clientId='single-ui-contract';
let frame=0,now=100000;
const context={input_kind:'scripted_test',synthetic:true,arm_order:['left']};
const journal=createSingleJournal({clientId,send:async batch=>{
  batches.push(structuredClone(batch));
  return {ok:true,last_sequence:batch.rows.at(-1).sequence};
}});
Date.now=()=>now;
// Execute the application's real pump, including its pre-stamping timestamp
// decision. Rebuilding packets here would miss dispatch-time corruption.
const app=readFileSync(new URL('../../collection/push_t/web/single-arm-app.js',import.meta.url),'utf8');
const transport=createContext({
  engine,clientId,sequence:0,connected:true,shutdown:false,packetFlight:false,
  eventQueue:[],latestInput:null,operation:null,operationError:'',
  runtime:{session:'ui-contract',control_epoch:2},
  Date:{now:()=>now+10},performance:{now:()=>now+10},lastNext:null,lastResultAt:0,
  crypto:{randomUUID:()=>`ui-contract-event:${frame}`},
  journal,context:()=>context,render:()=>{},logSingleControlResult,operationExplicitlyRejected:()=>false,
  jsonRequest:async(path,packet)=>{
    packets.push({frame,packet:structuredClone(packet)});
    return {ok:true,queued:true};
  },
});
runInContext(app.slice(app.indexOf('function stamped(packet)'),app.indexOf("function pause(reason=")),transport);
async function emit(shape='ILY',dx=0,dy=0){
  const result=shape===null?{landmarks:[]} : gestureResult([shape],{dx,dy});
  const measurement=handMeasurement(result,{id:`ui-contract:${frame}`,session:'ui-contract',epoch:2,
    capturedAt:frame*50,capturedEpochMs:now,inferenceMs:10,receivedAt:frame*50+10,width:640,height:480});
  const next=engine.update(measurement,now+10,.7);
  const telemetry=singleTelemetry(engine,next,{measurement},.7);
  journal.add('measurement',{measurement},context);measurements.push(structuredClone(measurement));
  const input=continuousPacket(next,telemetry);
  if(input){
    transport.latestInput=input;
    await runInContext('pump()',transport);
  }
  frame++;now+=50;
}
for(let i=0;i<42;i++)await emit('PALM');
for(let i=0;i<20;i++)await emit('ILY',i*.001);
engine.resetMotion();
const start=await journal.flushBoundary(journal.freeze(now));
const startFrame=frame;
for(let i=0;i<20;i++)await emit('ILY',i*.001);
for(let i=0;i<5;i++)await emit('UNKNOWN');
for(let i=0;i<15;i++)await emit('ILY',i*.001);
for(let i=0;i<20;i++)await emit('PALM');
for(let i=0;i<5;i++)await emit(null);
for(let i=0;i<20;i++)await emit('ILY',i*.001);
const endFrame=frame;
const end=journal.freeze(now-1);
await emit('ILY'); // A result completed after the fixed save prefix.
const saved=await journal.flushBoundary(end);
const directionPackets=[];
for(const [direction,dx,dy,displayDelta] of [
  ['right',-.01,0,[1,0]],['left',.01,0,[-1,0]],
  ['up',0,-.01,[0,1]],['down',0,.01,[0,-1]],
]) {
  engine.resetMotion();
  for(let i=0;i<6;i++)await emit('ILY');
  await emit('ILY',dx,dy);
  directionPackets.push({direction,displayDelta,packet:structuredClone(packets.at(-1).packet)});
}
await journal.flush();
const manualPackets=[];
transport.jsonRequest=async(path,packet)=>{manualPackets.push(structuredClone(packet));return {ok:true,queued:true};};
transport.journal={add:()=>{}};
transport.logSingleControlResult=()=>{};
transport.latestInput={command:'pause',sentAt:now-500};
await runInContext('pump()',transport);
transport.operation={eventId:'manual-record-start'};
transport.latestInput={command:'record_start',eventId:'manual-record-start',sentAt:now-500};
await runInContext('pump()',transport);
process.stdout.write(JSON.stringify({batches,packets,measurements,start,saved,startFrame,endFrame,
  manualPackets,dispatchEpochMs:now+10,directionPackets,
  expectedRecordingIds:measurements.slice(startFrame,endFrame).map(m=>m.measurement_id)}));
