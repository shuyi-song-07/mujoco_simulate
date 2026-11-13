import test from 'node:test';
import assert from 'node:assert/strict';
import {MeasurementJournal} from './measurement-journal.js';
const row=i=>({measurement:{measurement_id:`m${i}`,capture_epoch_ms:i,hands:[{gesture:'None',image_landmarks_21:Array.from({length:21},()=>[.2,.3,0])}]}});

test('save seals a fixed prefix while new observations keep arriving',async()=>{
  let release;const gate=new Promise(r=>release=r);
  const journal=new MeasurementJournal({clientId:'x',send:async b=>{await gate;return{ok:true,last_sequence:b.rows.at(-1).sequence};}});
  journal.add('measurement',row(100),{});
  const boundary=journal.freeze(105);
  const saving=journal.flushBoundary(boundary);
  journal.add('measurement',row(110),{});
  release();const sealed=await saving;
  assert.equal(sealed.last_sequence,1);assert.equal(sealed.acknowledged_sequence,1);
  assert.equal(sealed.pending,0);assert.equal(sealed.end_capture_epoch_ms,105);
  assert.equal(sealed.last_acknowledged_capture_ms,100);
  assert.equal(sealed.completed,1);assert.equal(journal.queue.length,1);
});

test('save boundary rejects loss and upload failure, but an identical retry succeeds',async()=>{
  let fail=true;
  const journal=new MeasurementJournal({clientId:'x',send:async b=>{if(fail)throw Error('offline');return{ok:true,last_sequence:b.rows.at(-1).sequence};}});
  journal.add('measurement',row(10),{});const boundary=journal.freeze(20);
  await assert.rejects(journal.flushBoundary(boundary),/offline/);
  fail=false;assert.equal((await journal.flushBoundary(boundary)).pending,0);
  const lost=new MeasurementJournal({clientId:'x',capacity:0,send:async()=>{}});
  lost.add('measurement',row(1),{});
  await assert.rejects(lost.flushBoundary(lost.freeze(2)),/research_boundary_loss/);
});
test('slow research transport retains every completed Unknown observation independently of controls',async()=>{
  let release;const first=new Promise(r=>release=r),received=[];
  const journal=new MeasurementJournal({clientId:'x',send:async b=>{await first;received.push(...b.rows);return{ok:true,last_sequence:b.rows.at(-1).sequence};}});
  journal.add('measurement',row(0),{});const inFlight=journal.flush();
  for(let i=1;i<31;i++)journal.add('measurement',row(i),{});
  assert.equal(journal.completed,31);const saveBarrier=journal.flush();release();await Promise.all([inFlight,saveBarrier]);
  assert.equal(journal.written,31);assert.equal(journal.queue.length,0);
  assert.equal(new Set(received.map(r=>r.measurement.measurement_id)).size,31);
  assert.equal(received[30].measurement.hands[0].image_landmarks_21.length,21);
});
test('lost ACK retries identical rows; bounded overflow is counted and its sequence range reported',async()=>{
  const sent=[];let fail=true;
  const journal=new MeasurementJournal({clientId:'x',capacity:3,send:async b=>{sent.push(b);if(fail)throw Error('offline');return{ok:true,last_sequence:b.rows.at(-1).sequence};}});
  for(let i=0;i<6;i++)journal.add('measurement',row(i),{});
  assert.equal(journal.completed,6);assert.equal(journal.droppedMeasurements,3);
  assert.deepEqual(journal.lossRanges,[[4,6]]);
  await assert.rejects(journal.flush(),/offline/);assert.equal(journal.queue.length,3);
  fail=false;await journal.flush();assert.deepEqual(sent[0].rows,sent[1].rows);
  assert.equal(journal.written+journal.droppedMeasurements,6);
});


test('unknown handedness has an explicit reason without deleting its raw 2D points',async()=>{
  const {handMeasurement}=await import('./hand-measurement.js');
  const {resultPair}=await import('../../../tests/task3/hand_fixture.mjs');
  const result=resultPair();result.handedness[0][0].categoryName='Unknown';
  const m=handMeasurement(result,{id:'identity',session:'s',epoch:0,width:640,height:480});
  assert.equal(m.hands[0].valid,false);assert.equal(m.hands[0].invalid_reason,'unknown_handedness');
  assert.equal(m.hands[0].image_landmarks_21.length,21);
});
