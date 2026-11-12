import test from 'node:test';
import assert from 'node:assert/strict';
import {handMeasurement} from './hand-measurement.js';
import {MeasurementJournal} from './measurement-journal.js';

const landmarks = () => Array.from({length:21}, (_,i) => ({x:.2+i*.02,y:.4,z:0}));
const meta = {id:'c:1',session:'s',epoch:2,width:640,height:480,
  capturedAt:10,capturedEpochMs:1000,recognizedEpochMs:1008,receivedAt:18,inferenceMs:8};

test('unknown gestures retain original image/world landmarks for later processing', () => {
  const result={landmarks:[landmarks(),landmarks()],worldLandmarks:[landmarks(),landmarks()],
    handedness:[[{categoryName:'Right',score:.8}],[{categoryName:'Left',score:.9}]],
    gestures:[[{categoryName:'None',score:.3}],[{categoryName:'ILoveYou',score:.9}]]};
  const raw=handMeasurement(result,meta);
  assert.equal(raw.hands.length,2);
  assert.equal(raw.hands[0].raw_label,'right');
  assert.equal(raw.hands[0].selected_for_control,true);
  assert.equal(raw.hands[1].selected_for_control,false);
  assert.equal(raw.hands[0].image_landmarks_21.length,21);
  assert.equal(raw.hands[0].hand_center_landmarks_m_21.length,21);
  assert.equal(raw.hands[0].gesture,'None');
  result.landmarks[0][0].x=.99;
  assert.equal(raw.hands[0].image_landmarks_21[0][0],.2);
  assert.match(raw.coordinate_semantics.hand_center,/not camera\/robot\/world/);
});

test('missing and invalid estimates remain identifiable without inventing coordinates', () => {
  assert.deepEqual(handMeasurement({},meta).hands,[]);
  const lm=landmarks();lm[3].x=NaN;
  const hand=handMeasurement({landmarks:[lm]},meta).hands[0];
  assert.equal(hand.image_landmarks_21[3][0],null);
  assert.equal(hand.hand_center_landmarks_m_21,null);
  assert.equal(hand.valid,false);
  assert.equal(hand.invalid_reason,'incomplete_landmarks');
});

test('retry keeps the same rows; a save flushes only its frozen prefix', async () => {
  const calls=[];let fail=true;
  const journal=new MeasurementJournal({clientId:'c',batchSize:1,send:async batch=>{
    calls.push(structuredClone(batch.rows));
    if(fail){fail=false;throw Error('temporary offline');}
    return {ok:true,last_sequence:batch.rows.at(-1).sequence};
  }});
  const raw=handMeasurement({landmarks:[landmarks()]},meta);
  journal.add('measurement',{measurement:raw},{gain:1});
  raw.hands[0].image_landmarks_21[0][0]=.9;
  await assert.rejects(journal.flush(),/temporary offline/);
  const boundary=journal.freeze(1010);
  journal.add('measurement',{measurement:{...raw,measurement_id:'c:2',capture_epoch_ms:1020}},{});
  const ack=await journal.flushBoundary(boundary);
  assert.deepEqual(calls[0],calls[1]);
  assert.equal(calls[1][0].measurement.hands[0].image_landmarks_21[0][0],.2);
  assert.equal(ack.acknowledged_sequence,1);
  assert.equal(journal.status().pending,1);
  assert.equal(ack.pending,0);
  await journal.flush();
  assert.equal(journal.status().written,2);
});

test('overflow is reported and prevents a falsely complete save', async () => {
  const journal=new MeasurementJournal({clientId:'c',capacity:1,send:async()=>({})});
  journal.add('measurement',{measurement:{measurement_id:'1',hands:[]}},{});
  journal.add('measurement',{measurement:{measurement_id:'2',hands:[]}},{});
  assert.equal(journal.status().dropped_measurements,1);
  await assert.rejects(journal.flushBoundary(journal.freeze(1)),/boundary_loss/);
});
