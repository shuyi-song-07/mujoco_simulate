import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {handMeasurement, ApertureCalibration} from "./hand-measurement.js";

import {resultPair, measure} from "../../../tests/task3/hand_fixture.mjs";

test("H03 captured edge collapse remains observable but cannot authorize control", () => {
  const fixture=JSON.parse(readFileSync(new URL("../../../tests/task3/fixtures/hand_edge_20261013.json",import.meta.url),"utf8"));
  const c=new ApertureCalibration();c.phase="READY";c.endpoints={left:fixture.calibration};
  const measured=fixture.frames.map(frame=>{
    const point=([x,y,z])=>({x,y,z});
    return handMeasurement({landmarks:[frame.image_landmarks_21.map(point)],
      worldLandmarks:[frame.hand_center_landmarks_m_21.map(point)],
      handedness:[[{categoryName:"Left",score:frame.identity_score}]]},
      {width:fixture.width,height:fixture.height}).hands[0];
  });
  assert.ok(c.normalize("left",measured[0].aperture_ratio_image)>.4);
  assert.ok(c.normalize("left",measured[1].aperture_ratio_image)<.1);
  for(const [i,h] of measured.entries()) {
    assert.ok(Math.abs(h.aperture_ratio_image-fixture.frames[i].aperture_ratio_image)<1e-12);
    assert.equal(h.near_edge,true);
    assert.equal(h.valid,false,"H03 edge geometry must not remain trusted despite high identity score");
    assert.equal(h.invalid_reason,"hand_near_frame_edge");
    assert.equal(h.image_landmarks_21.length,21);
    assert.equal(h.world_valid,true);
  }
});

test("all four edge bands invalidate control without discarding raw geometry", () => {
  for(const [axis,value] of [["x",.059],["x",.941],["y",.059],["y",.941]]) {
    const r=resultPair();r.landmarks[0][20][axis]=value;
    const h=handMeasurement(r,{width:640,height:480}).hands[0];
    assert.equal(h.valid,false);assert.equal(h.invalid_reason,"hand_near_frame_edge");
    assert.ok(Number.isFinite(h.aperture_ratio_image));
  }
  const r=resultPair();r.landmarks[0][20]={x:.06,y:.94,z:0};
  assert.equal(handMeasurement(r,{width:640,height:480}).hands[0].valid,true);
});

test("complete coordinate sets and raw vs bound identities survive JSON", () => {
  const m=measure([.3,1.5],0,{swapLabels:true});
  assert.equal(m.hands[0].raw_label,"left"); assert.equal(m.hands[0].bound_side,"right");
  for(const h of JSON.parse(JSON.stringify(m)).hands) { assert.equal(h.image_landmarks_21.length,21); assert.equal(h.hand_center_landmarks_m_21.length,21); assert.equal(h.valid,true); }
  assert.match(m.coordinate_semantics.hand_center,/not camera/);
  assert.equal(m.recognition_epoch_ms,null); // Missing timestamps are not fabricated.
});
test("ratio invariant to image translation, scale and in-plane palm rotation", () => {
  for(const roll of [0,.5,1.1]) {
    const m=measure([.3,1.3],0,{roll,dx:.02,dy:-.02});
    assert.ok(Math.abs(m.hands[0].aperture_ratio_image-.3)<1e-10);
    assert.ok(Math.abs(m.hands[1].aperture_ratio_image-1.3)<1e-10);
  }
  const r=resultPair([.3,1.3]); r.landmarks=r.landmarks.map(ps=>ps.map(p=>({x:.5+(p.x-.5)*.6,y:.5+(p.y-.5)*.6,z:p.z})));
  assert.ok(Math.abs(handMeasurement(r,{width:640,height:480}).hands[0].aperture_ratio_image-.3)<1e-10);
});
test("missing world estimates do not become zero skeletons", () => {
  const r=resultPair(); delete r.worldLandmarks;
  const h=handMeasurement(r,{width:640,height:480}).hands[0];
  assert.equal(h.hand_center_landmarks_m_21,null);assert.equal(h.world_valid,false);assert.equal(h.image_valid,true);
});
test("out-of-frame, NaN, identity and stale results are invalid", () => {
  assert.equal(measure([1,1],0,{dx:.5}).hands[1].valid,false);
  assert.equal(measure([1,1],0,{dropReason:"stale_recognition"}).hands[0].valid,false);
  const r=resultPair(); r.landmarks[0][4].x=NaN;r.handedness[1][0].score=.7;
  assert.ok(handMeasurement(r,{width:640,height:480}).hands.every(h=>!h.valid));
});
test("both hands calibrate independently, invalid/repeated samples cannot advance", () => {
  const c=new ApertureCalibration();
  for(let t=0;t<=1600;t+=40)c.update(measure([.15,.25],t),true);
  assert.equal(c.phase,"OPEN");
  for(let t=1640;t<=3240;t+=40)c.update(measure([1.1,1.5],t),true);
  assert.equal(c.phase,"READY");assert.ok(Math.abs(c.normalize("left",.625)-.5)<1e-10);
  assert.ok(Math.abs(c.normalize("right",.875)-.5)<1e-10);
  const bad=new ApertureCalibration();for(let n=0;n<80;n++)bad.update(measure([.1,.1],0),true);
  assert.equal(bad.phase,"CLOSE");
});
test("calibration cannot finish from unbound or one-handed input", () => {
  const c=new ApertureCalibration();
  for(let t=0;t<3000;t+=40) {const m=measure([.1,.1],t);m.hands.pop();c.update(m,true);}
  assert.equal(c.phase,"CLOSE");
  for(let t=3000;t<6000;t+=40)c.update(measure([.1,.1],t),false);
  assert.equal(c.phase,"CLOSE");
});
