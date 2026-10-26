import test from "node:test";
import assert from "node:assert/strict";
import {gestureProfile,fingerGeometry} from "./hand-measurement.js";
import {gestureMeasure} from "../../../tests/task3/gesture_hand_fixture.mjs";
const profile=(shape,opts={})=>{const m=gestureMeasure([shape,shape],0,opts);return gestureProfile(m.hands[0],m.width,m.height);};
test("projected finger angles survive image-plane rotation without image-y extension rules",()=>{
  for(const roll of [0,.7,Math.PI/2,Math.PI]) {
    const p=profile("PALM",{roll,ratios:[.2,.2]});
    assert.equal(p.geometry.method,"image-pixel-angles-v1");assert.equal(p.geometry.full_palm,true);
    assert.ok(Math.abs(p.geometry.fingers.middle.pip_angle_deg-180)<1e-5);
  }
});
test("idle-finger enable is independent of thumb/index posture and None classification",()=>{
  const open=profile("PINCH",{ratios:[.2,.2]}),closed=profile("PINCH_CLOSED",{ratios:[.2,.2]});
  assert.equal(open.pinch,true);assert.equal(closed.pinch,true);
  assert.equal(closed.geometry.fingers.index.state,"FLEXED");assert.equal(closed.full_palm,false);
});
test("a degenerate idle bone is UNKNOWN rather than permission to continue pinch",()=>{
  const p=profile("UNKNOWN");assert.equal(p.geometry.fingers.middle.state,"UNKNOWN");
  assert.equal(p.geometry.valid,false);assert.equal(p.pinch,false);
});
test("full palm combines a low engineering label gate with geometry, without aperture threshold",()=>{
  assert.equal(profile("PALM",{ratios:[.2,.2],scores:[.6,.6]}).full_palm,true);
  assert.equal(profile("PALM",{scores:[.59,.59]}).full_palm,false);
  assert.equal(profile("PINCH_CLOSED",{gestures:["Open_Palm","Open_Palm"]}).full_palm,false);
});
test("Fist and ILoveYou require matching geometry as well as their motion-label score",()=>{
  assert.equal(profile("FIST").fist,true);assert.equal(profile("ILY").xy,true);
  assert.equal(profile("PALM",{gestures:["Closed_Fist","Closed_Fist"]}).fist,false);
  assert.equal(profile("PALM",{gestures:["ILoveYou","ILoveYou"]}).xy,false);
  assert.equal(profile("ILY",{scores:[.74,.74]}).xy,false);
});
test("complete but short projected bones preserve raw measurements and cannot become geometry permission",()=>{
  const m=gestureMeasure(["PALM","PALM"]),image=structuredClone(m.hands[0].image_landmarks_21);
  image[11]=[image[10][0]+2/640,image[10][1],image[10][2]];
  const g=fingerGeometry(image,640,480);assert.equal(g.fingers.middle.valid,false);assert.equal(g.idle_extended,false);
  assert.equal(m.hands[0].image_landmarks_21.length,21);
});
