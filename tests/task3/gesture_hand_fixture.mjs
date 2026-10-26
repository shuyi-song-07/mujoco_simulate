// New anatomically shaped fixtures. The historical hand_fixture.mjs is frozen.
import {handMeasurement} from "../../collection/teleoperation/mediapipe/hand-measurement.js";
export function gestureResult(shapes=["PALM","PALM"],{ratios=[1.2,1.2],roll=0,dx=0,dy=0,gestures=null,scores=[.95,.95]}={}) {
  const landmarks=shapes.map((shape,i)=>{
    const cx=(i?.72:.28)+(Array.isArray(dx)?dx[i]:dx),cy=.6+(Array.isArray(dy)?dy[i]:dy);
    const ps=Array.from({length:21},()=>({x:cx,y:cy,z:0}));
    ps[0]={x:cx,y:cy+.07,z:0};
    const chain=(indices,x,extended)=>{
      const pts=extended?[[x,cy],[x,cy-.045],[x,cy-.09],[x,cy-.135]]:[[x,cy],[x,cy-.035],[x+.025,cy-.015],[x+.005,cy+.008]];
      indices.forEach((id,k)=>ps[id]={x:pts[k][0],y:pts[k][1],z:0});
    };
    const flex=shape==="FIST",ily=shape==="ILY";
    chain([5,6,7,8],cx-.045,!flex&&shape!=="PINCH_CLOSED");
    chain([9,10,11,12],cx-.015,!flex&&!ily);
    chain([13,14,15,16],cx+.02,!flex&&!ily);
    chain([17,18,19,20],cx+.055,!flex);
    const tip=shape==="PINCH_CLOSED"?{x:ps[8].x+ratios[i]*.10,y:ps[8].y,z:0}:{x:cx-.045-ratios[i]*.10,y:cy-.135,z:0};
    // Thumb is straight along a diagonal; pinch changes its endpoint without
    // participating in the three idle-finger extension condition.
    const base={x:cx-.03,y:cy+.025,z:0};
    [1,2,3,4].forEach((id,k)=>ps[id]={x:base.x+(tip.x-base.x)*k/3,y:base.y+(tip.y-base.y)*k/3,z:0});
    if(shape==="UNKNOWN")ps[11]={...ps[10]};
    return ps.map(p=>{const x=(p.x-cx)*640,y=(p.y-cy)*480;return{x:cx+(x*Math.cos(roll)-y*Math.sin(roll))/640,y:cy+(x*Math.sin(roll)+y*Math.cos(roll))/480,z:p.z};});
  });
  const labels=gestures??shapes.map(s=>s==="ILY"?"ILoveYou":s==="FIST"?"Closed_Fist":s==="PALM"?"Open_Palm":"None");
  return {landmarks,worldLandmarks:landmarks.map(ps=>ps.map(p=>({x:(p.x-.5)*.2,y:(p.y-.5)*.15,z:0}))),handedness:[[{categoryName:"Left",score:.99}],[{categoryName:"Right",score:.99}]],gestures:labels.map((g,i)=>[{categoryName:g,score:scores[i]}])};
}
export function gestureMeasure(shapes,t=0,opts={}) {return handMeasurement(gestureResult(shapes,opts),{id:`gesture-${t}`,session:"test",epoch:0,capturedAt:t,capturedEpochMs:100000+t,inferenceMs:20,receivedAt:t+25,width:640,height:480});}
