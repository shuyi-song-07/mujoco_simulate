import {handMeasurement} from "../../collection/teleoperation/mediapipe/hand-measurement.js";
export function resultPair(ratios=[1.2, 1.3], {roll=0, dx=0, dy=0}={}) {
  const landmarkSets = ratios.map((ratio, i) => {
    const cx = (i ? .72 : .28)+dx, cy=.55+dy;
    const points = Array.from({length:21}, () => ({x:cx,y:cy-.06,z:0}));
    points[0]={x:cx,y:cy+.1,z:0}; points[9]={x:cx,y:cy-.08,z:0};
    points[5]={x:cx-.05,y:cy,z:0}; points[17]={x:cx+.05,y:cy,z:0};
    points[4]={x:cx-ratio*.05,y:cy-.08,z:0}; points[8]={x:cx+ratio*.05,y:cy-.08,z:0};
    return points.map(p => { const x=(p.x-cx)*640,y=(p.y-cy)*480; return {x:cx+(x*Math.cos(roll)-y*Math.sin(roll))/640,y:cy+(x*Math.sin(roll)+y*Math.cos(roll))/480,z:p.z}; });
  });
  return {landmarks:landmarkSets, worldLandmarks:landmarkSets.map(ps => ps.map(p => ({x:(p.x-.5)*.2,y:(p.y-.5)*.15,z:p.z}))),
    handedness:[[{categoryName:"Left",score:.99}],[{categoryName:"Right",score:.99}]], gestures:[[{categoryName:"Open_Palm",score:.7}],[{categoryName:"Open_Palm",score:.7}]]};
}
export const measure = (ratios, t=0, options={}) => handMeasurement(resultPair(ratios,options), {id:`frame-${t}`,session:"test",epoch:0,capturedAt:t,capturedEpochMs:100000+t,inferenceMs:20,receivedAt:t+25,width:640,height:480,...options});

