/** Category-gated gripper goals. The backend owns rate limiting and grasp safety. */
export const RATE_CONTROL_PROTOCOL="gesture-rate-gripper-v3";
const sides=["left","right"];
const zeros=()=>Object.fromEntries(sides.map(s=>[s,[0,0,0]]));

function packet(engine,modes,delta,widths,reason=null) {
  engine.lastWireModes=[...modes];engine.mode=modes.every(m=>m==="XY_HOLD")?"XY":"Z";
  engine.reason=reason;engine.takeover=false;
  return engine.output({command:"continuous_input",control_protocol:RATE_CONTROL_PROTOCOL,mode:engine.mode,
    interaction_modes:[...modes],delta_task_m:delta,widths_m:[...widths],
    release_intent:modes.map(mode=>mode==="Z_OPEN"),measurement_id:engine.measurement.measurement_id},
    modes.some(mode=>mode==="Z_OPEN"||mode==="Z_CLOSE")||Object.values(delta).some(d=>Math.hypot(...d)>0),delta);
}
function hold(engine,reason) {
  const widths=sides.map(side=>engine.health.grippers?.[side]?.command_width_m);
  if(widths.some(width=>!Number.isFinite(width)))return engine.pause("backend_state_unknown");
  return packet(engine,["HOLD","HOLD"],zeros(),widths,reason);
}
function category(hand,p) {
  const threshold={ILoveYou:p.motion_score_min??.75,Closed_Fist:p.close_score_min??.8,Open_Palm:p.open_score_min??.55}[hand.gesture];
  if(!Number.isFinite(hand.gesture_score)||threshold===undefined||hand.gesture_score<threshold)return "HOLD";
  return {ILoveYou:"XY_HOLD",Closed_Fist:"Z_CLOSE",Open_Palm:"Z_OPEN"}[hand.gesture];
}
export function updateRate(engine,detections,now) {
  engine.lastUpdate=now;
  const m=engine.measurement,observed=sides.map(side=>m?.hands?.find(hand=>hand.bound_side===side));
  const trusted=Boolean(m&&m.capture_browser_monotonic_ms===now&&m.accepted_recognition!==false&&observed.every(hand=>hand?.valid));
  const tracked=engine.tracker.update(trusted?detections:[],now);
  if(!trusted||(!tracked.valid&&tracked.reason!=="tracking_recovery"))
    return engine.pause(observed.some(h=>h?.invalid_reason==="hand_near_frame_edge")?"hand_near_frame_edge":tracked.reason??"untrusted_measurement");
  if(["FAIL","DONE"].includes(engine.health.task_state)||engine.health.saving)
    return engine.pause(engine.health.failure_reason??(engine.health.saving?"saving":"task_done"));
  if(!engine.tracker.calibrated)return engine.pause(tracked.reason??"calibration_required");
  if(engine.manualPaused)return engine.pause("user_pause");
  // Valid HOLD measurements advance backend recovery. Repeated pause packets
  // here would keep restarting its five-sample recovery window indefinitely.
  if(!tracked.valid)return hold(engine,"tracking_recovery");
  if(engine.tracker.controlResumeRequired) {
    engine.tracker.controlResumeRequired=false;engine.resetGestureModes();return hold(engine,"tracking_recovered");
  }
  const p=engine.gestureParameters,candidates=observed.map(hand=>category(hand,p));
  if(engine.tracker.palmBindingGuard) {
    if(!candidates.every(mode=>mode==="XY_HOLD"||mode==="Z_CLOSE"))return hold(engine,"binding_complete");
    engine.tracker.palmBindingGuard=false;engine.resetGestureModes();
    return hold(engine,"binding_gesture_released");
  }
  const widths=sides.map(side=>engine.health.grippers?.[side]?.command_width_m);
  if(widths.some(width=>!Number.isFinite(width)))return engine.pause("backend_state_unknown");
  const modes=[],delta=zeros();let reason=null;
  for(let i=0;i<2;i++) {
    const side=sides[i],state=engine.interaction[side],candidate=candidates[i];
    state.current_width_m=widths[i];state.desired_width_m=candidate==="Z_OPEN"?.08:candidate==="Z_CLOSE"?0:widths[i];
    state.takeover=false;state.lastAt=now;
    if(candidate!==state.candidate) {
      state.candidate=candidate;state.candidateSince=now;state.mode="HOLD";state.motionAnchor=null;
    }
    if(candidate==="HOLD") {modes.push("HOLD");reason??="rate_gesture_unqualified";continue;}
    // The active backend contract supplies this window; an explicit zero
    // disables confirmation without changing historical contracts' default.
    const confirmMs=p.mode_confirm_ms??200;
    if(now-state.candidateSince<confirmMs) {modes.push("HOLD");reason??="gesture_mode_confirmation";continue;}
    const changed=state.mode!==candidate;state.mode=candidate;
    widths[i]=state.desired_width_m;modes.push(candidate);
    const wrist=engine.tracker.hands[side].wrist_filtered;
    if(changed||!state.motionAnchor) {state.motionAnchor=[...wrist];reason??="gesture_motion_baseline";continue;}
    const du=wrist[0]-state.motionAnchor[0],dv=wrist[1]-state.motionAnchor[1];
    const x=Math.abs(du)>.002?du:0,y=Math.abs(dv)>.002?dv:0;
    const gain=.6*Math.max(.1,Math.min(1,engine.config.motion_speed_scale));
    const d=candidate==="XY_HOLD"?[y*gain,-x*gain,0]:[0,0,-y*gain];
    const scale=Math.min(1,.004/Math.max(1e-9,Math.hypot(...d)));delta[side]=d.map(v=>v*scale);
    if(candidate==="XY_HOLD"&&x)state.motionAnchor[0]=wrist[0];
    if(y)state.motionAnchor[1]=wrist[1];
  }
  if(modes.some(mode=>mode==="XY_HOLD")&&!modes.every(mode=>mode==="XY_HOLD")) {
    for(const side of sides)engine.interaction[side].motionAnchor=null;
    return hold(engine,"mixed_motion_planes");
  }
  return packet(engine,modes,delta,widths,reason);
}
