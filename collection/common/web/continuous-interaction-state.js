import {DualInteractionState} from "./dual-interaction-state.js";
import {ApertureCalibration,DEFAULT_GESTURE_PARAMETERS,gestureProfile} from "./hand-measurement.js";
import {RATE_CONTROL_PROTOCOL,updateRate} from "./rate-interaction-state.js";

const sides=["left","right"];
const clamp=(v,lo=0,hi=1)=>Math.max(lo,Math.min(hi,v));
const zeros=()=>Object.fromEntries(sides.map(s=>[s,[0,0,0]]));
export const GESTURE_CONTROL_PROTOCOL="gesture-grasp-release-v2";
export function anchoredWidth(u,u0,w0,maximum=.08) {
  if(![u,u0,w0,maximum].every(Number.isFinite)||u0<0||u0>1||maximum<=0) return null;
  u=clamp(u);w0=clamp(w0,0,maximum);
  if(Math.abs(u-u0)<1e-12)return w0;
  // At a saturated hand endpoint there is no input range on one side of the
  // anchor. Keep the issued width exactly and map only the available direction;
  // inventing the missing range would open/close the jaw during takeover.
  if(u0<=1e-9)return w0+(maximum-w0)*u;
  if(u0>=1-1e-9)return w0*u;
  return clamp(u<=u0?w0*u/u0:w0+(maximum-w0)*(u-u0)/(1-u0),0,maximum);
}
export class ContinuousInteractionState extends DualInteractionState {
  constructor(config={}) {
    super(config); this.continuous=true; this.apertureCalibration=new ApertureCalibration();
    this.measurement=null; this.takeover=true; this.widths=null; this.selection=null; this.selectionSince=null;
    this.manualPaused=false; this.modePreference="auto"; this.lastInputAt=null;
    this.gestureParameters={...DEFAULT_GESTURE_PARAMETERS};this.lastWireModes=["HOLD","HOLD"];
    this.resetGestureModes();
  }
  setMeasurement(m) {this.measurement=m;}
  pause(reason) {
    if(reason==="user_pause")this.manualPaused=true;
    this.takeover=true;this.widths=null;this.lastInputAt=null;
    this.resetGestureModes();
    return super.pause(reason);
  }
  resume() {this.manualPaused=false;this.tracker.requireRecovery("operator_resume");return super.pause("tracking_recovery");}
  unbind(reason="binding_released") {this.apertureCalibration.reset();this.measurement=null;this.takeover=true;return super.unbind(reason);}
  resetInput(reason) {this.takeover=true;this.widths=null;this.selection=null;this.lastInputAt=null;this.resetGestureModes();return super.resetInput(reason);}
  syncHealth(health, now) {
    const previous=this.gestureProtocol();super.syncHealth(health,now);this.gripperLatch=health.gripper_latch??null;
    this.gestureParameters={...DEFAULT_GESTURE_PARAMETERS,...health.action_contract?.gesture_parameters};
    if(previous!==this.gestureProtocol())this.resetGestureModes();
  }
  gestureProtocol() {return this.health?.physical_contact?.control_protocol??this.health?.action_contract?.control_protocol;}
  resetGestureModes() {
    const old=this.interaction??{};
    this.interaction=Object.fromEntries(sides.map(side=>[side,{mode:"HOLD",candidate:null,candidateSince:null,
      wristReference:null,stillSince:null,moving:true,takeover:true,baseline_u:null,baseline_width_m:null,
      desired_width_m:null,current_width_m:null,filtered_width_m:null,motionAnchor:null,lastAt:null,
      opening_blocked:Boolean(old[side]?.opening_blocked||old[side]?.mode==="OPENING"||this.health?.physical_contact?.release_opening?.[side]),
      release_cycle_guard:Boolean(old[side]?.release_cycle_guard),profile:null,still_ms:0} ]));
    this.lastWireModes=["HOLD","HOLD"];
  }
  telemetry(recordable=false, delta=null) {
    const v2=this.gestureProtocol()===GESTURE_CONTROL_PROTOCOL;
    const v3=this.gestureProtocol()===RATE_CONTROL_PROTOCOL;
    return {...super.telemetry(recordable,delta), input_mapping:v3?RATE_CONTROL_PROTOCOL:v2?"gesture-grasp-release-v2":"continuous-aperture-grasp-intent-v2",
      aperture_calibration:v3?{phase:"NOT_REQUIRED",reason:"categorical_rate_control"}:this.apertureCalibration?.status()??null, aperture_takeover:v3?false:this.takeover,
      measurement_id:this.measurement?.measurement_id??null, measurement_valid:Boolean(this.measurement?.hands?.length===2&&this.measurement.hands.every(h=>h.valid)),
      mode_preference:v2||v3?"gesture":this.modePreference??"auto",
      interaction_state:v3?Object.fromEntries(sides.map((side,i)=>{const s=this.interaction?.[side]??{};return[side,{mode:this.lastWireModes?.[i]??"HOLD",candidate_mode:s.candidate,current_width_m:s.current_width_m,desired_width_m:s.desired_width_m}];})):v2?Object.fromEntries(sides.map(side=>{const s=this.interaction?.[side]??{};return[side,{mode:s.mode??"HOLD",candidate_mode:s.candidate,takeover:s.takeover,moving:s.moving,still_ms:s.still_ms??0,
        baseline_u:s.baseline_u,baseline_width_m:s.baseline_width_m,desired_width_m:s.desired_width_m,current_width_m:s.current_width_m,
        idle_fingers_extended:s.profile?.pinch??false,full_palm:s.profile?.full_palm??false,
        opening_blocked:s.opening_blocked,release_cycle_guard:s.release_cycle_guard}];})):null,
      pinch_mapping:v2?{version:"anchored-piecewise-v2",...Object.fromEntries(sides.map((side,i)=>[side,{baseline_u:this.interaction?.[side]?.baseline_u??null,baseline_width_m:this.interaction?.[side]?.baseline_width_m??null,
        active:this.lastWireModes?.[i]==="PINCH"&&this.interaction?.[side]?.takeover===false}]))}:null};
  }
  update(detections, now) {
    if(this.gestureProtocol()===RATE_CONTROL_PROTOCOL)return updateRate(this,detections,now);
    return this.gestureProtocol()===GESTURE_CONTROL_PROTOCOL?this.updateGesture(detections,now):this.updateLegacy(detections,now);
  }
  gesturePacket(modes,delta,widths,intents,reason=null) {
    this.lastWireModes=[...modes];this.mode=modes.every(m=>m==="XY_HOLD")?"XY":"Z";this.reason=reason;
    this.takeover=sides.some(s=>this.interaction[s].mode==="PINCH"&&this.interaction[s].takeover);
    return this.output({command:"continuous_input",control_protocol:GESTURE_CONTROL_PROTOCOL,mode:this.mode,
      interaction_modes:[...modes],delta_task_m:delta,widths_m:[...widths],release_intent:[...intents],measurement_id:this.measurement.measurement_id},modes.some(m=>["PINCH","RELEASE","OPENING"].includes(m))||Object.values(delta).some(d=>Math.hypot(...d)>0),delta);
  }
  holdGesture(reason) {
    const widths=sides.map(s=>this.health.grippers?.[s]?.command_width_m);
    if(widths.some(v=>!Number.isFinite(v)))return this.pause("backend_state_unknown");
    return this.gesturePacket(["HOLD","HOLD"],zeros(),widths,[false,false],reason);
  }
  updateGesture(detections,now) {
    this.lastUpdate=now;
    const m=this.measurement, observed=sides.map(s=>m?.hands?.find(h=>h.bound_side===s));
    const trusted=Boolean(m&&m.capture_browser_monotonic_ms===now&&m.accepted_recognition!==false&&observed.every(h=>h?.valid));
    const tracked=this.tracker.update(trusted?detections:[],now);
    if(!trusted||(!tracked.valid&&tracked.reason!=="tracking_recovery")) {
      if(this.apertureCalibration.phase!=="READY")this.apertureCalibration.clearWindow();
      return this.pause(observed.some(h=>h?.invalid_reason==="hand_near_frame_edge")?"hand_near_frame_edge":tracked.reason);
    }
    if(["FAIL","DONE"].includes(this.health.task_state)||this.health.saving)return this.pause(this.health.failure_reason??(this.health.saving?"saving":"task_done"));
    if(!this.tracker.calibrated)return this.pause(tracked.reason??"calibration_required");
    if(this.apertureCalibration.phase!=="READY") {
      if(!tracked.valid)this.apertureCalibration.clearWindow();else this.apertureCalibration.update(m,true);
      return this.pause(`aperture_${this.apertureCalibration.phase.toLowerCase()}`);
    }
    if(this.manualPaused)return super.pause("user_pause");
    this.tracker.palmBindingGuard=false;
    if(!tracked.valid)return this.holdGesture("tracking_recovery");
    if(this.tracker.controlResumeRequired) {this.tracker.controlResumeRequired=false;this.resetGestureModes();return this.holdGesture("tracking_recovered");}
    const p=this.gestureParameters,contact=this.health.physical_contact??{},widths=sides.map(s=>this.health.grippers?.[s]?.command_width_m);
    if(widths.some(v=>!Number.isFinite(v)))return this.pause("backend_state_unknown");
    const modes=[],delta=zeros(),intents=[false,false];let reason=null;
    for(let i=0;i<2;i++) {
      const side=sides[i],state=this.interaction[side],h=observed[i],profile=gestureProfile(h,m.width,m.height,p);
      state.profile=profile;state.current_width_m=widths[i];state.desired_width_m=null;
      const wrist=[h.image_landmarks_21[0][0],h.image_landmarks_21[0][1]];
      if(!state.wristReference||state.lastAt===null||now-state.lastAt>this.config.hand_lost_grace_ms) {state.wristReference=[...wrist];state.stillSince=now;state.moving=true;}
      const displaced=Math.hypot(wrist[0]-state.wristReference[0],wrist[1]-state.wristReference[1])>p.wrist_motion_threshold;
      if(displaced) {state.wristReference=[...wrist];state.stillSince=now;state.moving=true;state.takeover=true;state.baseline_u=state.baseline_width_m=null;}
      state.still_ms=Math.max(0,now-state.stillSince);
      if(!displaced&&state.still_ms>=p.wrist_still_ms)state.moving=false;
      const dt=state.lastAt===null?0:Math.min(.1,Math.max(0,(now-state.lastAt)/1000));state.lastAt=now;
      const opening=Boolean(contact.release_opening?.[side]);
      const releaseCycle=contact.release_cycle_state?.[side];
      if(!opening&&state.mode==="OPENING")state.release_cycle_guard=releaseCycle==="COMPLETE";
      if(!opening)state.opening_blocked=false;
      // ABORTED retains the backend's operation lock until a new full-palm
      // authorization. Its reentry flag must not prevent earning that window.
      // COMPLETE instead needs an explicit exit before the free PINCH channel.
      if(releaseCycle==="ABORTED")state.release_cycle_guard=false;
      else if(releaseCycle==="COMPLETE"&&contact.release_reentry_pending?.[side])state.release_cycle_guard=true;
      const locked=Number.isFinite(contact.hold_width_m?.[side])||Number.isFinite(contact.operation_hold_width_m?.[side]);
      let candidate="HOLD";
      if(opening) {
        const explicitMove=["ILoveYou","Closed_Fist"].includes(h.gesture)&&h.gesture_score>=p.motion_score_min;
        if(explicitMove||displaced||state.opening_blocked||!profile.geometry.valid) {state.opening_blocked=true;reason="opening_cancelled";}
        else candidate="OPENING";
      } else if(state.release_cycle_guard) {
        if(profile.xy||profile.fist||(profile.pinch&&!profile.geometry.full_palm)) {state.release_cycle_guard=false;state.candidate=null;reason="release_cycle_exit";}
        else reason="release_cycle_hold";
      } else if(profile.xy)candidate="XY_HOLD";
      else if(profile.fist)candidate="Z_HOLD";
      else if(locked) {if(profile.full_palm&&!state.moving)candidate="RELEASE";else reason=profile.full_palm?"release_waiting_for_stop":"operation_lock_hold";}
      else if(profile.pinch&&!(["ILoveYou","Closed_Fist"].includes(h.gesture)&&h.gesture_score>=p.motion_score_min))candidate=state.moving?"Z_APPROACH":"PINCH";
      else reason="gesture_geometry_hold";
      if(candidate!==state.candidate) {
        state.candidate=candidate;state.candidateSince=now;state.mode="HOLD";state.takeover=true;
        state.baseline_u=state.baseline_width_m=null;state.filtered_width_m=null;state.motionAnchor=null;
      }
      if(candidate==="HOLD") {state.mode="HOLD";modes.push("HOLD");continue;}
      if(candidate!=="OPENING"&&now-state.candidateSince<p.mode_confirm_ms) {modes.push("HOLD");reason??="gesture_mode_confirmation";continue;}
      const changed=state.mode!==candidate;state.mode=candidate;
      if(candidate==="OPENING"||candidate==="RELEASE") {modes.push(candidate);intents[i]=candidate==="RELEASE";state.takeover=true;continue;}
      if(candidate==="PINCH") {
        const u=this.apertureCalibration.normalize(side,h.aperture_ratio_image);state.desired_width_m=u*.08;
        if(state.takeover) {
          if(!Number.isFinite(u)||Math.abs(u*.08-widths[i])>p.aperture_takeover_tolerance_m) {modes.push("HOLD");reason??="aperture_takeover";continue;}
          state.takeover=false;state.baseline_u=u;state.baseline_width_m=widths[i];state.filtered_width_m=widths[i];
          state.motionAnchor=[...this.tracker.hands[side].wrist_filtered];modes.push("HOLD");reason??="pinch_baseline";continue;
        }
        const goal=anchoredWidth(u,state.baseline_u,state.baseline_width_m);
        if(!Number.isFinite(goal)) {state.takeover=true;modes.push("HOLD");reason??="aperture_takeover";continue;}
        state.filtered_width_m+=(1-Math.exp(-dt/.06))*(goal-state.filtered_width_m);
        widths[i]=state.filtered_width_m;modes.push("PINCH");continue;
      }
      const filtered=this.tracker.hands[side].wrist_filtered;
      if(changed||!state.motionAnchor) {state.motionAnchor=[...filtered];modes.push(candidate);reason??="gesture_motion_baseline";continue;}
      const du=filtered[0]-state.motionAnchor[0],dv=filtered[1]-state.motionAnchor[1];
      const x=Math.abs(du)>.002?du:0,y=Math.abs(dv)>.002?dv:0,gain=.6*Math.max(.1,Math.min(1,this.config.motion_speed_scale));
      let d=candidate==="XY_HOLD"?[y*gain,-x*gain,0]:[0,0,-y*gain];
      const scale=Math.min(1,.004/Math.max(1e-9,Math.hypot(...d)));delta[side]=d.map(v=>v*scale);
      if(candidate==="XY_HOLD"&&x)state.motionAnchor[0]=filtered[0];if(y)state.motionAnchor[1]=filtered[1];
      modes.push(candidate);
    }
    if(modes.some(v=>v==="XY_HOLD")&&!modes.every(v=>v==="XY_HOLD")) {
      for(const s of sides) {this.interaction[s].takeover=true;this.interaction[s].motionAnchor=null;}
      return this.holdGesture("mixed_motion_planes");
    }
    return this.gesturePacket(modes,delta,widths,intents,reason);
  }
  updateLegacy(detections, now) {
    this.lastUpdate=now;
    const m=this.measurement;
    const observed=sides.map(s=>m?.hands?.find(h=>h.bound_side===s));
    const trusted=Boolean(m&&m.capture_browser_monotonic_ms===now&&observed.every(h=>h?.valid));
    const tracked=this.tracker.update(trusted?detections:[],now);
    if(!tracked.valid) {
      if(this.apertureCalibration.phase!=="READY")this.apertureCalibration.clearWindow();
      const edgeUnsafe=observed.some(h=>h?.valid===false&&h.invalid_reason==="hand_near_frame_edge");
      return this.pause(edgeUnsafe?"hand_near_frame_edge":tracked.reason);
    }
    if(tracked.calibrationConfirmed)return this.pause("binding_complete");
    if(["FAIL","DONE"].includes(this.health.task_state)||this.health.saving)return this.pause(this.health.failure_reason??(this.health.saving?"saving":"task_done"));
    // Calibration measures the hands without actuation, even while manually paused.
    if(this.apertureCalibration.phase!=="READY") {
      this.apertureCalibration.update(m,this.tracker.calibrated);
      return this.pause(`aperture_${this.apertureCalibration.phase.toLowerCase()}`);
    }
    if(this.manualPaused)return super.pause("user_pause");
    this.tracker.palmBindingGuard=false;
    if(this.tracker.controlResumeRequired) {this.tracker.controlResumeRequired=false;return this.pause("tracking_recovered");}
    const angles=observed.map(h=>Math.abs(h.palm_roll_rad)*180/Math.PI);
    const candidate=this.modePreference!=="auto"?this.modePreference:angles.every(a=>a<=35)?"XY":angles.every(a=>a>=55&&a<=125)?"Z":null;
    if(!candidate) {this.selection=null;return this.pause("palm_clutch");}
    if(this.selection!==candidate) {
      this.selection=candidate;this.selectionSince=now;this.clearAnchors();this.takeover=true;this.widths=null;
    }
    if(now-this.selectionSince<200) {
      this.mode="PAUSE";this.reason="continuous_mode_confirmation";
      return this.output({command:"pause",reason:this.reason});
    }
    const desired=observed.map((h,i)=>this.apertureCalibration.normalize(sides[i],h.aperture_ratio_image)*.08);
    if(this.takeover) {
      const held=sides.map(s=>this.health.grippers?.[s]?.command_width_m);
      // Contact-latched jaws do not follow aperture until the backend permits
      // release. Only free jaws need width matching; all wrists re-anchor.
      const contact=this.health.physical_contact;
      const locked=sides.map(s=>contact?.control_protocol==="grasp-hold-release-v1"&&Number.isFinite(contact.hold_width_m?.[s]));
      if(held.some((v,i)=>!Number.isFinite(v)||(!locked[i]&&Math.abs(v-desired[i])>.012)))return this.pause("aperture_takeover");
      this.takeover=false;this.widths=[...held];
      for(const s of sides)this.tracker.hands[s].anchor=[...this.tracker.hands[s].wrist_filtered];
      this.lastInputAt=now;this.mode=candidate;this.reason="continuous_baseline";
      return this.output();
    }
    this.mode=candidate;this.reason=null;
    const dt=Math.min(.1,Math.max(0,(now-this.lastInputAt)/1000));
    this.lastInputAt=now;
    // Separate low-pass aperture from movement/identity enablement.
    const alpha=1-Math.exp(-dt/.06);
    this.widths=this.widths.map((v,i)=>v+alpha*(desired[i]-v));
    const delta={};
    for(const side of sides) {
      const h=this.tracker.hands[side];
      if(!h.anchor) {h.anchor=[...h.wrist_filtered];return this.pause("continuous_baseline");}
      const du=h.wrist_filtered[0]-h.anchor[0],dv=h.wrist_filtered[1]-h.anchor[1];
      const x=Math.abs(du)>.002?du:0,y=Math.abs(dv)>.002?dv:0;
      const gain=.6*Math.max(.1,Math.min(1,this.config.motion_speed_scale));
      let d=candidate==="XY"?[y*gain,-x*gain,0]:[0,0,-y*gain];
      const scale=Math.min(1,.004/Math.max(1e-9,Math.hypot(...d)));d=d.map(v=>v*scale);
      delta[side]=d;
      if(candidate==="XY"&&x)h.anchor[0]=h.wrist_filtered[0];
      if(y)h.anchor[1]=h.wrist_filtered[1];
    }
    const threshold=this.health.action_contract?.release_parameters?.release_intent_aperture_normalized??.85;
    return this.output({command:"continuous_input",mode:candidate,delta_task_m:delta,
      widths_m:[...this.widths],release_intent:desired.map(v=>v/.08>=threshold),
      measurement_id:m.measurement_id},true,delta);
  }
}
