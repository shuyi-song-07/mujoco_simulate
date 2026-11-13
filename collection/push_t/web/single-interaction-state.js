/** One real hand and one real arm. No synthetic right-hand observations. */
import {MeasurementJournal} from './measurement-journal.js';

export function createSingleJournal(options) {
  return new MeasurementJournal({...options,batchSize:4});
}
export function logSingleControlResult(journal,fields,context) {
  journal.add('camera_event',{...fields,event:'control_result'},context);
}
export function operationExplicitlyRejected(error) {
  return error?.status>=400&&error.status<500&&error.response?.queued!==true;
}

export const SINGLE_DEFAULTS = Object.freeze({
  binding_ms:2000, binding_score_min:.60, identity_score_min:.80,
  mode_confirm_ms:200, motion_score_min:.75, close_score_min:.80, open_score_min:.55,
  recovery_ms:200, recovery_samples:5, maximum_age_ms:150,
  wrist_dead_zone:.002, motion_gain:.6, max_step_m:.004,
});

export class SingleInteractionState {
  constructor(parameters={}) { this.parameters={...SINGLE_DEFAULTS,...parameters};this.manualPaused=false;this.preference='auto';this.reset(); }
  reset() {
    this.boundLabel=null;this.bindingLabel=null;this.bindingSince=null;this.bindingWrist=null;this.bindingProgress=0;
    this.palmBindingGuard=true;this.lastCapture=null;this.lastId=null;this.lastWrist=null;
    this.recoverySince=null;this.recoveryCount=0;this.resetMotion();this.reason='binding_required';
  }
  resetMotion() {this.candidate='HOLD';this.candidateSince=null;this.mode='HOLD';this.anchor=null;}
  interrupt(reason) {this.resetMotion();this.recoverySince=null;this.recoveryCount=0;this.lastWrist=null;this.reason=reason;}
  setPaused(value) {this.manualPaused=Boolean(value);this.interrupt(value?'operator_pause':'tracking_recovery');}
  select(measurement) {
    const p=this.parameters;
    let hands=measurement.hands.filter(h=>h.valid&&h.identity_score>=p.identity_score_min);
    const required=this.boundLabel??(this.preference==='auto'?null:this.preference);
    if(required)hands=hands.filter(h=>h.raw_label===required);
    // Before binding, avoid silently picking one of two people/hands.
    if(!this.boundLabel)hands=hands.filter(h=>h.gesture==='Open_Palm'&&h.gesture_score>=p.binding_score_min);
    return hands.length===1?hands[0]:null;
  }
  update(measurement, nowEpochMs, speed=.7) {
    const p=this.parameters,now=measurement.capture_epoch_ms;
    const fresh=measurement.accepted_recognition===true&&Number.isFinite(now)&&nowEpochMs-now>=0&&nowEpochMs-now<=p.maximum_age_ms;
    const ordered=Number.isFinite(now)&&(this.lastCapture===null||now>this.lastCapture)&&measurement.measurement_id!==this.lastId;
    const gap=this.lastCapture===null||now-this.lastCapture>250;
    const hand=fresh&&ordered?this.select(measurement):null;
    this.lastId=measurement.measurement_id;if(Number.isFinite(now)&&ordered)this.lastCapture=now;
    if(!hand) {
      this.bindingSince=null;this.bindingProgress=0;this.bindingWrist=null;
      this.interrupt(!fresh?'stale_recognition':!ordered?'out_of_order_measurement':this.boundLabel?'hand_lost':'binding_required');
      const candidates=this.boundLabel?measurement.hands.filter(h=>h.raw_label===this.boundLabel):measurement.hands;
      const observed=candidates.length===1?{...candidates[0],valid:false,invalid_reason:candidates[0].invalid_reason??this.reason}:null;
      return this.output(observed,measurement);
    }
    const wrist=hand.image_landmarks_21[0].slice(0,2);
    if(!this.boundLabel) {
      if(gap||this.bindingLabel!==hand.raw_label||!this.bindingWrist||Math.hypot(...wrist.map((v,i)=>v-this.bindingWrist[i]))>.06) {
        this.bindingSince=now;this.bindingLabel=hand.raw_label;this.bindingWrist=[...wrist];
      }
      this.bindingProgress=Math.min(1,(now-this.bindingSince)/p.binding_ms);
      if(this.bindingProgress>=1) {this.boundLabel=hand.raw_label;this.palmBindingGuard=true;this.interrupt('binding_complete');}
      else this.reason='binding_in_progress';
      return this.output(hand,measurement);
    }
    if(gap||this.lastWrist&&Math.hypot(...wrist.map((v,i)=>v-this.lastWrist[i]))>.2)this.interrupt('tracking_recovery');
    this.lastWrist=[...wrist];
    if(this.manualPaused) {this.resetMotion();this.reason='operator_pause';return this.output(hand,measurement);}
    const map={ILoveYou:['XY_HOLD',p.motion_score_min],Closed_Fist:['Z_CLOSE',p.close_score_min],Open_Palm:['Z_OPEN',p.open_score_min]};
    const rule=map[hand.gesture],candidate=rule&&hand.gesture_score>=rule[1]?rule[0]:'HOLD';
    if(this.palmBindingGuard) {
      if(!['XY_HOLD','Z_CLOSE'].includes(candidate)) {this.reason='binding_complete';return this.output(hand,measurement);}
      this.palmBindingGuard=false;this.interrupt('binding_gesture_released');
    }
    this.recoverySince??=now;this.recoveryCount++;
    if(this.recoveryCount<p.recovery_samples||now-this.recoverySince<p.recovery_ms) {
      this.resetMotion();this.reason='tracking_recovery';return this.output(hand,measurement);
    }
    if(candidate==='HOLD') {this.resetMotion();this.reason='unknown_gesture';return this.output(hand,measurement);}
    if(candidate!==this.candidate) {this.resetMotion();this.candidate=candidate;this.candidateSince=now;}
    if(now-this.candidateSince<p.mode_confirm_ms) {this.reason='gesture_mode_confirmation';return this.output(hand,measurement);}
    this.mode=candidate;
    if(!this.anchor) {this.anchor=[...wrist];this.reason='gesture_motion_baseline';return this.output(hand,measurement);}
    const du=wrist[0]-this.anchor[0],dv=wrist[1]-this.anchor[1];
    const x=Math.abs(du)>p.wrist_dead_zone?du:0,y=Math.abs(dv)>p.wrist_dead_zone?dv:0;
    const gain=p.motion_gain*Math.max(.1,Math.min(1,Number(speed)||.7));
    // Same task-axis convention as the dual-arm UI: image up -> -X,
    // mirrored display right -> +Y; image up -> +Z in height mode.
    const delta=candidate==='XY_HOLD'?[y*gain,-x*gain,0]:[0,0,-y*gain];
    const scale=Math.min(1,p.max_step_m/Math.max(1e-9,Math.hypot(...delta)));
    if(candidate==='XY_HOLD'&&x)this.anchor[0]=wrist[0];if(y)this.anchor[1]=wrist[1];
    this.reason=candidate==='XY_HOLD'?'xy_motion':'z_motion';
    return this.output(hand,measurement,delta.map(v=>v===0?0:v*scale));
  }
  output(hand,measurement,delta=[0,0,0]) {
    const selected=hand?{...hand,bound_side:'left'}:null;
    // Preserve true handedness; left is the robot channel, not the human hand.
    measurement.hands=selected?[selected]:[];
    measurement.control_hand_raw_label=this.boundLabel;
    const active=this.boundLabel&&!this.manualPaused&&!this.palmBindingGuard&&selected?.valid;
    return {measurement,hand:selected,reason:this.reason,mode:active?this.mode:'HOLD',delta:active?delta:[0,0,0],
      calibrated:Boolean(this.boundLabel),bindingProgress:this.bindingProgress,
      bindingLabel:this.boundLabel??this.bindingLabel,palmBindingGuard:this.palmBindingGuard,
      canSendControl:Boolean(active)};
  }
}

export function continuousPacket(next,telemetry) {
  if(!next.canSendControl)return null;
  return {command:'continuous_input',control_protocol:'gesture-rate-gripper-v3',mode:next.mode==='XY_HOLD'?'XY':'Z',
    interaction_modes:[next.mode],delta_task_m:{left:[...next.delta]},widths_m:[0],release_intent:[next.mode==='Z_OPEN'],
    measurement_id:next.measurement.measurement_id,sentAt:next.measurement.capture_epoch_ms,telemetry};
}

/** The controller consumes visible; valid is retained for research diagnostics. */
export function singleHandTelemetry(next) {
  const valid=Boolean(next?.hand?.valid);
  return {left:{gesture:next?.hand?.gesture??'None',confidence:next?.hand?.gesture_score??0,visible:valid,valid}};
}

export function singleTelemetry(engine,next,diagnostic,speed=.7) {
  return {calibrated:Boolean(engine.boundLabel),
    calibration:{phase:engine.boundLabel?'READY':'BINDING',palm_binding_guard:engine.palmBindingGuard,bound_raw_label:engine.boundLabel},
    aperture_calibration:{phase:'NOT_REQUIRED',reason:'categorical_rate_control'},
    control_mode:next?.mode==='XY_HOLD'?'XY':next?.mode==='HOLD'?'PAUSE':'Z',
    mode:next?.mode??'HOLD',pause_reason:engine.reason,tracking_valid:Boolean(next?.hand?.valid),
    hands:singleHandTelemetry(next),operator_control:{motion_mapping:'incremental',motion_speed_scale:Number(speed)},
    recognition_result:diagnostic,recordable:Boolean(next?.canSendControl)};
}

/** Buttons follow confirmed backend recording state; task success never saves. */
export function recordingButtons(status, connected, pending=false) {
  const terminal=['DONE','FAIL'].includes(status.task_state);
  const idle=connected&&!status.saving&&!pending;
  return {record_start:idle&&!status.recording&&!terminal,
    record_save:idle&&status.recording&&status.task_state==='DONE',
    record_failure:idle&&status.recording,record_discard:idle&&status.recording,
    reset:idle&&!status.recording,record_stop:idle&&!status.recording};
}
