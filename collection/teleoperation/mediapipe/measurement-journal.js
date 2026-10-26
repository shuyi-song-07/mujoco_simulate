/** Research observations never share the replaceable control slot. */
export class MeasurementJournal {
  constructor({send, clientId, capacity=2000, batchSize=4, changed=()=>{}}) {
    Object.assign(this,{send,clientId,capacity,batchSize,changed});
    this.queue=[];this.sequence=0;this.completed=0;this.written=0;this.dropped=0;
    this.droppedMeasurements=0;this.error=null;this.flight=null;this.lossRanges=[];this.acknowledgedSequence=0;this.lastAcknowledgedCaptureMs=null;
  }
  status() {return {completed:this.completed,written:this.written,dropped:this.dropped,
    dropped_measurements:this.droppedMeasurements,pending:this.queue.length,error:this.error,
    last_sequence:this.sequence,acknowledged_sequence:this.acknowledgedSequence,
    last_acknowledged_capture_ms:this.lastAcknowledgedCaptureMs,loss_ranges:this.lossRanges.map(x=>[...x])};}
  add(kind,fields,context) {
    if(kind==='measurement')this.completed++;
    const sequence=++this.sequence;
    if(this.queue.length>=this.capacity) {
      this.dropped++;if(kind==='measurement')this.droppedMeasurements++;
      const last=this.lossRanges.at(-1);
      if(last&&last[1]===sequence-1)last[1]=sequence;else this.lossRanges.push([sequence,sequence]);
      this.error='research_queue_full';this.changed(this.status());return;
    }
    this.queue.push(structuredClone({sequence,kind,browser_epoch_ms:Date.now(),...fields,context}));
    this.changed(this.status());
  }
  freeze(endCaptureMs) {
    const last=this.queue.filter(row=>row.kind==='measurement').at(-1);
    return {version:'hand-save-boundary-v1',client_id:this.clientId,...this.status(),
      end_capture_epoch_ms:endCaptureMs,
      last_capture_epoch_ms:last?.measurement.capture_epoch_ms??this.lastAcknowledgedCaptureMs};
  }
  async flushBoundary(boundary) {
    if(boundary.dropped)throw Error('research_boundary_loss');
    await this.flush(boundary.last_sequence);
    if(this.acknowledgedSequence<boundary.last_sequence)throw Error('research_boundary_unacknowledged');
    // Newer camera results belong to the session, not to this fixed prefix.
    return {...boundary,written:boundary.completed,pending:0,error:null,
      last_acknowledged_capture_ms:boundary.last_capture_epoch_ms,
      acknowledged_sequence:boundary.last_sequence};
  }
  async flush(through=this.sequence) {
    // A save joining an older in-flight batch must also flush measurements
    // completed before this call, not just await the earlier caller's cutoff.
    while(this.queue.length&&this.queue[0].sequence<=through) {
      if(this.flight) {await this.flight;continue;}
      this.flight=this.drain(through);
      try {await this.flight;} finally {this.flight=null;}
    }
  }
  async drain(through) {
    while(this.queue.length&&this.queue[0].sequence<=through) {
      const rows=this.queue.filter(r=>r.sequence<=through).slice(0,this.batchSize),report=this.status();
      try {
        const ack=await this.send({client_id:this.clientId,rows,report});
        if(ack?.ok!==true||ack.last_sequence!==rows.at(-1).sequence)throw Error('research_ack_mismatch');
        this.acknowledgedSequence=rows.at(-1).sequence;
        for(const row of rows)if(row.kind==='measurement')this.lastAcknowledgedCaptureMs=row.measurement.capture_epoch_ms;
        this.queue.splice(0,rows.length);this.written+=rows.filter(r=>r.kind==='measurement').length;
        this.error=null;this.changed(this.status());
      } catch(error) {this.error=String(error.message??error);this.changed(this.status());throw error;}
    }
  }

}

/** Observe engine state without changing its decisions, filters or anchors. */
export function controlObservation(engine,next,measurement,before={}) {
  const telemetry=structuredClone(next.telemetry??{});delete telemetry.recognition_result;
  return {telemetry,tracking_recovery:{pending:Boolean(engine.tracker.recoveryPending),reason:engine.tracker.reason??null,
    candidate:engine.tracker.recoveryCandidate??null},proposed_command:next.command?Object.fromEntries(Object.entries(next.command).filter(([k])=>k!=='telemetry')):null,
    sides:Object.fromEntries(['left','right'].map((side,i)=>{
      const state=engine.interaction?.[side]??{},hand=measurement.hands.find(h=>h.bound_side===side);
      const mode=next.command?.interaction_modes?.[i]??'HOLD';
      const anchor=state.motionAnchor??null,old=before[side]??null;
      return [side,{mode,frontend_control_enabled:next.command?.command==='continuous_input'&&mode!=='HOLD',
        pause_reason:mode==='HOLD'?(next.reason??'hold'):null,
        hand_present:Boolean(hand),hand_valid:hand?.valid??false,
        invalid_reason:hand?.invalid_reason??(hand?null:'not_detected'),
        filtered_wrist_current:measurement.accepted_recognition===true&&engine.tracker.hands[side]?.last_seen_ms===measurement.capture_browser_monotonic_ms,
        motion_anchor:anchor,anchor_event:anchor&&!old?'rebuilt':!anchor&&old?'cleared':null,
        candidate_mode:state.candidate??null,candidate_since_ms:state.candidateSince??null}];
    }))};
}
