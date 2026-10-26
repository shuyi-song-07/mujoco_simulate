"""Full raw sampling, measurement causality and strictly separate derivation."""
import copy
import json
import queue
import time
import numpy as np
from .action_contract import V2, ACTION_NAMES_V2
from .recording_policy import physical_change

DERIVATION_VERSION = "physical-index-v2"
CONTROL_INTERFACE_VERSION = "grasp-hold-release-v1"
HAND_SUPERVISION_VERSION = "direct-action-alignment-v1"
GESTURE_INTERFACE_VERSION = "gesture-grasp-release-v2"
GESTURE_SUPERVISION_VERSION = "direct-pinch-alignment-v2"
RATE_INTERFACE_VERSION = "gesture-rate-gripper-v3"
RATE_SUPERVISION_VERSION = "categorical-rate-no-aperture-v3"
RATE_POLICY_VERSION = "gesture-rate-mode-policy-v3"
EDGE_TRIGGER_POLICY = "palm_fist_edge_v1"
EDGE_POLICY_VERSION = "gesture-edge-mode-policy-v1"
EDGE_RESUME_TRIGGER_POLICY = "palm_fist_edge_v2"
EDGE_POLICY_VERSIONS = {EDGE_TRIGGER_POLICY: EDGE_POLICY_VERSION,
                        EDGE_RESUME_TRIGGER_POLICY: "gesture-edge-mode-policy-v2"}
LEVEL_TRIGGER_POLICY = "palm_fist_level_v2"
LEVEL_POLICY_VERSION = "gesture-level-mode-policy-v2"
LEVEL_RELEASE_VERSION = "independent-palm-rate-opening-v2"
RATE_CALIBRATION = {"phase": "NOT_REQUIRED", "reason": "categorical_rate_control"}
PINCH_MAPPING_VERSION = "anchored-piecewise-v2"
SUPPORTED_PINCH_MAPPINGS = ("anchored-piecewise-v1", PINCH_MAPPING_VERSION)
SUPPORTED_CONTROL_INTERFACES = (CONTROL_INTERFACE_VERSION, GESTURE_INTERFACE_VERSION, RATE_INTERFACE_VERSION)
INTERACTION_MODES = ("XY_HOLD", "Z_HOLD", "Z_APPROACH", "PINCH", "RELEASE", "OPENING", "HOLD")
INTERACTION_MODE_CONTRACT = {"names": ["left.interaction_mode", "right.interaction_mode"],
    "shape": [2], "type": "string", "allowed": list(INTERACTION_MODES), "required_for_policy": True}
RATE_INTERACTION_MODES = ("XY_HOLD", "Z_OPEN", "Z_CLOSE", "HOLD")
RATE_MODE_CONTRACT = {**INTERACTION_MODE_CONTRACT, "allowed": list(RATE_INTERACTION_MODES)}
INTENT_TRAINING_TIMELINE = {"version": "raw-control-intent-v1", "source": "raw_control_frames",
    "control_hz": 60, "image_hz": 30, "preserve_confirmation_time": True,
    "allow_physical_index_only": False}
SIDES = ("left", "right")


def valid_pinch_mapping(mapping, side):
    """An active, finite takeover anchor is required for direct pinch labels."""
    if not isinstance(mapping, dict) or mapping.get('version') not in SUPPORTED_PINCH_MAPPINGS:
        return False
    anchor = mapping.get(side)
    if not isinstance(anchor, dict) or anchor.get('active') is not True:
        return False
    u, width = anchor.get('baseline_u'), anchor.get('baseline_width_m')
    if any(type(v) not in (int, float) or not np.isfinite(v) for v in (u, width)):
        return False
    bounded = 0 <= u <= 1 and 0 <= width <= .08
    # v1 could not represent a nonmatching saturated endpoint. Do not relax
    # historical labels when v2 adds the one-sided, no-jump endpoint mapping.
    return bool(bounded and (mapping['version'] == PINCH_MAPPING_VERSION
                or ((u > 0 or abs(width) < 1e-9) and (u < 1 or abs(width-.08) < 1e-9))))


def control_label_fields(status, masks, operation, observed_id, sources, action_id, wall_ms):
    """Keep requested release, physical execution and hand alignment distinct.

    A valid observed hand is not automatically a direct action teacher while
    the contact controller holds or releases the gripper. Sources are only
    requests already persisted in THIS episode, never pre-recording state.
    """
    contact = status.get("physical_contact") or {}
    protocol = contact.get("control_protocol")
    if protocol not in SUPPORTED_CONTROL_INTERFACES:
        return {}
    gesture = protocol == GESTURE_INTERFACE_VERSION
    rate = protocol == RATE_INTERFACE_VERSION
    action = sources.get(action_id) or {}
    labels = {"version": protocol,
              "release_intent": {}, "release_executed": {}, "release_opening": {}, "hold_width_m": {},
              "intent_source_event_id": {}, "intent_source_measurement_id": {},
              "release_event_id": {}, "intent_supervision_mask": {},
              "intervention_mask": {}, "intervention_reasons": {}}
    if gesture or rate:
        labels.update(interaction_modes={}, interaction_mode_event_id={}, aperture_direct_enabled={}, operation_hold_width_m={}, raw_release_intent={},
                      automatic_opening_mask={})
        if gesture: labels['pinch_mapping'] = copy.deepcopy(action.get('pinch_mapping'))
        if rate and contact.get('gripper_trigger_policy') in EDGE_POLICY_VERSIONS:
            labels['gripper_trigger_policy'] = contact['gripper_trigger_policy']
            labels['gripper_edge'] = copy.deepcopy(contact.get('gripper_edge'))
        elif rate and contact.get('gripper_trigger_policy') == LEVEL_TRIGGER_POLICY:
            labels['gripper_trigger_policy'] = LEVEL_TRIGGER_POLICY
    direct = {}
    for index, side in enumerate(SIDES):
        intent = contact["release_intent"][side]
        held = contact["hold_width_m"][side]
        executed = contact["release_executed"][side]
        opening = contact["release_opening"][side]
        mode = contact.get("interaction_modes", {}).get(side)
        operation_held = contact.get("operation_hold_width_m", {}).get(side)
        source_id = contact["release_intent_event_id"][side]
        source = sources.get(source_id) or {}
        source_age = wall_ms - source.get("capture_epoch_ms", float("-inf"))
        intent_valid = bool(source and not status["paused"] and 0 <= source_age <= 150
                            and source["release_intent"][index] is intent)
        if gesture and opening:
            intent_valid = False  # Automatic continuation is not repeated human intent.
        reasons = []
        if status["paused"]: reasons.append("paused_or_interface")
        if held is not None:
            reasons.append("interrupted_release_hold" if rate and contact.get('rate_interrupted_release_hold', {}).get(side) else "grasp_hold")
        if intent and held is not None: reasons.append("release_confirmation")
        if opening: reasons.append("release_execution")
        if gesture:
            if operation_held is not None: reasons.append("operation_hold")
            if mode != "PINCH": reasons.append("mode_not_pinch")
            if mode == 'PINCH' and contact.get('aperture_direct_enabled', {}).get(side) is not True:
                reasons.append('aperture_not_enabled')
            if intent and "release_confirmation" not in reasons: reasons.append("release_confirmation")
        if rate and mode == 'HOLD': reasons.append('mode_hold')
        edge_state = (contact.get('gripper_edge') or {}).get(side) or {}
        if (rate and contact.get('gripper_trigger_policy') == EDGE_RESUME_TRIGGER_POLICY
                and edge_state.get('state') == 'HOLD'
                and edge_state.get('reason') == 'motion_rejected_wait_resume'):
            reasons.append('motion_rejected_wait_resume')
        requested, projected = action.get("requested_target"), action.get("projected_target")
        if requested is not None and projected is not None:
            offset = 7 * index
            if np.any(np.abs(np.asarray(requested[offset:offset+7])-np.asarray(projected[offset:offset+7])) > 1e-7):
                reasons.append("action_projection")
        labels["release_intent"][side] = intent
        labels["release_executed"][side] = executed
        labels["release_opening"][side] = opening
        labels["hold_width_m"][side] = held
        labels["intent_source_event_id"][side] = source_id if source else None
        labels["intent_source_measurement_id"][side] = source.get("measurement_id")
        labels["release_event_id"][side] = contact["release_event_id"][side]
        labels["intent_supervision_mask"][side] = intent_valid
        labels["intervention_mask"][side] = bool(reasons)
        labels["intervention_reasons"][side] = reasons
        if gesture or rate:
            labels["interaction_modes"][side] = mode
            labels['interaction_mode_event_id'][side] = contact.get('interaction_mode_event_id', {}).get(side)
            labels['aperture_direct_enabled'][side] = contact.get('aperture_direct_enabled', {}).get(side, False)
            labels["operation_hold_width_m"][side] = operation_held
            labels["raw_release_intent"][side] = intent
            labels["automatic_opening_mask"][side] = opening if gesture else False
        direct[side] = bool(not rate and masks[side] and operation and action and observed_id
                            and observed_id == action.get("measurement_id") and not reasons
                            and (not gesture or action.get("interaction_modes", [None,None])[index] == "PINCH"
                                 and contact.get('interaction_mode_event_id', {}).get(side) == action_id
                                 and valid_pinch_mapping(action.get('pinch_mapping'), side)))
    return {"control_labels": labels, "hand_supervision_version": RATE_SUPERVISION_VERSION if rate else GESTURE_SUPERVISION_VERSION if gesture else HAND_SUPERVISION_VERSION,
            "direct_action_alignment_mask": direct,
            "morphology_applicable_mask": dict(direct), "hand_supervision_mask": dict(direct)}


def persisted_control_source(event):
    """Return a compact source only for accepted, versioned public actions."""
    result = event.get("result") or {}
    request = event.get("request") or {}
    if (event.get("kind") != "request_executed" or not result.get("accepted")
            or result.get("control_protocol") not in SUPPORTED_CONTROL_INTERFACES
            or not request.get("eventId")):
        return None
    intent = result.get("release_intent")
    if not isinstance(intent, list) or len(intent) != 2 or any(type(v) is not bool for v in intent):
        return None
    source = {"event_id": request["eventId"], "measurement_id": request.get("measurement_id"),
            "capture_epoch_ms": request.get("sentAt", event["wall_time_ms"]),
            "release_intent": intent, "requested_target": result.get("requested_target"),
            "projected_target": result.get("projected_target")}
    if result.get("control_protocol") in (GESTURE_INTERFACE_VERSION, RATE_INTERFACE_VERSION):
        rate = result['control_protocol'] == RATE_INTERFACE_VERSION
        modes=result.get("interaction_modes")
        allowed = RATE_INTERACTION_MODES if rate else INTERACTION_MODES
        if not isinstance(modes,list) or len(modes)!=2 or any(m not in allowed for m in modes):return None
        if rate and any(intent[i] is not (modes[i] == 'Z_OPEN') for i in range(2)): return None
        source.update(interaction_modes=modes, physical_contact=result.get('physical_contact'))
        if not rate: source['pinch_mapping'] = copy.deepcopy((request.get('telemetry') or {}).get('pinch_mapping'))
    return source


def compact_event(fields):
    """Keep full landmarks once per observation, references in transport rows."""
    result=copy.deepcopy(fields)
    request=result.get('request')
    if isinstance(request,dict):
        request.pop('_admission',None)
        telemetry=request.get('telemetry')
        recognition=telemetry.get('recognition_result') if isinstance(telemetry,dict) else None
        if isinstance(recognition,dict):
            m=recognition.pop('measurement',None)
            if isinstance(m,dict):recognition['measurement_id']=m.get('measurement_id')
    return result


def terminal_snapshot(r, outcome):
    """Freeze first DONE, or the manual save boundary for unfinished attempts."""
    s=getattr(r, 'episode_terminal_snapshot', None)
    if s is None:s=r.controller.snapshot()
    previous=r.last_training_snapshot
    fields=('time','state','action','qpos','qvel','ctrl','body_pos','body_quat')
    np.savez_compressed(r.pending/'terminal_snapshot.npz',**{k:s[k] for k in fields})
    report={'outcome':outcome,'time':s['time'],'last_raw_frame':r.frames-1,
            'last_raw_time':previous['time'] if previous else None,
            'sample_to_terminal_s':s['time']-previous['time'] if previous else None,
            'max_qpos_delta':float(np.max(np.abs(s['qpos']-previous['qpos']))) if previous else None,
            'log_dropped':r.paper_event_dropped,'status':s['status'],
            'episode_end':getattr(r,'episode_end',None)}
    (r.pending/'terminal.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (r.pending/'episode_metadata.json').write_text(json.dumps(r.metadata(outcome),ensure_ascii=False,indent=2),encoding='utf-8')


def paper_metadata(recorder, metadata):
    c = recorder.controller
    result = {**metadata, "schema_version": V2, "action_names": ACTION_NAMES_V2,
        "episode_end":copy.deepcopy(getattr(recorder,'episode_end',None)),
        "action_units": ["m", "m", "m", "rad", "rad", "rad", "m"]*2,
        "input_kind": getattr(recorder.args, "input_kind", "unspecified"), "quality_review": "pending",
        "sampling_policy": {"version": "raw_fixed_phase_v2", "fps": 30, "control_hz":60,
                            "clock":"physics", "stationary_frames_retained":True, "off_phase_event_frames":False},
        "confirmation_gate": False, "hand_schema": "task3-hand-v1", "camera_video_saved":False,
        "operator_view": c.config.get("operator_view"), "training_views":"opaque",
        "aperture_calibration": copy.deepcopy(recorder.recording_calibration),
        "control_epoch": recorder.control_epoch,
        "hand_action_timing_version": "guard-check-v1",
        "model_dimensions": {k:int(getattr(c.model,k)) for k in ("nq","nv","nu","nbody")},
        "joint_mapping": {a.side: {"joint_names":[f"{a.side}_joint{i}" for i in range(1,8)],
            "qpos_indices":a.qpos_indices.tolist(),"finger_qpos_indices":a.finger_qpos_indices.tolist(),
            "actuator_ids":a.actuator_ids.tolist(),"gripper_actuator_id":int(a.gripper_actuator_id)} for a in c.arms},
        "object_qpos_adr":c.object_qpos_adr, "object_dof_adr":c.object_dof_adr,
        "clock_semantics":"browser/worker monotonic clocks remain separate; epoch and health request/response pairs provide clock correspondence; request_executed.hand_guard_check records the server epoch value passed to hand freshness validation, before IK/protection and execution-log emission"}
    contact = getattr(c, "contact_control", None)
    protocol=contact.status().get("control_protocol") if contact is not None else None
    if protocol in SUPPORTED_CONTROL_INTERFACES:
        result.update(control_interface_version=protocol,
            hand_supervision_version=HAND_SUPERVISION_VERSION,
            confirmation_gate={"categorical_gesture": False, "continuous_aperture": False, "release_intent": True},
            release_intent_contract={"names": ["left.release_intent", "right.release_intent"],
                "shape": [2], "type": "bool", "required_for_policy": True,
                "meaning": "explicit request, distinct from physical release execution"},
            policy_compatibility="14D executed actions alone cannot release; M12 must implement the separate 2D intent output",
            intent_training_timeline=copy.deepcopy(INTENT_TRAINING_TIMELINE),
            hand_supervision_semantics="raw measurement validity is unchanged; direct-action/morphology masks exclude hold, release confirmation/execution, pause/interface and projected commands")
        if protocol == GESTURE_INTERFACE_VERSION:
            result.update(hand_supervision_version=GESTURE_SUPERVISION_VERSION,
                policy_contract_version="gesture-mode-intent-policy-v2",
                confirmation_gate={"categorical_gesture":True,"continuous_aperture":False,"release_intent":True},
                interaction_mode_contract=copy.deepcopy(INTERACTION_MODE_CONTRACT),
                pinch_mapping_contract={"version":PINCH_MAPPING_VERSION,
                    "accepted_source_versions":list(SUPPORTED_PINCH_MAPPINGS),
                    "source":"accepted request telemetry.pinch_mapping",
                    "semantics":"u0 maps to current issued width. Interior anchors use two linear segments to zero/full width. At u0=0: w=w0+(0.08-w0)*u; at u0=1: w=w0*u. Saturated anchors preserve width and only map the available input direction; the unavailable endpoint requires another takeover or authorized full-palm opening. v1 sources retain their strict matching endpoint rule; downstream filtering and protection still apply"},
                release_authorization_semantics={"version":"palm-confirmed-auto-opening-v1",
                    "raw_intent":"full-palm request only in RELEASE; false in OPENING is not cancellation",
                    "authorization":"release_event_id references the accepted RELEASE event that authorized opening",
                    "continuation":"OPENING uses fresh accepted inputs and physical protection; no repeated human release intent",
                    "interruption":"pause, lost tracking or cancellation stops opening and requires new confirmation"},
                policy_compatibility="14D actions and the former 2D intent alone are incompatible: policy must provide explicit per-side interaction modes plus raw release intent and use the shared authorization controller",
                hand_supervision_semantics="direct aperture/morphology supervision only for causally aligned trusted PINCH with a valid active recorded takeover anchor, without physical/operation lock, pause, release confirmation/automatic opening or target projection; conditional mapping plus filtering, not raw normalized aperture times 80 mm; holding/moving modes are not direct aperture teachers")
        elif protocol == RATE_INTERFACE_VERSION:
            result.update(hand_supervision_version=RATE_SUPERVISION_VERSION,
                policy_contract_version=RATE_POLICY_VERSION,
                aperture_calibration=copy.deepcopy(RATE_CALIBRATION),
                confirmation_gate={"categorical_gesture":True,"continuous_aperture":False,"release_intent":True},
                interaction_mode_contract=copy.deepcopy(RATE_MODE_CONTRACT),
                release_authorization_semantics={"version":"fresh-palm-rate-opening-v3",
                    "raw_intent":"true exactly in Z_OPEN, including free opening without a grasp",
                    "authorization":"release_event_id exists only for an actual protected grasp release and references an accepted Z_OPEN event",
                    "continuation":"both opening and closing require fresh accepted categorical inputs; no latched automatic opening",
                    "interruption":"mode change, pause or lost/stale input cancels the remaining gripper travel"},
                policy_compatibility="legacy ACT and v1/v2 gesture checkpoints are incompatible; M12 must implement explicit per-side rate modes and the shared protection controller",
                hand_supervision_semantics="all direct-action, morphology and hand-supervision masks are false; raw 21-point hands and causal accepted modes are retained for future study, not direct continuous aperture targets or proof of a learned gesture prior")
            trigger = c.config.get('gesture_control', {}).get('gripper_trigger_policy')
            if trigger == LEVEL_TRIGGER_POLICY:
                result.update(policy_contract_version=LEVEL_POLICY_VERSION,
                    gripper_trigger_policy=trigger,
                    confirmation_gate={"categorical_gesture":True,"continuous_aperture":False,"release_intent":False},
                    gripper_trigger_semantics={
                        "version": trigger,
                        "trigger": "fresh accepted Z_OPEN ramps open; Z_CLOSE ramps closed; no opposite-shape transition required",
                        "categorical_confirm_ms": c.config.get('gesture_control', {}).get('mode_confirm_ms', 100),
                        "pause_recovery": "fresh_confirmed_current_mode",
                        "xy": "XY_HOLD freezes issued width and cancels remaining gripper travel",
                        "independent_sides": True,
                        "source": "causal accepted interaction modes and physical_contact per side",
                        "compatibility": "same rate v3 wire modes; level execution is distinct from historical edge policies"},
                    release_authorization_semantics={
                        "version": LEVEL_RELEASE_VERSION,
                        "raw_intent": "true exactly in Z_OPEN; each side opens independently",
                        "authorization": "fresh accepted palm after categorical confirmation; no additional release timer or opposite-shape transition",
                        "independent_sides": True, "extra_confirmation_seconds": 0,
                        "support_required": False, "stationarity_required": False,
                        "continuation": "fresh matching accepted mode drives bounded servo ramp; Z motion may continue",
                        "interruption": "XY_HOLD, HOLD, pause or stale input stops the ramp and holds issued width"},
                    policy_compatibility=f"{LEVEL_POLICY_VERSION} records independent level modes; do not interpret it as an edge-transition executor")
            elif trigger in EDGE_POLICY_VERSIONS:
                result.update(policy_contract_version=EDGE_POLICY_VERSIONS[trigger],
                    gripper_trigger_policy=trigger,
                    gripper_trigger_semantics={
                        "version": trigger,
                        "trigger": "accepted Z_OPEN to Z_CLOSE closes once; Z_CLOSE to Z_OPEN opens once",
                        "initial_baseline": "known fully open reset state; not fabricated camera evidence",
                        "pause_recovery": "baseline_only_then_opposite",
                        "xy": "hold issued width; cancel remaining stroke; preserve last palm/fist baseline",
                        "source": "physical_contact.gripper_edge per side records authorization event, count and state",
                        "compatibility": "same wire modes as rate v3; different stateful execution, never silently mix training policies"},
                    release_authorization_semantics={
                        "version": "edge-palm-confirmed-opening-v1",
                        "raw_intent": "true exactly in Z_OPEN; not equivalent to an authorized opening edge",
                        "authorization": "opposite palm/fist transition; protected dual grasp requires both palms and 700 ms confirmation; table support and stationarity are not required",
                        "support_required": False, "stationarity_required": False,
                        "continuation": "fresh trusted matching mode; bounded servo ramp; opening may coexist with Z motion or occur in air; real grasp stops further closure",
                        "interruption": "loss/pause/expired stream revokes authorization; first recovered shape is baseline only"},
                    policy_compatibility=f"{EDGE_POLICY_VERSIONS[trigger]} requires its versioned stateful edge executor; legacy ACT, level-mode and other edge-version checkpoints are incompatible")
                if trigger == EDGE_RESUME_TRIGGER_POLICY:
                    result['gripper_trigger_semantics'].update(
                        trusted_hold="freeze issued width; retain unfinished intent while trusted HOLD input remains fresh; matching confirmed mode resumes without another edge",
                        stale_stream="250 ms without trusted fresh input revokes intent and baseline; trusted HOLD or a motion feasibility rejection is not a stale stream",
                        motion_rejection="roll back unaccepted edges and stop motion; retain last accepted baseline and unfinished intent for recoverable motion feasibility rejection; restart release confirmation")
                    result['release_authorization_semantics'].update(
                        version="edge-palm-confirmed-opening-v2",
                        interruption="trusted HOLD freezes travel and restarts incomplete confirmation; resume matching palm with fresh input; real loss/pause/expired stream cancels intent and baseline")
    return result


def capture_paper(recorder, sampler, jsonable, *, diagnostics_only=False):
    r=recorder; c=r.controller
    now=time.monotonic(); wall=time.time()*1000
    # HTTP publishes an observation only after enqueuing its raw event. Read
    # that reference BEFORE draining, so a concurrent receive cannot leave a
    # control row pointing forward to an event not yet written.
    observed=getattr(r,"latest_observation",None)
    if getattr(r, "_paper_source_file", None) is not r.raw_file:
        r._paper_source_file = r.raw_file
        r._paper_control_sources = {}
    while True:
        try:event=r.paper_events.get_nowait()
        except queue.Empty:break
        end=getattr(r,'episode_end',None)
        if (end and event.get('kind')!='episode_capture_end'
                and event.get('wall_time_ms',0)>end['wall_time_ms']):
            continue  # Retained in session diagnostics, outside this episode.
        r.raw_file.write(json.dumps(jsonable(event),ensure_ascii=False)+"\n")
        source = persisted_control_source(event)
        if source: r._paper_control_sources[source["event_id"]] = source
    snapshot=c.snapshot(); status=snapshot["status"]
    # Always retain the raw timeline. Force/save events do not insert an
    # off-phase training frame, and no stationary filter runs here.
    recordable=not diagnostics_only and sampler.sample(snapshot["time"],True,False)
    index=None
    if recordable:
        r.record_commands.put({"type":"frame","snapshot":snapshot},timeout=2)
        index=r.frames;r.frames+=1;r.last_frame_at=snapshot["time"]
        r.last_training_snapshot=snapshot
    measurement=observed["measurement"] if observed else None
    age_ms=(wall-measurement["capture_epoch_ms"]) if measurement else None
    fresh=bool(measurement and measurement.get('source_session')==r.session and measurement.get('source_control_epoch')==r.control_epoch and measurement.get("accepted_recognition") is True and age_ms is not None and 0<=age_ms<=150)
    masks={s: bool(fresh and any(h.get("bound_side")==s and h.get("valid") for h in measurement["hands"])) for s in ("left","right")}
    # Interface/calibration poses remain raw observations, not morphology
    # supervision for manipulation. Missing hands are never zero-filled.
    protocol=(status.get('physical_contact') or {}).get('control_protocol')
    calibration=r.telemetry.get('aperture_calibration') or {}
    operation=not status["paused"] and (calibration==RATE_CALIBRATION if protocol==RATE_INTERFACE_VERSION else calibration.get("phase")=="READY")
    row={"kind":"control_frame","session":r.session,"control_epoch":r.control_epoch,
         "tick":r.tick,"physics_tick":round(snapshot["time"]/c.model.opt.timestep),
         "sim_time":snapshot["time"],"wall_time_ms":wall,"server_monotonic_ns":time.monotonic_ns(),
         "frame_index":index,"state":snapshot["state"],"action":snapshot["action"],"status":status,
         "observation_measurement_id":measurement.get("measurement_id") if measurement else None,
         "measurement_age_ms":age_ms,"hand_measurement_mask":masks,
         "hand_supervision_mask":{s: valid and operation for s,valid in masks.items()},
         "action_measurement_id":getattr(r,"action_measurement_id",None),
         "action_event_id":getattr(r,"action_event_id",None),
         "events":r.pending_events,"sampling_policy":"raw_fixed_phase_v2",
         "log_dropped":r.paper_event_dropped}
    row.update(control_label_fields(status, masks, operation, row["observation_measurement_id"],
                                   r._paper_control_sources, row["action_event_id"], wall))
    r.raw_file.write(json.dumps(jsonable(row),ensure_ascii=False)+"\n");r.raw_file.flush()
    r.pending_events=[];r.force_frame=False
    r._capture_status={"state":"raw_timeline","training_eligible":True,"last_frame_recorded":recordable}


def pause_boundaries(control_rows):
    """Map every observed 60 Hz pause transition to its next raw image frame.

    A pause/resume between two images must still split action chunks. Older
    archives already contain these control rows; no state is interpolated.
    """
    boundaries={};previous=None;pending=False
    for row in control_rows:
        paused=bool(row['status'].get('paused'))
        if previous is not None and paused!=previous:pending=True
        index=row.get('frame_index')
        if index is not None and pending:
            boundaries[index]=['pause_transition'];pending=False
        previous=paused
    return boundaries


def derive_frame_index(snapshots, frame_rows, control_rows=None):
    """Return an index only; keep raw images/times untouched. Shared audit rule."""
    retained=[];previous=None;previous_contacts=None;segment=-1;last_index=-2;last_pause=None
    boundaries=pause_boundaries(control_rows if control_rows is not None else frame_rows)
    pending_cuts=[]
    for i,(snap,row) in enumerate(zip(snapshots,frame_rows)):
        pending_cuts.extend(boundaries.get(i,[]))
        contacts={s:row["status"]["contacts"].get(s) for s in ("left","right")}
        contact_boundary=previous_contacts is not None and contacts!=previous_contacts
        changed=physical_change(snap,previous,"")  # No state-name exemption.
        keep=changed or contact_boundary
        pause=bool(row["status"].get("paused"))
        if keep:
            if i!=last_index+1:pending_cuts.append('filtered_gap' if retained else 'first_retained_frame')
            if pause!=last_pause and retained:pending_cuts.append('pause_transition')
            if pending_cuts:segment+=1
            retained.append({"raw_frame_index":i,"segment_id":segment,"original_time_s":float(snap["time"]),
                             "derived_time_s":len(retained)/30.,"reason":"contact_boundary" if contact_boundary else "physical_change",
                             "cut_reasons":sorted(set(pending_cuts)),
                             "hand_supervision_mask":row["hand_supervision_mask"]})
            for key in ("hand_supervision_version", "control_labels", "direct_action_alignment_mask", "morphology_applicable_mask"):
                if key in row: retained[-1][key] = copy.deepcopy(row[key])
            pending_cuts=[]
            previous=snap;last_index=i;last_pause=pause
        elif previous is None:
            previous=snap
        previous_contacts=contacts
    return retained


def timeline_statistics(control_rows, retained):
    """Integrate observed control intervals; never infer missing human frames.

    Retained/filtered pause duration uses the preceding raw-frame index for
    each control interval. Recovery deltas cover adjacent control states,
    not a claim about human perception or a guarantee of safe motion.
    """
    kept={row['raw_frame_index'] for row in retained}
    result={'pause_duration_s':0.,'retained_pause_s':0.,'filtered_pause_s':0.,
            'pause_reasons_s':{},'recoveries':[],
            'duration_basis':'observed physics intervals; preceding raw frame owns interval'}
    active_frame=None;pause_start=None
    for i,row in enumerate(control_rows):
        if row.get('frame_index') is not None:active_frame=row['frame_index']
        paused=bool(row['status'].get('paused'))
        if paused and pause_start is None:pause_start=row
        if not paused and pause_start is not None and i:
            before=control_rows[i-1]
            a=np.asarray(before['action'],dtype=float).reshape(2,7)
            b=np.asarray(row['action'],dtype=float).reshape(2,7)
            rotations=(b[:,3:6]-a[:,3:6]+np.pi)%(2*np.pi)-np.pi
            actual={}
            for side in ('left','right'):
                previous=before['status'].get('actual_ee',{}).get(side,{})
                current=row['status'].get('actual_ee',{}).get(side,{})
                if 'position' in previous and 'position' in current:
                    actual[side]=float(np.linalg.norm(np.asarray(current['position'])-previous['position']))
            result['recoveries'].append({'sim_time_s':float(row['sim_time']),
                'pause_duration_s':float(row['sim_time']-pause_start['sim_time']),
                'control_interval_s':float(row['sim_time']-before['sim_time']),
                'target_position_delta_m':np.linalg.norm(b[:,:3]-a[:,:3],axis=1).tolist(),
                'target_rpy_delta_rad':rotations.tolist(),
                'target_aperture_delta_m':(b[:,6]-a[:,6]).tolist(),
                'actual_ee_position_delta_m':actual})
            pause_start=None
        if i+1==len(control_rows):continue
        duration=float(control_rows[i+1]['sim_time']-row['sim_time'])
        if duration<0:raise ValueError('control timeline moved backwards')
        if paused:
            reason=row['status'].get('pause_reason') or 'unspecified'
            result['pause_duration_s']+=duration
            result['retained_pause_s' if active_frame in kept else 'filtered_pause_s']+=duration
            result['pause_reasons_s'][reason]=result['pause_reasons_s'].get(reason,0.)+duration
    return result


def action_chunks(index, length):
    """Complete contiguous chunks only. Call separately per episode/session."""
    if not isinstance(length,int) or length<1:raise ValueError('positive chunk length required')
    for start in range(len(index)-length+1):
        rows=index[start:start+length]
        if all((row['segment_id'],row.get('episode_index'),row.get('session'))==
               (rows[0]['segment_id'],rows[0].get('episode_index'),rows[0].get('session')) for row in rows):
            if all(b['raw_frame_index']==a['raw_frame_index']+1 for a,b in zip(rows,rows[1:])):
                yield [row['raw_frame_index'] for row in rows]
