"""Shared raw hand-timeline and snapshot checks used by the formal audit."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from simulation.mujoco.dual_arm.action_contract import V2, ACTION_NAMES_V2, STATE_NAMES
from simulation.mujoco.dual_arm.recording_timeline import (
    CONTROL_INTERFACE_VERSION, HAND_SUPERVISION_VERSION, INTENT_TRAINING_TIMELINE,
    GESTURE_INTERFACE_VERSION, GESTURE_SUPERVISION_VERSION, SUPPORTED_CONTROL_INTERFACES,
    INTERACTION_MODES, INTERACTION_MODE_CONTRACT,
    RATE_INTERFACE_VERSION, RATE_SUPERVISION_VERSION, RATE_POLICY_VERSION,
    RATE_INTERACTION_MODES, RATE_MODE_CONTRACT, RATE_CALIBRATION,
    EDGE_RESUME_TRIGGER_POLICY, EDGE_POLICY_VERSIONS,
    LEVEL_TRIGGER_POLICY, LEVEL_POLICY_VERSION, LEVEL_RELEASE_VERSION,
)
from simulation.mujoco.dual_arm.hand_measurement import inspect_measurement, inspect_release_intent

SIDES = ('left', 'right')


def audit_control_labels(event, frozen, sources, check):
    """Rebuild eligibility from physical status and preceding accepted inputs."""
    contact = event.get('status', {}).get('physical_contact') or {}
    labels = event.get('control_labels')
    protocol=frozen.get('control_interface_version');gesture=protocol==GESTURE_INTERFACE_VERSION
    rate=protocol==RATE_INTERFACE_VERSION
    check(contact.get('control_protocol') == protocol, 'control status protocol missing or changed')
    check(event.get('hand_supervision_version') == (RATE_SUPERVISION_VERSION if rate else GESTURE_SUPERVISION_VERSION if gesture else HAND_SUPERVISION_VERSION), 'hand supervision version missing or changed')
    if not isinstance(labels, dict):
        check(False, 'versioned control labels missing'); return
    check(labels.get('version') == protocol, 'control label version mismatch')
    trigger = frozen.get('gripper_trigger_policy')
    edge = rate and trigger in EDGE_POLICY_VERSIONS
    if rate and trigger == LEVEL_TRIGGER_POLICY:
        check(contact.get('gripper_trigger_policy') == trigger, 'level policy missing from physical status')
        check(labels.get('gripper_trigger_policy') == trigger, 'level policy missing from control labels')
    if edge:
        check(contact.get('gripper_trigger_policy') == trigger, 'edge policy missing from physical status')
        check(labels.get('gripper_trigger_policy') == trigger, 'edge policy missing from control labels')
        check(labels.get('gripper_edge') == contact.get('gripper_edge'), 'edge label differs from execution state')
        states = contact.get('gripper_edge')
        valid_states = {'WAIT_BASELINE', 'READY', 'CLOSING', 'OPENING', 'WAIT_SUPPORT', 'WAIT_BOTH',
                        'CONFIRMING', 'HOLD', 'COMPLETE', 'ABORTED'}
        valid = isinstance(states, dict) and set(states) == set(SIDES)
        if valid:
            for side in SIDES:
                state = states[side]
                valid = valid and isinstance(state, dict) and state.get('baseline_mode') in (None, 'Z_OPEN', 'Z_CLOSE')
                valid = valid and state.get('authorized_mode') in (None, 'Z_OPEN', 'Z_CLOSE')
                valid = valid and type(state.get('edge_count')) is int and state['edge_count'] >= 0
                valid = valid and state.get('state') in valid_states
                if not valid: break
                event_id = state.get('event_id')
                valid = valid and (event_id is None or isinstance(event_id, str) and bool(event_id))
        check(valid, 'invalid edge execution state')
    def pair(value, predicate):
        return isinstance(value, dict) and set(value) == set(SIDES) and all(predicate(v) for v in value.values())
    predicates = {
        'release_intent': lambda v: type(v) is bool,
        'release_executed': lambda v: type(v) is bool,
        'release_opening': lambda v: type(v) is bool,
        'release_armed': lambda v: type(v) is bool,
        'release_progress_s': lambda v: type(v) in (int, float) and np.isfinite(v) and v >= 0,
        'hold_width_m': lambda v: v is None or (type(v) in (int, float) and np.isfinite(v) and 0 <= v <= .080001),
        'release_intent_event_id': lambda v: v is None or isinstance(v, str) and bool(v),
        'release_event_id': lambda v: v is None or isinstance(v, str) and bool(v),
    }
    if gesture or rate:
        predicates.update(interaction_modes=lambda v:v in (RATE_INTERACTION_MODES if rate else INTERACTION_MODES),
            operation_hold_width_m=predicates['hold_width_m'],
            aperture_direct_enabled=lambda v:type(v) is bool,
            interaction_mode_event_id=predicates['release_event_id'])
    if any(not pair(contact.get(k), predicate) for k, predicate in predicates.items()):
        check(False, 'invalid grasp intent physical status'); return
    for name in ('release_intent','release_executed','release_opening','intent_supervision_mask','intervention_mask') + (('raw_release_intent','automatic_opening_mask') if gesture or rate else ()):
        check(pair(labels.get(name), lambda v: type(v) is bool), 'invalid control label boolean map: ' + name)
    for name in ('direct_action_alignment_mask', 'morphology_applicable_mask', 'hand_supervision_mask'):
        check(pair(event.get(name), lambda v: type(v) is bool), 'invalid versioned hand alignment mask')
    action = sources.get(event.get('action_event_id')) or {}
    expected_direct = {}
    expected = {'release_intent': {}, 'release_executed': {}, 'release_opening': {}, 'hold_width_m': {},
                'intent_source_event_id': {}, 'intent_source_measurement_id': {},
                'release_event_id': {}, 'intent_supervision_mask': {},
                'intervention_mask': {}, 'intervention_reasons': {}}
    if gesture or rate:
        expected.update(interaction_modes={},interaction_mode_event_id={},aperture_direct_enabled={},operation_hold_width_m={},raw_release_intent={},automatic_opening_mask={})
        if gesture: expected['pinch_mapping']=action.get('pinch_mapping')
        if rate: check('pinch_mapping' not in labels, 'rate control must not claim a pinch takeover mapping')
    paused = event['status']['paused']
    for i, side in enumerate(SIDES):
        intent, held, released = (contact[k][side] for k in ('release_intent', 'hold_width_m', 'release_executed'))
        opening=contact['release_opening'][side]
        mode=contact.get('interaction_modes',{}).get(side)
        operation_held=contact.get('operation_hold_width_m',{}).get(side)
        check(not opening or released, 'release opening without an executed release')
        if gesture or rate:
            if rate:
                # Mode IDs describe the last accepted action. A rate-lease
                # cancellation can revoke live intent before global pause.
                check(not intent or (not paused and mode=='Z_OPEN'),
                      'live rate opening intent requires unpaused Z_OPEN mode')
                suspended = edge and mode == 'HOLD' and (contact.get('gripper_edge', {}).get(side) or {}).get('state') == 'HOLD' and (contact.get('gripper_edge', {}).get(side) or {}).get('authorized_mode') == 'Z_OPEN'
                check(not opening or mode=='Z_OPEN' or suspended,
                      'rate opening requires continuing Z_OPEN mode or authorized edge suspension')
                check(contact['aperture_direct_enabled'][side] is False and operation_held is None,
                      'rate control cannot enable direct aperture or a permanent operator lock')
            else:
                check(not intent or mode=='RELEASE','raw release intent outside RELEASE mode')
                check(not opening or mode in ('RELEASE','OPENING'),'automatic opening outside authorized mode')
            check(not contact['aperture_direct_enabled'][side]
                  or mode=='PINCH' and not paused and held is None and operation_held is None and not opening,
                  'direct aperture permission conflicts with control protection')
            mode_source=sources.get(contact['interaction_mode_event_id'][side])
            if rate and not paused and mode!='HOLD':
                check(mode_source is not None, 'active rate mode lacks a preceding accepted source')
            if mode_source:
                check(mode_source.get('interaction_modes',[None,None])[i]==mode,
                      'interaction mode differs from its accepted input')
        sid = contact['release_intent_event_id'][side]
        source = sources.get(sid) or {}
        if sid is not None and source:
            check(source['release_intent'][i] is intent, 'physical release intent differs from its accepted request')
        age = event['wall_time_ms'] - source.get('capture_epoch_ms', float('-inf'))
        intent_valid = bool(source and not paused and 0 <= age <= 150 and source['release_intent'][i] is intent)
        if gesture and opening:intent_valid=False
        release_source = sources.get(contact['release_event_id'][side])
        if released:
            check(release_source is not None and release_source['release_intent'][i] is True,
                  'release execution has no preceding true intent source')
            if release_source:
                if gesture or rate:check((release_source.get('interaction_modes') or [None,None])[i]==('Z_OPEN' if rate else 'RELEASE'),
                                 'release authorization lacks the matching source mode')
                execution = release_source.get('physical_contact') or {}
                check((execution.get('release_executed') or {}).get(side) is True
                      and (execution.get('release_event_id') or {}).get(side) == contact['release_event_id'][side],
                      'release execution was not confirmed by its execution event')
        else:
            check(contact['release_event_id'][side] is None, 'unexecuted release carries an execution source')
        reasons = []
        if paused: reasons.append('paused_or_interface')
        if held is not None:
            reasons.append('interrupted_release_hold' if rate and contact.get('rate_interrupted_release_hold', {}).get(side) else 'grasp_hold')
        if intent and held is not None: reasons.append('release_confirmation')
        if opening: reasons.append('release_execution')
        if gesture:
            if operation_held is not None:reasons.append('operation_hold')
            if mode!='PINCH':reasons.append('mode_not_pinch')
            if mode=='PINCH' and contact['aperture_direct_enabled'][side] is not True:reasons.append('aperture_not_enabled')
            if intent and 'release_confirmation' not in reasons:reasons.append('release_confirmation')
        if rate and mode=='HOLD':reasons.append('mode_hold')
        edge_state = (contact.get('gripper_edge') or {}).get(side) or {}
        if (rate and trigger == EDGE_RESUME_TRIGGER_POLICY and edge_state.get('state') == 'HOLD'
                and edge_state.get('reason') == 'motion_rejected_wait_resume'):
            reasons.append('motion_rejected_wait_resume')
        requested, projected = action.get('requested_target'), action.get('projected_target')
        if requested is not None and projected is not None:
            offset = i * 7
            if any(abs(a-b) > 1e-7 for a,b in zip(requested[offset:offset+7], projected[offset:offset+7])):
                reasons.append('action_projection')
        expected['release_intent'][side] = intent
        expected['release_executed'][side] = released
        expected['release_opening'][side] = opening
        expected['hold_width_m'][side] = held
        expected['intent_source_event_id'][side] = sid if source else None
        expected['intent_source_measurement_id'][side] = source.get('measurement_id')
        expected['release_event_id'][side] = contact['release_event_id'][side]
        expected['intent_supervision_mask'][side] = intent_valid
        expected['intervention_mask'][side] = bool(reasons)
        expected['intervention_reasons'][side] = reasons
        if gesture or rate:
            expected['interaction_modes'][side]=mode
            expected['interaction_mode_event_id'][side]=contact['interaction_mode_event_id'][side]
            expected['aperture_direct_enabled'][side]=contact['aperture_direct_enabled'][side]
            expected['operation_hold_width_m'][side]=operation_held
            expected['raw_release_intent'][side]=intent
            expected['automatic_opening_mask'][side]=opening if gesture else False
        calibrated = (frozen.get('aperture_calibration') or {}).get('phase') == 'READY'
        mapping = action.get('pinch_mapping') or {}
        anchor = mapping.get(side) or {}
        u, width = anchor.get('baseline_u'), anchor.get('baseline_width_m')
        anchor_valid = (mapping.get('version') in ('anchored-piecewise-v1','anchored-piecewise-v2') and anchor.get('active') is True
                        and all(type(v) in (int,float) and np.isfinite(v) for v in (u,width)))
        if anchor_valid:
            anchor_valid = (0 <= u <= 1 and 0 <= width <= .08
                            and (mapping['version']=='anchored-piecewise-v2'
                                 or ((u > 0 or abs(width) < 1e-9) and (u < 1 or abs(width-.08) < 1e-9))))
        if gesture and mapping.get('version')=='anchored-piecewise-v2':
            check((frozen.get('pinch_mapping_contract') or {}).get('version')=='anchored-piecewise-v2',
                  'v2 pinch mapping requires a v2 frozen contract')
        expected_direct[side] = bool(not rate and event.get('hand_measurement_mask', {}).get(side) and not paused and calibrated
                                    and action and event.get('observation_measurement_id')
                                    and event['observation_measurement_id'] == action.get('measurement_id') and not reasons
                                    and (not gesture or action.get('interaction_modes',[None,None])[i]=='PINCH'
                                         and contact['interaction_mode_event_id'][side]==event.get('action_event_id') and anchor_valid))
    for key, value in expected.items():
        check(labels.get(key) == value, 'control label mismatch: ' + key)
    for key in ('direct_action_alignment_mask', 'morphology_applicable_mask', 'hand_supervision_mask'):
        check(event.get(key) == expected_direct, 'control alignment mismatch: ' + key)


def read_snapshots(directory, *, allow_partial_last=False):
    blocks=[];cursor=0
    paths=sorted(Path(directory).glob('snapshots_*.npz'))
    for path in paths:
        if int(path.stem.split('_')[-1])!=cursor:raise ValueError('snapshot block gap')
        try:
            with np.load(path,allow_pickle=False) as data:
                block={key:data[key] for key in data.files}
        except Exception:
            if allow_partial_last and path==paths[-1] and blocks:break
            raise
        n=len(block['time'])
        if not n or any(len(v)!=n or not np.isfinite(v).all() for v in block.values()):raise ValueError('invalid snapshot values')
        blocks.append(block);cursor+=n
    if not blocks:raise ValueError('no persisted snapshots')
    return {k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}


def audit_hand_timeline(research, frozen):
    """Check provenance independently of the recorder's validity labels."""
    errors=[];measurements={};executed={};received={};coverage=0;epochs=set();control_sources={}
    protocol=frozen.get('control_interface_version')
    versioned = protocol in SUPPORTED_CONTROL_INTERFACES
    gesture = protocol == GESTURE_INTERFACE_VERSION
    rate = protocol == RATE_INTERFACE_VERSION
    timing={'version':'guard-time-or-receipt-execution-bounds-v1','exact_checks':0,
            'bounded_checks':0,'uncertain_checks':0,'uncertain_events':[]}
    def check(ok,message):
        if not ok and message not in errors:errors.append(message)
    def mask_valid(mask):
        return isinstance(mask,dict) and set(mask)=={'left','right'} and all(type(v) is bool for v in mask.values())
    configured=(frozen.get('config') or {}).get('contact_control',{}).get('control_protocol')
    if configured in SUPPORTED_CONTROL_INTERFACES:
        check(protocol==configured, 'configured control protocol requires versioned intent metadata')
    if versioned:
        check(frozen.get('hand_supervision_version') == (RATE_SUPERVISION_VERSION if rate else GESTURE_SUPERVISION_VERSION if gesture else HAND_SUPERVISION_VERSION), 'frozen hand supervision version mismatch')
        level = rate and (frozen.get('config') or {}).get('gesture_control', {}).get('gripper_trigger_policy') == LEVEL_TRIGGER_POLICY
        check(frozen.get('confirmation_gate') == {'categorical_gesture':gesture or rate,'continuous_aperture':False,'release_intent':not level},
              'frozen confirmation gates mismatch')
        if gesture:
            check(frozen.get('interaction_mode_contract')==INTERACTION_MODE_CONTRACT,
                  'separate interaction mode policy contract missing or changed')
            authorization=frozen.get('release_authorization_semantics') or {}
            check(authorization.get('version')=='palm-confirmed-auto-opening-v1',
                  'automatic opening authorization semantics missing')
            check(frozen.get('policy_contract_version')=='gesture-mode-intent-policy-v2',
                  'gesture policy incompatibility marker missing')
            check((frozen.get('pinch_mapping_contract') or {}).get('version') in ('anchored-piecewise-v1','anchored-piecewise-v2'),
                  'pinch takeover mapping contract missing')
        if rate:
            check(frozen.get('interaction_mode_contract')==RATE_MODE_CONTRACT,
                  'rate interaction mode policy contract missing or changed')
            trigger = (frozen.get('config') or {}).get('gesture_control', {}).get('gripper_trigger_policy', 'level_v1')
            edge = trigger in EDGE_POLICY_VERSIONS
            check(trigger in ('level_v1', LEVEL_TRIGGER_POLICY) or edge, 'unknown gripper trigger policy')
            check(frozen.get('policy_contract_version')==(LEVEL_POLICY_VERSION if level else EDGE_POLICY_VERSIONS.get(trigger, RATE_POLICY_VERSION)),
                  'rate policy incompatibility marker missing')
            if edge:
                check(frozen.get('gripper_trigger_policy') == trigger, 'frozen edge trigger policy missing')
                semantics = frozen.get('gripper_trigger_semantics') or {}
                check(semantics.get('version') == trigger and
                      semantics.get('pause_recovery') == 'baseline_only_then_opposite', 'edge trigger semantics missing')
                if trigger == EDGE_RESUME_TRIGGER_POLICY:
                    check(all(isinstance(semantics.get(k), str) and semantics[k] for k in
                              ('trusted_hold', 'stale_stream', 'motion_rejection')),
                          'resumable edge safety semantics missing')
            elif level:
                check(frozen.get('gripper_trigger_policy') == trigger, 'frozen level trigger policy missing')
                semantics = frozen.get('gripper_trigger_semantics') or {}
                parameters = (frozen.get('config') or {}).get('gesture_control', {})
                check(semantics.get('version') == trigger and semantics.get('independent_sides') is True
                      and semantics.get('pause_recovery') == 'fresh_confirmed_current_mode'
                      and semantics.get('categorical_confirm_ms') == parameters.get('mode_confirm_ms', 100),
                      'level trigger semantics missing')
                authorization = frozen.get('release_authorization_semantics') or {}
                check(authorization.get('independent_sides') is True
                      and authorization.get('extra_confirmation_seconds') == 0
                      and authorization.get('support_required') is False
                      and authorization.get('stationarity_required') is False,
                      'independent level release semantics missing')
                check((frozen.get('config') or {}).get('contact_control', {}).get('release_confirm_seconds') == 0,
                      'independent level release must not add a confirmation timer')
            else:
                check(frozen.get('gripper_trigger_policy') in (None, 'level_v1'), 'level configuration cannot claim edge policy')
            check(frozen.get('aperture_calibration')==RATE_CALIBRATION,
                  'rate control must declare aperture calibration not required')
            check((frozen.get('release_authorization_semantics') or {}).get('version')==
                  (LEVEL_RELEASE_VERSION if level else
                   'edge-palm-confirmed-opening-v2' if trigger == EDGE_RESUME_TRIGGER_POLICY else
                   'edge-palm-confirmed-opening-v1' if edge else 'fresh-palm-rate-opening-v3'),
                  'fresh categorical opening semantics missing')
            check('pinch_mapping_contract' not in frozen,'rate metadata cannot declare pinch takeover')
        contract = frozen.get('release_intent_contract') or {}
        check(contract.get('names') == ['left.release_intent','right.release_intent'] and contract.get('shape') == [2]
              and contract.get('type') == 'bool' and contract.get('required_for_policy') is True,
              'separate release intent policy contract missing')
        timeline=frozen.get('intent_training_timeline')
        check(timeline==INTENT_TRAINING_TIMELINE and isinstance(timeline,dict)
              and timeline.get('preserve_confirmation_time') is True and timeline.get('allow_physical_index_only') is False,
              'release intent training must retain the raw confirmation timeline')
    elif frozen.get('control_interface_version') is not None:
        check(False, 'unsupported control interface version')
    for event in research:
        kind=event.get('kind')
        if kind=='request_received' and isinstance(event.get('request'),dict):
            received.setdefault(event['request'].get('eventId'),event)
        if kind=='hand_measurement' and event.get('schema_valid'):
            m=event['measurement'];epoch=event.get('control_epoch')
            problem=inspect_measurement(m,session=frozen['session'],epoch=epoch)
            check(event.get('session')==frozen['session'] and problem is None,'invalid trusted measurement context or contents')
            check('control_epoch' not in frozen or epoch==frozen['control_epoch'],'measurement epoch differs from frozen episode')
            if problem is not None:continue
            epochs.add(epoch)
            check(len(epochs)==1,'measurement epoch changed inside episode')
            mid=m['measurement_id']
            check(mid not in measurements or measurements[mid]==m,'measurement ID reused with changed contents')
            measurements[mid]=m
        if kind=='request_executed' and (event.get('result') or {}).get('accepted'):
            request=event['request']
            result=event['result']
            if versioned and result.get('control_protocol') == protocol:
                intent=result.get('release_intent');aid=request.get('eventId')
                valid_intent = isinstance(intent,list) and len(intent)==2 and all(type(v) is bool for v in intent)
                requested_intent=request.get('release_intent',[False,False])
                valid_request=(isinstance(requested_intent,list) and len(requested_intent)==2
                               and all(type(v) is bool for v in requested_intent))
                check(valid_intent,'accepted action release intent must be two booleans')
                check(valid_intent and valid_request and intent==requested_intent,
                      'accepted release intent differs from request')
                modes=result.get('interaction_modes')
                if gesture or rate:
                    mode_valid=isinstance(modes,list) and len(modes)==2 and all(m in (RATE_INTERACTION_MODES if rate else INTERACTION_MODES) for m in modes)
                    check(mode_valid,'accepted interaction modes must be two supported modes')
                    adapter=result.get('interaction_mode_adapter')
                    if adapter is None:
                        check(mode_valid and request.get('interaction_modes')==modes,
                              'accepted interaction modes differ from request')
                    else:
                        check(not rate and isinstance(adapter,dict) and adapter.get('name')=='trusted_python_gesture_mode_defaults_v1'
                              and adapter.get('source') in ('trusted_python','scripted_test')
                              and adapter.get('network_eligible') is False and adapter.get('interaction_modes')==modes
                              and request.get('command')!='continuous_input' and request.get('interaction_modes') is None,
                              'invalid trusted-only interaction mode adapter')
                    check(not valid_intent or not mode_valid or all(intent[i] is (modes[i]=='Z_OPEN') if rate else not intent[i] or modes[i]=='RELEASE' for i in range(2)),
                          'accepted release intent differs from interaction mode')
                check(isinstance(aid,str) and bool(aid) and aid not in control_sources,
                      'control intent action ID missing or repeated')
                valid_targets=True
                for key in ('requested_target','projected_target'):
                    value=result.get(key)
                    good=isinstance(value,list) and len(value)==14 and all(type(v) in (int,float) and np.isfinite(v) for v in value)
                    check(good,'accepted versioned action has invalid '+key)
                    valid_targets=valid_targets and good
                if valid_intent and valid_targets and isinstance(aid,str) and aid:
                    control_sources[aid]={'measurement_id':request.get('measurement_id'),
                        'release_intent':intent,'capture_epoch_ms':request.get('sentAt',event['wall_time_ms']),
                        'requested_target':result['requested_target'],'projected_target':result['projected_target'],
                        'physical_contact':result.get('physical_contact')}
                    if (gesture or rate) and mode_valid:
                        control_sources[aid]['interaction_modes']=modes
                        if gesture:control_sources[aid]['pinch_mapping']=copy.deepcopy((request.get('telemetry') or {}).get('pinch_mapping'))
            elif versioned and request.get('command') in ('continuous_input','metric_action'):
                check(False,'accepted public action lacks control protocol')
            if request.get('command')=='continuous_input':
                aid=request.get('eventId');mid=request.get('measurement_id');m=measurements.get(mid)
                check(isinstance(aid,str) and bool(aid) and aid not in executed,'accepted action ID missing or executed twice')
                check(m is not None,'accepted action has no preceding raw measurement')
                if m:
                    if versioned:
                        restored=copy.deepcopy(request)
                        restored.setdefault('telemetry',{}).setdefault('recognition_result',{})['measurement']=m
                        if rate:
                            from simulation.mujoco.dual_arm.hand_measurement import inspect_rate_input
                            problem=inspect_rate_input(restored,parameters=(frozen.get('config') or {}).get('gesture_control'))
                            check(problem is None,'accepted rate mode disagrees with recorded hand input: '+str(problem))
                        elif gesture:
                            from simulation.mujoco.dual_arm.hand_measurement import inspect_gesture_input
                            problem=inspect_gesture_input(restored,parameters=(frozen.get('config') or {}).get('gesture_control'))
                            check(problem is None,'accepted interaction mode disagrees with recorded hand geometry: '+str(problem))
                        else:
                            threshold=(frozen.get('config') or {}).get('contact_control',{}).get('release_intent_aperture_normalized',.85)
                            check(inspect_release_intent(restored,threshold=threshold) is None,
                                  'accepted release intent disagrees with recorded hand geometry')
                    telemetry=request.get('telemetry') or {}
                    recognition=telemetry.get('recognition_result') or {}
                    check(recognition.get('measurement_id')==mid,'accepted request measurement references differ')
                    check(request.get('session')==event.get('session')==m['source_session']==frozen['session'] and
                          request.get('control_epoch')==event.get('control_epoch')==m['source_control_epoch'],
                          'accepted action measurement context differs')
                    check(request.get('sentAt')==m['capture_epoch_ms'],'accepted action capture timestamp replaced')
                    guard=event.get('hand_guard_check')
                    if isinstance(guard,dict):
                        checked=guard.get('epoch_ms');mono=guard.get('monotonic_ns')
                        valid_clock=(type(checked) in (int,float) and np.isfinite(checked) and
                                     type(mono) is int and mono>=0)
                        check(valid_clock,'invalid hand freshness check clock')
                        if valid_clock:
                            check(0<=checked-m['capture_epoch_ms']<=150,'accepted action measurement expired at guard check')
                            check(checked<=event['wall_time_ms'],'hand guard check occurs after execution log')
                            if 'server_monotonic_ns' in event:
                                check(mono<=event['server_monotonic_ns'],'hand guard monotonic check occurs after execution log')
                            receipt=received.get(aid)
                            if receipt is not None:
                                check(receipt.get('received_at_ms',receipt['wall_time_ms'])<=checked,
                                      'hand guard check occurs before request receipt')
                        timing['exact_checks']+=1
                    else:
                        check(frozen.get('hand_action_timing_version')!='guard-check-v1',
                              'recording requires exact hand guard check time')
                        # Historical archives have no guard timestamp. A
                        # receipt/execution interval is evidence, not an
                        # invented point estimate of the missing check time.
                        receipt=received.get(aid)
                        check(receipt is not None,'legacy accepted action lacks receipt-time lower bound')
                        if receipt is not None:
                            lower=receipt.get('received_at_ms',receipt['wall_time_ms'])-m['capture_epoch_ms']
                            upper=event['wall_time_ms']-m['capture_epoch_ms']
                            check(lower<=upper and lower<=150 and upper>=0,'legacy accepted action cannot have passed freshness guard')
                            if 0<=lower<=upper<=150:
                                timing['bounded_checks']+=1
                            else:
                                timing['uncertain_checks']+=1
                                timing['uncertain_events'].append({'event_id':aid,'age_lower_ms':lower,
                                    'age_upper_ms':upper,'guard_time_known':False})
                    pair=[h for h in m['hands'] if h.get('bound_side') in ('left','right') and h.get('valid') is True]
                    check(m.get('accepted_recognition') is True and len(pair)==2 and
                          {h['bound_side'] for h in pair}=={'left','right'},'accepted action lacks two trusted recognized hands')
                    calibration=telemetry.get('aperture_calibration')
                    check(telemetry.get('calibrated') is True and isinstance(calibration,dict) and
                          (calibration==RATE_CALIBRATION if rate else calibration.get('phase')=='READY') and calibration==frozen.get('aperture_calibration'),
                          'accepted action calibration differs from frozen episode')
                executed[aid]=mid
        if kind!='control_frame':continue
        if versioned:
            audit_control_labels(event,frozen,control_sources,check)
        aid=event.get('action_event_id');mid=event.get('action_measurement_id')
        check(bool(aid)==bool(mid),'action source IDs incomplete')
        if aid:check(aid in executed and executed[aid]==mid,'action has no preceding accepted continuous event')
        mask=event.get('hand_measurement_mask');supervision=event.get('hand_supervision_mask')
        check(mask_valid(mask) and mask_valid(supervision),'hand masks must contain two boolean sides')
        if not mask_valid(mask) or not mask_valid(supervision):continue
        check(all(not supervision[s] or mask[s] for s in mask),'supervision mask exceeds valid measurement mask')
        check(not event['status']['paused'] or not any(supervision.values()),'paused interface pose is supervision')
        if any(supervision.values()):
            calibration=frozen.get('aperture_calibration')
            check(isinstance(calibration,dict) and calibration.get('phase')=='READY','supervision lacks frozen ready calibration')
        if any(mask.values()):
            m=measurements.get(event.get('observation_measurement_id'))
            check(m is not None,'valid hand mask has no preceding raw observation')
            if m:
                check(m.get('accepted_recognition') is True,'valid hand mask uses rejected recognition')
                check(event.get('session',frozen['session'])==m['source_session'] and
                      event.get('control_epoch',m['source_control_epoch'])==m['source_control_epoch'],
                      'control observation context differs')
                check(abs(event['wall_time_ms']-m['capture_epoch_ms']-event['measurement_age_ms'])<.01 and
                      0<=event['measurement_age_ms']<=150,'measurement age invalid')
                for side,valid in mask.items():
                    matches=[h for h in m['hands'] if h.get('bound_side')==side and h.get('valid') is True]
                    check(not valid or len(matches)==1,'mask claims missing or ambiguous hand valid')
        if event.get('frame_index') is not None:coverage+=all(supervision.values())
    return errors,{'measurements':len(measurements),'supervision_frames':coverage,'hand_freshness_audit':timing,
                  'control_interface_version':frozen.get('control_interface_version'),
                  'intent_source_events':len(control_sources)}
