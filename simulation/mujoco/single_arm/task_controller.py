"""One physical Panda, source Push-T geometry, metric S9/A7 recording interface."""
from __future__ import annotations
import copy
import math
import time
import numpy as np
import mujoco
from simulation.mujoco.dual_arm.panda_arm_controller import PandaArmController, matrix_to_rpy, rpy_to_matrix
from simulation.mujoco.dual_arm.task_states import TaskState, ControlMode, GripperLatch
from simulation.mujoco.dual_arm.config import config_hash
from simulation.mujoco.dual_arm.task_metrics import DualArmMetrics
from tasks.push_t.scene import create_model, initial_state
from tasks.push_t.success import PushTSuccess

SCHEMA='single-panda-push-v1'
STATE_NAMES=[f'left_joint{i}' for i in range(1,8)]+['left_gripper_yaw_rad','left_finger_width']
ACTION_NAMES=[f'left_{name}' for name in ('x','y','z','roll','pitch','yaw','gripper_width_m')]


class PushArm(PandaArmController):
    def __init__(self, model, data, config):
        super().__init__(model,data,'left',config)
        self.yaw_joint_id=model.joint('left_gripper_yaw_joint').id
        self.yaw_qpos_index=int(model.jnt_qposadr[self.yaw_joint_id])
        self.yaw_dof_index=int(model.jnt_dofadr[self.yaw_joint_id])
        self.yaw_actuator_id=model.actuator('left_gripper_yaw_actuator').id

    def reset(self, home_qpos=None, gripper='closed'):
        # Use the source joint home exactly, rather than solving a different pose.
        q=np.asarray(self.config['scene']['home_qpos'] if home_qpos is None else home_qpos,dtype=float)
        self.data.qpos[self.qpos_indices]=q;self.data.ctrl[self.actuator_ids]=q
        self.data.qpos[self.yaw_qpos_index]=0.;self.data.ctrl[self.yaw_actuator_id]=0.
        self.data.qpos[self.finger_qpos_indices]=0.
        self.gripper_requested_width_m=self.gripper_goal_width_m=0.;self._write_gripper_width(0.)
        mujoco.mj_forward(self.model,self.data)
        self.command_position,self.command_rotation=self.get_actual_pose()
        self.fixed_rotation=self.command_rotation.copy()
        self._ik_tick_time=None;self._ik_tick_origin=None;self._gripper_tick=None
        self.workspace_clamped=False;self.workspace_violation_count=0

    def state_vector(self):
        return np.r_[self.data.qpos[self.qpos_indices],self.data.qpos[self.yaw_qpos_index],
                     np.sum(self.data.qpos[self.finger_qpos_indices])].astype(np.float32)

    def freeze(self):
        # Retain issued targets. A pause must not invent a Cartesian or yaw jump.
        self.hold_gripper()


class PushContact:
    """Common recorder protocol facade. Push-T never authorizes finger opening."""
    rate_control=True
    edge_control=False
    protocol='gesture-rate-gripper-v3'

    def __init__(self,controller):
        self.controller=controller
        self.limits=controller.config.get('contact_control',{})
        self.release_clock='simulation';self.clock=lambda:float(controller.data.time)

    def set_input_clock(self,clock,name):
        self.clock=clock;self.release_clock=name

    def cancel_release(self,reason):
        # The shared input mailbox also uses this simulation-thread hook when
        # an admitted pause is superseded or an input lease is rejected. There
        # is no release lease, but the motion interruption must still survive.
        return self.controller.pause(reason)

    def status(self):
        c=self.controller;mode=c.interaction_mode
        output={'state':'FREE','control_protocol':self.protocol,'gripper_trigger_policy':'push_locked_v1',
            'policy_contract_version':'single-push-rate-v1','ever_dual':False,'released':False,
            'support_ready':c.support_force_n>=.1*.1*9.81,'support_force_n':c.support_force_n,
            'support_surface':'table','support_height_gap_m':float(c.data.xpos[c.object_id,2]-.015-.4),
            'contact_forces_n':{'left':list(c.finger_forces)},'grasp_stable':{'left':False},'stable_grasps':{'left':False},
            'interaction_modes':{'left':mode},'interaction_mode_event_id':{'left':c.mode_event_id},
            'fixed_gripper':True,'fixed_gripper_reason':'push_task_closed_gripper',
            'gripper_execution':{'left':{'pending':False,'state':'LOCKED','reason':'push_task_closed_gripper'}},
            'gripper_authorization':{'left':{'active':False}},'gripper_baseline':{'left':None}}
        for key in ('release_intent','release_executed','release_opening','release_armed','release_reentry_pending',
                    'rate_interrupted_release_hold','aperture_direct_enabled','gripper_rate_enabled'):
            output[key]={'left':False}
        for key in ('release_intent_event_id','release_event_id'):
            output[key]={'left':None}
        output['hold_width_m']={'left':0.};output['operation_hold_width_m']={'left':0.}
        output['gripper_goal_width_m']={'left':0.};output['gripper_command_width_m']={'left':0.}
        output['release_cycle_state']={'left':'IDLE'};output['release_progress_s']={'left':0.}
        output['gripper_rate_direction']={'left':0.};output['release_reason']={'left':'push_task_closed_gripper'}
        return output


class SingleArmTaskController:
    arm_order=('left',)

    def __init__(self,config,model=None,data=None):
        self.config=copy.deepcopy(config);self.config_hash=config_hash(self.config)
        self.model=create_model(self.config) if model is None else model
        self.data=mujoco.MjData(self.model) if data is None else data
        self.left=PushArm(self.model,self.data,self.config);self.arms=(self.left,)
        self.continuous_gripper=True;self.continuous_task=True;self.rigid_control=None
        self.object_id=self.model.body('cooperative_object').id
        self.object_geom_id=self.model.geom('object_geom').id
        self.object_geoms=set(np.flatnonzero(self.model.geom_bodyid==self.object_id))
        self.target_id=self.model.body('target').id
        self.object_joint_id=self.model.joint('object_freejoint').id
        self.object_qpos_adr=int(self.model.jnt_qposadr[self.object_joint_id])
        self.object_dof_adr=int(self.model.jnt_dofadr[self.object_joint_id])
        self.finger_geoms={'left':[set(np.flatnonzero(self.model.geom_bodyid==self.model.body(name).id))
            for name in ('left_left_finger','left_right_finger')]}
        self.robot_geoms={'left':set(i for i,b in enumerate(self.model.geom_bodyid)
            if self.model.body_rootid[b]==self.model.body('left_link0').id)}
        self.episode_timeout_enabled=True
        self.contact_control=PushContact(self)
        self.reset(0)

    def reset(self,seed=0):
        self.seed=int(seed);mujoco.mj_resetData(self.model,self.data)
        sampled=initial_state(self.config,self.seed)
        self.data.qpos[self.object_qpos_adr:self.object_qpos_adr+7]=np.r_[sampled['object_position'],sampled['object_quaternion_wxyz']]
        self.model.body_pos[self.target_id]=sampled['target_position']
        self.model.body_quat[self.target_id]=[1,0,0,0]
        self.target_position=np.asarray(sampled['target_position']);self.target_yaw=sampled['target_yaw_rad']
        self.left.reset();self.data.qvel[:]=0.;mujoco.mj_forward(self.model,self.data)
        self.task_state=TaskState.PREGRASP;self.control_mode=ControlMode.IDLE
        self.gripper_latch=GripperLatch.CLOSED;self.paused=False;self.pause_reason=None;self.failure_reason=None
        self.interaction_mode='HOLD';self.mode_event_id=None;self.hand_state={};self.telemetry={}
        self.metrics=DualArmMetrics();self.success_monitor=PushTSuccess(self.config)
        self.success_tail=None;self.success_result={};self.last_action_result=None
        self.gripper_event=False;self.last_gripper_event=None;self._seen_event_ids=set()
        self._task_started_at=None;self._physics_remainder=0.;self._last_command_sim_time=None
        self._last_sent_at=-math.inf;self.user_delta={'left':[0.,0.,0.]};self.common_delta=[0.,0.,0.]
        self.input_disagreement_m=0.;self.relative_pose_error_m=0.;self.relative_orientation_error_rad=0.
        self.finger_forces=[0.,0.];self.support_force_n=0.;self.contacts={}
        self.ee_speeds=[0.];self._previous_ee_positions=[self.left.get_actual_pose()[0]]
        self.initial_state={**sampled,'config_hash':self.config_hash,'arm_order':['left'],
            'base_transforms':{'left':{'position':[0.,0.,0.],'quaternion_wxyz':[1.,0.,0.,0.]}},
            'task_frame':{'position':[0.,0.,0.],'quaternion_wxyz':[1.,0.,0.,0.]}}
        self._update_contacts();return self.get_status()

    @property
    def gripper_width_limits(self): return [self.left.max_gripper_width_m]

    @property
    def release_ready(self): return False

    @property
    def task_elapsed_seconds(self):
        return 0. if self._task_started_at is None else float(self.data.time)-self._task_started_at

    def set_episode_timeout_enabled(self,enabled): self.episode_timeout_enabled=bool(enabled)

    def start_task_timer(self):
        if self._task_started_at is None:self._task_started_at=float(self.data.time)
        return self._task_started_at

    def pause(self,reason='manual_pause',fail=False):
        if not self.paused:self.metrics.pause_count+=1
        self.paused=True;self.pause_reason=str(reason);self.control_mode=ControlMode.PAUSE
        self.interaction_mode='HOLD';self.mode_event_id=None;self.user_delta={'left':[0.,0.,0.]};self.left.freeze()
        if fail:self.task_state=TaskState.FAIL;self.failure_reason=str(reason)
        return self._result(True,reason)

    def begin_recording(self):
        """Keep the pose, but never attribute recording frames to practice input.

        The common service invokes this before the first recording snapshot.
        Event history remains intact: reusing a practice event is still invalid.
        A fresh guarded camera sample is required before target tracking resumes.
        """
        return self.pause('record_start_baseline')

    def resume(self):
        if self.task_state in (TaskState.DONE,TaskState.FAIL):return self._result(False,'episode_ended_reset_required')
        self.paused=False;self.pause_reason=None;self.control_mode=ControlMode.IDLE
        return self._result(True)

    def update_telemetry(self,telemetry):
        if not isinstance(telemetry,dict):return False
        self.telemetry=copy.deepcopy(telemetry)
        if isinstance(telemetry.get('hands'),dict):self.hand_state=copy.deepcopy(telemetry['hands'])
        if self.task_state in (TaskState.DONE,TaskState.FAIL):return True
        reason=self._telemetry_pause_reason(telemetry)
        if reason:self.pause(reason)
        # Telemetry is observational. It cannot itself resume an old target.
        return reason!='invalid_telemetry_mode'

    @staticmethod
    def _telemetry_pause_reason(telemetry):
        try:mode=ControlMode(str(telemetry.get('control_mode','IDLE')).upper())
        except ValueError:return 'invalid_telemetry_mode'
        hands=telemetry.get('hands')
        if hands is not None and (not isinstance(hands,dict) or not isinstance(hands.get('left'),dict)):
            return 'invalid_tracking_metadata'
        lost=isinstance(hands,dict) and hands.get('left',{}).get('visible') is not True
        if lost:return 'hand_lost'
        if telemetry.get('calibrated') is False:return 'calibration_required'
        if mode==ControlMode.PAUSE:return telemetry.get('pause_reason') or 'operator_pause'
        return None

    def _result(self,accepted,reason=None,requested=None,modes=None):
        if not accepted:self.metrics.rejected_command_count+=1
        modes=modes or [self.interaction_mode]
        result={'accepted':bool(accepted),'reason':reason,'arm_order':['left'],
            'control_protocol':'gesture-rate-gripper-v3','gripper_trigger_policy':'push_locked_v1',
            'policy_contract_version':'single-push-rate-v1','interaction_modes':list(modes),
            'release_intent':[m=='Z_OPEN' for m in modes],
            'requested_target':self.action_vector_m().tolist() if requested is None else np.asarray(requested).tolist(),
            'projected_target':self.action_vector_m().tolist(),'accepted_target':self.action_vector_m().tolist(),
            'physical_contact':self.contact_control.status(),'task_state':str(self.task_state)}
        self.last_action_result=result;return result

    def apply_command(self,payload,now_ms=None):
        if not isinstance(payload,dict):return self._result(False,'invalid_payload')
        command=payload.get('command')
        if command=='pause':return self.pause(payload.get('reason','manual_pause'))
        if command=='resume':return self.resume()
        if command=='metric_action':return self.apply_action(payload.get('action'),schema_version=payload.get('schema_version',SCHEMA),source='policy')
        if command!='continuous_input':return self._result(False,'unsupported_single_push_command')
        if self.task_state in (TaskState.DONE,TaskState.FAIL):return self._result(False,'episode_ended_reset_required')
        # HTTP has a raw-hand guard as well. Keep ordering/age enforcement here
        # for direct callers and prevent rejected packets from advancing clocks.
        clock=time.time()*1000 if now_ms is None else now_ms
        sent_at=payload.get('sentAt',clock)
        if type(clock) not in (int,float) or type(sent_at) not in (int,float) or not math.isfinite(clock) or not math.isfinite(sent_at):
            return self._result(False,'invalid_timestamp')
        age=clock-sent_at
        if age>self.config['control']['max_command_age_ms'] or age<0:
            self.metrics.stale_command_count+=1;return self._result(False,'stale_command')
        if sent_at<self._last_sent_at:
            self.metrics.stale_command_count+=1;return self._result(False,'out_of_order_command')
        modes=payload.get('interaction_modes');deltas=payload.get('delta_task_m')
        if not isinstance(modes,(tuple,list)) or len(modes)!=1 or type(modes[0]) is not str or modes[0] not in {'XY_HOLD','Z_OPEN','Z_CLOSE','HOLD'}:
            return self._result(False,'invalid_interaction_modes')
        try:delta=np.asarray(deltas['left'],dtype=float)
        except (KeyError,TypeError,ValueError):return self._result(False,'invalid_continuous_input')
        if delta.shape!=(3,) or not np.isfinite(delta).all():return self._result(False,'invalid_continuous_input')
        mode=modes[0]
        plane=payload.get('mode')
        if plane not in ('XY','Z') or (mode=='XY_HOLD' and plane!='XY') or (mode in ('Z_OPEN','Z_CLOSE') and plane!='Z'):
            return self._result(False,'interaction_plane_mismatch')
        if payload.get('control_protocol')!='gesture-rate-gripper-v3':return self._result(False,'gesture_control_protocol_required')
        intent=payload.get('release_intent')
        if not isinstance(intent,(list,tuple)) or len(intent)!=1 or type(intent[0]) is not bool or intent[0]!=(mode=='Z_OPEN'):
            return self._result(False,'release_intent_mode_mismatch')
        telemetry=payload.get('telemetry')
        if telemetry is not None:
            if not isinstance(telemetry,dict):return self._result(False,'invalid_tracking_metadata')
            reason=self._telemetry_pause_reason(telemetry)
            if reason:self.pause(reason);return self._result(False,reason)
        if ((mode=='XY_HOLD' and abs(delta[2])>1e-12) or (mode in {'Z_OPEN','Z_CLOSE'} and np.any(np.abs(delta[:2])>1e-12))
                or (mode=='HOLD' and np.any(np.abs(delta)>1e-12))):return self._result(False,'interaction_motion_forbidden')
        event=payload.get('eventId')
        if type(event) is not str or not 1<=len(event)<=256:return self._result(False,'event_id_required')
        if event in self._seen_event_ids:return self._result(False,'duplicate_event')
        if len(self._seen_event_ids)>=65536:return self._result(False,'event_history_full_reset_scene')
        length=np.linalg.norm(delta);cap=self.config['control']['max_ee_step_m']
        if length>cap:delta*=cap/length
        requested=self.action_vector_m().copy();requested[:3]+=delta
        if length>1e-12:
            if not self._submit_delta(delta):return self._result(False,'ik_target_rejected',requested,modes)
            self.start_task_timer()
        self._seen_event_ids.add(event);self._last_sent_at=sent_at
        self.paused=mode=='HOLD';self.pause_reason='unassigned_gesture' if self.paused else None
        self.interaction_mode=mode;self.mode_event_id=event
        self.control_mode=ControlMode.XY if mode=='XY_HOLD' else ControlMode.Z if mode in ('Z_OPEN','Z_CLOSE') else ControlMode.IDLE
        self.user_delta={'left':delta.tolist()};self._last_command_sim_time=float(self.data.time)
        return self._result(True,'push_gripper_locked',requested,modes)

    def _submit_delta(self,delta):
        """Report an explicit IK veto as a rejection, never as accepted motion."""
        position=self.left.command_position.copy();rotation=self.left.command_rotation.copy();ctrl=self.data.ctrl.copy()
        result=self.left.move_delta_task_frame(*delta)
        rejected=result is False or (isinstance(result,dict) and result.get('accepted') is False)
        if rejected:
            self.left.command_position=position;self.left.command_rotation=rotation;self.data.ctrl[:]=ctrl
            return False
        return True

    def apply_action(self,action,schema_version=SCHEMA,source='scripted_test',**kwargs):
        try:a=np.asarray(action,dtype=float)
        except (TypeError,ValueError):return self._result(False,'invalid_action')
        if schema_version!=SCHEMA or a.shape!=(7,) or not np.isfinite(a).all():return self._result(False,'invalid_action_schema')
        if abs(a[6])>1e-9:return self._result(False,'push_gripper_must_remain_closed',a)
        if self.task_state in (TaskState.DONE,TaskState.FAIL):return self._result(False,'episode_ended_reset_required',a)
        # Full RPY is explicit in A7, but this pushing protocol fixes source orientation.
        wanted=rpy_to_matrix(a[3:6]);angle=np.arccos(np.clip((np.trace(self.left.fixed_rotation.T@wanted)-1)/2,-1,1))
        if angle>.01:return self._result(False,'push_orientation_locked',a)
        delta=a[:3]-self.left.world_to_task(self.left.command_position)
        cap=self.config['control']['max_ee_step_m'];length=np.linalg.norm(delta)
        if length>cap:delta*=cap/length
        if not self._submit_delta(delta):return self._result(False,'ik_target_rejected',a)
        self.paused=False;self.pause_reason=None;self.control_mode=ControlMode.XY
        self.interaction_mode='XY_HOLD';self.mode_event_id=None;self._last_command_sim_time=float(self.data.time)
        self.start_task_timer()
        return self._result(True,None,a)

    def _update_contacts(self):
        forces=[0.,0.];self.support_force_n=0.;robot_contact=False;force=np.zeros(6)
        table=self.model.geom('table_geom').id
        for i in range(self.data.ncon):
            con=self.data.contact[i];g1,g2=int(con.geom1),int(con.geom2)
            if not ({g1,g2}&self.object_geoms):continue
            other=g2 if g1 in self.object_geoms else g1
            mujoco.mj_contactForce(self.model,self.data,i,force);normal=max(0.,float(force[0]))
            if other==table:self.support_force_n+=normal
            if other in self.robot_geoms['left']:robot_contact=True
            for j,ids in enumerate(self.finger_geoms['left']):
                if other in ids:forces[j]+=normal
        self.finger_forces=forces;flags=[f>=.05 for f in forces]
        self.contacts={'left':all(flags),'left_fingers':flags,'finger_normal_force_n':{'left':forces},
            'left_grip_forces_n':forces,'robot_object':robot_contact,'support_force_n':self.support_force_n}

    def step(self,dt=1/60):
        if not np.isfinite(dt) or dt<=0:raise ValueError('positive finite step required')
        budget=dt+self._physics_remainder;n=int(budget/self.model.opt.timestep+1e-9)
        self._physics_remainder=max(0.,budget-n*self.model.opt.timestep)
        if not self.paused and self.task_state not in (TaskState.DONE,TaskState.FAIL):self.left.track_target()
        previous=self.left.get_actual_pose()[0]
        for _ in range(n):
            # Source protocol keeps the fingers closed throughout reposition and pushing.
            self.left.gripper_goal_width_m=0.;self.left._write_gripper_width(0.)
            self.data.ctrl[self.left.yaw_actuator_id]=0.
            mujoco.mj_step(self.model,self.data)
        self._update_contacts()
        self.ee_speeds=[float(np.linalg.norm(self.left.get_actual_pose()[0]-previous)/dt)]
        self.success_result=self.success_monitor.update(self)
        if self.task_state not in (TaskState.DONE,TaskState.FAIL) and self.success_result['success']:
            self.task_state=TaskState.DONE;self.pause('task_done')
            self.success_tail={'active':False,'success_time':float(self.data.time),'end_time':float(self.data.time),'seconds':0.}
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():self.pause('nonfinite_physics',fail=True)
        elif self.data.xpos[self.object_id,2]<.35:self.pause('object_fell',fail=True)
        elif (self.episode_timeout_enabled and self._task_started_at is not None
              and self.task_elapsed_seconds>=self.config['evaluation']['max_episode_seconds']
              and self.task_state not in (TaskState.DONE,TaskState.FAIL)):self.pause('timeout',fail=True)
        return self.get_status()

    def consume_gripper_event(self):
        self.gripper_event=False;return None

    def state_vector(self):return self.left.state_vector()

    def action_vector_m(self):
        return np.r_[self.left.world_to_task(self.left.command_position),
            matrix_to_rpy(self.left.task_rotation.T@self.left.command_rotation),0.].astype(np.float32)

    action_vector=action_vector_m

    def action_contract(self):
        return {'version':SCHEMA,'arm_order':['left'],'state_names':STATE_NAMES,'action_names':ACTION_NAMES,
            'state_units':['rad']*8+['m'],'action_units':['m']*3+['rad']*3+['m'],
            'coordinate_frame':'task_frame','rotation_convention':'Rz(yaw) @ Ry(pitch) @ Rx(roll)',
            'gripper_max_width_m':[self.left.max_gripper_width_m],
            'capabilities':{'continuous_gripper':False,'fixed_closed_gripper':True,'free_orientation':False},
            'control_protocol':'gesture-rate-gripper-v3','gripper_trigger_policy':'push_locked_v1',
            'policy_contract_version':'single-push-rate-v1'}

    def get_status(self):
        pose,rotation=self.left.get_actual_pose();obj=self.data.xpos[self.object_id];rot=self.data.xmat[self.object_id].reshape(3,3)
        velocity=self.data.qvel[self.object_dof_adr:self.object_dof_adr+6]
        return {'task_id':'push_t','arm_order':['left'],'task_mode':'single_arm','task_state':str(self.task_state),
            'control_mode':str(self.control_mode),'paused':self.paused,'pause_reason':self.pause_reason,
            'failure_reason':self.failure_reason,'success':self.task_state==TaskState.DONE,
            'success_tail':copy.deepcopy(self.success_tail),'push_t_coverage':self.success_monitor.coverage,
            'push_success_evidence':dict(self.success_result),'task_elapsed_seconds':self.task_elapsed_seconds,
            'episode_timeout_enabled':self.episode_timeout_enabled,'gripper_latch':'CLOSED','release_ready':False,
            'physical_contact':self.contact_control.status(),'contacts':copy.deepcopy(self.contacts),
            'grippers':{'left':self.left.gripper_status()},'actual_ee':{'left':{'position':pose.tolist(),'rotation':rotation.tolist()}},
            'commanded_ee':{'left':{'position':self.left.command_position.tolist(),'rotation':self.left.command_rotation.tolist()}},
            'object_position':obj.tolist(),'object_rotation':rot.tolist(),'object_linear_velocity_mps':velocity[:3].tolist(),
            'object_angular_velocity_radps':velocity[3:].tolist(),'target_position':self.target_position.tolist(),
            'target_yaw_rad':self.target_yaw,'ee_speeds_mps':list(self.ee_speeds),'metrics':self.metrics.as_dict(),
            'relative_pose_error_m':0.,'relative_orientation_error_rad':0.,'input_disagreement_m':0.,
            'user_delta':self.user_delta,'common_delta':[0.,0.,0.],'simulation_time':float(self.data.time)}

    def snapshot(self):
        return {'time':float(self.data.time),'qpos':self.data.qpos.copy(),'qvel':self.data.qvel.copy(),'ctrl':self.data.ctrl.copy(),
            'body_pos':self.model.body_pos.copy(),'body_quat':self.model.body_quat.copy(),
            'state':self.state_vector(),'action':self.action_vector_m(),'status':self.get_status(),
            'initial_state':copy.deepcopy(self.initial_state)}
