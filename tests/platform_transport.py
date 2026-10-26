"""Real contact transport of the high-wall tray; no object pose edits after reset.

Scripted end-effector targets exercise the same PolicyExecutor used by Rollout.
This is an engineering physics acceptance, never a learned-policy success rate.
"""
from pathlib import Path
import argparse
import copy
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tasks.registry import get, hook
from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
from simulation.mujoco.dual_arm.panda_arm_controller import matrix_to_rpy
from simulation.mujoco.dual_arm.rigid_control import rz
from rollout.executor import PolicyExecutor


def run(seed=0, media=False, output=None, *, controller=None, on_step=None, placement_yaw_offset=0.):
    cfg=get('tray_to_platform').config()
    out=Path(output or ROOT/'.local/checks/tray_to_platform'/f'transport_seed{seed}_{time.time_ns()}')
    out.mkdir(parents=True,exist_ok=False)
    c=controller if controller is not None else DualArmTaskController(cfg)
    if controller is None:c.reset(seed)
    executor=PolicyExecutor(c)
    monitor=hook(cfg,'rollout_monitor')(cfg,c.initial_state,minimum_open_fraction=.5,min_lift_m=.03)
    snapshots=[];phases=[];ticks=0;trace=[]
    initial_free_qpos=c.data.qpos[c.object_qpos_adr:c.object_qpos_adr+7].copy()
    def hold(action, seconds, label):
        nonlocal ticks
        rejections=[]
        for _ in range(round(seconds*60)):
            applied=executor.apply(action)
            if not applied['accepted']:rejections.append(applied)
            c.step(1/60);ticks+=1
            if on_step:on_step()
            measured=monitor.update(c)
            if media and ticks%2==0:snapshots.append(c.snapshot())
            if ticks%3==0:trace.append({'time':float(c.data.time),'phase':label,
                'object_position':c.data.xpos[c.object_id].tolist(),
                'object_velocity':c.data.qvel[c.object_dof_adr:c.object_dof_adr+6].tolist(),
                'grippers':c.get_status()['grippers'],'contacts':copy.deepcopy(c.contacts),
                'support':dict(c.contact_control.support_forces_n),'state':str(c.task_state),
                'relative_error_m':float(c.relative_pose_error_m)})
            if str(c.task_state)=='FAIL':break
            if str(c.task_state)=='DONE':break
        status=c.get_status()
        phases.append({'phase':label,'state':status,'monitor':measured,'rejections':rejections})
        print(label,'t=',round(c.data.time,3),'state=',str(c.task_state),'obj=',np.round(c.data.xpos[c.object_id],4),
              'contacts=',[c.contacts[s] for s in ('left','right')], 'support=',c.contact_control.support_surface,
              'failure=',c.failure_reason,flush=True)
        if media:phase_snapshots[label]=c.snapshot()
        assert not rejections,(label,rejections[:1])
        assert str(c.task_state)!='FAIL',(label,c.failure_reason)
        return measured
    phase_snapshots={}
    try:
        action=c.action_vector_m().astype(float)
        hold(action,.4,'initial_settle')
        actual=c.data.xpos[c.object_id].copy();rotation=c.data.xmat[c.object_id].reshape(3,3).copy()
        yaw=float(np.arctan2(rotation[1,0],rotation[0,0]))
        for arm,o,offset in zip(c.arms,(0,7),cfg['scene']['grasp_offsets']):
            pos=actual+rotation@np.array(offset)
            action[o:o+3]=arm.world_to_task(pos);action[o+2]=.55
            action[o+3:o+6]=matrix_to_rpy(arm.task_rotation.T@rz(yaw)@arm.fixed_rotation)
        hold(action,2.,'approach')
        for o,offset in zip((0,7),cfg['scene']['grasp_offsets']):action[o+2]=actual[2]+offset[2]
        hold(action,2.,'descend')
        action[[6,13]]=0.
        hold(action,1.5,'grasp')
        assert all(c.contact_control.stable.values()),'Both true opposed grasps required'
        refs=copy.deepcopy(c.contact_control.reference)
        def target(center, angle):
            command=c.action_vector_m().astype(float);r=rz(angle)
            for arm,o,ref in zip(c.arms,(0,7),refs):
                command[o:o+3]=arm.world_to_task(np.asarray(center)+r@ref[0])
                command[o+3:o+6]=matrix_to_rpy(arm.task_rotation.T@r@ref[1])
            return command
        center=c.data.xpos[c.object_id].copy()
        center[2]=c.target_position[2]+cfg['scene']['platform_task']['transport_clearance_m']
        hold(target(center,yaw),2.,'lift')
        assert c.contact_control.bottom_height_m >= cfg['scene']['placement_platform']['height_m'],'Tray must clear original table by configured platform height'
        center[:2]=c.target_position[:2]
        hold(target(center,yaw),3.,'transfer')
        hold(target(center,c.target_yaw+placement_yaw_offset),1.5,'align')
        hold(target(c.target_position,c.target_yaw+placement_yaw_offset),2.,'land')
        assert c.contact_control.support_ready and c.contact_control.support_surface=='placement_platform'
        action=c.action_vector_m().astype(float);action[[6,13]]=.08
        result=hold(action,3.,'release')
        assert str(c.task_state)=='DONE' and result['task_success'],'Release must reach actual DONE'
        # Freeze accepted end-effector goals and simulate independent supported settling.
        for _ in range(60):
            c.step(1/60);ticks+=1
            if on_step:on_step()
            if media and ticks%2==0:snapshots.append(c.snapshot())
        final=monitor.update(c)
        assert final['task_success'] and c.metrics.collision_count==0
        assert c.contact_control.support_surface=='placement_platform'
        passed=True
    except Exception as error:
        passed=False;failure=repr(error)
    report={'kind':'scripted_real_physics','human_teleop_tested':False,'learned_policy':False,
            'seed':seed,'task_binding':cfg['task_binding'],'config':cfg,'passed':passed,
            'failure':None if passed else failure,'phases':phases,'trace':trace,
            'initial_object_free_qpos':initial_free_qpos.tolist(),'object_pose_writes_after_reset':0,
            'constraint_count':int(c.model.neq),'collision_count':c.metrics.collision_count}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if media:
        from simulation.mujoco.dual_arm.render_workers import MultiCameraRenderer,apply_snapshot
        from PIL import Image
        import av,mujoco
        data=mujoco.MjData(c.model)
        renderer=MultiCameraRenderer(c.model,config=cfg,camera_names=['top','front','oblique'])
        try:
            for phase,snapshot in phase_snapshots.items():
                apply_snapshot(c.model,data,snapshot)
                for camera,image in renderer.render(data).items():Image.fromarray(image).save(out/f'{phase}_{camera}.png')
        finally:renderer.close()
        renderer=MultiCameraRenderer(c.model,config=cfg,camera_names=['oblique'])
        try:
            with av.open(str(out/'transport_oblique.mp4'),'w') as video:
                stream=video.add_stream('libx264',rate=30);stream.width=640;stream.height=480;stream.pix_fmt='yuv420p'
                for snapshot in snapshots:
                    apply_snapshot(c.model,data,snapshot)
                    frame=av.VideoFrame.from_ndarray(renderer.render(data)['oblique'],format='rgb24')
                    for packet in stream.encode(frame):video.mux(packet)
                for packet in stream.encode():video.mux(packet)
        finally:renderer.close()
        with av.open(str(out/'transport_oblique.mp4')) as video:
            decoded=sum(1 for _ in video.decode(video=0))
        assert decoded==len(snapshots)
        print('video frames',decoded,flush=True)
    print('REPORT',out/'report.json',flush=True)
    if not passed:raise AssertionError(failure)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=0);parser.add_argument('--media',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args();run(args.seed,args.media,args.output)
