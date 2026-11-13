"""Upstream Push-T 81x81 coverage, 90% entry / 85% exit, 0.5s dwell."""
import math
import numpy as np

CRITERION='upstream-push-t-coverage-v1'


def make_push_t_sample_points(resolution=81):
    x=np.linspace(-.15,.15,resolution); y=np.linspace(-.15,.15,resolution)
    xx,yy=np.meshgrid(x,y,indexing='xy')
    mask=((xx>=.06)&(xx<=.15)&(np.abs(yy)<=.15))|((xx>=-.15)&(xx<=.06)&(np.abs(yy)<=.045))
    return np.column_stack((xx[mask],yy[mask]))


def push_t_coverage(object_xy, object_rotation_xy, target_xy, target_rotation_xy, sample_points):
    points=(sample_points@object_rotation_xy.T+object_xy-target_xy)@target_rotation_xy
    bar=(points[:,0]>=.055)&(points[:,0]<=.155)&(np.abs(points[:,1])<=.155)
    stem=(points[:,0]>=-.155)&(points[:,0]<=.065)&(np.abs(points[:,1])<=.050)
    return float(np.mean(bar|stem))


class PushTSuccess:
    def __init__(self, config):
        self.parameters=config['scene']['push_t']; self.samples=make_push_t_sample_points(self.parameters['coverage_resolution'])
        self.since=None; self.success=False; self.coverage=0.

    def update(self, controller):
        c=controller; data=c.data
        self.coverage=push_t_coverage(data.xpos[c.object_id,:2],data.xmat[c.object_id].reshape(3,3)[:2,:2],
            data.xpos[c.target_id,:2],data.xmat[c.target_id].reshape(3,3)[:2,:2],self.samples)
        if not self.success:
            if self.coverage>=self.parameters['success_coverage']:
                if self.since is None: self.since=float(data.time)
                if data.time-self.since>=self.parameters['dwell_seconds']: self.success=True
            else: self.since=None
        elif self.coverage<self.parameters['exit_coverage']:
            self.success=False; self.since=None
        rotation=data.xmat[c.object_id].reshape(3,3);target_rotation=data.xmat[c.target_id].reshape(3,3)
        yaw=math.atan2(rotation[1,0],rotation[0,0]);target_yaw=math.atan2(target_rotation[1,0],target_rotation[0,0])
        failed=str(c.task_state)=='FAIL';success=bool(self.success and not failed)
        # Error values are diagnostics, not extra acceptance thresholds: retain
        # the upstream coverage/dwell definition for this registered task.
        errors={'xy_m':float(np.linalg.norm(data.xpos[c.object_id,:2]-data.xpos[c.target_id,:2])),
                'z_m':float(data.xpos[c.object_id,2]-(c.config['scene']['table_top_z']+.015)),
                'orientation_rad':abs(math.remainder(yaw-target_yaw,math.tau))}
        return {'success':success,'task_success':success,'physical_success':success,
                'terminal':success or failed,'reason':c.failure_reason if failed else None,
                'coverage':self.coverage,'criterion':CRITERION,'task_success_criterion':CRITERION,
                'success_criterion':CRITERION,'target_error':errors,
                'dwell_elapsed_s': max(0.,float(data.time)-self.since) if self.since is not None else 0.}


def collection_success(contact_control, controller, now, object_pos, object_rot, any_contact):
    return controller.success_monitor.update(controller)


def rollout_monitor(config, initial_state=None, **options):
    return PushTSuccess(config)
