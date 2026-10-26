"""IK math and collision protection must match full-forward scratch calculations."""
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from tasks.registry import get
from simulation.mujoco.dual_arm.task_controller import DualArmTaskController


class ComputationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = DualArmTaskController(get('tray_to_platform').config())

    def test_ik_pose_jacobian_and_targets_match_full_forward_without_dynamics(self):
        c=self.c;c.reset(0);rng=np.random.default_rng(901)
        for arm in c.arms:
            for _ in range(12):
                c.data.ctrl[arm.actuator_ids]=c.data.qpos[arm.qpos_indices]+rng.uniform(-.08,.08,7)
                live=c.data.qpos.copy()
                expected=mujoco.MjData(c.model)
                expected.qpos[:]=c.data.qpos
                expected.qpos[arm.qpos_indices]=c.data.ctrl[arm.actuator_ids]
                mujoco.mj_forward(c.model,expected)
                jp,jr=np.zeros_like(arm.jacp),np.zeros_like(arm.jacr)
                mujoco.mj_jacSite(c.model,expected,jp,jr,arm.site_id)
                with patch.object(mujoco,'mj_forward',side_effect=AssertionError('IK must not solve contact dynamics')):
                    arm._copy_commanded_configuration()
                np.testing.assert_allclose(arm.ik_data.site_xpos,expected.site_xpos,atol=1e-12)
                np.testing.assert_allclose(arm.ik_data.site_xmat,expected.site_xmat,atol=1e-12)
                mujoco.mj_jacSite(c.model,arm.ik_data,arm.jacp,arm.jacr,arm.site_id)
                np.testing.assert_allclose(arm.jacp,jp,atol=1e-12)
                np.testing.assert_allclose(arm.jacr,jr,atol=1e-12)
                target=expected.site_xpos[arm.site_id]+[.001,0,.001]
                rot=expected.site_xmat[arm.site_id].reshape(3,3).copy()
                with patch.object(mujoco,'mj_forward',side_effect=AssertionError('IK must not solve contact dynamics')):
                    arm._ik_iteration(target,rot,.01)
                mujoco.mj_forward(c.model,arm.ik_data)
                np.testing.assert_array_equal(c.data.qpos,live)

    def test_collision_contacts_match_full_forward_including_collision_cases(self):
        c=self.c;c.reset(0);rng=np.random.default_rng(902);blocked=0
        forward=mujoco.mj_forward
        for sample in range(30):
            for arm in c.arms:
                limits=c.model.jnt_range[arm.joint_ids]
                c.data.qpos[arm.qpos_indices]=rng.uniform(limits[:,0]+.01,limits[:,1]-.01)
                arm.ik_data.qpos[:]=c.data.qpos
                arm.ik_data.qpos[arm.qpos_indices]+=rng.uniform(-.01,.01,7)
            mujoco.mj_forward(c.model,c.data)
            if sample == 29:
                # Deliberate negative fixture: physical platform intersects a finger.
                platform=c.model.body('placement_platform').id
                finger=next(g for g in c.finger_geoms['left'][0] if c.model.geom_contype[g])
                c.model.body_pos[platform]=c.data.geom_xpos[finger]
                mujoco.mj_forward(c.model,c.data)
            live=c.data.qpos.copy()
            with patch.object(mujoco,'mj_forward',side_effect=AssertionError('Collision precheck must not solve forces')):
                optimized=c.rigid_control.collision_reason()
            contacts=[(int(x.geom1),int(x.geom2),float(x.dist)) for x in c.rigid_control.collision_data.contact]
            with patch.object(mujoco,'mj_fwdPosition',side_effect=forward):
                reference=c.rigid_control.collision_reason()
            self.assertEqual(optimized,reference)
            self.assertEqual(contacts,[(int(x.geom1),int(x.geom2),float(x.dist)) for x in c.rigid_control.collision_data.contact])
            blocked+=reference is not None
            np.testing.assert_array_equal(c.data.qpos,live)
        self.assertGreater(blocked,0)

class PreviewSchedulingTests(unittest.TestCase):
    def recorder(self):
        from types import SimpleNamespace
        from queue import Queue
        from simulation.mujoco.dual_arm.teleop_service import DualArmRecorder
        r=DualArmRecorder.__new__(DualArmRecorder)
        r.preview_process=SimpleNamespace(is_alive=lambda:True)
        r.preview_in=Queue(1);r.preview_out=Queue(1)
        r.controller=SimpleNamespace(snapshot=lambda:{'time':1.})
        r.control_epoch=2;r.session='test';r.preview_requested_at=100.
        r.preview_updated_at=99.9;r.preview_jpegs={};r.preview_error=None
        return r

    def test_pending_preview_is_replaced_by_latest_snapshot(self):
        r=self.recorder();r.preview_in.put({'time':0.})
        with patch('time.monotonic',return_value=100.):r.update_previews(100.)
        self.assertEqual(r.preview_in.get_nowait()['time'],1.)

    def test_old_scene_preview_cannot_reappear_after_reset(self):
        r=self.recorder()
        r.preview_out.put({'images':{'top':b'old'},'created_at':100.,'captured_at':99.95,'control_epoch':1,'session':'test'})
        r.update_previews(100.)
        self.assertEqual(r.preview_jpegs,{})
    def test_age_is_measured_from_snapshot_not_render_completion(self):
        r=self.recorder()
        r.preview_out.put({'images':{'top':b'old'},'created_at':100.,'captured_at':98.,'control_epoch':2,'session':'test'})
        r.update_previews(100.)
        self.assertEqual(r.preview_jpegs,{})


if __name__=='__main__':unittest.main()
