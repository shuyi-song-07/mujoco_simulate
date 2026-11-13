"""Single-arm source fidelity and real-contact engineering checks (no human claims)."""
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import mujoco
from tasks.registry import get
from tasks.push_t.scene import create_model, initial_state
from tasks.push_t.success import PushTSuccess
from simulation.mujoco.single_arm.task_controller import SingleArmTaskController, SCHEMA

ROOT=Path(__file__).resolve().parents[2]


class SinglePushTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=get('push_t').config()

    def test_original_physics_retained(self):
        assets={p.name:p.read_bytes() for p in (ROOT/'simulation/mujoco/assets/franka_emika_panda/assets').iterdir() if p.is_file()}
        src=ROOT/'tasks/push_t/assets'
        assets['panda.xml']=(src/'panda.xml').read_bytes().replace(b'meshdir="assets"',b'meshdir=""')
        original=mujoco.MjModel.from_xml_string((src/'push_t_scene.xml').read_text(),assets=assets)
        model=create_model(self.config)
        self.assertEqual((model.nq,model.nv,model.nu,model.nbody,model.ngeom),(17,16,9,16,79))
        for field in ('body_mass','body_inertia','geom_size','geom_friction','geom_solref','geom_solimp',
                      'jnt_range','dof_damping','actuator_gainprm','actuator_biasprm','actuator_ctrlrange',
                      'actuator_forcerange','body_pos','body_quat'):
            np.testing.assert_array_equal(getattr(original,field),getattr(model,field),err_msg=field)
        self.assertEqual(model.opt.timestep,.002)
        self.assertEqual(model.joint('left_gripper_yaw_joint').id,7)
        self.assertEqual(model.jnt_range[model.joint('left_finger_joint1').id,1],.05)
        self.assertEqual(model.jnt_range[model.joint('left_finger_joint2').id,1],.05)
        # Controller initialization must not silently calibrate the source
        # 100mm servo to the dual-arm scene's 80mm gripper.
        controller=SingleArmTaskController(self.config)
        for field in ('actuator_gainprm','actuator_biasprm','actuator_ctrlrange','jnt_range'):
            np.testing.assert_array_equal(getattr(original,field),getattr(controller.model,field))
        actuator=controller.left.gripper_actuator_id
        width=2*controller.model.actuator_gainprm[actuator,0]*255/-controller.model.actuator_biasprm[actuator,1]
        self.assertAlmostEqual(width,.1,places=8)

    def test_reproducible_bounded_initial_state(self):
        for seed in (0,1,2,910001):
            a=initial_state(self.config,seed);b=initial_state(self.config,seed)
            self.assertEqual(a,b)
            x,y,z=a['object_position'];tx,ty,tz=a['target_position']
            self.assertTrue(.46<=x<=.54 and -.26<=y<=.26)
            self.assertTrue(.57<=tx<=.66 and -.26<=ty<=.26)
            self.assertGreaterEqual(np.linalg.norm(np.array([x,y])-np.array([tx,ty])),np.hypot(.15,.15)+np.hypot(.155,.155)+.005)

    def test_idle_and_snapshot_replay(self):
        c=SingleArmTaskController(self.config)
        for _ in range(600):c.step(1/60)
        self.assertTrue(np.isfinite(c.data.qpos).all());self.assertEqual(c.state_vector().shape,(9,))
        self.assertEqual(c.action_vector_m().shape,(7,));self.assertEqual(len(c.arms),1)
        self.assertFalse(hasattr(c,'right'));self.assertAlmostEqual(c.data.time,10.,places=9)
        self.assertAlmostEqual(c.support_force_n,.981,places=4)
        self.assertGreater(c.left.max_gripper_width_m,.09999)
        snap=c.snapshot();from simulation.mujoco.dual_arm.render_workers import apply_snapshot
        other=create_model(self.config);data=mujoco.MjData(other);apply_snapshot(other,data,snap)
        np.testing.assert_allclose(data.qpos,c.data.qpos,atol=0,rtol=0)
        np.testing.assert_allclose(data.xpos[c.object_id],c.data.xpos[c.object_id],atol=1e-7)

    def test_input_lock_and_invalid_actions(self):
        c=SingleArmTaskController(self.config);ctrl=c.data.ctrl.copy()
        a=c.action_vector_m();a[0]=np.nan
        self.assertFalse(c.apply_action(a)['accepted']);np.testing.assert_array_equal(c.data.ctrl,ctrl)
        a=c.action_vector_m();a[-1]=.1
        self.assertEqual(c.apply_action(a)['reason'],'push_gripper_must_remain_closed')
        a=c.action_vector_m();a[3]+=.2
        self.assertEqual(c.apply_action(a)['reason'],'push_orientation_locked')
        for i,mode in enumerate(('Z_OPEN','Z_CLOSE','XY_HOLD')):
            answer=c.apply_command({'command':'continuous_input','interaction_modes':[mode],
                'mode':'XY' if mode=='XY_HOLD' else 'Z','release_intent':[mode=='Z_OPEN'],
                'control_protocol':'gesture-rate-gripper-v3',
                'delta_task_m':{'left':[0.,0.,0.]},'eventId':f'e{i}'})
            self.assertTrue(answer['accepted']);self.assertEqual(answer['release_intent'],[mode=='Z_OPEN'])
            self.assertFalse(answer['physical_contact']['release_intent']['left'])
            for _ in range(5):c.step(1/60)
            self.assertEqual(c.action_vector_m()[-1],0.);self.assertEqual(c.data.ctrl[c.left.gripper_actuator_id],0.)
        self.assertFalse(c.apply_command({'command':'continuous_input','interaction_modes':['XY_HOLD'],
            'mode':'XY','release_intent':[False],'control_protocol':'gesture-rate-gripper-v3',
            'delta_task_m':{'left':[0.,0.,.01]},'eventId':'wrong_plane'})['accepted'])
        self.assertFalse(c.apply_command({'command':'continuous_input','interaction_modes':['XY_HOLD'],
            'mode':'XY','release_intent':[False],'control_protocol':'gesture-rate-gripper-v3',
            'delta_task_m':{'left':[0.,0.,0.]},'eventId':'e2'})['accepted'])
        c.pause();held=c.data.ctrl.copy()
        for _ in range(10):c.step(1/60)
        np.testing.assert_array_equal(c.data.ctrl,held)

    def test_explicit_ik_veto_preserves_accepted_target(self):
        c=SingleArmTaskController(self.config);before=c.action_vector_m();ctrl=c.data.ctrl.copy()
        def reject(*delta):
            c.left.command_position+=.001;c.data.ctrl[c.left.actuator_ids]+=.01
            return False
        desired=before.copy();desired[0]+=.001
        with patch.object(c.left,'move_delta_task_frame',side_effect=reject):
            result=c.apply_action(desired)
        self.assertFalse(result['accepted']);self.assertEqual(result['reason'],'ik_target_rejected')
        np.testing.assert_array_equal(c.action_vector_m(),before);np.testing.assert_array_equal(c.data.ctrl,ctrl)

    def test_contact_push_completes_without_object_pose_writes(self):
        c=SingleArmTaskController(self.config);c.set_episode_timeout_enabled(False)
        # Explicit engineering initial fixture: within source XY ranges but
        # deliberately nearer than the randomized curriculum. Not human data.
        c.data.qpos[c.object_qpos_adr:c.object_qpos_adr+7]=[.54,0,.417,1,0,0,0]
        c.model.body_pos[c.target_id]=[.64,0,.401];c.target_position=c.model.body_pos[c.target_id].copy()
        mujoco.mj_forward(c.model,c.data)
        for _ in range(60):c.step(1/60)
        start=c.data.xpos[c.object_id].copy();peak=0.
        # From here on only real arm servo targets and mj_step move the object.
        for _ in range(400):
            a=c.action_vector_m();a[0]+=.0005
            self.assertTrue(c.apply_action(a,schema_version=SCHEMA)['accepted'])
            c.step(1/60);peak=max(peak,*c.finger_forces)
            if c.task_state=='DONE':break
        self.assertEqual(c.task_state,'DONE');self.assertGreaterEqual(c.success_monitor.coverage,.9)
        self.assertTrue(c.success_result['task_success'])
        self.assertEqual(set(c.success_result['target_error']),{'xy_m','z_m','orientation_rad'})
        self.assertGreater(peak,.05);self.assertGreater(np.linalg.norm(c.data.xpos[c.object_id]-start),.04)
        self.assertLess(abs(c.data.xpos[c.object_id,2]-.415),.001)

    def test_real_contact_push_responds_in_both_y_directions(self):
        # Extra controllability fixtures. They verify actual contact and signed
        # motion, not full random-scene automatic planning or human usability.
        for sign in (-1,1):
            with self.subTest(direction=sign):
                c=SingleArmTaskController(self.config);c.set_episode_timeout_enabled(False)
                c.data.qpos[c.object_qpos_adr:c.object_qpos_adr+7]=[.54,0,.417,1,0,0,0]
                c.model.body_pos[c.target_id]=[.54,sign*.1,.401]
                c.target_position=c.model.body_pos[c.target_id].copy()
                mujoco.mj_forward(c.model,c.data)
                # No object position/velocity is written after initialization.
                for waypoint in ([.35,0,.60],[.62,-sign*.22,.60],[.62,-sign*.22,.5234]):
                    for _ in range(250):
                        action=c.action_vector_m();action[:3]=waypoint
                        self.assertTrue(c.apply_action(action)['accepted']);c.step(1/60)
                start=c.data.xpos[c.object_id].copy();peak=0.
                for _ in range(700):
                    action=c.action_vector_m();action[1]+=sign*.0005
                    self.assertTrue(c.apply_action(action)['accepted']);c.step(1/60)
                    peak=max(peak,*c.finger_forces)
                self.assertGreater(sign*(c.data.xpos[c.object_id,1]-start[1]),.1)
                self.assertGreater(peak,.05);self.assertTrue(np.isfinite(c.data.qpos).all())
                self.assertLess(abs(c.data.xpos[c.object_id,2]-.415),.001)


if __name__=='__main__':unittest.main()
