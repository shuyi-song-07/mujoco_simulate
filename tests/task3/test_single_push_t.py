"""Single-arm source fidelity and real-contact engineering checks (no human claims)."""
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import mujoco
from tasks.registry import get
from tasks.push_t.scene import create_model, initial_state, footprint_xy
from tasks.push_t.success import PushTSuccess
from simulation.mujoco.single_arm.task_controller import SingleArmTaskController, SCHEMA
from simulation.mujoco.dual_arm.panda_arm_controller import rotation_angle

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
        p=self.config['scene']['push_t']
        for seed in (0,1,2,910001):
            a=initial_state(self.config,seed);b=initial_state(self.config,seed)
            self.assertEqual(a,b)
            x,y,z=a['object_position'];tx,ty,tz=a['target_position']
            self.assertTrue(p['object_x_range'][0]<=x<=p['object_x_range'][1] and p['object_y_range'][0]<=y<=p['object_y_range'][1])
            self.assertEqual(a['target_position'],self.config['scene']['target_position'])
            self.assertEqual(a['target_yaw_rad'],self.config['scene']['target_yaw_rad'])
            self.assertGreaterEqual(np.linalg.norm(np.array([x,y])-np.array([tx,ty])),p['object_min_separation_m'])

    def test_target_pose_is_fixed_while_object_starts_vary_on_both_sides(self):
        c=SingleArmTaskController(self.config);starts=[]
        for seed in range(64):
            c.reset(seed);state=c.initial_state
            self.assertEqual(state['target_position'],self.config['scene']['target_position'])
            self.assertEqual(state['target_yaw_rad'],self.config['scene']['target_yaw_rad'])
            np.testing.assert_allclose(c.data.xpos[c.target_id],self.config['scene']['target_position'],atol=1e-10)
            yaw=np.arctan2(c.data.xmat[c.target_id].reshape(3,3)[1,0],c.data.xmat[c.target_id].reshape(3,3)[0,0])
            self.assertAlmostEqual(yaw,self.config['scene']['target_yaw_rad'])
            starts.append(state['object_position'][:2])
        self.assertEqual(len({tuple(xy) for xy in starts}),64)
        starts=np.array(starts)
        self.assertLess(starts[:,1].min(),0);self.assertGreater(starts[:,1].max(),0)
        self.assertGreater(np.ptp(starts[:,0]),.03)

    def test_configured_fixed_target_orientation_reaches_the_model(self):
        import copy
        config=copy.deepcopy(self.config);config['scene']['target_position']=[.4,0.,.401]
        config['scene']['target_yaw_rad']=np.pi/4
        c=SingleArmTaskController(config)
        for seed in (0,1,21):
            c.reset(seed)
            rotation=c.data.xmat[c.target_id].reshape(3,3)
            self.assertAlmostEqual(np.arctan2(rotation[1,0],rotation[0,0]),np.pi/4)
            self.assertEqual(c.initial_state['target_yaw_rad'],np.pi/4)

    def test_actual_geometry_and_pusher_margin_fit_for_randomized_placements(self):
        c=SingleArmTaskController(self.config);p=self.config['scene']['push_t'];margin=p['push_clearance_m']
        workspace=c.left.workspace[:2]
        for seed in range(128):
            c.reset(seed)
            outlines=[]
            for body in (c.object_id,c.target_id):
                points=[]
                for gid in np.flatnonzero(c.model.geom_bodyid==body):
                    half=c.model.geom_size[gid,:2]
                    corners=np.array([[x,y,0.] for x in (-half[0],half[0]) for y in (-half[1],half[1])])
                    points.extend((corners@c.data.geom_xmat[gid].reshape(3,3).T+c.data.geom_xpos[gid])[:,:2])
                points=np.array(points);outlines.append(points)
                self.assertLessEqual(np.linalg.norm(points,axis=1).max()+margin,p['safe_reach_radius_m']+1e-9)
                self.assertTrue(np.all(points.min(axis=0)-margin>=workspace[:,0]))
                self.assertTrue(np.all(points.max(axis=0)+margin<=workspace[:,1]))
                table=np.array(self.config['scene']['table_position'][:2]);half=np.array(self.config['scene']['table_half_size_xy'])
                self.assertTrue(np.all(points.min(axis=0)-margin>=table-half))
                self.assertTrue(np.all(points.max(axis=0)+margin<=table+half))
            self.assertLessEqual(c.get_status()['push_t_coverage'],p['initial_max_coverage'])
            for contact in c.data.contact:
                geoms={contact.geom1,contact.geom2}
                if geoms&c.object_geoms and geoms&c.robot_geoms['left']:
                    self.assertGreaterEqual(contact.dist,0.,f'initial robot penetration at seed {seed}')

    def test_close_placements_settle_without_home_pusher_collisions(self):
        c=SingleArmTaskController(self.config)
        for seed in (0,1,2,21,63):
            with self.subTest(seed=seed):
                c.reset(seed);before=c.data.xpos[c.object_id,:2].copy();c.step(.4)
                self.assertFalse(c.contacts['robot_object'])
                self.assertLess(np.linalg.norm(c.data.xpos[c.object_id,:2]-before),.001)

    def test_unreachable_footprints_are_rejected_even_when_centres_fit(self):
        import copy
        config=copy.deepcopy(self.config)
        config['scene']['push_t'].update(object_x_range=[.63,.64])
        with self.assertRaisesRegex(ValueError,'footprint'):
            initial_state(config,0)

    def test_unreachable_fixed_target_is_rejected(self):
        import copy
        config=copy.deepcopy(self.config);config['scene']['target_position']=[.64,0.,.401]
        with self.assertRaisesRegex(ValueError,'target footprint'):
            initial_state(config,0)

    def test_extreme_approach_points_have_fixed_orientation_ik_solutions(self):
        c=SingleArmTaskController(self.config);a=c.left;p=self.config['scene']['push_t']
        offsets=p['push_clearance_m']*np.array([[1,0],[-1,0],[0,1],[0,-1]])
        points=[]
        for seed in range(128):
            state=initial_state(self.config,seed)
            for outline in (footprint_xy(state['object_position'],state['object_yaw_rad']),
                            footprint_xy(state['target_position'],state['target_yaw_rad'],target=True)):
                points.extend((outline[:,None,:]+offsets).reshape(-1,2))
        points=np.array(points)
        indices=set(np.argsort(np.linalg.norm(points,axis=1))[-8:])
        for axis in (0,1):indices.update((np.argmin(points[:,axis]),np.argmax(points[:,axis])))
        for index in indices:
            for z in (a.workspace[2,0],.60):
                with self.subTest(xy=points[index],z=z):
                    target=np.r_[points[index],z];a._copy_commanded_configuration()
                    for _ in range(180):a._ik_iteration(target,a.fixed_rotation,.1)
                    self.assertLess(np.linalg.norm(a.ik_data.site_xpos[a.site_id]-target),.001)
                    self.assertLess(rotation_angle(a.ik_data.site_xmat[a.site_id].reshape(3,3),a.fixed_rotation),.01)

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
