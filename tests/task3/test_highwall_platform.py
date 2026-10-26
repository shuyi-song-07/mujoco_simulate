"""Task-local tray geometry/initialization/static IK; not a transport success test."""
import copy
import unittest

import mujoco
import numpy as np

from tasks.registry import get, hook
from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
from rollout.protocol import _waypoint_checks


class HighwallPlatformGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = get("tray_to_platform").config()
        cls.controller = DualArmTaskController(cls.config)

    def setUp(self):
        self.c = self.controller
        self.c.reset(0)

    def test_real_shell_grips_free_body_and_unchanged_robot(self):
        c, s = self.c, self.config["scene"]
        m = c.model
        parts = [m.geom("object_geom"), *[m.geom(f"object_part_{i}") for i in range(1, 7)]]
        self.assertEqual(len(c.object_geoms), 7)
        for geom in parts:
            self.assertEqual(int(m.geom_type[geom.id]), int(mujoco.mjtGeom.mjGEOM_BOX))
            self.assertGreater(m.geom_contype[geom.id], 0)
            self.assertGreater(m.geom_conaffinity[geom.id], 0)
        np.testing.assert_allclose(parts[0].size*2, [.24, .32, .008])
        np.testing.assert_allclose(parts[0].pos, [0, 0, -.030])
        for index in (1, 4):
            np.testing.assert_allclose(parts[index].size*2, [.008, .32, .05])
        for index in (2, 5):
            np.testing.assert_allclose(parts[index].size*2, [.224, .01, .06])
        for index in (3, 6):
            np.testing.assert_allclose(parts[index].size*2, [.036, .12, .052])
        self.assertEqual(int(m.jnt_type[c.object_joint_id]), int(mujoco.mjtJoint.mjJNT_FREE))
        self.assertEqual(float(m.body_gravcomp[c.object_id]), 0.)
        np.testing.assert_allclose(m.opt.gravity, [0, 0, -9.81])
        for i in range(m.neq):
            if int(m.eq_type[i]) == int(mujoco.mjtEq.mjEQ_WELD):
                self.assertNotIn(c.object_id, (int(m.eq_obj1id[i]), int(m.eq_obj2id[i])))
        for arm, offset in zip(c.arms, s["grasp_offsets"]):
            self.assertAlmostEqual(arm.max_gripper_width_m, .08)
            np.testing.assert_allclose(m.jnt_range[arm.finger_joint_ids], [[0, .04], [0, .04]])
            site = m.site(arm.side+"_grasp_site")
            np.testing.assert_allclose(site.pos, offset)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, site.quat)
            np.testing.assert_allclose(rotation.reshape(3, 3), s["ee_rotation_matrix"], atol=1e-12)
        baseline = get("tray_transfer").config()
        for name in ("table_top_z",
                     "left_base_quaternion_wxyz", "right_base_quaternion_wxyz"):
            self.assertEqual(s[name], baseline["scene"][name])
        np.testing.assert_allclose(m.geom('table_geom').size[:2], s['table_half_size_xy'])
        for side in ('left', 'right'):
            np.testing.assert_allclose(m.body(side+'_link0').pos, s[side+'_base_position'])
        self.assertLess(s['table_half_size_xy'][0], baseline['scene']['table_half_size_xy'][0])
        self.assertLess(s['table_half_size_xy'][1], baseline['scene']['table_half_size_xy'][1])
        self.assertEqual(self.config["training_camera"], baseline["training_camera"])
        self.assertEqual(self.config["recording"], baseline["recording"])
        self.assertEqual(self.config["gesture_control"], baseline["gesture_control"])
        self.assertEqual(self.config["control"], baseline["control"])
        existing = get("tray_raised_placement").config()["scene"]["cameras"]
        for camera in ("front", "side"):
            for field in ("projection", "position", "target"):
                self.assertEqual(s["cameras"][camera][field], existing[camera][field])
            self.assertGreaterEqual(s["cameras"][camera]["fovy"], existing[camera]["fovy"])
        # Task-local auxiliary views may expand to include randomized endpoints.
        # The ACT oblique camera above remains unchanged.
        self.assertEqual(s["cameras"]["top"]["position"], [0., 0., 3.35])
        self.assertGreaterEqual(s["cameras"]["top"]["fovy"], .66)

    def test_pedestals_follow_robot_bases_and_touch_table(self):
        m, s = self.c.model, self.config['scene']
        for side in ('left', 'right'):
            base = m.body(side+'_link0').pos
            support = m.body(side+'_pedestal').pos
            size = m.geom(side+'_pedestal_geom').size
            np.testing.assert_allclose(support[:2], base[:2])
            self.assertAlmostEqual(support[2]+size[2], base[2])
            self.assertAlmostEqual(support[2]-size[2], 0.)
            self.assertAlmostEqual(abs(support[1])-size[1], s['table_half_size_xy'][1])
        # A later parameter edit must move both supports, not just the robots.
        from simulation.mujoco.dual_arm.task_controller import create_model
        cfg = copy.deepcopy(self.config)
        cfg['scene']['left_base_position'] = [-.08, -.63, .37]
        changed = create_model(cfg)
        np.testing.assert_allclose(changed.body('left_pedestal').pos, [-.08, -.63, .185])
        self.assertAlmostEqual(changed.geom('left_pedestal_geom').size[2], .185)
        # Other tasks keep their original supports.
        original = create_model(get('tray_transfer').config())
        np.testing.assert_allclose(original.body('left_pedestal').pos, [0., -.75, .175])

    def test_volume_mass_center_and_inertia_are_consistent(self):
        c = self.c
        ids = sorted(c.object_geoms)
        sizes = c.model.geom_size[ids]
        centers = c.model.geom_pos[ids]
        volumes = 8*np.prod(sizes, axis=1)
        masses = .35*volumes/volumes.sum()
        center = np.sum(masses[:, None]*centers, axis=0)/masses.sum()
        inertia = np.zeros((3, 3))
        for mass, half, position in zip(masses, sizes, centers):
            widths = 2*half
            diagonal = mass/12*np.array([widths[1]**2+widths[2]**2,
                                        widths[0]**2+widths[2]**2,
                                        widths[0]**2+widths[1]**2])
            offset = position-center
            inertia += np.diag(diagonal)+mass*(np.dot(offset, offset)*np.eye(3)-np.outer(offset, offset))
        self.assertAlmostEqual(c.model.body_mass[c.object_id], .35, places=12)
        np.testing.assert_allclose(c.model.body_ipos[c.object_id], center, atol=1e-12)
        np.testing.assert_allclose(np.sort(c.model.body_inertia[c.object_id]), np.linalg.eigvalsh(inertia), atol=1e-12)
        self.assertGreater(float(np.min(c.model.body_inertia[c.object_id])), 0)
        # Empty shell: the COM lies below its bounding-box origin.
        self.assertLess(center[2], 0.)

    def test_seed_reset_platform_height_nonpenetration_and_gravity(self):
        c, s = self.c, self.config["scene"]
        platform = c.model.geom("placement_platform_geom").id
        platform_height = s["placement_platform"]["height_m"]
        platform_top = s["table_top_z"] + platform_height
        np.testing.assert_allclose(2*c.model.geom_size[platform], [.28, .36, platform_height])
        self.assertIn(platform, c.obstacle_geoms)
        self.assertAlmostEqual(c.data.geom_xpos[platform][2]+c.model.geom_size[platform][2], platform_top)
        for seed in range(20):
            c.reset(seed)
            initial = copy.deepcopy(c.initial_state)
            expected = hook(self.config, "initial_state")(self.config, seed)
            np.testing.assert_allclose(initial["object_position"], expected["object_position"], atol=1e-12)
            np.testing.assert_allclose(c.target_position, expected["target_position"], atol=1e-12)
            self.assertAlmostEqual(initial["object_position"][2]-.034, .3502)
            self.assertAlmostEqual(c.target_position[2]-s["object_half_size"][2], platform_top)
            for contact in c.data.contact[:c.data.ncon]:
                pair = {int(contact.geom1), int(contact.geom2)}
                if platform in pair and pair.intersection(c.object_geoms):
                    self.assertGreaterEqual(float(contact.dist), -1e-6)
            c.reset(seed)
            self.assertEqual(c.initial_state, initial)
        for _ in range(24):
            c.step(1/60)
        self.assertTrue(c.contact_control.support_ready)
        self.assertEqual(c.contact_control.support_surface, "table")
        self.assertLess(abs(c.data.xpos[c.object_id][2]-.034-.35), .0015)
        np.testing.assert_array_equal(c.data.xfrc_applied, 0)
        np.testing.assert_array_equal(c.data.qfrc_applied, 0)

    def test_true_platform_clearance_waypoints_have_static_ik(self):
        c, s = self.c, self.config["scene"]
        height = s["table_top_z"]+s["placement_platform"]["height_m"]+s["object_half_size"][2]+s["platform_task"]["transport_clearance_m"]
        for seed in (0, 7, 19):
            c.reset(seed)
            initial = np.asarray(c.initial_state["object_position"])
            quat = c.initial_state["object_quaternion_wxyz"]
            yaw = 2*np.arctan2(quat[3], quat[0])
            for position, angle, label in ((initial, yaw, "initial_grasp"),
                    (np.r_[initial[:2], height], yaw, "clear_platform"),
                    (np.r_[c.target_position[:2], height], c.target_yaw, "above_platform"),
                    (c.target_position, c.target_yaw, "place")):
                for check in _waypoint_checks(c, position, angle, label):
                    self.assertTrue(check["inside_workspace"], (seed, check))
                    self.assertTrue(check["static_ik_pass"], (seed, check))

    def test_randomized_grasp_extremes_fit_layout_and_controller_workspace(self):
        import itertools
        s = self.config['scene']
        center = np.asarray(s['object_position'], dtype=float)
        extent = np.asarray(s['randomization']['object_xy_range_m'])
        yaw_max = s['randomization']['object_yaw_range_rad']
        clear_z = (s['table_top_z'] + s['placement_platform']['height_m']
                   + s['object_half_size'][2] + s['platform_task']['transport_clearance_m'])
        for sx, sy, yaw, z in itertools.product((-1, 1), (-1, 1), (-yaw_max, yaw_max),
                                               (center[2], clear_z, .55)):
            position = np.r_[center[:2] + extent * [sx, sy], z]
            for check in _waypoint_checks(self.c, position, yaw, 'distribution_extreme'):
                self.assertTrue(check['inside_workspace'], check)
                self.assertTrue(check['static_ik_pass'], check)


if __name__ == "__main__":
    unittest.main()
