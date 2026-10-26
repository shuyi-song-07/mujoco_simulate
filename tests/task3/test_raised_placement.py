"""S3 scene contracts and real contact transport; no policy/human success claim."""
import copy
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

from simulation.mujoco.dual_arm.config import load_config
from simulation.mujoco.dual_arm.task_scenes import scenario_config, placement_geometry
from simulation.mujoco.dual_arm.task_controller import DualArmTaskController, create_model
from simulation.mujoco.dual_arm.panda_arm_controller import matrix_to_rpy
from simulation.mujoco.dual_arm.rigid_control import rz
from simulation.mujoco.dual_arm.render_workers import apply_snapshot
from rollout.executor import PolicyExecutor
from rollout.protocol import SuccessMonitor, sample_initial_state


def raised_config():
    return scenario_config(load_config('config/collection.json'), 'S3')


class RaisedPlacementTests(unittest.TestCase):
    def test_reset_geometry_support_and_snapshot_roundtrip(self):
        c = DualArmTaskController(raised_config())
        platform = c.model.geom('placement_platform_geom').id
        np.testing.assert_allclose(c.model.geom_size[platform]*2, [.25, .33, .05])
        self.assertIn(platform, c.obstacle_geoms)
        for seed in range(50):
            c.reset(seed)
            np.testing.assert_allclose(c.target_position, [.15, 0., .422])
            np.testing.assert_allclose(sample_initial_state(c.config, seed)['object_position'], c.initial_state['object_position'])
            np.testing.assert_allclose(sample_initial_state(c.config, seed)['target_position'], c.target_position)
            for con in c.data.contact[:c.data.ncon]:
                self.assertFalse(platform in (con.geom1, con.geom2) and con.dist < -.0001)
        for _ in range(24):
            c.step(1/60)
        self.assertEqual(c.contact_control.support_surface, 'table')
        self.assertTrue(c.contact_control.support_ready)
        self.assertFalse(placement_geometry(c)['contained'])
        snap = c.snapshot()
        model = create_model(c.config)
        data = mujoco.MjData(model)
        apply_snapshot(model, data, snap)
        np.testing.assert_allclose(model.body_pos, c.model.body_pos)
        np.testing.assert_allclose(data.xpos, c.data.xpos, atol=1e-8)
        self.assertEqual(len(snap['state']), 16)
        self.assertEqual(len(snap['action']), 14)

    def test_invalid_platform_configuration(self):
        for field, value in [('height_m', float('nan')), ('height_m', -.05),
                             ('extra_size_xy_m', [.01]), ('extra_size_xy_m', [True, .01])]:
            cfg = raised_config()
            cfg['scene']['placement_platform'][field] = value
            with self.assertRaises(ValueError):
                load_config(cfg)

    def test_off_center_tray_and_old_table_are_not_target_support(self):
        # Synthetic stationary pose fixtures, NOT a transport demonstration.
        c = DualArmTaskController(raised_config())
        adr = c.object_qpos_adr
        c.data.qpos[adr:adr+3] = c.target_position
        c.data.qpos[adr+3:adr+7] = [np.cos(c.target_yaw/2), 0, 0, np.sin(c.target_yaw/2)]
        mujoco.mj_forward(c.model, c.data)
        self.assertTrue(placement_geometry(c)['contained'])
        c.data.qpos[adr] += .02  # Inside old 45 mm radius but outside this platform.
        mujoco.mj_forward(c.model, c.data)
        self.assertFalse(placement_geometry(c)['contained'])
        c.reset()
        for _ in range(24):
            c.step(1/60)
        status = c.get_status()
        self.assertTrue(status['physical_contact']['support_ready'])
        result = SuccessMonitor(c.config, c.initial_state).update(c)
        self.assertFalse(result['task_success'])

    def test_real_grasp_lift_transfer_land_release(self):
        cfg = raised_config()
        cfg['scene']['randomization'] = dict(object_xy_range_m=[0., 0.], target_xy_range_m=[0., 0.], object_yaw_range_rad=0.)
        c = DualArmTaskController(cfg)
        c.set_episode_timeout_enabled(False)
        executor = PolicyExecutor(c)
        monitor = SuccessMonitor(cfg, c.initial_state, minimum_open_fraction=.5)
        phases = []

        def run(action, seconds, phase):
            for _ in range(round(seconds*60)):
                result = executor.apply(action)
                self.assertTrue(result['accepted'], (phase, result))
                c.step(1/60)
                result = monitor.update(c)
                self.assertNotEqual(str(c.task_state), 'FAIL', (phase, c.get_status()))
                if result['task_success'] or str(c.task_state) == 'DONE':
                    break
            phases.append({'phase': phase, 'status': c.get_status(), 'scorer': result})

        action = c.action_vector_m().astype(float)
        action[[0, 7]] = -.16
        action[[1, 8]] = [-.22, .22]
        run(action, 2., 'approach_above')
        action[[2, 9]] = .372
        run(action, 2., 'descend')
        action[[6, 13]] = 0.
        run(action, 1.5, 'close')
        self.assertTrue(all(c.contact_control.stable.values()), c.get_status())
        refs = copy.deepcopy(c.contact_control.reference)

        def target(center, yaw):
            action = c.action_vector_m().astype(float)
            rotation = rz(yaw)
            for arm, offset, ref in zip(c.arms, (0, 7), refs):
                action[offset:offset+3] = arm.world_to_task(np.asarray(center)+rotation @ ref[0])
                action[offset+3:offset+6] = matrix_to_rpy(arm.task_rotation.T @ rotation @ ref[1])
            return action

        center = c.data.xpos[c.object_id].copy()
        center[2] += .10
        run(target(center, 0.), 2., 'lift_clear_of_platform')
        center[:2] = c.target_position[:2]
        run(target(center, 0.), 3., 'transfer')
        run(target(center, c.target_yaw), 1.5, 'align_yaw')
        run(target(c.target_position, c.target_yaw), 2., 'land')
        self.assertEqual(c.contact_control.support_surface, 'placement_platform')
        self.assertTrue(c.contact_control.support_ready, c.get_status())
        self.assertTrue(placement_geometry(c)['contained'])
        action = c.action_vector_m().astype(float)
        action[[6, 13]] = .08
        run(action, 2., 'release')
        self.assertTrue(phases[-1]['scorer']['task_success'], phases[-1])
        # The independent half-open scorer can succeed before the historical
        # collection FSM's full-open stability timer. Verify both separately.
        for _ in range(120):
            if str(c.task_state) == 'DONE':
                break
            c.step(1/60)
        self.assertEqual(str(c.task_state), 'DONE', c.get_status())
        phases.append({'phase': 'collection_done', 'status': c.get_status(), 'scorer': monitor.update(c)})
        self.assertEqual(c.metrics.collision_count, 0)
        output = Path('.local/checks/raised-placement')
        output.mkdir(parents=True, exist_ok=True)
        (output/'physical-transport.json').write_text(json.dumps({'kind':'scripted_physics_not_policy_or_human', 'phases':phases}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    unittest.main()
