"""Control checks for the 9D ACT rollout using the real Panda model."""

import math
import unittest

import mujoco
import numpy as np

from evaluation.act_variation_rollout import (
    DEFAULT_SCENE,
    RolloutPanda,
    apply_cartesian_action,
    checkpoint_camera_names,
    placement_status,
)
from simulation.mujoco.record_mujoco_panda import parse_args, rotation_matrix_to_rpy


class VariationRolloutControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = parse_args(["--model-path", str(DEFAULT_SCENE), "--random-seed", "10000"])
        cls.env = RolloutPanda(args)

    def setUp(self):
        self.env.reset_seed(10000)

    def action(self, yaw=0.6):
        env = self.env
        base = env.data.xmat[env.base_id].reshape(3, 3)
        position = base.T @ (env.data.xpos[env.hand_id] - env.data.xpos[env.base_id])
        c, s = math.cos(yaw), math.sin(yaw)
        yaw_rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        rotation = env.initial_hand_rotation_in_base @ yaw_rotation
        return np.r_[position + [0.25, 0.1, 0], rotation_matrix_to_rpy(rotation), 255.0]

    def test_joint_and_yaw_commands_stay_rate_limited(self):
        env = self.env
        action = self.action()
        for _ in range(80):
            before = env.data.ctrl.copy()
            apply_cartesian_action(env, action)
            self.assertLessEqual(np.max(np.abs(env.data.ctrl[:7] - before[:7])), 0.030001)
            self.assertLessEqual(abs(env.data.ctrl[env.gripper_yaw_actuator_id]
                                     - before[env.gripper_yaw_actuator_id]), 0.020001)
            self.assertLessEqual(np.max(np.abs(env.data.ctrl[:7] - env.data.qpos[:7])), 0.120001)
            for _ in range(33):
                mujoco.mj_step(env.model, env.data)
        self.assertAlmostEqual(env.data.ctrl[env.gripper_yaw_actuator_id], 0.6, places=3)

    def test_equivalent_roll_branches_produce_same_command(self):
        action = self.action()
        apply_cartesian_action(self.env, action)
        first = self.env.data.ctrl.copy()
        self.env.reset_seed(10000)
        action[3] += 2 * math.pi
        apply_cartesian_action(self.env, action)
        np.testing.assert_allclose(self.env.data.ctrl, first, atol=1e-9)

    def test_stable_plate_contact_counts_beyond_old_center_threshold(self):
        env = self.env
        plate_xy = env.data.xpos[env.target_plate_id, :2].copy()
        plate_geom = int(env.model.body_geomadr[env.target_plate_id])
        plate_top = float(
            env.model.body_pos[env.target_plate_id, 2]
            + env.model.geom_size[plate_geom, 1]
        )
        cube_half_height = float(env.model.geom_size[env.cube_geom_id, 2])
        address = int(env.cube_qpos_address)
        env.data.qpos[address:address + 3] = [
            plate_xy[0] + 0.06, plate_xy[1], plate_top + cube_half_height + 0.003
        ]
        env.data.qpos[address + 3:address + 7] = [1, 0, 0, 0]
        dof = int(env.model.jnt_dofadr[env.cube_joint_id])
        env.data.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(env.model, env.data)
        for _ in range(1000):
            mujoco.mj_step(env.model, env.data)

        offset = np.linalg.norm(
            env.data.xpos[env.cube_id, :2] - env.data.xpos[env.target_plate_id, :2]
        )
        self.assertGreater(offset, env.args.target_radius)
        self.assertTrue(placement_status(env)["on_plate"])


class CheckpointInputTests(unittest.TestCase):
    def test_camera_names_follow_checkpoint_order(self):
        class Config:
            input_features = {
                "observation.state": object(),
                "observation.images.overview": object(),
                "observation.images.camera_2": object(),
                "observation.images.camera_3": object(),
            }

        self.assertEqual(
            checkpoint_camera_names(Config()),
            ("overview", "camera_2", "camera_3"),
        )

    def test_unknown_checkpoint_camera_is_rejected(self):
        class Config:
            input_features = {"observation.images.wrist": object()}

        with self.assertRaisesRegex(ValueError, "unsupported cameras"):
            checkpoint_camera_names(Config())


if __name__ == "__main__":
    unittest.main()
