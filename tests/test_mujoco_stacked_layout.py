"""Physical stability checks for randomized pick-and-place object stacks."""

import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from simulation.mujoco.record_mujoco_panda import (
    DEFAULT_MODEL_PATH,
    DISTRACTOR_NAMES,
    INITIAL_ARM_QPOS,
    MujocoPandaRecorder,
)


def layout_state(stack_probability: float) -> MujocoPandaRecorder:
    state = object.__new__(MujocoPandaRecorder)
    state.model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH))
    state.data = mujoco.MjData(state.model)
    state.args = SimpleNamespace(
        task_mode="pick_place", table_top_z=0.4, target_top_z=0.416,
        cube_x_min=0.37, cube_x_max=0.68, cube_y_min=-0.24, cube_y_max=0.24,
        plate_x_min=0.37, plate_x_max=0.68,
        plate_y_min=-0.24, plate_y_max=0.24,
        safe_reach_radius=0.72, object_min_separation=0.16,
        stack_probability=stack_probability,
    )
    state.hand_id = state.model.body("hand").id
    state.cube_id = state.model.body("cube").id
    state.cube_joint_id = int(state.model.body_jntadr[state.cube_id])
    state.cube_qpos_address = int(state.model.jnt_qposadr[state.cube_joint_id])
    state.cube_geom_id = int(state.model.body_geomadr[state.cube_id])
    state.distractor_ids = [state.model.body(name).id for name in DISTRACTOR_NAMES]
    state.randomized_geom_ids = [
        state.cube_geom_id,
        *(int(state.model.body_geomadr[body_id]) for body_id in state.distractor_ids),
    ]
    state.target_plate_id = state.model.body("target_plate").id
    state.initial_target_z = float(state.model.body_pos[state.target_plate_id][2])
    state.rng = np.random.default_rng(0)
    return state


class StackedLayoutTest(unittest.TestCase):
    def test_stacked_rigid_boxes_settle_without_penetration_or_jump(self):
        state = layout_state(stack_probability=1.0)
        for seed in range(16):
            with self.subTest(seed=seed):
                state.rng = np.random.default_rng(seed)
                mujoco.mj_resetData(state.model, state.data)
                state.data.qpos[:7] = INITIAL_ARM_QPOS
                state.data.ctrl[:7] = INITIAL_ARM_QPOS
                state.data.ctrl[7] = 255.0
                state.randomize_object_positions()
                mujoco.mj_forward(state.model, state.data)

                layout = {entry["name"]: entry for entry in state.current_layout}
                stacked = [entry for entry in state.current_layout if entry.get("stacked_on")]
                self.assertEqual(len(stacked), 1)
                top = stacked[0]
                base = layout[top["stacked_on"]]
                self.assertIn("cube", (top["name"], base["name"]))
                if base["name"] == "cube":
                    self.assertAlmostEqual(top["yaw_rad"], base["yaw_rad"])
                self.assertTrue(state.layout_reachability["checked"])
                for pose in state.layout_reachability["checked_poses"]:
                    self.assertTrue(state._layout_pose_clear(np.array(pose)))
                self.assertTrue(state._safe_xy(
                    np.asarray(top["xy_m"]), top["half_size_m"]
                ))
                top_id = state.model.body(top["name"]).id
                base_id = state.model.body(base["name"]).id
                initial_top_xy = state.data.xpos[top_id][:2].copy()
                for contact in state.data.contact:
                    if state.cube_geom_id in (contact.geom1, contact.geom2):
                        other_geom = (
                            contact.geom2 if contact.geom1 == state.cube_geom_id
                            else contact.geom1
                        )
                        self.assertIn(int(state.model.geom_bodyid[other_geom]),
                                      (base_id, top_id),
                                      "stacked target must not spawn inside the Panda gripper")

                # The rotated top footprint fits entirely on the lower box.
                yaw_delta = top["yaw_rad"] - base["yaw_rad"]
                footprint_half = top["half_size_m"] * (
                    abs(np.cos(yaw_delta)) + abs(np.sin(yaw_delta))
                )
                world_offset = np.asarray(top["xy_m"]) - np.asarray(base["xy_m"])
                c, s = np.cos(base["yaw_rad"]), np.sin(base["yaw_rad"])
                local_offset = np.array([
                    c * world_offset[0] + s * world_offset[1],
                    -s * world_offset[0] + c * world_offset[1],
                ])
                self.assertLess(float(np.max(np.abs(local_offset))) + footprint_half,
                                base["half_size_m"] - 0.003)

                for body_id, geom_id in zip(
                    [state.cube_id, *state.distractor_ids], state.randomized_geom_ids
                ):
                    joint_id = int(state.model.body_jntadr[body_id])
                    self.assertEqual(state.model.jnt_type[joint_id], mujoco.mjtJoint.mjJNT_FREE)
                    half = float(state.model.geom_size[geom_id][0])
                    self.assertAlmostEqual(state.model.geom_rbound[geom_id], np.sqrt(3) * half)
                    np.testing.assert_allclose(state.model.geom_aabb[geom_id][3:], half)

                for step in range(1000):
                    mujoco.mj_step(state.model, state.data)
                    if step % 16 == 0:
                        state.enforce_layout_support()

                top_z = float(state.data.xpos[top_id][2])
                base_z = float(state.data.xpos[base_id][2])
                gap = top_z - top["half_size_m"] - base_z - base["half_size_m"]
                self.assertLess(abs(gap), 0.002)
                self.assertLess(float(np.linalg.norm(
                    state.data.xpos[top_id][:2] - initial_top_xy
                )), 0.005)
                for body_id, geom_id in zip(
                    [state.cube_id, *state.distractor_ids], state.randomized_geom_ids
                ):
                    rotation = state.data.xmat[body_id].reshape(3, 3)
                    extent = float(np.abs(rotation[2]) @ state.model.geom_size[geom_id])
                    if body_id != top_id:
                        self.assertGreaterEqual(
                            float(state.data.xpos[body_id][2]) - extent, 0.398
                        )

    def test_plain_layout_and_resized_bounds_remain_stable_after_reset(self):
        state = layout_state(stack_probability=0.0)
        for seed in range(16):
            with self.subTest(seed=seed):
                state.rng = np.random.default_rng(seed)
                mujoco.mj_resetData(state.model, state.data)
                state.data.qpos[:7] = INITIAL_ARM_QPOS
                state.data.ctrl[:7] = INITIAL_ARM_QPOS
                state.data.ctrl[7] = 255.0
                state.randomize_object_positions()
                mujoco.mj_forward(state.model, state.data)
                self.assertFalse(any(
                    entry.get("stacked_on") for entry in state.current_layout
                ))
                initial_xy = {
                    body_id: state.data.xpos[body_id][:2].copy()
                    for body_id in [state.cube_id, *state.distractor_ids]
                }
                for entry in state.current_layout:
                    self.assertAlmostEqual(
                        entry["z_m"],
                        state.args.table_top_z + entry["half_size_m"] + 0.002,
                    )
                for body_id, geom_id in zip(
                    [state.cube_id, *state.distractor_ids], state.randomized_geom_ids
                ):
                    half = float(state.model.geom_size[geom_id][0])
                    self.assertAlmostEqual(
                        state.model.geom_rbound[geom_id], np.sqrt(3) * half
                    )
                    np.testing.assert_allclose(state.model.geom_aabb[geom_id][3:], half)
                for step in range(1000):
                    mujoco.mj_step(state.model, state.data)
                    if step % 16 == 0:
                        state.enforce_layout_support()
                for body_id, geom_id in zip(
                    [state.cube_id, *state.distractor_ids], state.randomized_geom_ids
                ):
                    rotation = state.data.xmat[body_id].reshape(3, 3)
                    extent = float(np.abs(rotation[2]) @ state.model.geom_size[geom_id])
                    self.assertGreaterEqual(
                        float(state.data.xpos[body_id][2]) - extent, 0.398
                    )
                    self.assertLess(float(np.linalg.norm(
                        state.data.xpos[body_id][:2] - initial_xy[body_id]
                    )), 0.005)

        # A subsequent reset can change the same geometry from a large base
        # to a small independent box without leaving stale collision bounds.
        state.args.stack_probability = 1.0
        state.rng = np.random.default_rng(19)
        state.randomize_object_positions()
        self.assertTrue(any(entry.get("stacked_on") for entry in state.current_layout))
        state.args.stack_probability = 0.0
        mujoco.mj_resetData(state.model, state.data)
        state.rng = np.random.default_rng(20)
        state.randomize_object_positions()
        for geom_id in state.randomized_geom_ids:
            half = float(state.model.geom_size[geom_id][0])
            self.assertAlmostEqual(state.model.geom_rbound[geom_id], np.sqrt(3) * half)
            np.testing.assert_allclose(state.model.geom_aabb[geom_id][3:], half)


if __name__ == "__main__":
    unittest.main()
