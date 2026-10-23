"""Camera transition checks for the red cube grasp approach."""

import unittest
from types import SimpleNamespace

import numpy as np

from simulation.mujoco.record_mujoco_panda import (
    MujocoPandaRecorder,
    gripper_centered_over_cube,
)


class GripperApproachViewTest(unittest.TestCase):
    def test_requires_centered_xy_and_grasp_height(self):
        cube = np.array([0.5, 0.0, 0.43])
        half_size = 0.03
        self.assertFalse(gripper_centered_over_cube(np.array([0.54, 0.0, 0.56]), cube, half_size))
        self.assertFalse(gripper_centered_over_cube(np.array([0.5, 0.0, 0.70]), cube, half_size))
        self.assertTrue(gripper_centered_over_cube(np.array([0.51, 0.0, 0.56]), cube, half_size))
        self.assertFalse(gripper_centered_over_cube(np.array([0.515, 0.0, 0.56]), cube, half_size))

    def test_switch_waits_for_stable_position_and_clears_on_exit(self):
        positions = np.array([
            [0.5, 0.0, 0.43],  # cube
            [0.5, 0.0, 0.56],  # hand
            [0.8, 0.0, 0.4],   # plate
        ])
        state = SimpleNamespace(
            cube_id=0,
            hand_id=1,
            target_plate_id=2,
            cube_geom_id=0,
            data=SimpleNamespace(xpos=positions, time=0.0),
            model=SimpleNamespace(geom_size=np.array([[0.03, 0.03, 0.03]])),
            args=SimpleNamespace(target_radius=0.05, target_exit_margin=0.02),
            gripper_above_cube_view=False,
            gripper_above_cube_since=None,
            is_holding_cube=lambda: False,
        )
        update = lambda: MujocoPandaRecorder.update_gripper_approach_view(state)
        update()
        self.assertFalse(state.gripper_above_cube_view)
        state.data.time = 0.049
        update()
        self.assertFalse(state.gripper_above_cube_view)
        state.data.time = 0.05
        update()
        self.assertTrue(state.gripper_above_cube_view)
        positions[1, 0] = 0.55
        update()
        self.assertFalse(state.gripper_above_cube_view)
        self.assertIsNone(state.gripper_above_cube_since)


if __name__ == "__main__":
    unittest.main()
