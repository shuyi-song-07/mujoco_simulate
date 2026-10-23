"""Ensure variation policies use only the overview camera from multiview data."""

import unittest

from training.ACT.train_act_variation_baseline import policy_input_features as act_inputs
from training.Diffusion.train_diffusion_variation_baseline import policy_input_features as diffusion_inputs


class VariationCameraInputTests(unittest.TestCase):
    def test_act_uses_state_and_overview_only(self):
        self.assert_camera_filter(act_inputs)

    def test_diffusion_uses_state_and_overview_only(self):
        self.assert_camera_filter(diffusion_inputs)

    def assert_camera_filter(self, select_inputs):
        features = {
            "observation.state": object(),
            "observation.images.overview": object(),
            "observation.images.camera_2": object(),
            "observation.images.camera_3": object(),
            "observation.gesture": object(),
        }

        self.assertEqual(
            set(select_inputs(features)),
            {"observation.state", "observation.images.overview"},
        )


if __name__ == "__main__":
    unittest.main()
