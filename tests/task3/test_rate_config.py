import copy
import unittest
from simulation.mujoco.dual_arm.config import load_config


class RateConfigurationTests(unittest.TestCase):
    def test_default_rate_mode_and_invalid_thresholds(self):
        cfg = load_config('config/collection.json')
        self.assertEqual(cfg['contact_control']['control_protocol'], 'gesture-rate-gripper-v3')
        for key in ('open_score_min', 'close_score_min', 'motion_score_min', 'mode_confirm_ms'):
            for value in (None, False, float('nan'), float('inf'), -.1):
                bad = copy.deepcopy(cfg); bad['gesture_control'][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    load_config(bad)
        for key in ('open_score_min', 'close_score_min', 'motion_score_min'):
            for value in (0, 1.01):
                bad = copy.deepcopy(cfg); bad['gesture_control'][key] = value
                with self.assertRaises(ValueError): load_config(bad)
        self.assertEqual(cfg['gesture_control']['mode_confirm_ms'], 100)
        self.assertEqual(cfg['contact_control']['release_confirm_seconds'], 0)
        bad = copy.deepcopy(cfg); bad['gesture_control']['mode_confirm_ms'] = 0
        load_config(bad)
        bad['gesture_control']['gripper_trigger_policy'] = 'level_v1'
        with self.assertRaises(ValueError): load_config(bad)
