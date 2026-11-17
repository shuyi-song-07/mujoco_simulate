import unittest

from dataset.common.validate_recordings import _request_modes_match


class SinglePauseAuditTests(unittest.TestCase):
    def test_single_pause_accepts_only_non_releasing_hold(self):
        request = {'command': 'pause', 'reason': 'hand_lost'}
        self.assertTrue(_request_modes_match(request, {
            'interaction_modes': ['HOLD'], 'release_intent': [False]}, push_locked=True))
        for modes, intent in [(['XY_HOLD'], [False]), (['HOLD'], [True]),
                              (['HOLD', 'HOLD'], [False, False])]:
            self.assertFalse(_request_modes_match(request, {
                'interaction_modes': modes, 'release_intent': intent}, push_locked=True))

    def test_motion_and_dual_arm_still_require_matching_modes(self):
        for command in ('continuous_input', 'pause'):
            self.assertFalse(_request_modes_match({'command': command}, {
                'interaction_modes': ['HOLD'], 'release_intent': [False]}))
        self.assertFalse(_request_modes_match({'command': 'continuous_input'}, {
            'interaction_modes': ['HOLD'], 'release_intent': [False]}, push_locked=True))
        self.assertFalse(_request_modes_match({'command': 'pause', 'interaction_modes': ['XY_HOLD']}, {
            'interaction_modes': ['HOLD'], 'release_intent': [False]}, push_locked=True))


if __name__ == '__main__':
    unittest.main()
