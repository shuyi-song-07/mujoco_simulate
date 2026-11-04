"""HTML reports frozen physical thresholds without changing outcome evidence."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from rollout.rollout import summarize


class RolloutReportingTests(unittest.TestCase):
    def report(self, minimum_lift_m, *, settings_present=True):
        metadata = {'seeds': [1], 'mode': 'preview', 'run': 'structural_fixture',
                    'checkpoint': 'last', 'max_seconds': 60, 'protocol_signature': None,
                    'settings': {'minimum_lift_m': minimum_lift_m, 'success_open_fraction': .5}}
        row = {'seed': 1, 'success': False, 'failure_reason': 'timeout', 'sim_seconds': 60.,
               'final_xy_error_m': .1, 'intervention_fraction': 0., 'video': 'oblique.mp4',
               'evidence': {'min_lift_m': minimum_lift_m}}
        if not settings_present:
            del metadata['settings']['minimum_lift_m']
        original = deepcopy((metadata, row))
        with tempfile.TemporaryDirectory(prefix='rollout_report_fixture_') as directory:
            summary = summarize(Path(directory), metadata, [row])
            report = (Path(directory)/'report.html').read_text(encoding='utf-8')
        self.assertEqual((metadata,row), original)
        self.assertEqual((summary['successes'],summary['failures']), (0,1))
        return report

    def test_html_uses_frozen_lift_requirement_in_centimeters(self):
        for meters, text in ((.03, '抬高至少3厘米'), (.045, '抬高至少4.5厘米')):
            with self.subTest(minimum_lift_m=meters):
                report = self.report(meters)
                self.assertIn(text, report)
                self.assertNotIn('抬高至少2厘米', report)

    def test_html_uses_saved_physical_evidence_when_setting_is_absent(self):
        self.assertIn('抬高至少2.5厘米', self.report(.025, settings_present=False))


if __name__ == '__main__':
    unittest.main()
