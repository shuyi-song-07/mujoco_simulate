"""Push completion is task evidence, never inferred from open grippers or DONE."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from test_intent_curation import evidence_rows, execution
from dataset.common.curation_rules import select_rows, load_config
from dataset.h_frame_push_align.evidence import PushAlignmentEvidenceAdapter
from dataset.push_r.evidence import PushREvidenceAdapter


def push_rows():
    rows = evidence_rows(40)
    for i, r in enumerate(rows):
        f = r['evidence']
        f['task_semantics'] = {'contact': 'nonprehensile_push', 'completion': 'task_monitor'}
        f['task_completion'] = {'kind': 'push_alignment', 'verified': i >= 16}
        f['task'].update(done=i >= 16, success=i >= 16)
        for s in ('left', 'right'):
            f['grippers'][s].update(actual_width_m=0., command_width_m=0., goal_width_m=0.)
            f['contact'][s].update(any_finger=True, double_finger=True, stable=False)
    return rows


class PushTaskEvidenceTests(unittest.TestCase):
    def test_closed_contact_done_crops_only_verified_tail(self):
        out = select_rows(push_rows())
        self.assertEqual(out['terminal_release']['completion_raw_frame'], 16)
        self.assertEqual(out['terminal_release']['last_kept_raw_frame'], 19)
        self.assertFalse(any(r['keep'] for r in out['rows'][20:]))
        self.assertEqual(out['rows'][16]['policy_reason'], ['verified_task_complete'])

    def test_missing_evidence_or_failed_source_never_crops_tail(self):
        for failure in ('no_evidence', 'not_done', 'not_success', 'source_invalid', 'source_missing', 'moving', 'servo_pending'):
            rows = push_rows()
            for r in rows:
                f = r['evidence']
                if failure == 'no_evidence': f.pop('task_completion')
                if failure == 'not_done': f['task']['done'] = False
                if failure == 'not_success': f['task']['success'] = False
                if failure == 'source_invalid': f['source']['valid'] = False
                if failure == 'source_missing': f['source']['available'] = False
                if failure == 'moving': f['object']['linear_speed_mps'] = 1.
                if failure == 'servo_pending': f['grippers']['right']['goal_width_m'] = .08
            self.assertIsNone(select_rows(rows)['terminal_release'], failure)

    def test_real_close_adjustment_ends_without_fictitious_grasp(self):
        rows = push_rows()
        for r in rows: r['evidence']['task_completion']['verified'] = False
        execution(rows, 2, side='both')
        out = select_rows(rows)
        self.assertEqual(len(out['execution_intervals']), 2)
        for interval in out['execution_intervals']:
            self.assertEqual(interval['end_reason'], 'push_gripper_closed_and_stable')
            self.assertFalse(interval['review_required'])
        self.assertTrue(out['rows'][2]['grasp_protected'])
        self.assertFalse(out['rows'][30]['grasp_protected'])

    def test_later_opening_is_not_cut_by_an_earlier_done(self):
        rows = push_rows()
        execution(rows, 24, side='both', opening=True)
        for i, r in enumerate(rows[24:], 24):
            for s in ('left', 'right'):
                r['evidence']['grippers'][s].update(goal_width_m=.08,
                    actual_width_m=.04 if i < 28 else .08)
        terminal = select_rows(rows)['terminal_release']
        self.assertGreaterEqual(terminal['completion_raw_frame'], 28)
        self.assertLess(terminal['completion_raw_frame'], 40)

    def test_adapter_requires_recorded_physical_monitor_and_current_pose(self):
        for task, cls in (('h_frame_push_align', PushAlignmentEvidenceAdapter), ('push_r', PushREvidenceAdapter)):
            adapter = cls()
            row = evidence_rows(1)[0]
            row['evidence']['object']['rotation'] = np.eye(3).tolist()
            conditions = {k: True for k in ('position_xy', 'yaw', 'upright', 'inside_table',
                'table_support', 'linear_speed', 'angular_speed', 'not_failed')}
            physical = {'support_checks': {'table_load': True, 'clearance': True, 'penetration': True},
                'task_completion': {'task_success_criterion': adapter.criterion, 'task_success': True,
                    'final_conditions': conditions, 'alignment_stable_seconds': .5,
                    'completion_requirements': {'position_tolerance_m': .01, 'yaw_tolerance_rad': .035,
                        'max_tilt_rad': .1, 'stable_seconds': .5}}}
            ep = {'metadata': {'task_id': task, 'schema_version': 'task3-papera-v2'},
                  'frames': [{'status': {'target_position': [0., 0., .4], 'target_yaw_rad': 0.,
                                        'physical_contact': physical}}]}
            for invalid in (None, 'missing_monitor', 'bad_condition', 'wrong_criterion', 'no_support', 'pose_moved'):
                e, r = copy.deepcopy(ep), copy.deepcopy(row)
                p = e['frames'][0]['status']['physical_contact']
                if invalid == 'missing_monitor': p.pop('task_completion')
                if invalid == 'bad_condition': p['task_completion']['final_conditions']['yaw'] = False
                if invalid == 'wrong_criterion': p['task_completion']['task_success_criterion'] = 'wrong'
                if invalid == 'no_support': p['support_checks']['table_load'] = False
                if invalid == 'pose_moved': r['evidence']['object']['position'][0] = .1
                with patch('dataset.task3.frame_evidence.build_frame_evidence', return_value={'rows': [r]}):
                    out = adapter.build_evidence(e, load_config())
                self.assertEqual(out['rows'][0]['evidence']['task_completion']['verified'], invalid is None)

    def test_rollout_settings_use_task_criterion_and_push_milestones(self):
        import json
        from pathlib import Path
        from types import SimpleNamespace
        from tasks.stages import stage_adapter
        root = Path(__file__).resolve().parents[2]
        for task in ('h_frame_push_align', 'push_r'):
            a = stage_adapter(task, 'rollout')
            settings = json.loads((root/'tasks'/task/'evaluation.json').read_text(encoding='utf-8'))
            a.validate_success(settings)
            bad = settings | {'success_criterion': 'papera-physical-half-open-v1'}
            with self.assertRaises(ValueError): a.validate_success(bad)
            stages = {}
            a.update_stage_progress(SimpleNamespace(contacts={'left_fingers': [True, False]}),
                {'final_conditions': {'position_xy': True, 'yaw': False}, 'task_success': False}, stages, 1.)
            self.assertEqual(stages, {'left_push_contact': 1., 'position_aligned': 1.})


if __name__ == '__main__':
    unittest.main()
