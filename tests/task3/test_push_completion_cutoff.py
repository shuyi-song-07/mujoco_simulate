"""A recorded push success owns its 0.5s tail; grasp release stays physical."""
import copy
import unittest
from unittest.mock import patch

from test_push_task_evidence import push_rows
from test_terminal_tail import release_rows
from dataset.common.curation_rules import select_rows, load_config, derive_episode
from dataset.common.task_evidence import build_task_evidence


class PushCompletionCutoffTests(unittest.TestCase):
    def test_half_second_after_first_success_not_last_settled_pose(self):
        rows = push_rows()
        for r in rows[18:]:
            r['evidence']['task_completion']['verified'] = False
            r['evidence']['object']['angular_speed_radps'] = .1
        out = select_rows(rows)
        end = out['terminal_release']
        self.assertIsNotNone(end)
        self.assertEqual(end['completion_raw_frame'], 16)
        self.assertEqual(end['last_kept_raw_frame'], 31)
        self.assertTrue(all(r['keep'] for r in out['rows'][16:32]))
        self.assertFalse(any(r['keep'] for r in out['rows'][32:]))

    def test_uses_original_time_instead_of_fifteen_remaining_rows(self):
        rows = push_rows()
        for i, r in enumerate(rows):
            r['original_time_s'] = i * .04
            r['evidence']['source']['observation_period_s'] = .04
        end = select_rows(rows)['terminal_release']
        self.assertEqual(end['last_kept_raw_frame'], 28)
        self.assertLessEqual(rows[28]['original_time_s'], rows[16]['original_time_s']+.5)

    def test_grasp_opening_completion_and_three_frame_tail_are_unchanged(self):
        out = select_rows(release_rows(right_open_at=21))
        self.assertEqual(out['terminal_release']['completion_raw_frame'], 25)
        self.assertEqual(out['terminal_release']['last_kept_raw_frame'], 28)

    def test_short_recording_is_not_padded_to_half_second(self):
        rows = push_rows()[:20]
        out = select_rows(rows)
        self.assertEqual(len(out['rows']), 20)
        self.assertEqual(out['terminal_release']['last_kept_raw_frame'], 19)
        self.assertEqual(out['terminal_release']['retained_post_frames'], 3)

    def test_exact_backend_success_time_between_images_controls_cutoff(self):
        rows = push_rows()
        rows[16]['evidence']['task_completion']['recorded_success_time_s'] = 15.5 / 30
        end = select_rows(rows)['terminal_release']
        self.assertEqual(end['clock_source'], 'recorded_success_time')
        self.assertEqual(end['last_kept_raw_frame'], 30)

    def test_stale_or_future_completion_time_cannot_shift_cutoff(self):
        for stamp in (-100., 100.):
            with self.subTest(stamp=stamp):
                rows = push_rows()
                rows[16]['evidence']['task_completion']['recorded_success_time_s'] = stamp
                end = select_rows(rows)['terminal_release']
                self.assertEqual(end['clock_source'], 'first_verified_done_sample')
                self.assertEqual(end['last_kept_raw_frame'], 31)

    def test_custom_push_budget_does_not_change_grasp_budget(self):
        cfg = load_config()
        cfg['waits']['post_push_success_seconds'] = 0.
        self.assertEqual(select_rows(push_rows(), cfg)['terminal_release']['last_kept_raw_frame'], 16)
        self.assertEqual(select_rows(release_rows(right_open_at=21), cfg)['terminal_release']['last_kept_raw_frame'], 28)

    def test_single_arm_requires_recorded_coverage_dwell_evidence(self):
        criterion = 'upstream-push-t-coverage-v1'
        for elapsed, expected in ((.2, False), (.5, True)):
            with self.subTest(elapsed=elapsed):
                row = {'evidence': {'push_alignment': {'criterion': criterion}}}
                adapter = type('Adapter', (), {'build_evidence': lambda self, ep, cfg: {'rows': [row]}})()
                episode = {'metadata': {'config': {'task_binding': {'task_id': 'push_t'},
                    'scene': {'push_t': {'dwell_seconds': .5}}}},
                    'frames': [{'status': {'push_success_evidence': {'criterion': criterion,
                        'task_success': True, 'dwell_elapsed_s': elapsed}}}]}
                with patch('tasks.registry.hook', return_value=lambda: adapter):
                    evidence = build_task_evidence(episode, {})['rows'][0]['evidence']
                self.assertEqual(evidence['task_completion']['verified'], expected)
                self.assertEqual(evidence['task_semantics']['completion'], 'task_monitor')

    def test_derive_uses_the_registered_task_evidence_not_generic_facts(self):
        adapter = type('Adapter', (), {'build_evidence': lambda self, ep, cfg: {'rows': push_rows()}})()
        episode = {'metadata': {'config': {'task_binding': {'task_id': 'h_frame_push_align'}}}}
        with patch('tasks.registry.hook', return_value=lambda: adapter), \
                patch('dataset.task3.frame_evidence.build_frame_evidence', side_effect=AssertionError('generic bypass')):
            result = derive_episode(episode)
        self.assertEqual(result['terminal_release']['last_kept_raw_frame'],31)


if __name__ == '__main__': unittest.main()
