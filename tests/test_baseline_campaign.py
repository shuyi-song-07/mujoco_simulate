"""Supervisor-only regression checks; no training or MuJoCo is started."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_baseline_campaign as campaign


class CampaignTests(unittest.TestCase):
    def config_pair(self):
        act = {'test_seeds': list(range(50)), 'max_seconds': 60, 'observation_hz': 30, 'control_hz': 60,
               'execution': 'temporal_ensemble', 'n_action_steps': 1, 'checkpoint_selection': 'best_validation_prior_action_l1'}
        dp = dict(act, policy_algorithm='diffusion_policy', execution='chunked', n_action_steps=8,
                  checkpoint_selection='best_validation_ema_noise_mse')
        return act, dp

    def test_same_case_budget_and_distinct_execution(self):
        act, dp = self.config_pair()
        campaign.validate_evaluations(act, dp)
        for field, value in (('max_seconds', 180), ('test_seeds', list(range(1, 51))), ('n_action_steps', 1)):
            bad = dict(dp, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                campaign.validate_evaluations(act, bad)

    def test_attach_requires_exact_module_and_output(self):
        run = campaign.ROOT / 'outputs/training/测试实验'
        valid = ['python.exe', '-m', 'training.papera', '--output', str(run)]
        campaign.validate_act_command(valid, run)
        for bad in ([*valid[:2], 'training.diffusion', *valid[3:]], [*valid[:-1], str(run) + '2']):
            with self.assertRaises(ValueError):
                campaign.validate_act_command(bad, run)

    def test_run_uses_frozen_pack_and_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'steps': 40000, 'batch_size': 16, 'model': {'chunk_size': 30}}
            plan = {'task_id': 'test', 'pack': str(root / 'pack'), 'training_configs': {'act': config}}
            source = {'variant': 'baseline', 'task_id': 'test', 'pack': plan['pack'], 'config': config}
            campaign.write(root / 'run.json', {'provenance': source})
            self.assertFalse(campaign.training_complete(root, 'act', plan))
            for field, value in (('variant', 'gesture'), ('pack', str(root / 'another_pack'))):
                campaign.write(root / 'run.json', {'provenance': dict(source, **{field: value})})
                with self.assertRaises(ValueError):
                    campaign.validate_run(root, 'act', plan)
            changed = copy.deepcopy(source)
            changed['config']['steps'] = 50000
            campaign.write(root / 'run.json', {'provenance': changed})
            with self.assertRaises(ValueError):
                campaign.validate_run(root, 'act', plan)

    def test_incomplete_training_cannot_reach_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'steps': 40000}
            plan = {'task_id': 'test', 'pack': str(root / 'pack'), 'training_configs': {'dp': config}}
            campaign.write(root / 'run.json', {'algorithm': 'diffusion_policy', 'config': config,
                'task_id': 'test', 'pack_path': plan['pack']})
            campaign.write(root / 'completion.json', {'status': 'failed', 'step': 1000, 'budget_steps': 40000, 'engineering': False})
            with self.assertRaises(ValueError):
                campaign.training_complete(root, 'dp', plan)

    def dp_cli_run(self, root):
        """The DP CLI materializes its explicit --variant in the frozen config."""
        expected = {'steps': 40000, 'batch_size': 16, 'model': {
            'prediction_horizon': 30, 'state_dim': 16, 'action_dim': 14,
            'down_dims': [512, 1024, 2048]}}
        actual = copy.deepcopy(expected)
        actual['variant'] = 'baseline'
        actual['model']['variant'] = 'baseline'
        plan = {'task_id': 'test', 'pack': str(root / 'pack'),
                'training_configs': {'dp': expected}}
        run = {'algorithm': 'diffusion_policy', 'variant': 'baseline',
               'task_id': 'test', 'pack_path': plan['pack'], 'config': actual}
        campaign.write(root / 'run.json', run)
        return plan, run

    def test_dp_cli_materialized_baseline_variant_matches_the_frozen_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, run = self.dp_cli_run(root)
            original_plan = copy.deepcopy(plan)
            self.assertEqual(campaign.validate_run(root, 'dp', plan), run)
            self.assertEqual(plan, original_plan)  # No rewriting an active plan.

    def test_dp_variant_normalization_does_not_accept_gesture_or_other_model_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, run = self.dp_cli_run(root)
            mutations = [
                lambda r: r['config'].update(variant='gesture'),
                lambda r: r['config']['model'].update(variant='gesture'),
                lambda r: r['config']['model'].update(prediction_horizon=16),
                lambda r: r['config']['model'].update(down_dims=[256, 512, 1024]),
                lambda r: r.update(variant='gesture'),
            ]
            for index, mutate in enumerate(mutations):
                changed = copy.deepcopy(run)
                mutate(changed)
                campaign.write(root / 'run.json', changed)
                with self.subTest(mutation=index), self.assertRaises(ValueError):
                    campaign.validate_run(root, 'dp', plan)

    def test_protocols_require_same_source_and_initial_states(self):
        a = {'seeds': list(range(50)), 'source': {'scene': 'frozen'},
             'preflight': {'cases': [{'seed': 1, 'initial_state': {'x': 2}}]}}
        campaign.validate_protocol_pair(a, copy.deepcopy(a))
        b = copy.deepcopy(a)
        b['preflight']['cases'][0]['initial_state']['x'] = 3
        with self.assertRaises(ValueError):
            campaign.validate_protocol_pair(a, b)

    def test_preview_timeout_is_valid_but_cpu_and_igpu_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episodes/seed_910001'
            campaign.write(path / 'result.json', {'infrastructure_error': False, 'failure_reason': 'timeout',
                'mean_inference_ms': 3.0, 'p95_inference_ms': 4.0})
            runtime = {'device': 'cuda', 'gpu': 'NVIDIA RTX 5090',
                       'opengl': {'vendor': 'NVIDIA Corporation', 'renderer': 'NVIDIA RTX 5090'}}
            campaign.write(path / 'runtime.json', runtime)
            self.assertEqual(campaign.gpu_preview(directory)['failure_reason'], 'timeout')
            for bad in (dict(runtime, device='cpu'), dict(runtime, opengl={'vendor': 'Intel', 'renderer': 'Intel UHD'})):
                campaign.write(path / 'runtime.json', bad)
                with self.assertRaises(ValueError):
                    campaign.gpu_preview(directory)

    def test_child_failure_is_recorded_and_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'campaign'
            plan = {'python': 'python.exe', 'training_configs': {}, 'evaluation_configs': {}}
            supervisor = campaign.Campaign(root, plan)
            with self.assertRaises(FileExistsError):
                campaign.Campaign(root, plan)
            with patch.object(campaign.subprocess, 'Popen') as launch:
                launch.return_value.pid = 123
                launch.return_value.wait.return_value = 9
                with self.assertRaises(RuntimeError):
                    supervisor.command('failed_test', ['-m', 'nothing'])
            state = json.loads((root / 'status.json').read_text(encoding='utf-8'))
            self.assertEqual(state['stages'][-1]['state'], 'failed')
            self.assertEqual(state['stages'][-1]['exit_code'], 9)

    def test_incomplete_50_case_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign.write(root / 'summary.json', {'complete': False, 'planned_episodes': 50, 'completed_episodes': 49})
            campaign.write(root / 'run.json', {'device': 'cuda', 'max_seconds': 60})
            with self.assertRaises(ValueError):
                campaign.evaluation_result(root)

    def test_complete_result_uses_all_query_latencies_and_checks_each_device(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign.write(root / 'summary.json', {'complete': True, 'planned_episodes': 50, 'completed_episodes': 50,
                'infrastructure_errors': [], 'successes': 20, 'failures': 30, 'task_success_rate': .4,
                'failure_reasons': {'timeout': 30}, 'wilson95_given_checkpoint': [.2, .6]})
            campaign.write(root / 'run.json', {'device': 'cuda', 'max_seconds': 60, 'seeds': list(range(50))})
            runtime = {'device': 'cuda', 'gpu': 'NVIDIA RTX 5090',
                       'opengl': {'vendor': 'NVIDIA Corporation', 'renderer': 'RTX 5090'}}
            for seed in range(50):
                case = root / 'episodes' / f'seed_{seed}'
                campaign.write(case / 'result.json', {'seed': seed, 'infrastructure_error': False, 'success': seed < 20})
                campaign.write(case / 'runtime.json', runtime)
                np.savez(case / 'trajectory.npz', inference_ms=[seed + 1., 0.], prediction_query_mask=[True, False])
            result = campaign.evaluation_result(root)
            self.assertEqual(result['successes'], 20)
            self.assertAlmostEqual(result['mean_inference_ms'], 25.5)
            campaign.write(root / 'episodes/seed_49/runtime.json', dict(runtime, device='cpu'))
            with self.assertRaises(ValueError):
                campaign.evaluation_result(root)


if __name__ == '__main__':
    unittest.main()
