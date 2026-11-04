"""Policy report contracts using structural fixtures, not actual policy results."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.summarize_policy_comparison import (validate_group_inputs, paired,
    wilson_interval, verify_rate_and_interval, verify_initial_conditions,
    audit_training_budget, training_identity, execution_diagnostics)
from scripts.plot_policy_comparison import read_metrics, prepare_rows, draw_closed_loop


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class ComparisonContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="policy_comparison_")
        self.root = Path(self.tmp.name)
        self.pack = self.root/'pack'
        write(self.pack/'manifest.json', {'engineering_fixture': True})
        self.sha = hashlib.sha256((self.pack/'manifest.json').read_bytes()).hexdigest()
        self.task = {'task_id': 'engineering_fixture', 'digest': 'fixture'}
        self.binding = {'split': {'train': [1], 'validation': [2]}, 'train_keys': ['a'],
                        'validation_keys': ['b'], 'stats': {'mean': [0]}, 'episodes': ['a', 'b']}
        self.directories = {}
        for algo in ('act', 'dp'):
            for variant in ('baseline', 'gesture'):
                name = algo+'_'+variant
                train, evaluation = self.root/('train_'+name), self.root/('eval_'+name)
                config = {'seed': 1, 'steps': 60, 'batch_size': 2, 'image_size': [640,480],
                          'variant': variant, 'model': {'variant': variant}}
                if algo == 'act':
                    run = {'signature': name, 'runtime': {'engineering_smoke': False,
                           'engineering_overfit': False, 'validation_scope': 'heldout_sessions'},
                           'provenance': {'variant': variant, 'model': {'variant': variant},
                           'pack': str(self.pack), 'task_binding': self.task,
                           'pack_binding': deepcopy(self.binding), 'config': config,
                           'validation_scope': 'heldout_sessions'}}
                    complete = {'completed': True, 'step': 60, 'target_steps': 60,
                                'validation_scope': 'heldout_sessions'}
                else:
                    run = {'algorithm': 'diffusion_policy', 'config': config,
                           'model_config': {'variant': variant}, 'pack_path': str(self.pack)}
                    write(train/'data_binding.json', dict(self.binding, task_binding=self.task))
                    complete = {'status': 'complete', 'step': 60, 'budget_steps': 60, 'engineering': False}
                write(train/'run.json', run)
                write(train/'completion.json', complete)
                policy = {'variant': ('diffusion_gesture' if variant == 'gesture' else 'diffusion')
                          if algo == 'dp' else variant, 'run_dir': str(train), 'pack': str(self.pack),
                          'checked_hashes': {str(self.pack/'manifest.json'): self.sha},
                          'checkpoint': str(train/'checkpoints/last.pt'), 'checkpoint_step': 60,
                          'checkpoint_sha256': 'a'*64}
                if algo == 'dp':
                    policy['algorithm'] = 'diffusion_policy'
                write(evaluation/'run.json', {'run': str(train), 'policy': policy,
                      'scene': {'task_binding': self.task, 'physics': {'mass': 1}}})
                self.directories[name] = evaluation

    def tearDown(self):
        self.tmp.cleanup()

    def validate(self):
        return validate_group_inputs(self.directories, self.root/'out.json')

    def edit(self, path, update):
        doc = json.loads(path.read_text())
        update(doc)
        write(path, doc)

    def test_four_typed_structural_fixtures_accepted(self):
        self.assertEqual(len(self.validate()), 4)

    def test_duplicate_evaluation_rejected(self):
        self.directories['act_gesture'] = self.directories['act_baseline']
        with self.assertRaisesRegex(ValueError, 'distinct evaluation'):
            self.validate()

    def test_swapped_variants_rejected(self):
        a, b = 'act_baseline', 'act_gesture'
        self.directories[a], self.directories[b] = self.directories[b], self.directories[a]
        with self.assertRaisesRegex(ValueError, 'algorithm/variant'):
            self.validate()

    def test_forged_rollout_label_rejected(self):
        self.edit(self.directories['dp_gesture']/'run.json', lambda x: x['policy'].update(variant='diffusion'))
        with self.assertRaisesRegex(ValueError, 'policy identity'):
            self.validate()

    def test_different_pack_rejected(self):
        pack = self.root/'second_pack'
        write(pack/'manifest.json', {'engineering_fixture': True})
        self.edit(self.root/'train_dp_gesture/run.json', lambda x: x.update(pack_path=str(pack)))
        self.edit(self.directories['dp_gesture']/'run.json', lambda x: x['policy'].update(pack=str(pack)))
        with self.assertRaisesRegex(ValueError, 'different Packs'):
            self.validate()

    def test_different_scene_rejected(self):
        self.edit(self.directories['dp_gesture']/'run.json', lambda x: x['scene']['physics'].update(mass=2))
        with self.assertRaisesRegex(ValueError, 'physical scene'):
            self.validate()

    def test_changed_manifest_rejected(self):
        write(self.pack/'manifest.json', {'engineering_fixture': 'changed'})
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            self.validate()

    def test_different_normalization_rejected(self):
        self.edit(self.root/'train_dp_gesture/data_binding.json', lambda x: x['stats'].update(mean=[1]))
        with self.assertRaisesRegex(ValueError, 'stats differs'):
            self.validate()

    def test_different_budget_rejected(self):
        self.edit(self.root/'train_dp_gesture/run.json', lambda x: x['config'].update(steps=61))
        with self.assertRaisesRegex(ValueError, 'not completed'):
            self.validate()

    def continuation(self):
        actual, child = self.root/'train_act_baseline', self.root/'completed_act_baseline'
        doc = json.loads((actual/'run.json').read_text())
        doc['provenance']['config']['steps'] = 90
        doc['provenance']['config']['lr_schedule'] = {'kind':'constant_checkpoint_lr',
            'start_step':60,'start_lrs':[.001,.0001]}
        doc['provenance']['continued_from'] = {'run': str(actual),
            'checkpoint': str(actual/'checkpoints/last.pt'), 'parent_signature': doc['signature'],
            'checkpoint_sha256': 'a'*64, 'parent_step': 60, 'optimizer_learning_rates':[.001,.0001]}
        doc['signature'] = 'completed_act_baseline'
        doc['runtime'].update(continued_checkpoint_sha256='a'*64, starting_step=60,
                              starting_learning_rates=[.001,.0001])
        write(child/'run.json', doc)
        write(child/'completion.json', {'completed': True, 'step': 90, 'target_steps': 90,
              'continued_from_step': 60, 'validation_scope': 'heldout_sessions'})
        return actual, child

    def test_explicit_budget_preserves_actual_parent_checkpoint(self):
        actual, child = self.continuation()
        # Candidate search in the gesture fixture also completed 90 updates.
        self.edit(self.root/'train_act_gesture/run.json', lambda x: x['provenance']['config'].update(steps=90))
        self.edit(self.root/'train_act_gesture/completion.json', lambda x: x.update(step=90,target_steps=90))
        with self.assertRaisesRegex(ValueError, 'completed training budget differs'):
            self.validate()  # Never discover a child implicitly.
        docs = validate_group_inputs(self.directories,self.root/'out.json',{'act_baseline':child})
        audit = docs['act_baseline']['_training_audit']
        self.assertEqual(audit['completed_optimizer_steps'],90)
        self.assertEqual(audit['actual_model_run'],str(actual))
        self.assertEqual(audit['completed_training_run'],str(child))
        self.assertEqual(docs['act_baseline']['policy']['checkpoint_step'],60)

    def test_invalid_continuation_evidence_rejected(self):
        actual, child = self.continuation()
        original_run = json.loads((child/'run.json').read_text())
        original_completion = json.loads((child/'completion.json').read_text())
        metadata = json.loads((self.directories['act_baseline']/'run.json').read_text())
        cases = [
            ('wrong_parent',lambda r,c,m:r['provenance']['continued_from'].update(run=str(self.root/'other'))),
            ('wrong_signature',lambda r,c,m:r['provenance']['continued_from'].update(parent_signature='other')),
            ('wrong_parent_step',lambda r,c,m:r['provenance']['continued_from'].update(parent_step=59)),
            ('wrong_hash',lambda r,c,m:r['provenance']['continued_from'].update(checkpoint_sha256='b'*64)),
            ('selected_best_not_last',lambda r,c,m:m['policy'].update(checkpoint=str(actual/'checkpoints/best.pt'))),
            ('incomplete',lambda r,c,m:c.update(completed=False)),
            ('smoke',lambda r,c,m:r['runtime'].update(engineering_smoke=True)),
            ('model',lambda r,c,m:r['provenance']['model'].update(hidden_dim=32)),
            ('data',lambda r,c,m:r['provenance']['pack_binding']['stats'].update(mean=[3])),
            ('runtime_step',lambda r,c,m:r['runtime'].update(starting_step=59)),
            ('optimizer',lambda r,c,m:r['provenance']['config'].update(learning_rate=.1)),
            ('schedule',lambda r,c,m:r['provenance']['config']['lr_schedule'].update(kind='cosine')),
        ]
        for label, mutate in cases:
            with self.subTest(label=label):
                run, completion, rollout = deepcopy(original_run),deepcopy(original_completion),deepcopy(metadata)
                mutate(run,completion,rollout)
                write(child/'run.json',run);write(child/'completion.json',completion)
                with self.assertRaises(ValueError):
                    audit_training_budget(actual,training_identity(actual),rollout,child)

    def test_output_inside_history_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overwrite an evaluation'):
            validate_group_inputs(self.directories, self.directories['act_baseline']/'report.json')

    def test_initial_physics_field_order_accepted(self):
        a, b = self.root/'a', self.root/'b'
        for path in (a,b):
            write(path/'initial_state.json', {'seed': 1})
        np.savez(a/'initial_physics.npz', qpos=[1.,2.], qvel=[0.])
        np.savez(b/'initial_physics.npz', qvel=[0.], qpos=[1.,2.])
        verify_initial_conditions(a,b)
        np.savez(b/'initial_physics.npz', qvel=[0.], qpos=[1.,3.])
        with self.assertRaisesRegex(ValueError, 'qpos differs'):
            verify_initial_conditions(a,b)
        write(b/'initial_state.json', {'seed': 2})
        with self.assertRaisesRegex(ValueError, 'metadata differs'):
            verify_initial_conditions(a,b)

    def test_continuation_deduplicates_copied_prefix(self):
        parent, child = self.root/'parent', self.root/'child'
        write(parent/'run.json', {})
        write(child/'run.json', {'provenance': {'continued_from': {'run': str(parent), 'parent_step': 2}}})
        first = [{'event':'train','step':1,'total':4.}, {'event':'validation','step':2,'loss':3.}]
        (parent/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in first), encoding='utf-8')
        newer = first + [{'event':'validation','step':2,'loss':2.}, {'event':'train','step':3,'total':1.}]
        (child/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in newer)+'{"event":', encoding='utf-8')
        rows = read_metrics(child)
        self.assertEqual([r['step'] for r in rows], [1,2,3])
        self.assertEqual(rows[1]['loss'], 2.)
        (child/'metrics.jsonl').write_text('{bad}\n'+json.dumps(newer[-1])+'\n', encoding='utf-8')
        with self.assertRaises(json.JSONDecodeError):
            read_metrics(child)

    def test_cyclic_lineage_rejected(self):
        path = self.root/'cycle'
        write(path/'run.json', {'provenance': {'continued_from': {'run': str(path), 'parent_step': 1}}})
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            read_metrics(path)


class StatisticsAndLoss(unittest.TestCase):
    def test_positive_queries_require_grasp_and_lift_evidence(self):
        for evidence in ({}, {'ever_dual': False}, {'lift_verified': False}):
            with self.subTest(evidence=evidence):
                row = {'seed': 1, 'success': False, 'failure_reason': 'timeout',
                       'sim_seconds': 60., 'stage_first_seconds': {}, 'evidence': evidence,
                       'model_queries': 1, 'mean_inference_ms': 10., 'intervention_fraction': 0.}
                with self.assertRaisesRegex(ValueError, 'grasp/lift evidence'):
                    execution_diagnostics([row])

    def test_empty_evidence_is_only_valid_for_first_query_failure(self):
        for success, reason in ((False, 'timeout'), (True, 'success')):
            with self.subTest(success=success, reason=reason):
                row = {'seed': 1, 'success': success, 'failure_reason': reason,
                       'sim_seconds': 0., 'stage_first_seconds': {}, 'evidence': {},
                       'model_queries': 0, 'mean_inference_ms': None, 'intervention_fraction': 0.}
                with self.assertRaisesRegex(ValueError, 'grasp/lift evidence'):
                    execution_diagnostics([row])

    def test_positive_queries_require_finite_latency_evidence(self):
        for latency in (None, float('nan'), float('inf')):
            with self.subTest(latency=latency):
                row = {'seed': 1, 'success': False, 'failure_reason': 'timeout',
                       'sim_seconds': 60., 'stage_first_seconds': {},
                       'evidence': {'ever_dual': False, 'lift_verified': False},
                       'model_queries': 1, 'mean_inference_ms': latency, 'intervention_fraction': 0.}
                with self.assertRaisesRegex(ValueError, 'finite inference latency'):
                    execution_diagnostics([row])

    def test_zero_query_failure_remains_in_execution_diagnostics(self):
        failed = {'seed': 1, 'success': False, 'failure_reason': 'invalid_policy_prediction',
                  'sim_seconds': 0., 'stage_first_seconds': {}, 'evidence': {},
                  'model_queries': 0, 'mean_inference_ms': None, 'intervention_fraction': 0.}
        grasp = dict(failed, seed=2, failure_reason='timeout', sim_seconds=60.,
                     model_queries=2, mean_inference_ms=10.,
                     evidence={'ever_dual': True, 'lift_verified': False})
        lifted = dict(failed, seed=3, failure_reason='timeout', sim_seconds=60.,
                      model_queries=1, mean_inference_ms=40.,
                      evidence={'ever_dual': False, 'lift_verified': True})
        result = execution_diagnostics([failed, grasp, lifted])
        self.assertEqual(result['query_weighted_mean_inference_ms'], 20.)
        self.assertEqual(result['ever_verified_dual_grasp'], 1)
        self.assertEqual(result['ever_verified_lift'], 1)
        self.assertEqual(len(result['per_seed']), 3)
        self.assertEqual(result['per_seed'][0]['failure_reason'], 'invalid_policy_prediction')
        self.assertIsNone(execution_diagnostics([failed])['query_weighted_mean_inference_ms'])

    def test_known_wilson(self):
        np.testing.assert_allclose(wilson_interval(38,50), [0.6258731624205708,0.8570260860300828], rtol=0, atol=1e-14)
        self.assertAlmostEqual(wilson_interval(0,50)[0], 0.)
        self.assertAlmostEqual(wilson_interval(50,50)[1], 1.)

    def test_saved_rate_and_interval_are_recomputed(self):
        outcomes = [True]*38+[False]*12
        group = {'task_success_rate': .76, 'wilson95_given_checkpoint': wilson_interval(38,50)}
        verify_rate_and_interval(group,outcomes)
        with self.assertRaisesRegex(ValueError, 'success rate differs'):
            verify_rate_and_interval(dict(group, task_success_rate=.8),outcomes)
        with self.assertRaisesRegex(ValueError, 'Wilson interval differs'):
            verify_rate_and_interval(dict(group, wilson95_given_checkpoint=[0.,1.]),outcomes)

    def test_exact_paired_test_and_effect(self):
        result = paired([False]*6,[True]*6)
        self.assertEqual(result['exact_mcnemar_two_sided_p'], .03125)
        self.assertEqual(result['success_rate_difference_gesture_minus_baseline'], 1.)
        self.assertEqual(paired([True,False],[True,False])['exact_mcnemar_two_sided_p'], 1.)
        with self.assertRaises(ValueError):
            paired([True],[True,False])
        with self.assertRaises(ValueError):
            paired([1],[True])

    def test_dp_total_includes_auxiliary(self):
        row = {'event':'train','step':1,'noise_mse':.2,'gesture_loss':.3}
        config = {'model':{'gesture_weight':.1}}
        self.assertAlmostEqual(prepare_rows([row],'dp','gesture',config)[0]['total'], .23)
        self.assertAlmostEqual(prepare_rows([row],'dp','baseline',config)[0]['total'], .2)
        self.assertNotIn('total', row)
        with self.assertRaisesRegex(ValueError, 'requires its logged auxiliary'):
            prepare_rows([{'event':'train','noise_mse':.2}],'dp','gesture',config)
        with self.assertRaisesRegex(ValueError, 'total differs'):
            prepare_rows([dict(row,total_loss=.2)],'dp','gesture',config)


class ClosedLoopRendering(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='policy_plot_fixture_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'synthetic_comparison_fixture.json'
        names = ('act_baseline', 'act_gesture', 'dp_baseline', 'dp_gesture')
        self.document = {
            'version': 'papera-four-group-comparison-v1', 'seeds': list(range(50)),
            'fixture_note': 'Synthetic rendering fixture, deleted on test exit; not policy results.',
            'groups': {name: {'successes': count, 'failures': 50-count,
                'task_success_rate': count/50, 'wilson95_given_checkpoint': wilson_interval(count,50),
                'failure_reasons': {'timeout': 50-count}, 'initial_physics_equal': True,
                'cuda_and_nvidia_checked': True} for name,count in zip(names,(0,1,25,50))}}

    def test_engineering_reports_are_rejected_before_rendering(self):
        for marker in ('engineering_self_check', 'not_four_group_results'):
            with self.subTest(marker=marker):
                write(self.source, dict(self.document, **{marker: True}))
                with self.assertRaisesRegex(ValueError, 'Engineering'):
                    draw_closed_loop(self.source, self.root)
                self.assertFalse((self.root/'closed_loop_comparison.png').exists())
                self.assertFalse((self.root/'closed_loop_comparison.pdf').exists())

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Rendering requires the collection environment')
    def test_zero_success_wilson_boundary_renders_png_and_pdf(self):
        write(self.source, self.document)
        # Render through the real interface, including matplotlib error bars.
        try:
            draw_closed_loop(self.source, self.root)
            self.assertTrue((self.root/'closed_loop_comparison.png').read_bytes().startswith(b'\x89PNG'))
            self.assertTrue((self.root/'closed_loop_comparison.pdf').read_bytes().startswith(b'%PDF'))
        finally:
            import matplotlib.pyplot as plt
            plt.close('all')


if __name__ == '__main__':
    unittest.main(verbosity=2)
