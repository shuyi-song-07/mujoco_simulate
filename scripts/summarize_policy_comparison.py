"""Audit and summarize four frozen 50-case policy evaluations (read-only).

Uses the existing evaluation audit and compares physical initial snapshots.
Does not select checkpoints, change outcomes, or treat validation as test data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_baseline_campaign import evaluation_result, read, write, require


def training_identity(directory):
    """Read the frozen training identity; CLI argument names are not evidence."""
    directory = Path(directory).resolve()
    run = read(directory/'run.json')
    if run.get('algorithm') == 'diffusion_policy':
        config = run['config']
        variant = config.get('variant', config['model'].get('variant', 'baseline'))
        require(config['model'].get('variant', variant) == variant
                and run['model_config'].get('variant', 'baseline') == variant,
                'DP training variant fields disagree')
        binding = read(directory/'data_binding.json')
        algorithm, pack = 'dp', run['pack_path']
        task_binding = binding['task_binding']
    else:
        provenance = run['provenance']
        variant, algorithm = provenance['variant'], 'act'
        require(provenance['model']['variant'] == variant, 'ACT training variant fields disagree')
        pack, task_binding = provenance['pack'], provenance['task_binding']
        binding = provenance['pack_binding']
        config = provenance['config']
    require(variant in ('baseline', 'gesture'), 'Unsupported training variant')
    return {'algorithm': algorithm, 'variant': variant, 'pack': Path(pack).resolve(),
            'task_binding': task_binding, 'binding': binding, 'config': config}


def completed_steps(directory, identity):
    """Budget evidence comes from completion, never from a directory name."""
    run, completion = read(directory/'run.json'), read(directory/'completion.json')
    step = completion['step']
    require(type(step) is int and step > 0 and step == identity['config']['steps'],
            'Training has not completed its configured budget')
    if identity['algorithm'] == 'act':
        require(completion.get('completed') is True and completion['target_steps'] == step,
                'ACT training is not complete')
        for record in (run['runtime'], run['provenance'], completion):
            require(record.get('engineering_smoke', False) is False
                    and record.get('engineering_overfit', False) is False
                    and record.get('validation_scope') == 'heldout_sessions',
                    'ACT budget evidence must be full-data heldout training')
    else:
        require(completion.get('status') == 'complete' and completion['budget_steps'] == step
                and completion.get('engineering') is False, 'DP training is not formally complete')
    return step


def audit_training_budget(actual, identity, rollout, completed=None):
    """Allow a deployed parent last only with explicit, verified ACT lineage."""
    actual = Path(actual).resolve()
    actual_steps = completed_steps(actual, identity)
    completed = actual if completed is None else Path(completed).resolve()
    evidence = None
    budget = actual_steps
    if completed != actual:
        child = training_identity(completed)
        require(identity['algorithm'] == child['algorithm'] == 'act', 'Completed-run override is ACT-only')
        require(identity['variant'] == child['variant'] and identity['pack'] == child['pack']
                and identity['task_binding'] == child['task_binding']
                and identity['binding'] == child['binding'], 'Continuation variant/Pack/data binding differs')
        parent_doc, child_doc = read(actual/'run.json'), read(completed/'run.json')
        parent_prov, child_prov = parent_doc['provenance'], child_doc['provenance']
        require(parent_prov['model'] == child_prov['model'], 'Continuation model differs')
        # Continuation intentionally adds candidate persistence and freezes the
        # checkpoint LR; verify that schedule separately instead of ignoring it.
        ignore = {'steps', 'output_root', '_help', 'candidate_save_every', 'lr_schedule'}
        require({k:v for k,v in identity['config'].items() if k not in ignore}
                == {k:v for k,v in child['config'].items() if k not in ignore},
                'Continuation training configuration differs')
        lineage = child_prov.get('continued_from', {})
        require(lineage and Path(lineage['run']).resolve() == actual
                and lineage['parent_signature'] == parent_doc['signature'],
                'Completed run does not continue this exact parent run')
        policy = rollout['policy']
        last = actual/'checkpoints/last.pt'
        require(Path(lineage['checkpoint']).resolve() == last
                and Path(policy['checkpoint']).resolve() == last
                and lineage['parent_step'] == policy['checkpoint_step'] == actual_steps,
                'Completed-run override requires the exact parent last checkpoint/step')
        sha = lineage['checkpoint_sha256']
        require(isinstance(sha, str) and len(sha) == 64 and policy['checkpoint_sha256'] == sha,
                'Evaluated parent checkpoint hash differs from continuation input')
        runtime, completion = child_doc['runtime'], read(completed/'completion.json')
        require(runtime['continued_checkpoint_sha256'] == sha
                and runtime['starting_step'] == completion['continued_from_step'] == actual_steps,
                'Continuation runtime/completion lineage differs')
        schedule = child['config'].get('lr_schedule', {})
        require(schedule.get('kind') == 'constant_checkpoint_lr'
                and schedule.get('start_step') == actual_steps
                and schedule.get('start_lrs') == lineage.get('optimizer_learning_rates')
                == runtime.get('starting_learning_rates')
                and bool(schedule.get('start_lrs')),
                'Continuation does not preserve recorded checkpoint learning rates')
        budget = completed_steps(completed, child)
        require(budget > actual_steps, 'Completed continuation did not extend the parent budget')
        evidence = {'parent_step': actual_steps, 'parent_signature': lineage['parent_signature'],
                    'parent_last_checkpoint_sha256': sha, 'completed_target_steps': budget}
    return {'actual_model_run': str(actual), 'completed_training_run': str(completed),
            'completed_optimizer_steps': budget, 'continuation_budget_evidence': evidence}


def validate_group_inputs(directories, output, completed_runs=None):
    """Reject mislabeled, repeated or different-task evaluations before reporting."""
    paths = [Path(path).resolve() for path in directories.values()]
    require(len(set(paths)) == len(paths), 'Each group requires a distinct evaluation directory')
    output = Path(output).resolve()
    metadata, identities, train_paths, budget_paths = {}, {}, [], []
    completed_runs = {} if completed_runs is None else completed_runs
    common_pack = common_scene = common_task = None
    for name, directory in zip(directories, paths):
        require(not output.is_relative_to(directory), 'Comparison output must not overwrite an evaluation')
        document = read(directory/'run.json')
        train = Path(document['run']).resolve()
        require(not output.is_relative_to(train), 'Comparison output must not overwrite a training run')
        identity = training_identity(train)
        algorithm, variant = name.split('_', 1)
        require((identity['algorithm'], identity['variant']) == (algorithm, variant),
                f'{name}: supplied training algorithm/variant does not match this group')
        policy = document['policy']
        expected_variant = ('diffusion_gesture' if variant == 'gesture' else 'diffusion') if algorithm == 'dp' else variant
        require(policy['variant'] == expected_variant
                and (policy.get('algorithm') == 'diffusion_policy') == (algorithm == 'dp'),
                f'{name}: rollout policy identity differs from its training run')
        require(Path(policy['run_dir']).resolve() == train
                and Path(policy['pack']).resolve() == identity['pack'],
                f'{name}: rollout Pack/run differs from its training run')
        require(document['scene']['task_binding'] == identity['task_binding'],
                f'{name}: rollout scene differs from its frozen training task')
        if common_pack is None:
            common_pack, common_scene, common_task = identity['pack'], document['scene'], identity['task_binding']
        require(identity['pack'] == common_pack and identity['task_binding'] == common_task,
                'Evaluation groups use different Packs or tasks')
        require(document['scene'] == common_scene, 'Frozen physical scene/controller/camera configurations differ')
        manifest = identity['pack']/'manifest.json'
        frozen = {Path(path).resolve(): sha for path, sha in policy['checked_hashes'].items()}
        require(frozen.get(manifest) == hashlib.sha256(manifest.read_bytes()).hexdigest(),
                f'{name}: Pack manifest differs from the evaluated identity')
        audit = audit_training_budget(train, identity, document, completed_runs.get(name))
        require(not output.is_relative_to(Path(audit['completed_training_run'])),
                'Comparison output must not overwrite a completed training run')
        document['_training_audit'] = audit
        train_paths.append(train)
        budget_paths.append(Path(audit['completed_training_run']))
        metadata[name], identities[name] = document, identity
    require(len(set(train_paths)) == len(train_paths), 'Each group requires a distinct training run')
    require(len(set(budget_paths)) == len(budget_paths), 'Each group requires distinct completed training')
    for algorithm in ('act', 'dp'):
        left, right = (identities[algorithm+'_'+variant] for variant in ('baseline', 'gesture'))
        for key in ('split', 'train_keys', 'validation_keys', 'stats', 'episodes'):
            require(left['binding'][key] == right['binding'][key], f'{algorithm}: training data {key} differs')
        for key in ('seed', 'batch_size', 'image_size'):
            require(left['config'][key] == right['config'][key], f'{algorithm}: training {key} differs')
        require(metadata[algorithm+'_baseline']['_training_audit']['completed_optimizer_steps']
                == metadata[algorithm+'_gesture']['_training_audit']['completed_optimizer_steps'],
                f'{algorithm}: completed training budget differs')
    return metadata


def wilson_interval(successes, count):
    require(type(count) is int and count > 0 and 0 <= successes <= count, 'Invalid Wilson counts')
    z, rate = 1.959963984540054, successes/count
    denominator = 1 + z*z/count
    center = (rate + z*z/(2*count))/denominator
    radius = z*math.sqrt(rate*(1-rate)/count + z*z/(4*count*count))/denominator
    return [max(0., center-radius), min(1., center+radius)]


def verify_rate_and_interval(group, outcomes):
    count, successes = len(outcomes), sum(outcomes)
    interval = wilson_interval(successes, count)
    require(math.isclose(group['task_success_rate'], successes/count, rel_tol=0, abs_tol=1e-12),
            'Saved success rate differs from individual outcomes')
    require(np.asarray(group['wilson95_given_checkpoint']).shape == (2,)
            and np.allclose(group['wilson95_given_checkpoint'], interval, rtol=0, atol=1e-12),
            'Saved Wilson interval differs from individual outcomes')
    group.update(task_success_rate=successes/count, wilson95_given_checkpoint=interval)


def verify_initial_conditions(reference, candidate):
    require(read(reference/'initial_state.json') == read(candidate/'initial_state.json'),
            'Initial-state metadata differs')
    with np.load(reference/'initial_physics.npz', allow_pickle=False) as left, \
         np.load(candidate/'initial_physics.npz', allow_pickle=False) as right:
        require(set(left.files) == set(right.files), 'Initial physics fields differ')
        for key in left.files:
            require(np.array_equal(left[key], right[key]), f'Initial physics {key} differs')


def paired(reference, candidate):
    require(len(reference) == len(candidate) and len(reference) > 0,
            'Paired outcomes must have the same nonzero case count')
    require(all(type(value) is bool for value in (*reference, *candidate)), 'Paired outcomes must be boolean')
    improved = sum(not a and b for a,b in zip(reference,candidate))
    worsened = sum(a and not b for a,b in zip(reference,candidate))
    n = improved+worsened
    p = min(1., 2*sum(math.comb(n,k) for k in range(min(improved,worsened)+1))/2**n) if n else 1.
    return {'baseline_failure_gesture_success': improved,
            'baseline_success_gesture_failure': worsened,
            'same_success': sum(a and b for a,b in zip(reference,candidate)),
            'same_failure': sum(not a and not b for a,b in zip(reference,candidate)),
            'success_rate_difference_gesture_minus_baseline': (improved-worsened)/len(reference),
            'exact_mcnemar_two_sided_p': p,
            'scope': 'Paired initial states for one trained checkpoint per group; not training-seed uncertainty'}


def execution_diagnostics(outcomes):
    """Separate ever-reached stages, final conditions and outcome statistics."""
    phases = sorted({key for row in outcomes for key in row['stage_first_seconds']})
    conditions = sorted({key for row in outcomes for key in row['evidence'].get('final_conditions', {})})
    successful_times = [row['sim_seconds'] for row in outcomes if row['success']]
    queried = [row for row in outcomes if row['model_queries'] > 0]
    for row in outcomes:
        first_query_failure = (row['model_queries'] == 0 and row['success'] is False
                               and row['failure_reason'] == 'invalid_policy_prediction'
                               and row['evidence'] == {})
        require(first_query_failure or all(type(row['evidence'].get(key)) is bool
                                          for key in ('ever_dual', 'lift_verified')),
                'Missing grasp/lift evidence requires a zero-query invalid-policy failure')
    require(all(isinstance(row.get('mean_inference_ms'), (int, float))
                and math.isfinite(row['mean_inference_ms']) for row in queried),
            'Positive model queries require finite inference latency evidence')
    queries = sum(row['model_queries'] for row in queried)
    return {
        'ever_reached_stage_counts': {key: sum(row['stage_first_seconds'].get(key) is not None
                                               for row in outcomes) for key in phases},
        'ever_verified_dual_grasp': sum(row['evidence'].get('ever_dual', False) is True for row in outcomes),
        'ever_verified_lift': sum(row['evidence'].get('lift_verified', False) is True for row in outcomes),
        'final_condition_counts': {key: sum(row['evidence'].get('final_conditions', {}).get(key) is True
                                            for row in outcomes) for key in conditions},
        'final_condition_note': 'Conditions at termination, not the number that ever met each condition.',
        'successful_sim_seconds_median': float(np.median(successful_times)) if successful_times else None,
        'material_intervention_fraction_mean': float(np.mean([row['intervention_fraction'] for row in outcomes])),
        'query_weighted_mean_inference_ms': (sum(row['mean_inference_ms']*row['model_queries']
                                                for row in queried)/queries) if queries else None,
        'inference_timing_scope': 'Observed workload; concurrent training/rendering can affect wall-clock latency. Not an isolated speed benchmark.',
        'per_seed': [{'seed': row['seed'], 'success': row['success'],
                      'failure_reason': row['failure_reason'], 'sim_seconds': row['sim_seconds']}
                     for row in outcomes],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ('act_baseline','act_gesture','dp_baseline','dp_gesture')
    for name in names:
        parser.add_argument('--'+name.replace('_','-'), type=Path, required=True)
    for variant in ('baseline', 'gesture'):
        parser.add_argument('--act-'+variant+'-completed-run', type=Path,
                            help='Explicit completed ACT child when deploying its exact parent last checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--decode-videos', action='store_true', help='Decode all saved observation videos again; no file hashes')
    args = parser.parse_args()
    directories = {name: getattr(args, name).resolve() for name in names}
    completed_runs = {name: getattr(args, name+'_completed_run') for name in ('act_baseline','act_gesture')}
    audited_metadata = validate_group_inputs(directories, args.output, completed_runs)
    result = {'version':'papera-four-group-comparison-v1', 'groups':{}, 'paired':{},
              'formal_multi_seed_paper_claim':False,
              'interval_scope':'95% Wilson intervals conditional on each trained checkpoint; not uncertainty across training seeds',
              'paired_p_value_scope':'Two descriptive exact McNemar comparisons; not multiplicity-adjusted'}
    seeds = None
    success = {}
    reference = None
    expected_rules = None
    for name in names:
        directory = directories[name]
        metadata = audited_metadata[name]
        group = evaluation_result(directory)
        if seeds is None:
            seeds, reference = metadata['seeds'],directory
            expected_rules = {k:metadata['settings'][k] for k in (
                'minimum_lift_m','lift_stable_seconds','success_open_fraction',
                'success_criterion','task_success_criterion','max_seconds')}
        require(metadata['seeds']==seeds, 'Evaluation initial-state seeds differ')
        require({k:metadata['settings'][k] for k in expected_rules}==expected_rules,
                'Task success rules differ')
        success[name] = []
        outcomes = []
        for seed in seeds:
            case = directory/'episodes'/f'seed_{seed}'
            outcome = read(case/'result.json')
            require(type(outcome['success']) is bool, 'Success outcome must be boolean')
            success[name].append(outcome['success'])
            outcomes.append(outcome)
            runtime = read(case/'runtime.json')['policy_metadata']
            for key in ('checkpoint_sha256', 'checkpoint_step', 'variant', 'action_horizon', 'inference_steps'):
                require(runtime.get(key) == metadata['policy'].get(key), f'{name}/{seed}: runtime policy {key} differs')
            media = outcome.get('observation_media',outcome.get('media',{}))
            require(media.get('complete_decode') is True and media.get('decoded_frames',0)>0,
                    f'{name}/{seed}: rollout did not record a complete video decode')
            require((media.get('width'),media.get('height'))==(640,480), 'Wrong observation video size')
            video = case/'oblique.mp4'
            require(video.is_file() and video.stat().st_size>0, 'Observation video missing or empty')
            if args.decode_videos:
                import av
                with av.open(str(video)) as container:
                    count=0
                    for frame in container.decode(video=0):
                        require((frame.width,frame.height)==(640,480), 'Decoded observation size differs')
                        count+=1
                require(count==media['decoded_frames'], 'Decoded video count differs from saved rollout')
            if outcome['success']:
                evidence = outcome['evidence']
                require(evidence['ever_dual'] and evidence['lift_verified'] and evidence['task_success'],
                        'Reported success lacks required physical history')
                require(evidence['final_conditions'] and all(evidence['final_conditions'].values()),
                        'Reported success has missing/false final condition')
            verify_initial_conditions(reference/'episodes'/f'seed_{seed}', case)
        verify_rate_and_interval(group, success[name])
        group.update(run=metadata['run'], checkpoint_step=metadata['policy']['checkpoint_step'],
                     checkpoint=metadata['checkpoint'],settings=metadata['settings'],
                     cuda_and_nvidia_checked=True, initial_physics_equal=True,
                     video_decode_evidence_checked=True, videos_decoded_again=args.decode_videos)
        group.update(metadata['_training_audit'])
        group['execution_diagnostics'] = execution_diagnostics(outcomes)
        result['groups'][name]=group
    result['seeds']=seeds
    for algorithm in ('act','dp'):
        a,b = (result['groups'][algorithm+'_'+v]['settings'] for v in ('baseline','gesture'))
        for key in ('execution','n_action_steps','observation_hz','control_hz','chunk_size',
                    'diffusion_inference_steps','diffusion_action_horizon'):
            require(a.get(key)==b.get(key), f'{algorithm} baseline/gesture execution {key} differs')
        result['paired'][algorithm]=paired(success[algorithm+'_baseline'],success[algorithm+'_gesture'])
    write(args.output,result)
    print(json.dumps({name:{k:g[k] for k in ('successes','failures','task_success_rate','checkpoint_step')}
                      for name,g in result['groups'].items()},indent=2))


if __name__=='__main__':
    main()
