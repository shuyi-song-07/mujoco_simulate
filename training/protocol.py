"""Experiment policy only; no changes to ACT or DP mathematics."""
VERSION = 'all_train_fixed_last_v1'


def fixed_last(config):
    value = config.get('experiment_protocol')
    if value not in (None, VERSION):
        raise ValueError(f'Unknown experiment_protocol: {value}')
    return value == VERSION


def validate_training_data(config, binding, *, engineering=False):
    if not fixed_last(config):
        return  # Frozen historical runs remain readable under their original policy.
    split = binding['split']
    if split.get('version') != 'papera-all-train-v1' or split['val'] or split['val_sessions']:
        raise ValueError('This experiment uses all eligible episodes for training. '
                         'Create a new Pack with dataset/manage_dataset.py all-train --pack <old> --output <new>; '
                         'the historical Pack is never changed in place.')
    if not engineering and binding['train_keys'] != split['train']:
        raise ValueError('Formal all-train run cannot select a subset of the Pack')
    if binding['stats']['training_keys'] != binding['train_keys']:
        raise ValueError('Normalization must be fitted on the actual training membership')


def validate_last(config, completion, step, checkpoint):
    """An interrupted last.pt is resumable, but is not a final experiment model."""
    if not fixed_last(config):
        return
    complete = completion.get('completed') is True or completion.get('status') == 'complete'
    budget = completion.get('target_steps', completion.get('budget_steps'))
    if (checkpoint != 'last' or not complete or budget != config['steps']
            or completion.get('step') != config['steps'] or step != config['steps']):
        raise ValueError('Fixed-budget evaluation requires the completed last.pt at exactly config.steps; '
                         'interrupted Last, Best and intermediate candidates are not final models')


def record(config):
    # Deployment defaults only; each evaluation freezes its actual execution settings.
    diffusion = 'prediction_horizon' in config.get('model', {})
    return {'version': VERSION, 'training_data': 'all_eligible_pack_episodes',
            'budget_steps': config['steps'], 'checkpoint_selection': 'completed_last',
            'validation_used_for_selection': False,
            'execution': 'chunked' if diffusion else 'temporal_ensemble',
            'execute_steps': 8 if diffusion else 1,
            'temporal_ensemble_coeff': None if diffusion else .01}
