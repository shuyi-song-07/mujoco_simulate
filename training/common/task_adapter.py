"""Task-dependent model availability; never modifies frozen resume configuration."""
def validate_model(algorithm, supported):
    if algorithm not in supported:
        raise ValueError(f'This task supports {supported}; {algorithm} has not been adapted')


def configure_defaults(config, task, algorithm, supported):
    validate_model(algorithm, supported)
    # This is the same ACT new-run dimension selection formerly in main().
    if algorithm == 'act':
        schema = task.manifest['schema']
        config['model'].update(state_dim=schema['state_dim'], action_dim=schema['action_dim'])
    return config
