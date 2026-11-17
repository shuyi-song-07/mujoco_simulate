"""Task-dependent model availability; never modifies frozen resume configuration."""
def validate_model(algorithm, supported):
    if algorithm not in supported:
        raise ValueError(f'This task supports {supported}; {algorithm} has not been adapted')


def configure_defaults(config, task, algorithm, supported):
    validate_model(algorithm, supported)
    # Only observation/action projections depend on the robot. Frozen resumes
    # never call this new-run default resolver.
    from tasks.data_schema import schema_for
    schema = schema_for(task.manifest)
    config['model'].update(state_dim=schema.state_dim, action_dim=schema.action_dim)
    return config
