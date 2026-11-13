"""Select stage implementations; legacy unregistered dual fixtures keep their defaults."""
def adapter(value):
    from tasks.stages import stage_adapter
    config = getattr(value, 'config', value)
    if isinstance(config, dict) and (config.get('task_id') or config.get('task_binding')):
        return stage_adapter(config, 'rollout')
    from rollout.common import dual_task
    return dual_task
