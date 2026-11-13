"""Controller dispatch from an explicitly validated task schema."""
def create_controller(config, **kwargs):
    from tasks.stages import stage_adapter
    if not config.get('task_binding'):
        from collection.common.adapters.dual_panda_v1 import create_controller as legacy
        return legacy(config, **kwargs)
    return stage_adapter(config, 'collection').create_controller(config, **kwargs)


def create_model(config):
    from tasks.stages import stage_adapter
    if not config.get('task_binding'):
        from collection.common.adapters.dual_panda_v1 import create_model as legacy
        return legacy(config)
    return stage_adapter(config, 'collection').create_model(config)
