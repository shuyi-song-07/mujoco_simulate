"""Discover per-task stage adapters without adding task-ID branches."""
import importlib
import re

STAGES = ('collection', 'dataset', 'training', 'rollout')


def _identity(value):
    if hasattr(value, 'manifest'): value = value.manifest
    if isinstance(value, str):
        value = {'task_id': value}
    if not isinstance(value, dict): raise ValueError('Stage requires a task or frozen task config')
    if isinstance(value.get('task_binding'), dict): value = value['task_binding']
    task_id = value.get('task_id')
    if not isinstance(task_id, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', task_id):
        raise ValueError('Invalid stage task ID')
    controller = value.get('controller')
    if controller is not None and (not isinstance(controller, str) or
            not re.fullmatch(r'[a-z][a-z0-9_]*', controller)):
        raise ValueError('Invalid controller ID')
    return task_id, controller


def stage_adapter(value, stage):
    if stage not in STAGES: raise ValueError(f'Unknown pipeline stage: {stage}')
    task_id, controller = _identity(value)
    name = f'{stage}.{task_id}.adapter'
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        # Errors *inside* an adapter must be reported, never hidden by fallback.
        if error.name not in (name, f'{stage}.{task_id}'): raise
    if controller is None:
        from tasks.registry import get
        controller = get(task_id).manifest['controller']
    name = f'{stage}.common.adapters.{controller}'
    try:
        adapter = importlib.import_module(name)
        if task_id not in getattr(adapter, 'TASK_IDS', (task_id,)):
            raise ValueError(f'{controller} is task-specific; add {stage}/{task_id}/adapter.py')
        return adapter
    except ModuleNotFoundError as error:
        if error.name not in (name, f'{stage}.common.adapters'): raise
        raise ValueError(f'No {stage} adapter for task {task_id!r}; add {stage}/{task_id}/adapter.py') from error
