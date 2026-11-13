"""Task-selected executor; historical class imports remain aliases."""
from rollout.common.dual_executor import (EXECUTOR_VERSION, DEFAULT_EXECUTOR_CONFIG,
    PolicyExecutor, PolicyContactControl)
from rollout.push_t.executor import SinglePushPolicyExecutor


def create_policy_executor(controller, config=None):
    from tasks.stages import stage_adapter
    if not getattr(controller, 'config', {}).get('task_binding'):
        # Historical standalone dual-controller callers predate Task Registry.
        return PolicyExecutor(controller, config)
    return stage_adapter(controller.config, 'rollout').create_executor(controller, config)
